import argparse
import glob
import itertools
import json
import logging
import os
import pdb
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders import MyDataset147
from losses import (MultiClassDiceLoss)
from models import get_model
from options import get_options
from trainers import (pmt_Trainer)
from utils import IterationCounter


class pmt_Trainer():
    def __init__(self, opt, tunelayer=None):
        self.opt = opt
        self.tunelayer = tunelayer
    
    def initialize(self):
        self.set_seed(self.opt['random_seed'])

        ### initialize dataloaders
        ##MyDataset147_hist
        self.train_dataloader = DataLoader(
            MyDataset147(self.opt['data_root'], self.opt['target_sites'], dataset_name=self.opt['dataset_name'], phase='train'),
            batch_size=self.opt['batch_size'],
            shuffle=True,
            drop_last=True,
            num_workers=self.opt['num_workers']
        )

        print('Length of training dataset: ', len(self.train_dataloader))

        self.val_dataloader = DataLoader(
            MyDataset147(self.opt['data_root'], self.opt['target_sites'], dataset_name=self.opt['dataset_name'], phase='val'),
            batch_size=self.opt['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=4
        )

        print('Length of validation dataset: ', len(self.val_dataloader))

        ## initialize the models

        self.model = get_model(self.opt)
        checkpoint = torch.load(self.opt['source_model_path'],map_location='cpu')
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model = self.model.to(self.opt['gpu_id'])

        for name, param in self.model.named_parameters(): 
            param.requires_grad = False

        self.total_epochs = self.opt['total_epochs']
        p = self.opt['source_model_path']
        print(f'loading {p}')

        ## losses
        self.criterian_dc  = MultiClassDiceLoss(self.opt)

        ## metrics
        self.best_avg_dice = 0

        # visualizations
        self.iter_counter = IterationCounter(self.opt)
        
        self.model_resume()

    
    def set_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print('Random seed for this experiment is {} !'.format(seed))

    def save_models(self, step, dice):
        # if step != 0:
        checkpoint_dir = self.opt['checkpoint_dir']
        state = {'model': self.model.state_dict()}
        print('saving', os.path.join(checkpoint_dir, 'saved_models', 'model_step_{}_dice_{:.4f}.pth'.format(step,dice)))
        torch.save(state, os.path.join(checkpoint_dir, 'saved_models', 'model_step_{}_dice_{:.4f}.pth'.format(step,dice)))

    def model_resume(self):
        if self.opt['continue_train']:
            if os.path.isfile(self.opt['resume']):
                print("=> Loading checkpoint '{}'".format(self.opt['resume']))
            state = torch.load(self.opt['resume'])
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.start_epoch = state['epoch']
        else:
            self.start_epoch = 0
            print("=> No checkpoint, train from scratch !")

    ###################### training logic ################################
    def train_one_step(self, data, epoch):
        bn_cnt = 0
        self.model.set_hook()
        for name, param in self.model.named_modules():
            if hasattr(param, 'running_mean') :

                # if bn_cnt <20:#>= 8 or bn_cnt < 2: # == self.tunelayer:#  
                print(f'training on {name}')
                bn_cnt += 1
                continue

        imgs = data[0]

        target_f, predict, _, _ = self.model(imgs)

        loss_dc, _, _ = self.criterian_dc(predict, data[1])
        bn_loss,_,_ = self.model.get_BNLoss()
        print(bn_loss)

        return predict, bn_loss.detach()
    
    @torch.no_grad()
    def validate_one_step(self, data):
        self.model.eval()
    
        imgs = data[0]
        feat, predict, _, _ = self.model(imgs)

        self.model.train()

        return predict, feat, 

    def launch(self):
        self.initialize()
        self.train()
        
    def train(self):

        val_predicts = []
        val_gts = []
        val_iterator = tqdm((self.val_dataloader), total = len(self.val_dataloader))

        best_dice = 0
        val_dice = []
        total_train_dice = []
        total_bn_loss = []
        total_proto_gt = []
        total_proto_pred = []

        for epoch in range(self.start_epoch,self.total_epochs):
            train_iterator = tqdm((self.train_dataloader), total = len(self.train_dataloader))

            for it, (images,segs,img_name) in enumerate(train_iterator):

                images = images.to(self.opt['gpu_id'])
                segs = segs.to(self.opt['gpu_id'])
                
                print(it)
                predicts, bn_losses  = self.train_one_step([images, segs], it)

                train_dice, train_dice_wobg, train_cls_wise = self.criterian_dc(torch.tensor(predicts), torch.tensor(segs))
                train_dice = 1 - train_dice.item()
                train_dice_wobg = 1 - train_dice_wobg.item()
                total_train_dice.append(train_dice_wobg)
                total_bn_loss.append(bn_losses.detach().cpu().numpy())
                logging.info(f'Trgt bn train loss {bn_losses}, dice {train_dice}, dice wo bg {train_dice_wobg}, cls wise {train_cls_wise}')

                trgt_predicts = []
                trgt_gts = []
                trgt_feat = []
                proto_center_gt = np.zeros((5, 64))
                proto_center_pred = np.zeros((5, 64))
                val_iterator = tqdm((self.val_dataloader), total = len(self.val_dataloader))

                if it == 10:
                    for _, (val_imgs, val_segs, val_names) in enumerate(val_iterator):

                        val_imgs = val_imgs.to(self.opt['gpu_id'])
                        val_segs = val_segs.to(self.opt['gpu_id'])

                        predict, feat = self.validate_one_step([val_imgs, val_segs])
                        
                        trgt_predicts.append(predict.detach().cpu().numpy())
                        trgt_gts.append(val_segs.detach().cpu().numpy())
                        trgt_feat.append(feat.detach().cpu().numpy())

                    trgt_predicts = np.concatenate(trgt_predicts,axis=0) # N, 5, 256, 256
                    trgt_gts = np.concatenate(trgt_gts,axis=0) # N 256， 256
                    trgt_feat = np.concatenate(trgt_feat,axis=0) #  (N, 64, 256, 256)

                    trgt_predicts_fla = torch.softmax(torch.tensor(trgt_predicts), 1)
                    trgt_predicts_fla = torch.argmax(trgt_predicts_fla, 1)
                    trgt_predicts_fla = rearrange(trgt_predicts_fla, 'b h w -> (b h w)')
                    dice, dice_wobg, cls_wise = self.criterian_dc(torch.tensor(trgt_predicts), torch.tensor(trgt_gts))
                    dice = 1 - dice.item()
                    dice_wobg = 1 - dice_wobg.item()
                    val_dice.append(dice_wobg)
                    logging.info(f'Trgt val new dice loss {dice}, dice wo bg {dice_wobg}, cls wise {cls_wise}')
                    if dice_wobg > best_dice:
                        logging.info(f'📀 {best_dice} -> {dice_wobg}')
                        best_dice = dice_wobg

                if it > 10:
                    break

        self.save_models(self.iter_counter.steps_so_far,dice_wobg)

        val_dice = np.array(val_dice)
        total_train_dice = np.array(total_train_dice)
        total_bn_loss = np.array(total_bn_loss)

        
        print(f'saving npy')

def ensure_dirs(opt, tunelayer=None):
 
    checkpoints_dir = opt['checkpoints_dir']
    
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
        
    curr_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    if opt['dev']:
        exp_name = 'dev'
    else:
        exp_name = 'T{}_tuneall_epoch10'.format(curr_time)

    opt['checkpoint_dir'] = os.path.join(opt['checkpoints_dir'],exp_name)
    checkpoint_dir = opt['checkpoint_dir']

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        with open(os.path.join(checkpoint_dir,'config.json'),'w') as f:
            json.dump(opt,f)

    root_logger = logging.getLogger()
    for h in root_logger.handlers:
        root_logger.removeHandler(h)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(f'{checkpoint_dir}/train.log'), logging.StreamHandler(sys.stdout)])
    logging.info(str(opt))
    
    if not os.path.exists(os.path.join(checkpoint_dir,'console_logs')):
        os.makedirs(os.path.join(checkpoint_dir,'console_logs'))

    if not os.path.exists(os.path.join(checkpoint_dir, 'tf_logs')):
        os.makedirs(os.path.join(checkpoint_dir, 'tf_logs'))

    if not os.path.exists(os.path.join(checkpoint_dir, 'saved_models')):
        os.makedirs(os.path.join(checkpoint_dir, 'saved_models'))

    if not os.path.exists(os.path.join(checkpoint_dir, 'visuals')):
        os.makedirs(os.path.join(checkpoint_dir, 'visuals'))
        
    if not os.path.exists(os.path.join(checkpoint_dir, 'source_codes')):
        os.makedirs(os.path.join(checkpoint_dir, 'source_codes'))
        
        source_folders = ['.']
        sources_to_save = list(itertools.chain.from_iterable(
            [glob.glob(f'{folder}/*.py') for folder in source_folders]))
        sources_to_save.extend(['./dataloaders', './models','./losses','./trainers','./utils'])
        for source_file in sources_to_save:
            if os.path.isfile(source_file):
                shutil.copy(source_file,os.path.join(checkpoint_dir, 'source_codes'))
            if os.path.isdir(source_file):
                if os.path.exists(os.path.join(checkpoint_dir, 'source_codes', source_file)):
                    os.removedirs(os.path.join(checkpoint_dir, 'source_codes', source_file))
                shutil.copytree(source_file,os.path.join(checkpoint_dir, 'source_codes', source_file),ignore=shutil.ignore_patterns('__pycache__'))
                

if __name__ == '__main__':
    import argparse

    import yaml

    from options import get_options

    parser = argparse.ArgumentParser(description='Adapt Source Model on Target Images')
    
    parser.add_argument("--config_file",type=str)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--dev", default=False, action='store_true')
    parser.add_argument("--note", default='', type=str)
    
    
    opt = vars(parser.parse_args())
    opt['config_file'] = 'configs/train_target_adapt_bn.yaml'
    with open(opt['config_file']) as f:
        config = yaml.safe_load(f)
        
    opt.update(config)
    opt["gpu_id"] = "cuda:%s"%opt["gpu_id"]
    opt['checkpoints_dir'] = os.path.join(opt['save_root'],opt['experiment_name'])
    opt['img_size'] = tuple(opt['img_size'])

    ensure_dirs(opt)
    trainer = pmt_Trainer(opt)
    trainer.launch()
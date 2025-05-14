import torch
import os,random
from models import get_model

from dataloaders import MyDataset147
from torch.utils.data import DataLoader
from losses import MultiClassDiceLoss, PixelCELoss

import numpy as np

from utils import IterationCounter, Visualizer, mean_dice

from tqdm import tqdm
import pdb

import logging
import psutil


class SourceDomainTrainer():
    def __init__(self, opt):
        self.opt = opt
    
    def initialize(self):
        self.set_seed(self.opt['random_seed'])
        ### 1. initialize dataloaders
        self.train_dataloader = DataLoader(
            MyDataset147(self.opt['data_root'], self.opt['source_sites'], dataset_name=self.opt['dataset_name'], phase='train'),
            batch_size=self.opt['batch_size'],
            shuffle=True,
            drop_last=True,
            num_workers=self.opt['num_workers']
        )
        print('Length of training dataset: ', len(self.train_dataloader))

        self.val_dataloader = DataLoader(
            MyDataset147(self.opt['data_root'], self.opt['source_sites'], dataset_name=self.opt['dataset_name'], phase='val'),
            batch_size=self.opt['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=4
        )
        print('Length of validation dataset: ', len(self.val_dataloader))

        self.trgt_val_dataloader = DataLoader(
            MyDataset147(self.opt['data_root'], self.opt['target_sites'], dataset_name=self.opt['dataset_name'], phase='val'),
            batch_size=self.opt['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=4
        )
        print('Length of trgt validation dataset: ', len(self.trgt_val_dataloader))
        
        ## 2. initialize the models
        self.model = get_model(self.opt)
        self.model = self.model.to(self.opt['gpu_id'])
        self.total_epochs = self.opt['total_epochs']

        ## 3. optimizers, schedulars
        self.optimizer, self.schedular = self.get_optimizers()
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

        ## losses
        self.criterian_pce = PixelCELoss(self.opt)
        self.criterian_dc  = MultiClassDiceLoss(self.opt)
        ## metrics
        self.best_avg_dice = 0

        # visualizations
        self.iter_counter = IterationCounter(self.opt)
        # self.set_seed(self.opt['random_seed'])
        self.model_resume()
    
    def set_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print('Random seed for this experiment is {} !'.format(seed))

    def save_models(self, epoch, dice):
        if epoch != 0:
            checkpoint_dir = self.opt['checkpoint_dir']
            state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, os.path.join(checkpoint_dir, 'saved_models', 'model_epoch_{}_dice_{:.4f}.pth'.format(epoch,dice)))

    
    def save_best_models(self, epoch, dice):
        checkpoint_dir = self.opt['checkpoint_dir']
        state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch}
        torch.save(state,os.path.join(checkpoint_dir, 'saved_models','best_model_epoch_{}_dice_{:.4f}.pth'.format(epoch,dice)))


    def get_optimizers(self):
        params = list(self.model.parameters())
        optimizer = torch.optim.Adam(params,lr=self.opt['lr'],betas=(0.9, 0.999), weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)
        return optimizer, scheduler
    
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
    def train_one_step(self, data):
        # zero out previous grads
        self.optimizer.zero_grad()
        
        # get losses
        imgs = data[0]
        segs = data[1]

        feat, predict, _, _ = self.model(imgs)


        loss_pce = self.criterian_pce(predict, segs)
        loss_dc, _, _ = self.criterian_dc(predict, segs)

        loss = loss_pce + loss_dc 
        
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        seg_losses = {}
        seg_losses['train_dc'] = loss_dc.detach()
        seg_losses['train_ce'] = loss_pce.detach()
        seg_losses['train_total'] = loss.detach()

        return predict, seg_losses
    
    @torch.no_grad()
    def validate_one_step(self, data):
        self.model.eval()

        imgs = data[0]
        segs = data[1]

        losses = {}
        _,predict,_, _ = self.model(imgs)
        losses['val_ce'] = self.criterian_pce(predict, segs).detach()

        loss_dc, loss_dc_wobg, _ = self.criterian_dc(predict, segs)
        losses['val_dc'] = loss_dc.detach()
        losses['val_dc_wobg'] = loss_dc_wobg.detach()


        self.model.train()

        return predict,losses

    def launch(self):
        self.initialize()
        self.train()
        
    def train(self):
        for epoch in range(self.start_epoch,self.total_epochs):
            train_iterator = tqdm((self.train_dataloader), total = len(self.train_dataloader))
            train_losses = {}
            for it, (images,segs,_) in enumerate(train_iterator):
                # pdb.set_trace()
                images = images.to(self.opt['gpu_id'])
                segs = segs.to(self.opt['gpu_id'])
                
                with self.iter_counter.time_measurement("train"):
                    predicts, losses = self.train_one_step([images, segs])
                    for k,v in losses.items():
                        train_losses[k] = v + train_losses.get(k,0) 
                    train_iterator.set_description(f'Train Epoch [{epoch}/{self.total_epochs}]')
                    train_iterator.set_postfix(ce_loss = train_losses['train_ce'].item()/(it+1), dc_loss = train_losses['train_dc'].item()/(it+1), total_loss = train_losses['train_total'].item()/(it+1))

                self.iter_counter.record_one_iteration()
            self.iter_counter.record_one_epoch()
            del train_losses

            if self.iter_counter.needs_evaluation():

                ## src val
                val_losses = None
                val_metrics = {}
                sample_dict = {}
                val_predicts = []
                val_gts = []
                val_iterator = tqdm((self.val_dataloader), total = len(self.val_dataloader))
                for it_val, (val_imgs, val_segs, val_names) in enumerate(val_iterator):

                    val_imgs = val_imgs.to(self.opt['gpu_id'])
                    val_segs = val_segs.to(self.opt['gpu_id'])

                    if val_losses is None:
                        predict, val_losses = self.validate_one_step([val_imgs, val_segs]) 
                    else:
                        predict, losses = self.validate_one_step([val_imgs, val_segs])
                        for k,v in losses.items():
                            val_losses[k] += v.item()
                    val_predicts.append(predict.detach().cpu().numpy())
                    val_gts.append(val_segs.detach().cpu().numpy())

                    val_iterator.set_description(f'Eval Epoch [{epoch}/{self.total_epochs}]')
                    val_iterator.set_postfix(ce_loss = val_losses['val_ce']/(it_val+1), dc_loss = val_losses['val_dc']/(it_val+1))
                # if self.opt['dataset_name'] == 'brats': # for brats2021
                #     dice = ((it_val+1) - val_losses['val_dc']) / ((it_val+1))
                #     dice_wobg = ((it_val+1) - val_losses['val_dc_wobg']) / ((it_val+1))
                #     logging.info(f'Source val new dice loss {dice}, dice wo bg {dice_wobg}, cls wise discard')
                # else:
                val_predicts = np.concatenate(val_predicts,axis=0) 
                val_gts = np.concatenate(val_gts,axis=0) 
                dice, dice_wobg, cls_wise = self.criterian_dc(torch.tensor(val_predicts), torch.tensor(val_gts))
                dice = 1 - dice.item()
                dice_wobg = 1 - dice_wobg.item()
                logging.info(f'Source val new dice loss {dice}, dice wo bg {dice_wobg}, cls wise {cls_wise}')
                          
                if dice_wobg > self.best_avg_dice:
                    logging.info(f'🔵 Better dice in src val {self.best_avg_dice} ->  {dice_wobg}!')
                    self.best_avg_dice = dice_wobg
                    self.save_best_models(self.iter_counter.epochs_so_far,dice_wobg)
                del val_losses


                ### trgt val 
                # trgt_val_losses = None
                # trgt_predicts = []
                # trgt_gts = []
                # trgt_val_iterator = tqdm((self.trgt_val_dataloader), total = len(self.trgt_val_dataloader))
                # for it_trgtval, (trgt_val_imgs, trgt_val_segs, _) in enumerate(trgt_val_iterator):

                #     trgt_val_imgs = trgt_val_imgs.to(self.opt['gpu_id'])
                #     trgt_val_segs = trgt_val_segs.to(self.opt['gpu_id'])

                #     if trgt_val_losses is None:
                #         trgt_predict, trgt_val_losses = self.validate_one_step([trgt_val_imgs, trgt_val_segs])
                #     else:
                #         trgt_predict, trgt_losses = self.validate_one_step([trgt_val_imgs, trgt_val_segs])
                #         for k,v in trgt_losses.items():
                #             trgt_val_losses[k] += v.item()

                #     if not self.opt['dataset_name'] == 'brats':
                #         trgt_predicts.append(trgt_predict.detach().cpu().numpy())
                #         trgt_gts.append(trgt_val_segs.detach().cpu().numpy())

                #     trgt_val_iterator.set_description(f'Eval Epoch [{epoch}/{self.total_epochs}]')
                #     trgt_val_iterator.set_postfix(ce_loss = trgt_val_losses['val_ce']/(it_trgtval+1), dc_loss = trgt_val_losses['val_dc']/(it_trgtval+1))
                    
                #     ## visual is del
                
                # if self.opt['dataset_name'] == 'brats':
                #     dice = ((it_val+1) - trgt_val_losses['val_dc']) / ((it_val+1))
                #     dice_wobg = ((it_val+1) - trgt_val_losses['val_dc_wobg']) / ((it_val+1))
                #     logging.info(f'Source val new dice loss {dice}, dice wo bg {dice_wobg}, cls wise discard')
                # else:
                #     trgt_predicts = np.concatenate(trgt_predicts,axis=0)
                #     trgt_gts = np.concatenate(trgt_gts,axis=0)
                
                #     dice, dice_wobg, cls_wise = self.criterian_dc(torch.tensor(trgt_predicts), torch.tensor(trgt_gts))
                #     dice = 1 - dice.item()
                #     dice_wobg = 1 - dice_wobg.item()
                #     logging.info(f'Trgt val new dice loss {dice}, dice wo bg {dice_wobg}, cls wise {cls_wise}')
                # del trgt_val_losses
            self.schedular.step()
       
import logging
import os
import pdb
import random

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders import MyDataset147
from losses import MultiClassDiceLoss, PseuLoss
from models import get_model
from utils import IterationCounter, Visualizer, mean_dice, MultiASD


class pmt_Test():
    def __init__(self, opt):
        self.opt = opt
    
    def initialize(self):
        self.set_seed(self.opt['random_seed'])
        ### 1. initialize dataloaders
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

        ## 2. initialize the models
        self.model = get_model(self.opt)
        checkpoint = torch.load(self.opt['source_model_path'],map_location='cpu')
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model = self.model.to(self.opt['gpu_id'])
        p = self.opt['source_model_path']
        print(f'loading {p}')


        ## only used for visualization pseu
        self.ori_model = get_model(self.opt)
        checkpoint = torch.load(self.opt['ori_model_path'],map_location='cpu')
        self.ori_model.load_state_dict(checkpoint['model'], strict=False)
        self.ori_model = self.ori_model.to(self.opt['gpu_id'])
        self.ori_model.eval()

        self.total_epochs = self.opt['total_epochs']
        
        ## losses
        self.criterian_dc  = MultiClassDiceLoss(self.opt)

        ## metrics
        self.best_avg_dice = 0

        # visualizations
        self.iter_counter = IterationCounter(self.opt)        

        ## pmt
        self.pmt_type = self.opt['pmt_type']

    def set_seed(self,seed):
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print('Random seed for this experiment is {} !'.format(seed))

    @torch.no_grad()
    def validate_one_step(self, data):
        self.model.eval()

        imgs = data[0]
        feat, predict, prompted_img, prompt = self.model(imgs, type=self.pmt_type, phase='val')

        ## only for pseu visual
        with torch.no_grad():
            _, pseu, _, _ = self.ori_model(imgs)
            prob = torch.softmax(pseu, 1)
            pseu = torch.argmax(prob, 1)

            # if self.opt['dataset_name'] == 'abdomen':
            #     percent = [70, 50, 50, 50, 50]
            # elif self.opt['dataset_name'] == 'cardiac':
            #     percent =  [40, 40, 40, 40, 40]
            # else:
            #     raise ('dataset name error, which should be in [cardiac, abdomen]')
            percent =  [40, 40, 40, 40, 40]

            prob = prob
            target = pseu.clone()
            entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

            for i in range(predict.shape[1]):
                if (target == i).sum():
                    thresh_i = np.percentile(
                        entropy[target == i].detach().cpu().numpy().flatten(), percent[i]
                    )
                    thresh_i = min(thresh_i, 0.4)
                    thresh_mask = entropy.ge(thresh_i).bool() & (target == i).bool()
                    target[thresh_mask] = 255


        return predict, feat, prompted_img, prompt, pseu, target

    def launch(self):
        self.initialize()
        # self.test(self.train_dataloader, phase = 'trgt_train')
        self.test(self.val_dataloader, phase = 'trgt_val')
        
    def test(self, val_data, phase):

        sample_dict = {}
        val_predicts = []
        val_images = []
        val_gts = []
        val_pmt = []
        val_pmtimages = []
        val_name_list = []
        pseu_list = []
        pseu_select_list = []
        val_iterator = tqdm((val_data), total = len(val_data))
        for it, (val_imgs, val_segs, val_names) in enumerate(val_iterator):
            val_imgs = val_imgs.to(self.opt['gpu_id'])
            val_segs = val_segs.to(self.opt['gpu_id'])
        
            predict, feat, prompted_img, prompt, pseu, pseu_select = self.validate_one_step([val_imgs, val_segs])

            # used in slice-wise dice
            val_predicts.append(predict.detach().cpu().numpy())
            val_gts.append(val_segs.detach().cpu().numpy())
            if not self.opt['dataset_name'] == 'brats':
                val_images.append(val_imgs.detach().cpu().numpy())
                val_pmt.append(prompt.detach().cpu().numpy())
                val_pmtimages.append(prompted_img.detach().cpu().numpy())
                val_name_list.append(val_names)
                pseu_list.append(pseu.detach().cpu().numpy())
                pseu_select_list.append(pseu_select.detach().cpu().numpy())

            # used in patient-wise dice
            if self.opt['dataset_name'] == 'abdomen':
                # abdomen files name : imgxxx_xxx.npy
                for i,name in enumerate(val_names):
                    sample_name,index = name.split('_')[0],int(name.split('_')[1][:-4])
                    sample_dict[sample_name] = sample_dict.get(sample_name,[]) + [(predict[i].detach().cpu(),val_segs[i].detach().cpu(),index)]

            elif self.opt['dataset_name'] == 'cardiac':
                # cardiac files name : ct_train_1001_iamge_0000.npy
                for i,name in enumerate(val_names):
                    sample_name,index = name.split('_')[2],int(name.split('_')[-1][:-4])
                    sample_dict[sample_name] = sample_dict.get(sample_name,[]) + [(predict[i].detach().cpu(),val_segs[i].detach().cpu(),index)]
                    
            elif self.opt['dataset_name'] == 'brats':
                # cardiac files name : ct_train_1001_iamge_0000.npy
                for i,name in enumerate(val_names):
                    if phase == 'trgt_val':
                        start_idx = len(self.opt['target_sites'][0])
                    else:
                        start_idx = len(self.opt['source_sites'][0])
                    sample_name,index = name.split('_')[1] + name.split('_')[2] + name.split('_')[3],int(name.split('_')[-1][start_idx:-4])
                    sample_dict[sample_name] = sample_dict.get(sample_name,[]) + [(predict[i].detach().cpu(),val_segs[i].detach().cpu(),index)]
 

        #######################
        ### slice-wise dice
        #######################
        # it should be cosist with the original result recorded in the training log
        val_predicts = np.concatenate(val_predicts,axis=0) # 410, 5, 256, 256
        val_gts = np.concatenate(val_gts,axis=0) # 410， 256， 256
        if not self.opt['dataset_name'] == 'brats':
            val_images = np.concatenate(val_images,axis=0)
            val_name_list = np.concatenate(val_name_list,axis=0)
            pseu_list = np.concatenate(pseu_list,axis=0)
            pseu_select_list = np.concatenate(pseu_select_list,axis=0)

            if self.opt['save']:
                np.save(os.path.join(self.opt['checkpoint_dir'], 'val_predicts.npy'), val_predicts)
                np.save(os.path.join(self.opt['checkpoint_dir'], 'val_gts.npy'), val_gts)
                np.save(os.path.join(self.opt['checkpoint_dir'], 'val_images.npy'), val_images)
                np.save(os.path.join(self.opt['checkpoint_dir'], 'val_name.npy'), val_name_list)
                np.save(os.path.join(self.opt['checkpoint_dir'], 'pseu_list.npy'), pseu_list)
                np.save(os.path.join(self.opt['checkpoint_dir'], 'pseu_select_list.npy'), pseu_select_list)
                print('saving npy files')

            if self.opt['arch'] == 'Pmt_UNet':
                val_pmt = np.concatenate(val_pmt,axis=0)
                val_pmtimages = np.concatenate(val_pmtimages,axis=0)
                if self.opt['save']:
                    np.save(os.path.join(self.opt['checkpoint_dir'], 'val_pmt.npy'), val_pmt)
                    np.save(os.path.join(self.opt['checkpoint_dir'], 'val_pmtimages.npy'), val_pmtimages)

        
        dice, dice_wobg, cls_wise = self.criterian_dc(torch.tensor(val_predicts), torch.tensor(val_gts))
        dice = 1 - dice.item()
        dice_wobg = 1 - dice_wobg.item()
        logging.info(f'Slice-wise DICE {dice}, DICE without background {dice_wobg} ')

        #######################
        ### patient-wise dice & asd
        #######################
        # 1. Put all 2D images' output into original 3D format
        # 2. Calculate the dice coefficient | asd in 3D volume
        # 3. Calculate average result
        total_dice = []
        total_dice_wobg = []
        total_asd = np.zeros((self.opt['num_classes'],))
        total_clswise = np.zeros((self.opt['num_classes'],))

        for k in sample_dict.keys():
            # Step 1
            sample_dict[k].sort(key=lambda ele: ele[-1])
            preds = []
            targets = []
            for pred,target,_ in sample_dict[k]:
                if target.sum()==0:
                    continue
                preds.append(pred)
                targets.append(target)
            # Step 2
            dice, dice_wobg, cls_wise = mean_dice(torch.stack(preds,dim=-1), torch.stack(targets,dim=-1), self.opt['num_classes'],self.opt['organ_list'])
            dice = dice.item()
            dice_wobg = dice_wobg.item()
            asd_list = MultiASD(torch.stack(preds,dim=-1), torch.stack(targets,dim=-1), self.opt['num_classes'])
            logging.info(f'Patient-wise of patient {k}: DICE {dice}, DICE without background {dice_wobg}, cls_wise {cls_wise}')
            logging.info(f'                             asd {asd_list}')
            
            
            total_asd += asd_list
            total_dice.append(dice)
            total_dice_wobg.append(dice_wobg)
            total_clswise += cls_wise

        # Step 3
        total_clswise /= len(sample_dict.keys())
        total_asd /= len(sample_dict.keys())
        logging.info('========== 3D result ==========')
        logging.info(f'3D Total DICE {np.mean(total_dice)}, DICE without background {np.mean(total_dice_wobg)}, cls wise {total_clswise}')
        logging.info(f'asd {total_asd}')
                
        
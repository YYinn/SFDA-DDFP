import torch
import os,random
from models import get_model

from dataloaders import MyDataset147
from torch.utils.data import DataLoader
from losses import MultiClassDiceLoss, PixelCELoss

import numpy as np
from utils import IterationCounter, Visualizer, mean_dice, MultiASD

from tqdm import tqdm
import pdb
import logging

class SourceDomainTest():
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
        print('Length of training dataset: ', len(self.train_dataloader), self.opt['source_sites'])

        self.val_dataloader = DataLoader(
            MyDataset147(self.opt['data_root'], self.opt['source_sites'], dataset_name=self.opt['dataset_name'], phase='val'),
            batch_size=self.opt['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=4
        )
        print('Length of validation dataset: ', len(self.val_dataloader), self.opt['source_sites'])

        self.trgt_val_dataloader = DataLoader(
            MyDataset147(self.opt['data_root'], self.opt['target_sites'], phase='val', dataset_name=self.opt['dataset_name']),
            batch_size=self.opt['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=4
        )
        print('Length of trgt validation dataset: ', len(self.trgt_val_dataloader), self.opt['target_sites'])

        ## 2. initialize the models

        self.model = get_model(self.opt)
        self.model = self.model.to(self.opt['gpu_id'])

        ## losses
        self.criterian_pce = PixelCELoss(self.opt)
        self.criterian_dc  = MultiClassDiceLoss(self.opt)

        # visualizations
        self.iter_counter = IterationCounter(self.opt)
        self.visualizer = Visualizer(self.opt)
        # self.set_seed(self.opt['random_seed'])
        # self.model_resume()
        if os.path.isfile(self.opt['source_model_path']):
            print("=> Loading checkpoint '{}'".format(self.opt['source_model_path']))
            state = torch.load(self.opt['source_model_path'])
            self.model.load_state_dict(state['model'])
        else:
            print("=> No checkpoint, train from scratch !")
    
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
        segs = data[1]

        losses = {}
        feat, predict,_, _ = self.model(imgs)
        losses['val_ce'] = self.criterian_pce(predict, segs).detach()

        loss_dc, _, _ = self.criterian_dc(predict, segs)
        losses['val_dc'] = loss_dc.detach()
    
        return predict, losses, feat

    def launch(self):
        self.initialize()
        # self.test(self.train_dataloader, phase='src_train')
        self.test(self.val_dataloader, phase='src_val')
        self.test(self.trgt_val_dataloader, phase='trgt_val')
        
    def test(self, val_data, phase):
        logging.info(f'Doing ---- {phase}')
        sample_dict = {}
        trgt_predicts = []
        trgt_gts = []
        val_iterator = tqdm((val_data), total = len(val_data))
        for it, (val_imgs, val_segs, val_names) in enumerate(val_iterator):

            val_imgs = val_imgs.to(self.opt['gpu_id'])
            val_segs = val_segs.to(self.opt['gpu_id'])

            predict, val_losses, feat = self.validate_one_step([val_imgs, val_segs])

            trgt_predicts.append(predict.detach().cpu().numpy())
            trgt_gts.append(val_segs.detach().cpu().numpy())
                    
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
 

        trgt_predicts = np.concatenate(trgt_predicts,axis=0) # 410, 5, 256, 256
        trgt_gts = np.concatenate(trgt_gts,axis=0) # 410， 256， 256

        dice, dice_wobg, cls_wise = self.criterian_dc(torch.tensor(trgt_predicts), torch.tensor(trgt_gts))
        dice = 1 - dice.item()
        dice_wobg = 1 - dice_wobg.item()
        logging.info(f'Slice-wise DICE {dice}, DICE without background {dice_wobg} cls_wise {cls_wise}')

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
                
        
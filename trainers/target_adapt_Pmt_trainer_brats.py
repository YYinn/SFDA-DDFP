import torch
import torch.nn.functional as F
import os,random
from einops import rearrange
from models import get_model
from dataloaders import MyDataset147
from torch.utils.data import DataLoader
from losses import PseuLoss, MultiClassDiceLoss, EntLoss
import numpy as np
import torch.nn as nn
from utils import IterationCounter, Visualizer, mean_dice, MultiASD

from tqdm import tqdm
import pdb
import logging


class brats_pmt_Trainer():
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
 
        ### 2. initialize the target model as self.model. IN TRAINING MODE
        self.model = get_model(self.opt)
        checkpoint = torch.load(self.opt['source_model_path'],map_location='cpu')
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model = self.model.to(self.opt['gpu_id'])        
        p = self.opt['source_model_path']
        print(f'loading {p}')

        ### 3. initialize the pseudo generation model as self.ori_model. IN EVAL MODE
        ### （default: same as the source_model used above. you can switch to use the bn model or ori model by lines below (only for ablation)) 
        self.ori_model = get_model(self.opt)

        ## use which model to generate pseudo labels
        ## abdominal ct 2 mr src 
        # checkpoint = torch.load('/mnt/ExtData/SFDA_DDFP/log/abdominal/src_train/UNet_Abdomen_CT_Seg/T2023-11-30 10:12:34_/saved_models/best_model_epoch_92_dice_0.9289.pth', map_location='cpu')
        # checkpoint = torch.load('/mnt/ExtData/SFDA_DDFP/log/abdominal/UNet_Abdomen_CT2MR_bn/T2024-04-16 16:13:47_tuneall_epoch10/saved_models/model_step_0_dice_0.7580.pth', map_location='cpu')
        ## abdominal mr 2 ct src 
        # checkpoint = torch.load('/mnt/ExtData/SFDA_DDFP/log/abdominal/src_train/UNet_abdomen_MR_Seg/T2023-12-02 10:05:54_/saved_models/best_model_epoch_95_dice_0.9410.pth', map_location='cpu')
        # checkpoint = torch.load('/mnt/ExtData/SFDA_DDFP/log/abdominal/UNet_Abdomen_MR2CT_bn/T2024-04-16 16:14:33_tuneall_epoch10/saved_models/model_step_0_dice_0.6420.pth', map_location='cpu')
        ## cardiac ct 2 mr
        # checkpoint = torch.load('/mnt/ExtData/SFDA_DDFP/log/cardiac/src_train/UNet_cardiac_CT_seg/T2024-01-19 14:20:49_/saved_models/best_model_epoch_92_dice_0.9113.pth', map_location='cpu')
        # checkpoint = torch.load('/mnt/ExtData/SFDA_DDFP/log/cardiac/UNet_cardiac_CT2MR_bn/T2024-04-16 16:20:50_tuneall_epoch10/saved_models/model_step_0_dice_0.4853.pth', map_location='cpu')
        ## cardiac mr 2 ct
        # checkpoint = torch.load('/mnt/ExtData/SFDA_DDFP/log/cardiac/src_train/UNet_cardiac_MR_seg/T2024-01-19 14:20:41_/saved_models/best_model_epoch_150_dice_0.8635.pth', map_location='cpu')
        self.ori_model.load_state_dict(checkpoint['model'], strict=False)
        self.ori_model = self.ori_model.to(self.opt['gpu_id'])
        self.ori_model.eval()

        self.total_epochs = self.opt['total_epochs']
      

        ### optimizers, schedulars
        self.optimizer, self.schedular = self.get_optimizers()
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=True)


        ### calculate pamras and flops 
        ## method 1
        # from thop import profile
        # inputs = torch.randn((1, 3, 256, 256), dtype=torch.float32).cuda()
        # flops, params = profile(self.model, (inputs,))
        # print(flops)
        # print(params)
        ## method 2
        # total_num = sum(p.numel() for p in self.model.parameters())
        # trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)        
        # print(total_num, trainable_num)
        # os._exit(0)

        ## losses
        self.criterian_dc  = MultiClassDiceLoss(self.opt)

        ## metrics
        self.best_avg_dice = 0

        # visualizations
        self.iter_counter = IterationCounter(self.opt)
        self.visualizer = Visualizer(self.opt)
        
        self.model_resume()

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

    def save_models(self, step, dice):
        if step != 0:
            checkpoint_dir = self.opt['checkpoint_dir']
            state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
            torch.save(state, os.path.join(checkpoint_dir, 'saved_models', 'model_step_{}_dice_{:.4f}.pth'.format(step,dice)))

    
    def save_best_models(self, step, dice):
        checkpoint_dir = self.opt['checkpoint_dir']
        for file in os.listdir(os.path.join(checkpoint_dir, 'saved_models')):
            if 'best_model' in file:
                os.remove(os.path.join(checkpoint_dir, 'saved_models', file))
        state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
        torch.save(state,os.path.join(checkpoint_dir, 'saved_models','best_model_step_{}_dice_{:.4f}.pth'.format(step,dice)))
        return os.path.join(checkpoint_dir, 'saved_models','best_model_step_{}_dice_{:.4f}.pth'.format(step,dice))


    def get_optimizers(self):
        '''
        set trainable params
        get optimizer and scheduler

        4 types of training mode: pmt | LSET1 | pmtinc | All
        pmt: only prompt
        LSET1: named as layer set 1 : prompt + Style related layers
        pmtinc:  prompt + inc layer [ONLY FOR ABLATION]
        All: prompt + all layers [ONLY FOR ABLATION]
        '''
        params = []
      
        if self.opt['train_type'] == 'pmt':
            for name, param in self.model.named_parameters():
                if not (name.startswith('fre') or name.startswith('prompt')):
                    param.requires_grad = False
                    # print('freeze', name)
                    continue
                params.append(param)
                print(f'pmt training on {name}')
        elif self.opt['train_type'] == 'LSET1':
            for name, param in self.model.named_parameters():
                if not (name.startswith('fre') or name.startswith('prompt') or name.startswith('inc.double_conv') or name.startswith('down1') or name.startswith('down2') or name.startswith('down3')):
                    param.requires_grad = False
                    # print('freeze', name)
                    continue
                params.append(param)
                print(f'LSET1 training on {name}')
        elif self.opt['train_type'] == 'pmtinc':
            for name, param in self.model.named_parameters():
                if not (name.startswith('fre') or name.startswith('prompt') or name.startswith('inc.double_conv')):
                    param.requires_grad = False
                    # print('freeze', name)
                    continue
                params.append(param)
                print(f'pmtinc training on {name}')
        elif self.opt['train_type'] == 'All':
            params = list(self.model.parameters())

        ## for ablation study
        elif self.opt['train_type'] == 'besideLSET1':
            for name, param in self.model.named_parameters():
                if (name.startswith('fre') or name.startswith('prompt') or name.startswith('inc.double_conv') or name.startswith('down1') or name.startswith('down2') or name.startswith('down3')):
                    param.requires_grad = False
                    continue
                params.append(param)
                print(f'besideLSET1 training on {name}')
        elif self.opt['train_type'] == 'LSET1wodown3':
            for name, param in self.model.named_parameters():
                if not (name.startswith('fre') or name.startswith('prompt') or name.startswith('inc.double_conv') or name.startswith('down1') or name.startswith('down2')):
                    param.requires_grad = False
                    continue
                params.append(param)
                print(f'LSET1wodown3 training on {name}')
        elif self.opt['train_type'] == 'LSET1wodown23':
            for name, param in self.model.named_parameters():
                if not (name.startswith('fre') or name.startswith('prompt') or name.startswith('inc.double_conv') or name.startswith('down1')):
                    param.requires_grad = False
                    continue
                params.append(param)
                print(f'LSET1wodown23 training on {name}')

        optimizer = torch.optim.Adam(params,lr=self.opt['lr'],betas=(0.9, 0.999), weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
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
    def train_one_step(self, data, epoch):

        self.optimizer.zero_grad()
        self.model.set_hook()

        imgs = data[0]

        target_f, predict, prompted_img, prompt = self.model(imgs, type=self.pmt_type, phase='train')

        with torch.no_grad():
            _, pseu, _, _ = self.ori_model(imgs)
            prob = torch.softmax(pseu, 1)
            pseu = torch.argmax(prob, 1)
        
        ### loss
        bn_loss, bn_mean, bn_var = self.model.get_BNLoss()
        pseu_loss = PseuLoss(predict, data[1], outputs_woada=pseu, prob=prob, datasetname=self.opt['dataset_name'], percent=self.opt['wcls'], glo_thresh=self.opt['wglo'], theta=self.opt['theta'])
        ent_loss = EntLoss(predict)
        loss = bn_loss*self.opt['w1'] +pseu_loss*self.opt['w2'] + ent_loss*self.opt['w3']
        logging.info(f'bn {bn_loss}, pseu {pseu_loss}, ent_loss {ent_loss}')


        ### bp
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        adapt_losses = {}
        adapt_losses['total_loss'] = loss.detach()

        return predict, adapt_losses, target_f, prompted_img, prompt , data[1]#refined_pseu
    
    @torch.no_grad()
    def validate_one_step(self, data):
        self.model.eval()

        imgs = data[0]
        _, predict, _, _ = self.model(imgs, type=self.pmt_type, phase='val')

        self.model.train()

        return predict

    def launch(self):
        self.initialize()
        best_model_path = self.train()
        return best_model_path
        
    def train(self):
        
        ### 1. val woada 
        val_predicts = []
        val_gts = []
        val_iterator = tqdm((self.val_dataloader), total = len(self.val_dataloader))

        for it, (val_imgs, val_segs, val_names) in enumerate(val_iterator):

            val_imgs = val_imgs.to(self.opt['gpu_id'])
            val_segs = val_segs.to(self.opt['gpu_id'])

            predict = self.validate_one_step([val_imgs, val_segs])

            val_predicts.append(predict.detach().cpu().numpy())
            val_gts.append(val_segs.detach().cpu().numpy())

        val_predicts = np.concatenate(val_predicts,axis=0) 
        val_gts = np.concatenate(val_gts,axis=0) 

        dice, dice_wobg, cls_wise = self.criterian_dc(torch.tensor(val_predicts), torch.tensor(val_gts))
        dice = 1 - dice.item()
        dice_wobg = 1 - dice_wobg.item()

        logging.info(f'wo ada')
        logging.info(f'TRGT val new dice loss {dice}, dice wo bg {dice_wobg}, cls wise {cls_wise}')
        logging.info('---------------------------------------------------------')


        ### 2. train
        cnt_es_step = 0
        previous_loss = 0
        total_train_dice = []
        # for brats
        total_val_dice = []
        total_val_dice_wobg = []
        for epoch in range(self.start_epoch,self.total_epochs):
            train_iterator = tqdm((self.train_dataloader), total = len(self.train_dataloader))
            for it, (images, segs, img_name) in enumerate(train_iterator):

                images = images.to(self.opt['gpu_id'])
                segs = segs.to(self.opt['gpu_id'])
                
                with self.iter_counter.time_measurement("train"):
                    predicts, losses, target_f, prompted_img, prompt, _  = self.train_one_step([images, segs], it)

                    train_dice, train_dice_wobg, train_cls_wise = self.criterian_dc(torch.tensor(predicts), torch.tensor(segs))
                    train_dice = 1 - train_dice.item()
                    train_dice_wobg = 1 - train_dice_wobg.item()

                    loss_i = losses['total_loss']
                    total_train_dice.append(train_dice)
                    logging.info(f'epoch {epoch} it {it} -- Trgt train loss {loss_i}, dice {train_dice}, dice wo bg {train_dice_wobg}, cls wise {train_cls_wise} es {cnt_es_step}')

                ### 3. val for each iteration
                with self.iter_counter.time_measurement("maintenance"):
                    if self.iter_counter.needs_evaluation_steps():
                        trgt_predicts = []
                        trgt_gts = []
                        trgt_val_dice = 0
                        trgt_val_dicewobg = 0
                        val_iterator = tqdm((self.val_dataloader), total = len(self.val_dataloader))

                        for it, (val_imgs, val_segs, val_names) in enumerate(val_iterator):

                            val_imgs = val_imgs.to(self.opt['gpu_id'])
                            val_segs = val_segs.to(self.opt['gpu_id'])

                            predict = self.validate_one_step([val_imgs, val_segs])
                            
                            ## otherwise been killed
                            # if self.opt['dataset_name'] == 'brats':
                            #     dice, dice_wobg, _ = self.criterian_dc(torch.tensor(trgt_predicts), torch.tensor(trgt_gts))
                            #     total_val_dice.append(1 - dice.item())
                            #     total_val_dice_wobg.append(1 - dice_wobg.item())
                            # else:
                            trgt_predicts.append(predict.detach().cpu().numpy())
                            trgt_gts.append(val_segs.detach().cpu().numpy())

                        # if self.opt['dataset_name'] == 'brats':
                        #     total_val_dice = total_val_dice.mean()
                        #     total_val_dice_wobg = total_val_dice_wobg.mean()
                        #     logging.info(f'Trgt val new dice loss {dice}, dice wo bg {dice_wobg} (because brats dataset is too large, cls-wise dice is discard)')
                        # else:
                        trgt_predicts = np.concatenate(trgt_predicts,axis=0) # 410, 5, 256, 256
                        trgt_gts = np.concatenate(trgt_gts,axis=0) # 410， 256， 256

                        dice, dice_wobg, cls_wise = self.criterian_dc(torch.tensor(trgt_predicts), torch.tensor(trgt_gts))
                        dice = 1 - dice.item()
                        dice_wobg = 1 - dice_wobg.item()

                        logging.info(f'Trgt val new dice loss {dice}, dice wo bg {dice_wobg}, cls wise {cls_wise}')

                        if dice_wobg > self.best_avg_dice:
                            logging.info(f'🔵 Better dice in src val {self.best_avg_dice} -> {dice_wobg}!')
                            self.best_avg_dice = dice_wobg
                            best_model_path = self.save_best_models(self.iter_counter.steps_so_far,dice_wobg)
                        else:
                            if self.iter_counter.needs_saving_steps():
                                self.save_models(self.iter_counter.steps_so_far,dice_wobg)

                self.iter_counter.record_one_iteration()

                ### early stop setting
                if it == 0:
                    previous_loss = losses['total_loss']
                if losses['total_loss'] > previous_loss:
                    cnt_es_step += 1
                else:
                    cnt_es_step = 0
                previous_loss = losses['total_loss']

                if cnt_es_step > self.opt['es_step']:
                    break

            self.schedular.step()
            if cnt_es_step > self.opt['es_step']:
                logging.info('early step !')
                break
            self.iter_counter.record_one_epoch()


        ## if test at the end of the training phase
        return best_model_path
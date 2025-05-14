from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Sequence
import pdb

from torch import Tensor
import numpy as np

import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage as sn
import matplotlib.pyplot as plt



class ContraLoss(nn.Module):
    def __init__(self):
        super(ContraLoss, self).__init__()
        self.temp = 1
    def forward(self, f_t, prob, label=None, source_proto=None):
        
        f_t = F.relu(f_t)
        prob = torch.softmax(prob, 1)
        pseu = torch.argmax(prob, 1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

        f_t = rearrange(f_t, 'b c h w -> (b h w) c')
        pseu = rearrange(pseu, 'b h w -> (b h w)')
        entropy = rearrange(entropy, 'b h w -> (b h w)')

        loss = 0
        sample = 10000

        ## calculate feature center of each class
        # cls_center = torch.zeros((5, f_t.shape[1]))
        # for i in range(5):
        #     if (pseu == i).sum():

        #         thresh_pos = np.percentile(
        #             entropy[pseu == i].detach().cpu().numpy().flatten(), 50
        #         )
        #         thresh_pos = min(thresh_pos, 0.4)
        #         pos = entropy.le(thresh_pos).bool() & (pseu == i).bool()
        #         if not pos.any():
        #             continue

        #         cls_center[i] = f_t[pos, ...].mean(0)
        cls_center = source_proto
        ## assign positive and negative samples for each class
        for i in range(1, 5):
            if (pseu == i).sum():
                thresh_pos = np.percentile(
                    entropy[pseu == i].detach().cpu().numpy().flatten(), 50
                )
                thresh_pos = min(thresh_pos, 0.4)
                pos = entropy.le(thresh_pos).bool() & (pseu == i).bool()
                if not pos.any():
                    continue

                with torch.no_grad():

                    proto_i = cls_center[i:i+1, ...] # 1,64

                    negative_proto_i = []
                    for class_id in range(1, 5):
                        if class_id == i:
                            continue
                        z = cls_center[class_id:class_id+1, ...] 
                        # z = self.mu[class_id] + self.N.sample((40, 64)).cuda() * torch.exp(self.sigma[class_id]/2)
                        negative_proto_i.append(z)
                    negative_proto_i = torch.concat(negative_proto_i)

                low_ent = f_t[pos, ...] #[42127, 64]
                sample_i = min(sample, low_ent.shape[0])
                anchor_index=  torch.randint(low_ent.shape[0], size=(sample_i,))
                anchor = low_ent[anchor_index].clone()
                
                positive_i = proto_i.unsqueeze(0).repeat(sample_i, 1, 1)
                negative_proto_i = negative_proto_i.unsqueeze(0).repeat(sample_i, 1, 1)
                
                print(anchor.shape)

                all_feat = torch.cat((positive_i, negative_proto_i), dim=1).cuda() # sample_i, 5, 64
                seg_logits = torch.cosine_similarity(anchor.unsqueeze(1), all_feat, dim=2).cuda() # 64, 5

                loss = loss + F.cross_entropy(
                    seg_logits / self.temp, torch.zeros(sample_i).long().cuda()
                ) 
        print(loss)
        # print(loss, kl_loss)
        # if source_proto is not None:
        #     for i in range(5):
        #         proto_i = source_proto[i:i+1]
        #         negative_proto_i = torch.concat([source_proto[:i], source_proto[i+1:-1]])
        #         positive_i = proto_i.unsqueeze(0)
        #         negative_proto_i = negative_proto_i.unsqueeze(0)
                
        #         anchor = cls_center[i:i+1, ...].cuda()
        #         all_feat = torch.cat((positive_i, negative_proto_i), dim=1).cuda() # sample_i, 5, 64
        #         seg_logits = torch.cosine_similarity(anchor.unsqueeze(1), all_feat, dim=2).cuda() # 64, 5
        #         loss = loss + F.cross_entropy(
        #             seg_logits / self.temp, torch.zeros(1).long().cuda()
        #         ) 
        #     print(loss)
        return loss# + kl_loss



class SpatialLoss(nn.Module):
    def __init__(self):
        super(SpatialLoss, self).__init__()
        self.num_class = 5

    def forward(self, img, feat, pred, visual=False, label=None, woada_pseu=None, wowada_prob=None):
        img = img.permute(0, 2, 3, 1)
        percent = [60, 60, 60, 60, 60]
        with torch.no_grad():
            prob = torch.softmax(pred, dim=1)
            pseu = torch.argmax(prob, dim=1)
            # prob = wowada_prob
            # pseu = woada_pseu

            ## drop pixels with high entropy
            entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

            for i in range(1, self.num_class):
                if (pseu == i).sum():
                    ## 1 find reliable pixel
                    thresh_i = np.percentile(
                        entropy[pseu == i].detach().cpu().numpy().flatten(), percent[i]
                    )
                    thresh_i = min(thresh_i, 0.4)
                    thresh_mask = entropy.ge(thresh_i).bool() & (pseu == i).bool()
                    
                    pseu[thresh_mask] = 255


                    ## 2 find spatial proto
                    proto_gray_cls_i = torch.mean(img[pseu == i], dim=0).cuda()
                    
                    ## 3. calculate spatial similarity
                    spatial_sim = - torch.log(torch.abs(torch.tensor(proto_gray_cls_i[..., 1]).unsqueeze(0).unsqueeze(0).unsqueeze(0) - torch.tensor(img[..., 1])) +0.01 )

                    ## 4. dis 
                    pseu_mask_i = pseu == i
                    dilation_pseu_mask_i = sn.binary_dilation(pseu_mask_i.detach().cpu().numpy(), iterations=10)
                    spatial_dis = torch.tensor(dilation_pseu_mask_i).cuda()

                    refine_pseu = spatial_sim*spatial_dis
                    refine_pseu[refine_pseu < 2.5 ] = 0
                    refine_pseu[refine_pseu > 2.5] = 1

                    pseu[refine_pseu == 1] = i
                    # ## 5. combine
                    # proto_feat_cls_i = torch.mean(feat[pseu == i], dim=0)
                    
                    # feat_sim = F.cosine_similarity(torch.tensor(proto_feat_cls_i).unsqueeze(0).unsqueeze(0).unsqueeze(0), torch.tensor(feat), dim=(-1))

                    # mask = F.mse_loss(feat_sim, spatial_sim, reduction='none') *spatial_dis

            # weight = batch_size * h * w / torch.sum(pseu != 255)

        loss = F.cross_entropy(pred, pseu, ignore_index=255) #+ entro_loss *10
        # print(loss)

        return loss, pseu


class ProtoLoss(nn.Module):

    """
    Official Implementaion of PCT (NIPS 2021)
    Parameters:
        - **nav_t** (float): temperature parameter (1 for all experiments)
        - **beta** (float): learning rate/momentum update parameter for learning target proportions
        - **num_classes** (int): total number of classes
        - **s_par** (float, optional): coefficient in front of the bi-directional loss. 0.5 corresponds to pct. 1 corresponds to using only t to mu. 0 corresponds to only using mu to t.

    Inputs: mu_s, f_t
        - **mu_s** (tensor): weight matrix of the linear classifier, :math:`mu^s`
        - **f_t** (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - mu_s: : math: `(K,F)`, f_t: :math:`(M, F)` where F means the dimension of input features.

    """

    def __init__(self, nav_t: float, beta: float, num_classes: int, device: torch.device, s_par: Optional[float] = 0.5, reduction: Optional[str] = 'mean'):
        super(ProtoLoss, self).__init__()
        self.nav_t = nav_t
        self.s_par = s_par
        self.beta = beta
        self.prop = (torch.ones((num_classes,1))*(1/num_classes)).to(device)
        # self.prop = torch.tensor([[0.90],[0.10]]).to(device)
        # self.prop = torch.tensor([[0.60],[0.10],[0.05],[0.05],[0.20]]).to(device)
        # self.prop = torch.tensor([[0.50],[0.20],[0.05],[0.05],[0.20]]).to(device)
        # self.mu_s = torch.load('/home/lzh/workspace/yinb/ProtoContraSFDA/log/UNet_Abdomen_CT_Seg/mu_s1.pt').to(device)
        # self.mu = torch.tensor(np.load('/home/yyinn/Downloads/lzh_sshfs/ProtoContraSFDA_lzh/ct2mr_mu_9273.npy')).cuda()
        # self.sigma = torch.tensor(np.load('/home/yyinn/Downloads/lzh_sshfs/ProtoContraSFDA_lzh/utils/sigma4.npy')).cuda()
        self.eps = 1e-6
        # self.N = torch.distributions.Normal(0, 1)
         
    def pairwise_cosine_dist(self, x, y):
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        return 1 - torch.matmul(x, y.T)

    def get_pos_logits(self, sim_mat, prop):
        log_prior = torch.log(prop + self.eps)
        return sim_mat/self.nav_t + log_prior

    def update_prop(self, prop):
        return (1 - self.beta) * self.prop + self.beta * prop 

    def forward(self, mu_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        # Update proportions
        # mu_s = self.mu
        # random shuffle mu_s along mu_s
        # mu_s = []
        # for i in range(5):
        #     mu_s.append(self.mu[i] + self.N.sample((1, 64)).cuda() * torch.exp(self.sigma[i]/2))
        # mu_s = torch.concat(mu_s)
        # f_t = F.relu(f_t)
        sim_mat = torch.matmul(mu_s, f_t.T)
        old_logits = self.get_pos_logits(sim_mat.detach(), self.prop)
        s_dist_old = F.softmax(old_logits, dim=0)
        prop = s_dist_old.mean(1, keepdim=True)
        self.prop = self.update_prop(prop)
        

        # Calculate bi-directional transport loss
        new_logits = self.get_pos_logits(sim_mat, self.prop)
        s_dist = F.softmax(new_logits, dim=0)
        t_dist = F.softmax(sim_mat/self.nav_t, dim=1)
        cost_mat = self.pairwise_cosine_dist(mu_s, f_t)
        t2p_loss = (self.s_par*cost_mat*s_dist).sum(0).mean() 
        p2t_loss = (((1-self.s_par)*cost_mat*t_dist).sum(1)*self.prop.squeeze(1)).sum()
        
        return t2p_loss, p2t_loss

class ProtoLoss_ours(nn.Module):
    from typing import Optional
    def __init__(self, nav_t: float, beta: float, num_classes: int, device: torch.device, s_par: Optional[float] = 0.5, reduction: Optional[str] = 'mean'):
        super(ProtoLoss_ours, self).__init__()
        self.nav_t = nav_t
        self.s_par = s_par
        self.beta = beta
        self.prop = (torch.ones((num_classes,1))*(1/num_classes)).to(device)
        self.eps = 1e-6

        # path = '/home/yyinn/workspace/SFDA_clear/chaos_log_mr2ct/UNet_Abdomen_MR2CT_Seg/data/saved_models/best_model_step_25_dice_0.6803.pth'
        # path = '/home/yyinn/workspace/SFDA_clear/chaos_log_ct2mr/UNet_Abdomen_CT2MR_pmt/data/saved_models/best_model_step_8_dice_0.8077.pth'
        # self.mu = torch.load(path)['model']['outc.conv.weight'][:, :, 0, 0]
        # self.mu = torch.load('/home/yyinn/workspace/SFDA_clear/chaos_log/UNet_Abdomen_CT2MR_Seg/bn_pseu/mu.pt').cuda()
        # self.sigma = torch.load('/home/yyinn/workspace/SFDA_clear/chaos_log/UNet_Abdomen_CT2MR_Seg/bn_pseu/sigma.pt').cuda()
        self.result = []
        self.alpha = 0.1
         
        # self.ratio = torch.tensor([0.92946043, 0.05191776, 0.00449301, 0.00475206, 0.0100895 ]).unsqueeze(0).cuda()
        # self.N = torch.distributions.Normal(0, 1)
    def pairwise_cosine_similarity(self, x, y):
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        return torch.matmul(x, y.T)

    def get_pos_logits(self, sim_mat, prop):
        log_prior = torch.log(prop + self.eps)
        return sim_mat/self.nav_t + log_prior

    def update_prop(self, prop):
        return (1 - self.beta) * self.prop + self.beta * prop 

    def forward(self, mu_s: torch.Tensor, f_t: torch.Tensor, outputs:torch.Tensor) -> torch.Tensor:
        self.mu = mu_s
        f_t = F.relu(f_t)
        with torch.no_grad():
            entropy = -torch.sum(torch.softmax(outputs, 1) * torch.log(torch.softmax(outputs, 1) + 1e-10), dim=1)
            entropy = rearrange(entropy, 'b h w -> (b h w)')
            pseu = torch.argmax(torch.softmax(outputs, 1), 1)
            target = rearrange(pseu, 'b h w -> (b h w)')

        # 
        # sim_mat = torch.matmul(self.mu, f_t.T)
        # sim_mat = F.softmax(sim_mat, dim=0)
        # class_ratio = sim_mat.mean(1, keepdim=True)
        # class_ratio = F.softmax(class_ratio, dim=0)
        
        # tmp = rearrange(torch.argmax(outputs, 1),  'b h w -> (b h w) ')
        # # class_ratio = tmp.mean(0, keepdim=True)
        # class_ratio = torch.zeros(5)
        
        # for i in range(5):
        #     class_ratio[i] = (len(tmp[tmp == i] )/ tmp.shape[0])
        # class_ratio = class_ratio.cuda()

        # kl_loss = -torch.sum(class_ratio * self.ratio.log()) + torch.sum(class_ratio * class_ratio.log()) 

        loss = 0
        temp = 1
        sample = 100
        for i in range(5):
            if (target == i).sum():
                # dimensions of pos and neg should be the same
                thresh_pos = np.percentile(
                    entropy[target == i].detach().cpu().numpy().flatten(), 50
                )
                thresh_pos = min(thresh_pos, 0.2) # incase the entropy threshold is too high 
                # thresh_neg = np.percentile(
                #     entropy[target == i].detach().cpu().numpy().flatten(), 80
                # )

                pos = entropy.le(thresh_pos).bool() & (target == i).bool()
                
                if not pos.any():
                    continue
                # neg = entropy.ge(thresh_neg).bool() & (target == i).bool()

                with torch.no_grad():
                    # print(pos.shape)
                    # positive =  torch.mean(f_t[pos, ...], dim=0, keepdim=True) # 1， 64
                    # print(positive.shape)

                    proto_i = self.mu[i:i+1, ...] # 1,64
                    # proto_i = self.mu[i] +  self.sigma[i]*self.N.sample((1, 64)).cuda()
                    # proto_i = self.mu[i] + self.N.sample((1, 64)).cuda() * torch.exp(self.sigma[i]/2)
                    # negative_proto_i = torch.concat((self.mu[:i, ...], self.mu[i+1:, ...])) # 4, 64
                    
                    negative_proto_i = []
                    for class_id in range(5):
                        if class_id == i:
                            continue
                        z = self.mu[class_id:class_id+1, ...] 
                        # z = self.mu[class_id] + self.N.sample((40, 64)).cuda() * torch.exp(self.sigma[class_id]/2)
                        negative_proto_i.append(z)
                    negative_proto_i = torch.concat(negative_proto_i)

                low_ent = f_t[pos, ...] #[42127, 64]
                if sample >1:
                    sample_i = min(sample, low_ent.shape[0])
                else:
                    sample_i = int(low_ent.shape[0]*sample)
                anchor_index=  torch.randint(low_ent.shape[0], size=(sample_i,))
                anchor = low_ent[anchor_index].clone()

                positive_i = proto_i.unsqueeze(0).repeat(sample_i, 1, 1)
                negative_proto_i = negative_proto_i.unsqueeze(0).repeat(sample_i, 1, 1)

                all_feat = torch.cat((positive_i, negative_proto_i), dim=1) # sample_i, 5, 64
                seg_logits = torch.cosine_similarity(anchor.unsqueeze(1), all_feat, dim=2).cuda() # 64, 5

                loss = loss + F.cross_entropy(
                    seg_logits / temp, torch.zeros(sample_i).long().cuda()
                ) 
                # print(loss)
        # print(loss, kl_loss)
        return loss# + kl_loss

def one_hot_encoder(input_tensor):
    tensor_list = []
    for i in range(5):
        temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob.unsqueeze(1))
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()

def rce(pred, labels):
    pred = F.softmax(pred, dim=1)
    conf, _ = torch.max(pred, dim=1)
        
    pred = torch.clamp(pred, min=1e-7, max=1.0)
    torch.autograd.set_detect_anomaly(True)

    mask = (labels != 255).float()
    labels[labels==255] = 5
    # label_one_hot = torch.nn.functional.one_hot(labels, 5).float().cuda()
    label_one_hot = one_hot_encoder(labels)
    label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
    rce = -(torch.sum(pred * torch.log(label_one_hot), dim=1) * mask).sum() / (mask.sum() + 1e-6) 
    
    return rce

def EntLoss(probs):
    probs = rearrange(probs, 'b c h w -> (b h w) c')
    probs = F.softmax(probs,dim=1)
    entropy_loss = torch.sum(-probs * torch.log(probs + 1e-6), dim=1).mean()
    return entropy_loss


def PseuLoss(predict, true_label, outputs_woada=None, prob=None,datasetname='cardiac'):
    # percent = [80, 60, 60, 60, 60] # ct2mr chaos
    # percent = [70, 50, 50, 50, 50] # mr2ct chaos
    if datasetname == 'abdomen':
        percent = [70, 50, 50, 50, 50]
    elif datasetname == 'cardiac':
        percent =  [40, 40, 40, 40, 40]
    else:
        raise ('dataset name error, which should be in [cardiac, abdomen]')
        
    # percent = [30, 30, 30, 30, 30]

    with torch.no_grad():
        prob = prob
        target = outputs_woada
        # target = true_label
        
        #https://github.com/Haochen-Wang409/U2PL/blob/main/train_semi.py
        batch_size, num_class, h, w = predict.shape
        # target[target == 0] = 255

        ## drop pixels with high entropy
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
        # entropy[target == 0] = 1

        conf, _ = torch.max(prob, dim=1)
        
        # # weight = []
        # # weight.append(0)
        for i in range(num_class):
            if (target == i).sum():
                thresh_i = np.percentile(
                    entropy[target == i].detach().cpu().numpy().flatten(), percent[i]
                )
                thresh_i = min(thresh_i, 0.4)
                thresh_mask = entropy.ge(thresh_i).bool() & (target == i).bool()
                
                
                target[thresh_mask] = 255
                # weight.append( (batch_size * h * w ) / (torch.sum(target == i) + 1e-5))
            # else:
                # weight.append(0.0001)
        # weight.append(0)

    # prob = F.softmax(outputs_woada, dim=1)
    # entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

    # entro_loss = entropy[target == 255].mean()

    # inter = target ==  true_label
    # for i in range(5):
        
    #     print(inter[target == i].sum() / torch.sum(target == i), inter[target == i].sum() / torch.sum(true_label == i))
    
    weight = batch_size * h * w / torch.sum(target != 255)
    # print(weight)

    # loss = F.cross_entropy(predict, target, ignore_index=255) #+ entro_loss *10
    loss =(torch.mean(F.cross_entropy(predict, target, ignore_index=255,reduction='none') * conf.exp())) / (weight*0.2)
    return loss
class Proto_with_KLProp_Loss(nn.Module):

    """
    Official Implementaion of PCT (NIPS 2021)
    Parameters:
        - **nav_t** (float): temperature parameter (1 for all experiments)
        - **beta** (float): learning rate/momentum update parameter for learning target proportions
        - **num_classes** (int): total number of classes
        - **s_par** (float, optional): coefficient in front of the bi-directional loss. 0.5 corresponds to pct. 1 corresponds to using only t to mu. 0 corresponds to only using mu to t.

    Inputs: mu_s, f_t
        - **mu_s** (tensor): weight matrix of the linear classifier, :math:`mu^s`
        - **f_t** (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - mu_s: : math: `(K,F)`, f_t: :math:`(M, F)` where F means the dimension of input features.

    """

    def __init__(self, nav_t: float, beta: float, num_classes: int, device: torch.device, s_par: Optional[float] = 0.5, reduction: Optional[str] = 'mean'):
        super(Proto_with_KLProp_Loss, self).__init__()
        self.nav_t = nav_t
        self.s_par = s_par
        self.beta = beta
        self.eps = 1e-6
         
    def pairwise_cosine_dist(self, x, y):
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        return 1 - torch.matmul(x, y.T)

    def get_pos_logits(self, sim_mat, prop):
        log_prior = torch.log(prop + self.eps)
        return sim_mat/self.nav_t + log_prior

    def update_prop(self, prop):
        return (1 - self.beta) * self.prop + self.beta * prop 

    def forward(self, mu_s: torch.Tensor, f_t: torch.Tensor, gt_prop) -> torch.Tensor:
        # Update proportions
        sim_mat = torch.matmul(mu_s, f_t.T)

        # Calculate bi-directional transport loss
        new_logits = self.get_pos_logits(sim_mat, gt_prop)
        s_dist = F.softmax(new_logits, dim=0)
        t_dist = F.softmax(sim_mat/self.nav_t, dim=1)
        
        cost_mat = self.pairwise_cosine_dist(mu_s, f_t)
        source_loss = (self.s_par*cost_mat*s_dist).sum(0).mean() 
        target_loss = (((1-self.s_par)*cost_mat*t_dist).sum(1)*gt_prop.squeeze(1)).sum()
        # est_prop = s_dist.mean(1, keepdim=True)
        # log_gt_prop = (gt_prop + 1e-6).log()
        # log_est_prop = (est_prop + 1e-6).log()
        # kl_loss = (1-self.s_par)*(-torch.sum(est_prop * log_gt_prop) + torch.sum(est_prop * log_est_prop))
        
        loss = source_loss + target_loss
        return loss

# class EntropyLoss(nn.Module):
#     def __init__(self,nav_t,num_classes,device,weights=None):
#         super(EntropyLoss, self).__init__()
#         self.nav_t = nav_t
#         if weights is not None:
#             self.weights = weights
#         else:
#             self.weights = (torch.ones((num_classes,1))*(1/num_classes)).to(device)
            
#     def get_prob_logits(self, x, y):
#         x = F.normalize(x, p=2, dim=1)
#         y = F.normalize(y, p=2, dim=1)
#         return torch.matmul(x, y.T)
    
#     def forward(self, mu_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
#         prob_logits = self.get_prob_logits(mu_s,f_t)/self.nav_t
#         probs = F.softmax(prob_logits,dim=0)
#         return torch.sum(-self.weights * probs * torch.log(probs + 1e-6), dim=0).mean()

# class KLPropLoss(nn.Module):
#     """
#     CE between proportions
#     """
#     def __init__(self, ):
#         super(KLPropLoss, self).__init__()

#     def formawrd(self, probs: Tensor, target: Tensor) -> Tensor:
#         est_prop = probs.mean(dim=1, keepdim=True)
#         log_gt_prop = (target + 1e-6).log()
#         log_est_prop = (est_prop + 1e-6).log()
#         loss = -torch.sum(est_prop * log_gt_prop) + torch.sum(est_prop * log_est_prop)
#         return loss

# class Entropy_KLProp_Loss(nn.Module):

#     """
#     Simplify Implementaion of Entropy and KLProp Loss (MICCAI 2020)
#     Parameters:
#         - **nav_t** (float): temperature parameter (1 for all experiments)
#         - **beta** (float): learning rate/momentum update parameter for learning target proportions
#         - **num_classes** (int): total number of classes
#         - **s_par** (float, optional): coefficient in front of the bi-directional loss. 0.5 corresponds to pct. 1 corresponds to using only t to mu. 0 corresponds to only using mu to t.

#     Inputs: mu_s, f_t
#         - **mu_s** (tensor): weight matrix of the linear classifier, :math:`mu^s`
#         - **f_t** (tensor): feature representations on target domain, :math:`f^t`

#     Shape:
#         - mu_s: : math: `(K,F)`, f_t: :math:`(M, F)` where F means the dimension of input features.

#     """

#     def __init__(self, nav_t: float, beta: float, num_classes: int, device: torch.device, s_par: Optional[float] = 0.5, reduction: Optional[str] = 'mean'):
#         super(Entropy_KLProp_Loss, self).__init__()
#         self.nav_t = nav_t
#         self.s_par = s_par
#         self.beta = beta
#         self.eps = 1e-6
         
#     def get_prob_logits(self, x, y):
#         x = F.normalize(x, p=2, dim=1)
#         y = F.normalize(y, p=2, dim=1)
#         return torch.matmul(x, y.T)


#     def forward(self, mu_s: torch.Tensor, f_t: torch.Tensor, gt_prop) -> torch.Tensor:
#         # Update proportions
#         prob_logits = self.get_prob_logits(mu_s,f_t)/self.nav_t
#         probs = F.softmax(prob_logits,dim=0)
#         est_prop = probs.mean(dim=1, keepdim=True)
        
#         log_gt_prop = (gt_prop + 1e-6).log()
#         log_est_prop = (est_prop + 1e-6).log()
        
#         weights = 1/gt_prop
#         weights = weights/torch.sum(weights)
        
#         # entropy_loss = torch.sum(-weights * probs * torch.log(probs + 1e-6), dim=0).mean()
#         entropy_loss = torch.sum(-probs * torch.log(probs + 1e-6), dim=0).mean()
#         klprop_loss = -torch.sum(est_prop * log_gt_prop) + torch.sum(est_prop * log_est_prop)
#         loss = self.s_par*entropy_loss + (1-self.s_par)*klprop_loss
        
#         return loss

class Entropy_KLProp_Loss(nn.Module):

    """
    Simplify Implementaion of Entropy and KLProp Loss (MICCAI 2020)
    Parameters:
        - **nav_t** (float): temperature parameter (1 for all experiments)
        - **beta** (float): learning rate/momentum update parameter for learning target proportions
        - **num_classes** (int): total number of classes
        - **s_par** (float, optional): coefficient in front of the bi-directional loss. 0.5 corresponds to pct. 1 corresponds to using only t to mu. 0 corresponds to only using mu to t.

    Inputs: mu_s, f_t
        - **mu_s** (tensor): weight matrix of the linear classifier, :math:`mu^s`
        - **f_t** (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - mu_s: : math: `(K,F)`, f_t: :math:`(M, F)` where F means the dimension of input features.

    """

    def __init__(self, nav_t: float, beta: float, num_classes: int, device: torch.device, s_par: Optional[float] = 0.5, reduction: Optional[str] = 'mean'):
        super(Entropy_KLProp_Loss, self).__init__()
        self.nav_t = nav_t
        self.s_par = s_par
        self.beta = beta
        self.eps = 1e-6
         
    def forward(self, probs, gt_prop) -> torch.Tensor:
        # Update proportions
        probs = rearrange(probs, 'b c h w -> (b h w) c')
        probs = F.softmax(probs,dim=1)
        est_prop = probs.mean(dim=0, keepdim=True)
        log_gt_prop = (gt_prop + 1e-6).log()
        log_est_prop = (est_prop + 1e-6).log()
        
        
        # entropy_loss = torch.sum(-weights * probs * torch.log(probs + 1e-6), dim=0).mean()
        entropy_loss = torch.sum(-probs * torch.log(probs + 1e-6), dim=1).mean()
        klprop_loss = -torch.sum(est_prop * log_gt_prop) + torch.sum(est_prop * log_est_prop)
        loss = self.s_par*entropy_loss + (1-self.s_par)*klprop_loss
        
        return loss
    
class EntropyLoss(nn.Module):
    def __init__(self,num_classes,device,weights=None):
        super(EntropyLoss, self).__init__()
        if weights is not None:
            self.weights = weights
        else:
            self.weights = (torch.ones((1,num_classes))*(1/num_classes)).to(device)

    
    def forward(self, probs) -> torch.Tensor:
        probs = rearrange(probs, 'b c h w -> (b h w) c')
        probs = F.softmax(probs,dim=1)
        
        return torch.sum(-probs * torch.log(probs + 1e-6), dim=1).mean()
    
class EntropyClassMarginals(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, probs):
        avg_p = probs.mean(dim=[2, 3]) # avg along the pixels dim h x w -> size is batch x n_classes
        entropy_cm = torch.sum(avg_p * torch.log(avg_p + 1e-6), dim=1).mean()
        return entropy_cm
    
# class PseudoLabel_Loss(nn.Module):
#     def __init__(self):
#         super(PseudoLabel_Loss,self).__init__()
#         self.eps = 1e-10
    
#     def forward(self, pred, pseudo_label_teacher, drop_percent, prob_teacher):
#         batch_size, num_class, h, w = pred.shape
#         with torch.no_grad():
#             # drop pixels with high entropy
            
#             entropy = -torch.sum(prob_teacher * torch.log(prob_teacher + self.eps), dim=1)

#             thresh = np.percentile(
#                 entropy[pseudo_label_teacher != 255].detach().cpu().numpy().flatten(), drop_percent
#             )
#             thresh_mask = entropy.ge(thresh).bool() * (pseudo_label_teacher != 255).bool()

#             pseudo_label_teacher[thresh_mask] = 255
#             weight = batch_size * h * w / torch.sum(pseudo_label_teacher != 255)

#         loss = weight * F.cross_entropy(pred, pseudo_label_teacher, ignore_index=255)  # [10, 321, 321]

#         return loss

    
class PseudoLabel_Loss(nn.Module):
    def __init__(self):
        super(PseudoLabel_Loss,self).__init__()
        self.eps = 1e-6
    
    def get_logits(self, prop):
        log_prior = torch.log(prop + self.eps)
        return log_prior
        
    def forward(self, pred, target, drop_percent, prob_teacher):
        # drop pixels with high entropy
        b, c, h, w  = pred.shape
        # neg_loss = 0
        # pdb.set_trace()
        with torch.no_grad():
            entropy = -torch.sum(prob_teacher * torch.log(prob_teacher + self.eps), dim=1)
            for i in range(c):
                if torch.sum(entropy[target == i]) > 10:

                    thresh = np.percentile(
                    entropy[target == i].detach().cpu().numpy().flatten(), drop_percent
                    )
                    thresh_mask = entropy.ge(thresh).bool() * (target == i).bool()
                    target[thresh_mask] = 255
                    # neg_prob = prob[:,i][thresh_mask]
                    # neg_target = torch.zeros_like(neg_prob).to(neg_prob.device)
                    # neg_target[neg_prob < 0.05] = 1
                    # neg_loss += -torch.mean(neg_target * self.get_logits(1-neg_prob))
        weight = b * h * w / torch.sum(target != 255)

        pos_loss = weight * F.cross_entropy(pred, target, ignore_index=255)  # [10, 321, 321]
        
        # loss = pos_loss + neg_loss

        return pos_loss

# class PseudoLabel_Loss(nn.Module):
#     def __init__(self):
#         super(PseudoLabel_Loss,self).__init__()
#         self.eps = 1e-6
    
#     def get_logits(self, prop):
#         log_prior = torch.log(prop + self.eps)
#         return log_prior
        
#     def forward(self, pred, prob, target, percent, entropy):
#         # drop pixels with high entropy
#         b, c, h, w  = pred.shape
#         neg_loss = 0
#         # pdb.set_trace()
#         for i in range(c):
            
#             thresh = np.percentile(
#             prob[:,i][target == i].detach().cpu().numpy().flatten(), percent
#         )
#             thresh_mask = prob[:,i].le(thresh).bool() * (target == i).bool()
#             target[thresh_mask] = 255
#             neg_prob = prob[:,i][thresh_mask]
#             neg_target = torch.zeros_like(neg_prob).to(neg_prob.device)
#             neg_target[neg_prob < 0.05] = 1
#             neg_loss += -torch.mean(neg_target * self.get_logits(1-neg_prob))

#         pos_loss = F.cross_entropy(pred, target, ignore_index=255)  # [10, 321, 321]
        
#         loss = pos_loss + neg_loss

#         return loss

class Curriculum_Style_Entropy_Loss(nn.Module):
    def __init__(self,alpha=0.002,gamma=2):
        super(Curriculum_Style_Entropy_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    
    def forward(self, probs) -> torch.Tensor:
        probs = rearrange(probs, 'b c h w -> (b h w) c')
        probs = F.softmax(probs,dim=1)
        entropy_map = torch.sum(-probs * torch.log(probs + 1e-6), dim=1)
        probs_hat = torch.mean(torch.exp(-3 * entropy_map).unsqueeze(dim=1) * probs, dim=0)
        loss_cel = self.alpha * ((1.7-entropy_map) ** self.gamma) * entropy_map
        loss_div = torch.sum(-probs_hat * torch.log(probs_hat + 1e-6))
        # pdb.set_trace()
        
        return loss_cel.mean()+loss_div


def intra_class_variance(prob, img):
    mean_std = torch.std(img * prob, dim=[2,3])
    return mean_std.mean()

def inter_class_variance(prob, img):
    mean_std = torch.std(torch.mean(img * prob, dim=[2,3]), dim=1)
    return mean_std.mean()
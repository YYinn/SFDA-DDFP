import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


def EntLoss(probs):
    probs = rearrange(probs, 'b c h w -> (b h w) c')
    probs = F.softmax(probs,dim=1)
    entropy_loss = torch.sum(-probs * torch.log(probs + 1e-6), dim=1).mean()
    return entropy_loss

def PseuLoss(predict, true_label, outputs_woada=None, prob=None, datasetname='cardiac', percent=None, glo_thresh=None, theta=None):
    '''
    #https://github.com/Haochen-Wang409/U2PL/blob/main/train_semi.py
    '''

    if percent is None:
        if datasetname == 'abdomen':
            percent = [70, 50, 50, 50, 50]
        elif datasetname == 'cardiac':
            percent =  [40, 40, 40, 40, 40]
        else:
            raise ('dataset name error, which should be in [cardiac, abdomen]')
    if glo_thresh is None: 
        glo_thresh = 0.4
    if theta is None:
        theta = 0.2

    with torch.no_grad():
        prob = prob
        target = outputs_woada.clone()
                
        batch_size, num_class, h, w = predict.shape

        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

        conf, _ = torch.max(prob, dim=1)                                                ## confidence

        for i in range(num_class):
            if (target == i).sum():
                thresh_i = np.percentile(
                    entropy[target == i].detach().cpu().numpy().flatten(), percent[i]   ## class threshold
                )
                thresh_i = min(thresh_i, 0.4)                                           ## global threshold
                thresh_mask = entropy.ge(thresh_i).bool() & (target == i).bool()
                
                target[thresh_mask] = 255                                               ## set undesirable pixel into 255, which will be ignored in the loss calculation 
               
    weight = batch_size * h * w / torch.sum(target != 255)  
    
    loss =(torch.mean(F.cross_entropy(predict, target, ignore_index=255,reduction='none') * conf.exp())) / (weight * theta)
    return loss


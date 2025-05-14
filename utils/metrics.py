import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
import surface_distance as surfdist
from medpy import metric
# git clone https://github.com/deepmind/surface-distance.git
# pip install surface-distance/

def connectivity_region_analysis(mask):
    s = [[0,1,0],
         [1,1,1],
         [0,1,0]]
    label_im, nb_labels = ndimage.label(mask)#, structure=s)

    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))

    # plt.imshow(label_im)        
    label_im[label_im != np.argmax(sizes)] = 0
    label_im[label_im == np.argmax(sizes)] = 1

    return label_im

def	cal_dice_score( input, target):
    smooth = 1
    input_flat = input.numpy().flatten()
    target_flat = target.numpy().flatten()
    intersection = input_flat * target_flat
    loss = 2 * (intersection.sum() + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
    return loss
 

def MultiDiceScore(preds,target,num_classes,include_bg=True):
    dice_score_list = []

    target = F.one_hot(target,num_classes).float()
    if isinstance(preds, dict):
        seg = preds['seg']
    else:
        seg = preds
    seg = F.one_hot(seg.argmax(dim=0),num_classes).float()

    if include_bg:
        for i in range(num_classes):
            dice_score = cal_dice_score(seg[...,i], target[...,i])
            dice_score_list.append(dice_score)
    else:
        for i in range(1,num_classes):
            dice_score = cal_dice_score(seg[...,i], target[...,i])
            dice_score_list.append(dice_score)
    return dice_score_list


def mean_dice(results, gt_seg_maps,num_classes,organ_list,type=1):

    total_dice_mat = []
    dice_metric = {}
    class_cls_wise = []
    # if type == 1:
    #     num_imgs = len(results)
    #     assert len(gt_seg_maps) == num_imgs
    #     for i in range(num_imgs):
    #         dice_coef = MultiDiceScore(results[i],gt_seg_maps[i],num_classes) #(5, 256, 256, 64) (256, 256, 64)
    #         total_dice_mat.append(dice_coef)
    # else:
    dice_coef = MultiDiceScore(results,gt_seg_maps,num_classes) #(5, 256, 256, 64) (256, 256, 64)
    total_dice_mat.append(dice_coef)
    
    total_dice_mat = np.array(total_dice_mat)
    # for j,organ in enumerate(organ_list):
    for j in range(num_classes):
        # dice_metric['{:}_dice'.format(organ)] = total_dice_mat[:,j].mean()
        class_cls_wise.append(total_dice_mat[:,j].mean())
    dice = total_dice_mat.mean()
    dice_wobg = total_dice_mat[:, 1:].mean()
    # dice_metric['dice_avg'] = total_dice_mat.mean()
    return dice, dice_wobg, class_cls_wise


import torch.nn as nn
class DiceLoss1(nn.Module):
    def __init__(self):
        super(DiceLoss1, self).__init__()
    def	forward(self, input, target):
        smooth = 1
        input_flat = input.flatten()
        target_flat = target.flatten()
        intersection = input_flat * target_flat
        loss = 2 * (intersection.sum() + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
        loss = 1 - loss
        return loss
 

class MultiClassDiceLoss(nn.Module):
    def __init__(self, config=None):
        super(MultiClassDiceLoss, self).__init__()

        self.config = config
        self.num_classes = 5
        self.dice_criterion = DiceLoss1()

    def forward(self, preds, target, weights=None, softmax=True):
        target = F.one_hot(target,self.num_classes).permute((0, 3, 1, 2)).float()
        totalLoss = 0
        seg = preds
        seg = F.softmax(seg, dim=1)
        count = 0
        class_wise = []
        dice_wobg = 0
        for i in range(self.num_classes):
            diceLoss = self.dice_criterion(seg[:,i], target[:,i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss
            if i != 0:
                dice_wobg += diceLoss
            class_wise.append(1 - diceLoss.item())
            count+=1
        return totalLoss/count, dice_wobg/(count-1), class_wise

def cal_average_surface_distance(input,target):
    input = input.cpu().numpy().astype(np.bool8)
    target = target.cpu().numpy().astype(np.bool8)
    surface_distances = surfdist.compute_surface_distances(input, target, spacing_mm=(1.0, 1.0, 1.0))
    avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
    return (avg_surf_dist[0]+avg_surf_dist[1])/2

def MultiASD(preds,target,num_classes,include_bg=True):

    asd_list = []
    target = F.one_hot(target,num_classes)
    if isinstance(preds, dict):
        seg = preds['seg']
    else:
        seg = preds
    # print(seg.shape) #torch.Size([5, 256, 256, 30])
    seg = F.one_hot(seg.argmax(dim=0),num_classes)
    # print(seg.shape,target.shape) #torch.Size([256, 256, 30, 5]) torch.Size([256, 256, 30, 5])
    if include_bg:
        for i in range(num_classes):
            asd = cal_average_surface_distance(seg[...,i], target[...,i])
            # asd = 0
            asd_list.append(asd)
    else:
        for i in range(1,num_classes):
            asd = cal_average_surface_distance(seg[...,i], target[...,i])
            # asd = 0
            asd_list.append(asd)
    return asd_list


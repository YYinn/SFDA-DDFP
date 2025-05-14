import torch.nn as nn
import torch.nn.functional as F


class PixelCELoss(nn.Module):
    def __init__(self, config=None):
        super(PixelCELoss, self).__init__()

        self.ignore_index = -1
        if config['ce_ignore_index']!=-1:
            self.ignore_index = config['ce_ignore_index']

        self.seg_criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def forward(self, preds, target):
        loss = self.seg_criterion(preds, target)
        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

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
        self.alpha = 1
        self.num_classes = self.config['num_classes']

        self.ignore_index = -1
        if self.config['dice_ignore_index']!=-1:
            self.ignore_index = self.config['dice_ignore_index']

        self.dice_criterion = DiceLoss()

    def forward(self, preds, target, weights=None):
        target = F.one_hot(target,self.num_classes).permute((0, 3, 1, 2)).float()
        seg = F.softmax(preds,dim=1)

        totalLoss = 0
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
        return totalLoss/count, dice_wobg/ (count- 1), class_wise
    

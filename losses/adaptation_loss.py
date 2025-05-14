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
    # np.save('oriout1.npy', outputs_woada.detach().cpu().numpy())
    # np.save('true_label1.npy', true_label.detach().cpu().numpy())
    
    mm = 0
    with torch.no_grad():
        prob = prob
        target = outputs_woada.clone()
        cls_out = target.clone()
                
        batch_size, num_class, h, w = predict.shape

        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
        # np.save('entropy1.npy', entropy.detach().cpu().numpy())

        conf, _ = torch.max(prob, dim=1)                                                ## confidence

        for i in range(num_class):
            if (target == i).sum():
                thresh_i = np.percentile(
                    entropy[target == i].detach().cpu().numpy().flatten(), percent[i]   ## class threshold
                )
                # if thresh_i > glo_thresh:
                #     mm = 1

                thresh_mask1 = entropy.ge(thresh_i).bool() & (target == i).bool()
                
                cls_out[thresh_mask1] = 255       
                thresh_i = min(thresh_i, glo_thresh)                                           ## global threshold
                thresh_mask = entropy.ge(thresh_i).bool() & (target == i).bool()
                
                target[thresh_mask] = 255                                               ## set undesirable pixel into 255, which will be ignored in the loss calculation 
    
    # if mm == 1:
    #     np.save('cls_target_1.npy', target.cpu().numpy(), )
    #     os._exit(0)       
    weight = batch_size * h * w / torch.sum(target != 255)  
    # if  mm == 1:
        
    #     import matplotlib.pyplot as plt
    #     color_list = [[0, 0, 0], [238,172,255], [240,130,40], [251,111,111], [121, 173, 214]]

    #     def add_color(mask):
    #         color_mask = np.zeros((mask.shape[0], mask.shape[1], 3))   
    #         for i in range(5):
    #             color_mask[mask == i, :] = np.array(color_list[i]) / 255

    #         color_mask[mask == 255, :] = np.array([255, 255, 255]) / 255
    #         return color_mask

    #     plt.figure(figsize=(15, 15))
    #     cls_out1 = cls_out
    #     cls_out1[cls_out1 == 255] = 0
    #     target1 = target
    #     target1[target1 == 255] = 0
    #     cnt = 1
    #     for i in range(16):
    #         plt.subplot(8, 10, cnt)
    #         plt.imshow(add_color(true_label[i].detach().cpu().numpy()), vmin=0, vmax= 6)
    #         plt.axis('off')
    #         cnt += 1
    #         plt.subplot(8, 10, cnt)
    #         plt.imshow(add_color(true_label[i].detach().cpu().numpy()), vmin=0, vmax= 6)
    #         plt.axis('off')
    #         cnt += 1
    #         plt.subplot(8, 10, cnt)
    #         plt.imshow(add_color(cls_out1[i].detach().cpu().numpy()), vmin=0, vmax= 6)
    #         plt.axis('off')
    #         cnt += 1
    #         plt.subplot(8, 10, cnt)
    #         plt.imshow(add_color(target1[i].detach().cpu().numpy()), vmin=0, vmax= 6)
    #         plt.axis('off')
    #         cnt += 1
    #         plt.subplot(8, 10, cnt)
    #         plt.imshow(add_color((cls_out1[i]-target1[i]).detach().cpu().numpy()), vmin=0, vmax = 1)
    #         plt.axis('off')
    #         # print(np.unique(clsgloout[i]-clsout[i]), (clsgloout[i]-clsout[i]).sum())
    #         cnt += 1
    #     plt.savefig('test.png')
    #     plt.close()

    loss =(torch.mean(F.cross_entropy(predict, target, ignore_index=255,reduction='none') * conf.exp())) / (weight * theta)
    return loss


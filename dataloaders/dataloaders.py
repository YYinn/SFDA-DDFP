import torch.utils.data as data
import os
import glob
import torch
import numpy as np
import random,math
from .transformations import get_transform
from skimage.exposure import match_histograms

class MyDataset147(data.Dataset):
    def __init__(self, rootdir, sites, dataset_name, phase='train'):
        '''
        rootdir: the root dir of the dataset
        sites: ct or mr or other modalities name
        dataset_name: dataset name, used for sepecific preprocessing (because different dataset has already preprocessed in different ways)
                    abdominal: not normalized yet
                    cardiac: normalized already
        phase: train or val
        split_train: whether to split train dataset
        '''

        self.rootdir = rootdir
        self.sites = sites
        self.phase = phase
        self.all_data_path = []
        self.name_list = []
        self.dataset_name = dataset_name

        self.augmenter = get_transform(self.phase, New_size=(256,256))

        for site in sites:
            data_dir = os.path.join(self.rootdir,f'{site}_{phase}')
            for data_name in os.listdir(data_dir):
                self.name_list.append(data_name)
                self.all_data_path.append(os.path.join(data_dir,data_name))

        ## remove empty foreground slices 
        print('before remove empty foreground, len of all_data_path: ', len(self.all_data_path))
        if self.dataset_name != 'brats':
            self.all_data_path = self.remove_empty_foreground(self.all_data_path)
            print('after remove empty foreground, len of all_data_path: ', len(self.all_data_path))

    def remove_empty_foreground(self, all_data_paths):
        new_data_paths = []
        for path in all_data_paths:
            data = np.load(path)
            seg = data[:, :, 2:] # non foreground labels
            if np.sum(seg) > 0:
                new_data_paths.append(path)
        return new_data_paths

                
    def __getitem__(self, index):
        data = np.load(self.all_data_path[index])
        name = self.name_list[index]

        ## 1. loading image and preprocessing
        img = (data[:, :, 0]).astype(np.float32)
        img = img[None, ...]
        
        if self.dataset_name == 'cardiac':
            # img = (img + 1) /2 # ori 
            img = img # new processed 
        elif self.dataset_name == 'abdomen':
            img -= img.min()
            img /= img.max()
        elif self.dataset_name == 'brats':
            img = img
        else:
            raise ('dataset name error, which should be in [cardiac, abdomen]')
        
        img = np.repeat(img, 3, axis=0).transpose((1,2,0))
        
        ## 2. loading segmentation and preprocessing
        if self.dataset_name == 'brats':
            seg = data[:, :, 1:]
            seg = seg[:, :, 1:].sum(axis=2)
        else:
            seg = data[:, :, 1:]
            for i in range(seg.shape[2]):
                a = seg[:,:,i]
                a[a == 1] = i
        # seg = seg[:, :, 0] + seg[:, :, 1] + seg[:, :, 2] + seg[:, :, 3] + seg[:, :, 4]
            seg = seg.sum(axis=-1)

        ## 3. augmentatioin
        transformed = self.augmenter(image=img, mask=seg)
        img = transformed['image']
        img = img.to(torch.float32)
        seg = transformed['mask']
        seg = seg.to(torch.long)
        return img, seg, name
    
    def __len__(self):
        return len(self.all_data_path)




class MyDataset147_hist(data.Dataset):
    def __init__(self, rootdir, sites, phase='train', split_train=False, weak_strong_aug=False):

        self.rootdir = rootdir
        self.sites = sites
        self.phase = phase
        self.weak_strong_aug = weak_strong_aug
        self.all_data_path = []
        self.name_list = []
        self.site_list = []

        self.ref_all_data_path = []
        self.ref_name_list = []
        self.ref_site_list = []

        self.augmenter = get_transform(self.phase,New_size=(256,256))
        for site in sites:
            if split_train:
                data_dir = os.path.join(self.rootdir,f'{site}_train')
            else:
                data_dir = os.path.join(self.rootdir,f'{site}_val')
            for data_name in os.listdir(data_dir):
                self.name_list.append(data_name)
                self.all_data_path.append(os.path.join(data_dir,data_name))
                self.site_list.append(site)

            if site == 'ct':
                ref_site = 'mr'
            else:
                ref_site = 'ct'
            ref_dir = os.path.join(self.rootdir,f'{ref_site}_train')
            for data_name in os.listdir(ref_dir):
                self.ref_name_list.append(data_name)
                self.ref_all_data_path.append(os.path.join(ref_dir,data_name))
                self.ref_site_list.append(ref_site)
            self.ref_all_data_path = self.remove_empty_foreground(self.ref_all_data_path)

        print('before remove empty foreground, len of all_data_path: ', len(self.all_data_path))
        self.all_data_path = self.remove_empty_foreground(self.all_data_path)
        print('after remove empty foreground, len of all_data_path: ', len(self.all_data_path))

    def remove_empty_foreground(self, all_data_paths):
        new_data_paths = []
        for path in all_data_paths:
            data = np.load(path)
            seg = data[:, :, 2:] # non foreground labels
            # seg = seg[20:-20, 20:-20]
            if np.sum(seg) > 0:
                new_data_paths.append(path)
        return new_data_paths

    def __getitem__(self, index):
        
        data = np.load(self.all_data_path[index])
        name = self.name_list[index]
        img = (data[:, :, 0]).astype(np.float32)

        ref_data = np.load(self.ref_all_data_path[np.random.randint(0, len(self.ref_all_data_path)-1)])
        ref_img = (ref_data[:, :, 0]).astype(np.float32)
        ref_img -= ref_img.min()
        ref_img /= ref_img.max()        
        
        img -= img.min()
        img /= img.max()

        ref_img_255 = ref_img *255
        hist_img = img * 255

        hist_img = match_histograms(np.array(hist_img), np.array(ref_img_255))
        hist_img = hist_img/255

        img = hist_img[None, ...]
        
        img = np.repeat(img, 3, axis=0).transpose((1,2,0))
        
        seg = data[:, :, 1:]
        for i in range(seg.shape[2]):
            a = seg[:,:,i]
            a[a == 1] = i
        seg = seg[:, :, 0] + seg[:, :, 1] + seg[:, :, 2] + seg[:, :, 3] + seg[:, :, 4]    

        transformed = self.augmenter(image=img, mask=seg)
        img = transformed['image']
        img = img.to(torch.float32)
        seg = transformed['mask']
        seg = seg.to(torch.long)

        # print(img.shape, ref_img.shape, a.shape)
        return img, seg, name
        # return img, seg, ref_img, hist_img
    # def __getitem__(self, index):
        
    #     data = np.load(self.all_data_path[index])
    #     name = self.name_list[index]
    #     img = (data[:, :, 0]).astype(np.float32)
    #     img = img[None, ...]
        
    #     img -= img.min()
    #     img /= img.max()
        
    #     img = np.repeat(img, 3, axis=0).transpose((1,2,0))
        
    #     seg = data[:, :, 1:]
    #     for i in range(seg.shape[2]):
    #         a = seg[:,:,i]
    #         a[a == 1] = i
    #     seg = seg[:, :, 0] + seg[:, :, 1] + seg[:, :, 2] + seg[:, :, 3] + seg[:, :, 4]


        
    #     ref_data = np.load(self.ref_all_data_path[np.random.randint(0, len(self.ref_all_data_path)-1)])
    #     ref_img = (ref_data[:, :, 0]).astype(np.float32)
    #     ref_img = ref_img[None, ...]
    #     ref_img = np.repeat(ref_img, 3, axis=0).transpose((1,2,0))
    #     ref_img -= ref_img.min()
    #     ref_img /= ref_img.max()

    #     from skimage.exposure import match_histograms

    #     ref_img_255 = ref_img *255
    #     hist_img = img * 255

    #     hist_img = match_histograms(np.array(hist_img), np.array(ref_img_255))
    #     # hist_img = torch.tensor(hist_img, dtype=torch.float32)
    #     hist_img = hist_img/255

    #     transformed = self.augmenter(image=hist_img, mask=seg)
    #     img = transformed['image']
    #     img = img.to(torch.float32)
    #     seg = transformed['mask']
    #     seg = seg.to(torch.long)

    #     # print(img.shape, ref_img.shape, a.shape)
    #     return img, seg, name
    #     # return img, seg, ref_img, hist_img
    
    def __len__(self):
        return len(self.all_data_path)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import cm
    # data_root = '/mnt/ExtData/Data/chaos'
    # data_root = '/mnt/ExtData/Data/processed/chaos/'
    data_root = '/mnt/ExtData/Data/PnpAda_release_data'
    sites = ['ct']
    # dataset = PatientDataset(data_root,sites,'abdomen','test',True,16)
    # patient_sampler = MyBatchSampler(dataset,16,False)
    # dataloader = data.DataLoader(dataset,batch_size=1,batch_sampler=patient_sampler,num_workers=8)
    # dataset = MyDataset147(data_root,sites, 'train',True)
    dataset = MyDataset147_hist(data_root,sites, 'train',True)
    # dataset = MyDataset(data_root,sites,'abdomen','train',True,weak_strong_aug=True)
    dataloader = data.DataLoader(dataset,batch_size=8,num_workers=4)

    # for image_s,segs,names in dataloader:
    #     print(image_s.shape,segs.shape,names)
    #     print(image_s.min(), image_s.max())
    #     # image_s is [8, 3, 256, 256], merge them to 256, 8*256, 3
    #     image_s = image_s.numpy().transpose((0,2,3,1))
    #     image_s = np.concatenate(image_s, axis=1)

    #     # the value of segs is 0 to 4, map the values to color
    #     segs = cm.get_cmap('jet')(segs/4)
    #     segs = segs[:, :, :, :3]
    #     segs = np.concatenate(segs, axis=1)        

    #     image_s = np.concatenate([image_s, segs], axis=0)
        
    #     plt.figure()
    #     plt.imshow(image_s)
    #     plt.show()
    #     break


    for image_s,segs,ref_img, hist_img in dataloader:
        # print(image_s.shape, segs.shape, ref_img.shape, hist_img.shape) 
        #torch.Size([8, 256, 256, 3]) torch.Size([8, 256, 256]) torch.Size([8, 256, 256, 3]) torch.Size([8, 256, 256, 3])
        # image_s = image_s.numpy().transpose((0,2,3,1))
        # ref_img = ref_img.numpy().transpose((0,2,3,1))
        # hist_img = hist_img.numpy().transpose((0,2,3,1))
        plt.figure()
        cnt = 1
        for i in range(8):
            plt.subplot(4, 6, cnt)
            plt.imshow(image_s[i])
            cnt += 1
            plt.subplot(4, 6, cnt)
            plt.imshow(ref_img[i])
            cnt += 1
            plt.subplot(4, 6, cnt)
            plt.imshow(hist_img[i])
            cnt +=1

        plt.show()
        break
import os
import glob
import numpy as np
from tqdm import tqdm
# from utils import read_list, read_nifti, config
import SimpleITK as sitk
import itk
from scipy.ndimage import zoom
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

## https://github.com/xmed-lab/GenericSSL

def write_txt(data, path):
    with open(path, 'w') as f:
        for val in data:
            f.writelines(val + '\n')

def convert_labels(label):
    label[label==205] = 1
    label[label==420] = 2
    label[label==500] = 3
    label[label==820] = 4
    label[label>4] = 0
    return label


def read_reorient2RAI(path):
    itk_img = itk.imread(path)

    filter = itk.OrientImageFilter.New(itk_img)
    filter.UseImageDirectionOn()
    filter.SetInput(itk_img)
    m = itk.Matrix[itk.D, 3, 3]()
    m.SetIdentity()
    filter.SetDesiredCoordinateDirection(m)
    filter.Update()
    itk_img = filter.GetOutput()

    itk_arr = itk.GetArrayViewFromImage(itk_img)
    return itk_arr


def getRangeImageDepth(label):
    d = np.any(label, axis=(1,2))
    h = np.any(label, axis=(0,2))
    w = np.any(label, axis=(0,1))

    if len(np.where(d)[0]) >0:
        d_s, d_e = np.where(d)[0][[0,-1]]
    else:
        d_s = d_e = 0

    if len(np.where(h)[0]) >0:
        h_s,h_e = np.where(h)[0][[0,-1]]
    else:
        h_s = h_e = 0

    if len(np.where(w)[0]) >0:
        w_s,w_e = np.where(w)[0][[0,-1]]
    else:
        w_s = w_e = 0
    return d_s, d_e, h_s, h_e, w_s, w_e



def process_npy():
    base_dir = '/Backup/mega'
    save_dir = '/mnt/ExtData/Data/MMWHS'
    num = 1
    for tag in [ 'ct', 'mr']:
        img_ids = []
        # for path in tqdm(glob.glob(os.path.join(base_dir, tag, f'imagesTr', '*.nii.gz'))):
        for path in tqdm(sorted(glob.glob(os.path.join(base_dir, f'{tag}_train', '*image.nii.gz')))):
            img_id = path.split('/')[-1].split('.')[0]
            idx = path.split('/')[-1].split('_')[2]
            # print(img_id)
            img_ids.append(img_id)

            label_id= img_id[:-5] + 'label'

            image_path = os.path.join(base_dir, f'{tag}_train', f'{img_id}.nii.gz')
            label_path = os.path.join(base_dir, f'{tag}_train', f'{label_id}.nii.gz')

            ## not working for some mr , so we change it into our code 
            image_arr = read_reorient2RAI(image_path)
            label_arr = read_reorient2RAI(label_path)
            image_arr = image_arr.astype(np.float32)

            # img = sitk.ReadImage(image_path)
            # label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
            # direction = img.GetDirection()
            # direction = [direction[8], direction[4], direction[0]]

            # if direction != [1, 1, 1]:
            #     image_arr = change_axes_of_image(sitk.GetArrayFromImage(img), direction, [1, 1, 1])
            #     label_arr = change_axes_of_image(label, direction, [1, 1, 1])

            label_arr = convert_labels(label_arr)

            if img_id == "mr_train_1002_image":
                label_arr[0:4, :, :] = 0
                label_arr[:, -10:-1, :] = 0
                label_arr[:, :, 0:4] = 0

            d_s, d_e, h_s, h_e, w_s, w_e = getRangeImageDepth(label_arr)
            d, h, w = image_arr.shape

            print(label_arr.shape, (d_s + d_e)/2, d_e - d_s, (h_e + h_s)/2, h_e - h_s, (w_e + w_s) /2,w_e - w_s)


            d_s = (d_s -10).clip(min=0, max=d)
            d_e = (d_e +10).clip(min=0, max=d)
            h_s = (h_s).clip(min=0, max=h)
            h_e = (h_e).clip(min=0, max=h)
            w_s = (w_s - 10).clip(min=0, max=w)
            w_e = (w_e + 10).clip(min=0, max=w)

            image_arr = image_arr[d_s:d_e, h_s:h_e, w_s: w_e]
            label_arr = label_arr[d_s:d_e, h_s:h_e, w_s: w_e]

            upper_bound_intensity_level = np.percentile(image_arr, 98)

            image_arr = image_arr.clip(min=0, max=upper_bound_intensity_level)
            # image_arr = (image_arr - image_arr.mean()) / (image_arr.std() + 1e-8)
            image_arr = (image_arr - image_arr.min()) / (image_arr.max() - image_arr.min())

            dn, hn, wn = image_arr.shape

            image_arr = zoom(image_arr, [256/dn, 1, 256/wn], order=0)
            label_arr = zoom(label_arr, [256/dn, 1, 256/wn], order=0)

            test_mr_ids = ["mr_train_1007_image",
                        "mr_train_1009_image",
                        "mr_train_1018_image",
                        "mr_train_1019_image"]
            test_ct_ids = ["ct_train_1003_image",
                        "ct_train_1008_image",
                        "ct_train_1014_image",
                        "ct_train_1019_image"]
            if img_id in test_mr_ids or img_id in test_ct_ids:
                save_root = os.path.join(save_dir,  f'{tag}_val')
            else:
                save_root = os.path.join(save_dir,  f'{tag}_train')
            if not os.path.exists(save_root):
                os.makedirs(save_root)

            label_arr = label_arr.astype(np.int64)
            onehot_label = F.one_hot(torch.tensor(label_arr, dtype=torch.long), 5) # [281, 256, 256, 5]

            for slice_idx in range(image_arr.shape[1]):
                saving_name = os.path.join(save_root, f'{img_id}_{str(slice_idx).zfill(4)}')
                slice_img = image_arr[:, slice_idx, ...]
                slice_img = np.expand_dims(slice_img, axis=-1)
                slice_label = np.array(onehot_label[:, slice_idx, ...])
                info = np.concatenate((slice_img, slice_label), axis=-1)

                print((np.load(f'{saving_name}.npy') == info).all())
                # np.save(saving_name, info)
                # print(saving_name)
            print(img_id, slice_idx)


if __name__ == '__main__':
    process_npy()

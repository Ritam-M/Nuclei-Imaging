import os
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import argparse
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset

from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split

img_size = 128
paths = glob('../input/dsb-2018-dataset-generation/train/*')

for i in tqdm(range(len(paths))):
    path = paths[i]
    img = cv2.imread(os.path.join(path, 'images',os.path.basename(path) + '.png'))
    mask = np.zeros((img.shape[0],img.shape[1]))
    
    for mask_path in glob(os.path.join(path,'masks','*')):
        mask_ = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)>127
        mask[mask_]=1
    
    if(len(img.shape)==2):
        img = np.tile(img[...,None], (1,1,3))
    if(img.shape[2]>3):
        img = img [..., :3]
        
    img = cv2.resize(img, (img_size,img_size))
    mask = cv2.resize(mask, (img_size, img_size))
    
    ## Change Directories as per convenience
    os.makedirs('/kaggle/working/dsb2018_%d/images' % img_size, exist_ok=True)
    os.makedirs('/kaggle/working/dsb2018_%d/masks/0' % img_size, exist_ok=True)
    
    ## Change Directories as per convenience
    cv2.imwrite(os.path.join('/kaggle/working/dsb2018_%d/images' % img_size, os.path.basename(path)+'.png'), img)
    cv2.imwrite(os.path.join('/kaggle/working/dsb2018_%d/masks/0' % img_size, os.path.basename(path)+'.png'), (mask*255).astype('uint8'))
 
class Data(Dataset):
    
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
    
    def __len__(self):
        return(len(self.img_ids))
    
    def __getitem__(self,idx):
        
        img_id = self.img_ids[idx]
        img = cv2.imread(os.path.join(self.img_dir,img_id + self.img_ext))
        
        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)
        
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32')/255
        img = img.transpose(2,0,1)
        
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, {'img_id':img_id}

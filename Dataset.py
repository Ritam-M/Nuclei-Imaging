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

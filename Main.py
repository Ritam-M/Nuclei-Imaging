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

if config['name'] is None:
    if config['deep_supervision']:
        config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
    else:
        config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
os.makedirs('models/%s' % config['name'], exist_ok=True)

print('-' * 20)
for key in config:
    print('%s: %s' % (key, config[key]))
print('-' * 20)

if config['loss'] == 'BCEWithLogitsLoss':
    criterion = nn.BCEWithLogitsLoss().cuda()
else:
    criterion = BCEDiceLoss().cuda()

cudnn.benchmark = True

# model = DUNet(config['num_classes'],config['input_channels'])
model = DUNet()
model = model.cuda()

params = filter(lambda p: p.requires_grad, model.parameters())

if config['optimizer'] == 'Adam':
    optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
elif config['optimizer'] == 'SGD':
    optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],nesterov=config['nesterov'], weight_decay=config['weight_decay'])
else:
    raise NotImplementedError
    
if config['scheduler'] == 'CosineAnnealingLR':
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
elif config['scheduler'] == 'ReduceLROnPlateau':
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],verbose=1, min_lr=config['min_lr'])
elif config['scheduler'] == 'MultiStepLR':
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
elif config['scheduler'] == 'ConstantLR':
    scheduler = None
else:
    raise NotImplementedError
    
img_ids = glob(os.path.join(config['dataset'], 'images', '*' + config['img_ext']))
img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

train_transform = Compose([transforms.RandomRotate90(),transforms.Flip(),OneOf([transforms.HueSaturationValue(),transforms.RandomBrightness(),
            transforms.RandomContrast(),], p=1),transforms.Resize(config['input_h'], config['input_w']),transforms.Normalize(),])

val_transform = Compose([transforms.Resize(config['input_h'], config['input_w']),transforms.Normalize(),])

train_dataset = Data(
        img_ids=train_img_ids,
        img_dir=os.path.join(config['dataset'], 'images'),
        mask_dir=os.path.join(config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)

val_dataset = Data(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=config['num_workers'],
    drop_last=False)

log = OrderedDict([
    ('epoch', []),
    ('lr', []),
    ('loss', []),
    ('iou', []),
    ('dice',[]),
    ('val_loss', []),
    ('val_iou', []),
    ('val_dice',[])
])

best_dice = 0
trigger = 0
for epoch in range(config['epochs']):
    print('Epoch [%d/%d]' % (epoch, config['epochs']))

    # train for one epoch
    train_log = train(config, train_loader, model, criterion, optimizer)
    # evaluate on validation set
    val_log = validate(config, val_loader, model, criterion)

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler.step()
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler.step(val_log['loss'])

    print('loss %.4f - iou %.4f - dice %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f'
          % (train_log['loss'], train_log['iou'], train_log['dice_coef'],val_log['loss'], val_log['iou'], val_log['dice_coef']))

    log['epoch'].append(epoch)
    log['lr'].append(config['lr'])
    log['loss'].append(train_log['loss'])
    log['iou'].append(train_log['iou'])
    log['dice'].append(train_log['dice_coef'])
    log['val_loss'].append(val_log['loss'])
    log['val_iou'].append(val_log['iou'])
    log['val_dice'].append(val_log['dice_coef'])
    
    pd.DataFrame(log).to_csv('models/%s/log.csv' %
                             config['name'], index=False)

    trigger += 1

    if val_log['dice_coef'] > best_dice:
        torch.save(model.state_dict(), 'models/%s/model.pth' %
                   config['name'])
        best_dice = val_log['dice_coef']
        print("=> saved best model")
        trigger = 0

    # early stopping
    if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
        print("=> early stopping")
        break

    torch.cuda.empty_cache()

print('-'*20)
for key in config.keys():
    print('%s: %s' % (key, str(config[key])))
print('-'*20)

cudnn.benchmark = True

#model = DUNet(config['num_classes'],config['input_channels'])
model = DUNet()
model = model.cuda()

# Data loading code
img_ids = glob(os.path.join(config['dataset'], 'images', '*' + config['img_ext']))
img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

_, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

model.load_state_dict(torch.load('models/%s/model.pth' %
                                 config['name']))
model.eval()

val_transform = Compose([
    transforms.Resize(config['input_h'], config['input_w']),
    transforms.Normalize(),
])

val_dataset = Data(
    img_ids=val_img_ids,
    img_dir=os.path.join('inputs', config['dataset'], 'images'),
    mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
    img_ext=config['img_ext'],
    mask_ext=config['mask_ext'],
    num_classes=config['num_classes'],
    transform=val_transform)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=config['num_workers'],
    drop_last=False)

avg_meter = AverageMeter()

for c in range(config['num_classes']):
    os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    
with torch.no_grad():
    for input, target, meta in tqdm(val_loader, total=len(val_loader)):
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            output = model(input)[-1]
        else:
            output = model(input)

        dice_score = dice_coef(output, target)
        avg_meter.update(dice_score, input.size(0))

        output = torch.sigmoid(output).cpu().numpy()

        for i in range(len(output)):
            for c in range(config['num_classes']):
                cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                            (output[i, c] * 255).astype('uint8'))

print('DICE: %.4f' % avg_meter.avg)

torch.cuda.empty_cache()

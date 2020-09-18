## Change them into Python-Arguments incase you are running from shell command.

config = {}
config['name']=None
config['epochs']=100
config['batch_size']=16

config['arch']='NestedUNet'
config['deep_supervision']=False
config['input_channels']=3
config['num_classes']=1
config['input_w']=128
config['input_h']=128

config['loss']='BCEDiceLoss'

config['dataset']='/kaggle/working/dsb2018_128'
config['img_ext']='.png'
config['mask_ext']='.png'

config['optimizer']='SGD'
config['lr']=1e-3
config['weight_decay']=1e-4
config['momentum']=0.9
config['nesterov']=False

config['scheduler']='CosineAnnealingLR'
config['min_lr']=1e-5
config['factor']=0.1
config['patience']=2
config['milestones']='1,2'
config['gamma']=2/3
config['early_stopping']=-1
config['num_workers']=4

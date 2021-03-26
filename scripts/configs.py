#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:50:11 2021

@author: hasan
"""
#configs
from yacs.config import CfgNode as CN
import yaml

cfg = CN()
#
cfg.save_path = '../results/LSTMRexNet_100_try1'
cfg.resume = False
#training and validation
cfg.trainval = CN()
cfg.trainval.epochs = 1
cfg.trainval.lr = 0.0001
cfg.trainval.batch_size = 64
cfg.trainval.val_batch_size = 64
cfg.trainval.accum_steps = 1
cfg.trainval.val_period = 1
cfg.trainval.fold = 0
cfg.trainval.select_folds = [1,2,3,4,5,6]
cfg.trainval.amp = True
cfg.trainval.max_grad_norm = 5.0
cfg.trainval.transform_number = 2
cfg.trainval.val_sample = 10000 // cfg.trainval.val_batch_size
cfg.trainval.device = 'cuda'
#model
cfg.model = CN()
cfg.model.encoder = 'resnet34'
cfg.model.pretrained = True
cfg.model.encoder_attention = True
cfg.model.decoder_type = 'lstm'
cfg.model.decoder_dim = 512
cfg.model.decoder_dropout = 0.2
cfg.model.vocab_size = 194
cfg.model.max_pred_length = 330
#loss
cfg.criterion = 'ce_loss'
#optimizer
cfg.optimizer = CN()
cfg.optimizer.optimizer = 'adam'
optimizer_kwargs = {
    'adam':
    {
         'lr' : cfg.trainval.lr,
         'betas' : (0.9, 0.999),
         'weight_decay': 1e-6,
         'amsgrad' : False
    },
    'adamw':
    {
         'lr' : cfg.trainval.lr,
         'betas' : (0.9, 0.999),
         'weight_decay' : 1e-6,     
    },
    'rmsprop':
    {
         'lr' : cfg.trainval.lr,
         'alpha' : 0.99,
         'weight_decay' : 1e-6,
         'momentum' : 0,
         'centered' : False
        
    }
}
cfg.optimizer.kwargs = CN(optimizer_kwargs[cfg.optimizer.optimizer])
#scheduler
cfg.scheduler = CN()
cfg.scheduler.scheduler = 'multisteplr'
scheduler_kwargs = {
    'polylr':{
        'epochs' : cfg.trainval.epochs,
        'ratio' :0.9
    },
    'multisteplr':{
        'milestones' : [5,7,9],
        'gamma' : 0.5
    },
    'cosine-anneal':{
        'T_max' : 5,
        'eta_min' : 1e-8
    },
    'cosine-anneal-wm':{
        'T_0' : 1,
        'T_mult' : 2,
        'eta_min' : 1e-8
    }
} 
cfg.scheduler.kwargs = CN(scheduler_kwargs[cfg.scheduler.scheduler])


def get_configs():
    return cfg.clone()
def convert_cfg_to_dict(cfg_node, key_list=[]):
    
    """ Convert a config node to dictionary """
    _VALID_TYPES = {tuple, list, str, int, float, bool}
    if not isinstance(cfg_node, CN):
        if type(cfg_node) not in _VALID_TYPES:
            print("Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg_node), _VALID_TYPES), )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_cfg_to_dict(v, key_list + [k])
        return cfg_dict
    
def save_cfg_as_yaml(dic,path):
    
    with open('/'.join([path,'configs.yaml']),'w+') as yamlfile:
              yaml.safe_dump(dic,yamlfile, default_flow_style=False)
    yamlfile.close()

def load_cfg_from_yaml(path):
    with open('/'.join([path,'configs.yaml']),'r') as yamlfile:
        loaded = yaml.safe_load(yamlfile)
    yamlfile.close()
    return CN(loaded)
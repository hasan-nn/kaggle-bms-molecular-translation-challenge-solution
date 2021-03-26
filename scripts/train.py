#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:49:06 2021

@author: hasan
"""
import json
import os
import gc
import os
import torch
import torch.nn as nn

from functools import partial
import time
from train_val import fit_Cnn_Rnn
from criterion import Criterion
from configs import get_configs,save_cfg_as_yaml,convert_cfg_to_dict
from dataset import ImageCaptioningDataset,collate_fn
from encoder_decoder_rnn import EncoderDecoder
from optimizers import get_optimizer
from schedulers import get_scheduler
from torch.utils.data import DataLoader

labels_path = '../data/extended_labels.csv'
captions_path = '../data/captions.json'
i2t_path = '../data/idx2token.json'
t2i_path = '../data/token2idx.json'
imgs_path = '../data/train'

t2i = json.loads(open(t2i_path,'r').read())
i2t = json.loads(open(i2t_path,'r').read())
try:
  os.makedirs('experiments')
except:
  print('Path {} already exists!!!'.format('experiments'))
  
  

#all is Here

def collect():
  torch.cuda.empty_cache()
  gc.collect()
  
#collect()

cfgs = get_configs()
try:
  os.makedirs(cfgs.save_path)
except:
  print('Path {} already exists!!!'.format(cfgs.save_path))
save_cfg_as_yaml(convert_cfg_to_dict(cfgs),cfgs.save_path)

device = cfgs.trainval.device

if(cfgs.resume):
  checkpoint = torch.load(f = cfgs.resume+'/best_model.pth',map_location='cpu')
#model

model = EncoderDecoder(encoder_name = cfgs.model.encoder,
                       pretrained_encoder = cfgs.model.pretrained,
                       vocab_size = cfgs.model.vocab_size,
                       use_encoder_attention=cfgs.model.encoder_attention,
                       decoder_dim=cfgs.model.decoder_dim,
                       decoder_dropout=cfgs.model.decoder_dropout,
                       device=device,
                       decoder_type = cfgs.model.decoder_type)
if(cfgs.resume):
  print('loading model from checkpoint')
  model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)

#loss func
criterion = Criterion(name = cfgs.criterion).to(device)
#optimizer
optimizer = get_optimizer(name = cfgs.optimizer.optimizer,
                          params = model.parameters(),
                          **convert_cfg_to_dict(cfgs.optimizer.kwargs))
#scheduler
scheduler = get_scheduler(name = cfgs.scheduler.scheduler,
                          optimizer = optimizer,
                          **convert_cfg_to_dict(cfgs.scheduler.kwargs))
#'''
#load captions
print('Loading Captions:')
st = time.time()
captions = json.loads(open(captions_path,'r').read())
end = time.time()
print('Loaded Captions in {:.5} seconds'.format(end-st))
Train_Dataset = ImageCaptioningDataset(images_path = imgs_path,
                                       labels_path = labels_path,
                                       captions = captions,
                                       train = True, 
                                       fold = cfgs.trainval.fold,
                                       select_folds = cfgs.trainval.select_folds,
                                       transform_number = cfgs.trainval.transform_number,
                                       shuffle=True,
                                       to_tensor=True)

Val_Dataset = ImageCaptioningDataset(images_path = imgs_path,
                                     labels_path = labels_path,
                                     captions = captions,
                                     train = False, 
                                     fold = cfgs.trainval.fold,
                                     select_folds = None,
                                     transform_number = 1,
                                     shuffle=True,
                                     to_tensor=True)
#'''
train_loader = DataLoader(Train_Dataset,
                          batch_size= cfgs.trainval.batch_size, 
                          shuffle=True,
                          num_workers=2, 
                          collate_fn=collate_fn,
                          pin_memory=True,
                          drop_last=False)

val_loader = DataLoader(Val_Dataset,
                        batch_size= cfgs.trainval.val_batch_size, 
                        shuffle=True,
                        num_workers=2, 
                        collate_fn=collate_fn,
                        pin_memory=True,
                        drop_last=False)

runner = partial(fit_Cnn_Rnn,
                model = model,
                epochs = cfgs.trainval.epochs,
                train_loader = train_loader,
                val_loader = val_loader,
                optimizer = optimizer,
                scheduler = scheduler,
                criterion = criterion,
                save_path = cfgs.save_path,
                sample_size = cfgs.trainval.val_sample,
                accumulation_steps = cfgs.trainval.accum_steps,
                max_pred_length = cfgs.model.max_pred_length,
                device = device,
                first_epoch = 1,
                amp = cfgs.trainval.amp,
                val_period = cfgs.trainval.val_period,
                max_grad_norm = cfgs.trainval.max_grad_norm )
history = runner()
#collect()

print(history)
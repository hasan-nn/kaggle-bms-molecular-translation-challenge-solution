#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:44:57 2021

@author: hasan
"""
from funcs import decode,encode
from meters import AverageMeter,InstantMeter,ShortTermMemoryMeter
from tqdm import tqdm
import Levenshtein
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast,GradScaler
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import pickle
stm_length = 100

def get_meters_info(meters):
  l = []
  for meter in meters:
    l.append(' {} : {:.5} '.format(meter.name,meter.get_update()))
  return '|'.join(l)+'|'

def decode_all(codes,stop_idxs,i2t):
  captions = []
  for code,stop_idx in zip(codes,stop_idxs):
    captions.append(''.join(decode(code[:stop_idx],i2t)))
  return captions

def get_levenshtein_scores(true_captions,pred_captions):
  scores = []
  for tc,pc in zip(true_captions,pred_captions):
    scores.append(Levenshtein.distance(tc,pc))
  return scores

def train_epoch_CnnRnn(train_loader,model,optimizer,scheduler,criterion,accumulation_steps=1,device='cpu',epoch=0,amp=False,scaler=None,max_grad_norm = 5):
  model.train()
  model.zero_grad()
  meters = [AverageMeter(),ShortTermMemoryMeter(stm_length),InstantMeter()]
  for meter in meters:
    meter.set_name(criterion.name)
  iterator = tqdm(enumerate(train_loader))
  for i,(x,y_gt,l) in iterator:
    x,y_gt,l = x.to(device),y_gt.to(device),l.to(device)

    with autocast(enabled=amp):
      y_pred,alphas,l_dec = model(x,y_gt,l)

    y_gt = y_gt[:,1:]
    y_gt = pack_padded_sequence(y_gt,l_dec,batch_first=True,enforce_sorted=False).data
    y_pred = pack_padded_sequence(y_pred,l_dec,batch_first=True,enforce_sorted=False).data

    with autocast(enabled=amp):
      loss = criterion(y_pred,y_gt,alphas) / accumulation_steps
      if(amp):
        scaler.scale(loss).backward()
      else:
        loss.backward()

    if(i % accumulation_steps == 0 or i == len(iterator) - 1):
      if(not amp):
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
        optimizer.step() 
      else:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  
        scaler.step(optimizer)
        scaler.update()
      optimizer.zero_grad()

    loss_value = loss.detach().item() * accumulation_steps

    for meter in meters:
      meter.update(loss_value)
    info = get_meters_info(meters=meters)
    iterator.set_postfix_str('TrainEpoch{} : {}'.format(epoch,info))
    #if(i == 10):break
  scheduler.step()
  return meters


def validate_epoch_CnnRnn(val_loader,model,sample_size=None,max_pred_length = 330,device = 'cpu',epoch = 0):
  model.eval()
  levenshtein_meter = AverageMeter()
  levenshtein_meter.set_name('Levenshtein')
  iterator = tqdm(enumerate(val_loader))
  for i,(x,y_gt,l_real) in iterator:
    x = x.to(device)
    with torch.no_grad():
      y_pred,stop_idxs_pred = model.predict(x)

    y_gt = y_gt[:,1:]
    stop_idxs_real = (l_real[:,0] - 2).tolist()
    true_codes = y_gt.tolist()
    pred_codes = torch.argmax(y_pred,-1).tolist() 
    
    true_captions = decode_all(true_codes,stop_idxs_real,i2t)
    pred_captions = decode_all(pred_codes,stop_idxs_pred,i2t)

    scores = get_levenshtein_scores(true_captions,pred_captions)
    levenshtein_meter.update_all(scores) 

    info = get_meters_info([levenshtein_meter])
    iterator.set_postfix_str('ValEpoch{} : {}'.format(epoch,info))
    if(sample_size is not None):
      if(i==sample_size-1):break
  return levenshtein_meter

def fit_Cnn_Rnn(model,epochs,train_loader,val_loader,optimizer,scheduler,criterion,save_path,sample_size=None,accumulation_steps=1,max_pred_length=330,device='cpu',first_epoch=1,amp=False,val_period = 1,max_grad_norm=5):
  epoch = first_epoch
  model.zero_grad()
  best_score = 99999
  if(amp and device =='cpu'):
    print('changing device to cuda...')
    device = 'cuda'
  scaler = GradScaler() if(amp) else None
  progress_dict = {'{}_avg'.format(criterion.name) : [],
                   '{}_short_term_{}'.format(criterion.name,stm_length) : [],
                   '{}_instant'.format(criterion.name) : [],
                   'Levenshtein_avg' : [],
                   'learning_rate' : [],
                   'best_epoch' : 0,
                   'best_at' : 0}

  print('Training Starts:\n')
  
  for epoch in range(epochs):
    curr_lr = optimizer.param_groups[0]['lr']
    progress_dict['learning_rate'].append(curr_lr)

    train_meters = train_epoch_CnnRnn(train_loader,model,optimizer,scheduler,criterion,accumulation_steps,device,epoch+1,amp,scaler,max_grad_norm)
    for train_meter in train_meters:
      progress_dict[train_meter.name].append(train_meter.get_update())
    
    if(epoch % val_period == 0 or epoch == epochs - 1):
      val_meter = validate_epoch_CnnRnn(val_loader,model,sample_size,max_pred_length,device,epoch+1)
      curr_score = val_meter.get_update()
      progress_dict[val_meter.name].append(curr_score)
      progress_dict['best_epoch'] = epoch+1
      progress_dict['best_at'] = epoch//val_period

      if(curr_score <= best_score):
        torch.save({'state_dict' : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'scheduler_state_dict' : scheduler.state_dict(),
                    'gradscaler_state_dict' : scaler.state_dict() if(amp) else None,
                    'best_epoch' : epoch+1,
                    'optimizer_param_groups' : optimizer.param_groups},
                    save_path+'/best_model.pth')
        
    with open(save_path+'/history.pickle','wb') as jar:
        pickle.dump(progress_dict,jar,protocol=pickle.HIGHEST_PROTOCOL)   

  return progress_dict
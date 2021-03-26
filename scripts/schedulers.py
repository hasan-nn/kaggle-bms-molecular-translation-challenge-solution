#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:42:50 2021

@author: hasan
"""
import torch

#schedulers
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,CosineAnnealingLR,MultiStepLR
class PolyLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, epochs,ratio=0.9):
        super().__init__(optimizer, lambda epoch: (1 - (epoch / epochs) ** ratio))

def get_scheduler(name,optimizer,**kwargs):
  name = name.lower()
  if(name == 'polylr'):
    return PolyLR(optimizer = optimizer,**kwargs) #kwargs : (epochs , ratio)
  elif(name == 'multisteplr'):
    return MultiStepLR(optimizer = optimizer,**kwargs)
  elif(name == 'cosine-anneal'):
    return CosineAnnealingLR(optimizer = optimizer,last_epoch=-1,**kwargs) #kwargs : (T_max, eta_min)
  elif(name == 'cosine-anneal-wm'):
    return CosineAnnealingWarmRestarts(optimizer = optimizer,last_epoch=-1,**kwargs) #kwargs: (T_0, T_mult, eta_min)
  else:
    raise ValueError('optimizer not found')
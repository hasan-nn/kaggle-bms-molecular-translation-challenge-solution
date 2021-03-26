#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:43:32 2021

@author: hasan
"""
import torch.nn as nn

class CrossEntropyLossWithAttReg(nn.CrossEntropyLoss):
  def __init__(self,alpha_c=1,**kwargs):
    super().__init__(**kwargs)
    self.alpha_c = alpha_c

  def forward(self,preds,targets,alphas):
    loss = super().forward(preds,targets)
    if(alphas is not None):
      att_regularization = self.alpha_c * ((1 - alphas.sum(1))**2).mean()
      loss += att_regularization
    return loss

def get_criterion(name = 'ce_loss',**kwargs):
  if(name.lower() == 'ce_loss'):
    return nn.CrossEntropyLoss(**kwargs)
  elif(name.lower() == 'ce_loss_war'):
    return CrossEntropyLossWithAttReg(**kwargs)
  else:
    raise ValueError('Loss function {} not found'.format(name))

class Criterion(nn.Module):
  def __init__(self,name = 'ce_loss',**kwargs):
    super().__init__()
    self.criterion = get_criterion(name,**kwargs)
    self.name = name.lower()
    self.ignore_alphas = True if(self.name == 'ce_loss') else False
  def forward(self,preds,targets,alphas=None):
    kwargs = {'alphas':alphas} if(not self.ignore_alphas) else {}
    return self.criterion(preds,targets,**kwargs)
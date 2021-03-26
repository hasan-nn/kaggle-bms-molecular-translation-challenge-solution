#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:28:13 2021

@author: hasan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.resnet import resnet34,resnet50,resnext50_32x4d,resnet18
from timm.models.efficientnet import efficientnet_b0,efficientnet_b1,efficientnet_b2,efficientnet_b3
from timm.models.regnet import regnetx_002,regnetx_004,regnetx_006,regnetx_008,regnetx_016,regnety_002,regnety_004,regnety_006,regnety_008,regnety_016
from timm.models.rexnet import rexnet_100,rexnet_130,rexnet_150,rexnet_200

resnets = {'resnet18':resnet18,'resnet34':resnet34,'resnet50':resnet50,'resnext50':resnext50_32x4d}
effnets = {'efficientnet_b0':efficientnet_b0,'efficientnet_b1':efficientnet_b1,'efficientnet_b2':efficientnet_b2,'efficientnet_b3':efficientnet_b3}
regnets = {'regnetx_002':regnetx_002, 'regnetx_004':regnetx_004, 'regnetx_008':regnetx_008,'regnetx_006':regnetx_006, 'regnetx_016':regnetx_016,
           'regnety_002':regnety_002, 'regnety_004':regnety_004,'regnety_006':regnety_006,'regnety_008':regnety_008,'regnety_016':regnety_016}
rexnets = {'rexnet_100':rexnet_100,'rexnet_130':rexnet_130,'rexnet_150':rexnet_150,'rexnet_200':rexnet_200}

class ResNetEncoder(nn.Module):
  def __init__(self,name='resnet34',pretrained = True):
    super().__init__()
    assert name in resnets.keys(), '{} is not valid for a resnet name'.format(name)
    self.model = resnets[name](pretrained = pretrained)
    self.out_channels = self.model.fc.in_features
    del self.model.global_pool
    del self.model.fc

  def forward(self,x):
    out = self.model.forward_features(x)
    return out

class EfficientNetEncoder(nn.Module):
  def __init__(self,name='efficientnet_b0',pretrained = True):
    super().__init__()
    assert name in effnets.keys(), '{} is not valid for a efficientnet name'.format(name)
    self.model = effnets[name](pretrained = pretrained)
    self.out_channels = self.model.classifier.in_features
    del self.model.global_pool
    del self.model.classifier

  def forward(self,x):
    out = self.model.forward_features(x)
    return out

class RegNetEncoder(nn.Module):
  def __init__(self,name='regnety_002',pretrained = True):
    super().__init__()
    assert name in regnets.keys(), '{} is not valid for a regnet name'.format(name)
    self.model = regnets[name](pretrained = pretrained)
    self.out_channels = self.model.head.fc.in_features
    del self.model.head

  def forward(self,x):
    out = self.model.forward(x)
    return out

class RexNetEncoder(nn.Module):
  def __init__(self,name = 'rexnet_100',keep_rounded_channels=True,pretrained = True):
    super().__init__()
    assert name in rexnets.keys(), '{} is not valid for a rexnet name'.format(name)
    self.model = rexnets[name](pretrained = pretrained)
    self.out_channels = self.model.head.fc.in_features if(keep_rounded_channels) else self.model.features[-1].in_channels
    del self.model.head
    if(not keep_rounded_channels) : self.model.features = self.model.features[:-1]

  def forward(self,x):
    out = self.model.forward_features(x)
    return out

def get_encoder(name='resnet50',pretrained = True,**kwargs):
  if('resnet' in name):
    op = ResNetEncoder
  elif('efficientnet' in name):
    op = EfficientNetEncoder
  elif('regnet' in name):
    op = RegNetEncoder
  elif('rexnet' in name):
    op = RexNetEncoder  
  else:
    raise ValueError('{} model is not an option!!'.format(name))
  return op(name = name,pretrained = pretrained,**kwargs)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:30:00 2021

@author: hasan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
  def __init__(self,in_channels):
    super().__init__()
    self.spatial = nn.Conv2d(in_channels=in_channels,out_channels=1,kernel_size=1,bias=False,padding=0,stride=1)
    self.sigmoid = nn.Sigmoid()
  def forward(self,x):
    s = self.sigmoid(self.spatial(x))
    return x * s

class ChannelAttention(nn.Module):
  def __init__(self,in_channels,reduction = 16):
    super().__init__()
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.squeeze = nn.Conv2d(in_channels=in_channels,out_channels=in_channels//reduction,kernel_size=1,padding=0,stride=1)
    self.expand = nn.Conv2d(in_channels=in_channels//reduction,out_channels=in_channels,kernel_size=1,padding=0,stride=1)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
  def forward(self,x):
    c = self.pool(x)
    c = self.expand(self.relu(self.squeeze(c)))
    c = self.sigmoid(c)
    return x * c

class SCSE(nn.Module):
  def __init__(self,in_channels,reduction = 16):
    super().__init__()
    self.sa = SpatialAttention(in_channels)
    self.ca = ChannelAttention(in_channels,reduction)
  def forward(self,x):
    s = self.sa(x)
    c = self.ca(x)
    return s + c
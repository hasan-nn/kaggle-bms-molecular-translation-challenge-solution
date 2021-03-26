#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:36:16 2021

@author: hasan
"""
from albumentations import Resize,HorizontalFlip,VerticalFlip,Compose,OneOf,NoOp,\
                            Cutout,InvertImg,Normalize,ChannelShuffle,\
                            HueSaturationValue,GaussianBlur,GaussNoise,MultiplicativeNoise,ShiftScaleRotate 

pos_augs = OneOf([
                  VerticalFlip(p=1.0),
                   HorizontalFlip(p=1.0),
                   Compose([
                            VerticalFlip(1.0),
                            HorizontalFlip(p=1.0),
                                
                     ],p=1.0),
                  ShiftScaleRotate (shift_limit_x=0.04, shift_limit_y=0.05,scale_limit=(-0.02,0.1), rotate_limit=30, interpolation=1, border_mode=1,p=1.0)
                ],p=0.5)

color_augs = OneOf([
                    ChannelShuffle(p=1.0),
                    HueSaturationValue(hue_shift_limit=15, sat_shift_limit=15, val_shift_limit=15,p=1.0),
                    #Sharpen(p=1.0),
                    InvertImg(p=1.0)
                ],p=0.4)

hard_augs = OneOf([
                   GaussianBlur(blur_limit=3,p=1.0),
                   GaussNoise(var_limit=(100.0, 200.0),mean=0,p=1.0),
                   Cutout(num_holes=8, max_h_size=20, max_w_size=25, fill_value=[0],p=1.0),
                   MultiplicativeNoise(multiplier=(0.7, 1.1), per_channel=True,p=1.0)
              ],p=0.3)

tfm0 = NoOp()

tfm1 = Resize(height = 256,
              width = 448,
              interpolation =1,
              p = 1.0)

tfm2 = Compose([
                tfm1,
                pos_augs,
              ],p=1.0)

tfm3 = Compose([
                tfm1,
                pos_augs,
                color_augs,
              ],p=1.0)

tfm4 = Compose([
                tfm1,
                pos_augs,
                color_augs,
                hard_augs,
              ],p=1.0)

def get_tfm(number):
  if(number == 0):
    return tfm0
  elif(number == 1):
    return tfm1
  elif(number == 2):
    return tfm2
  elif(number == 3):
    return tfm3
  elif(number == 4):
    return tfm4
  else:
    raise ValueError('Transform {} is not an option!!'.format(number))
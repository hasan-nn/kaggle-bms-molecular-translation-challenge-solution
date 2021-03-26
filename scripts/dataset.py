#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:37:01 2021

@author: hasan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from.transforms import get_tfm
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from funcs import get_path
import random
from skimage.io import imread,imsave,imshow
from skimage.color import gray2rgb
class ImageCaptioningDataset(Dataset):
  def __init__(self,images_path,labels_path,captions,train = True, fold = 0,select_folds = None,transform_number = 1,shuffle=True,to_tensor=True):
    super().__init__()
    self.data = []
    self.train = train
    self.fold = fold
    self.select_folds = select_folds
    self.tfm = get_tfm(transform_number)
    self.shuffle = shuffle
    self.tt = to_tensor
    df = pd.read_csv(labels_path)

    if(fold != None):

      if(train):
        if(select_folds):
          if(fold in select_folds):
            print('Warning : if the same fold is used in validation,there will be a leak!!!')
          df = df.loc[(df['fold'].isin(select_folds))]
        else:
          df = df.loc[(df['fold'] != self.fold)]
      else: 
        df = df.loc[(df['fold'] == self.fold)]
        
    #captions = json.loads(open(captions_path,'r').read())
    for _,row in df.iterrows():
      self.data.append({'image_id' : row['image_id'],
                        'inchi' : row['InChI'],
                        'fold' : row['fold'],
                        'caption': captions[row['image_id']],
                        'img_path' : get_path(row['image_id'],images_path)
                        })   
    del df
    #del captions
    if(self.shuffle):random.shuffle(self.data)
    self.regen()

  def __getitem__(self, index):
    self.idx = index
    curr = self.__get_current_data__()
    img = gray2rgb(imread(curr['img_path']))
    caption = curr['caption']
    img = self.tfm(image = img)['image']
    length = len(caption)
    if(self.tt):
      img = torch.from_numpy(img.transpose((2,0,1))).float()
      caption = torch.LongTensor(caption)
      length = torch.LongTensor([length])
    return img,caption,length

  def regen(self):
    self.idx = 0

  def change_tfm(self,tfm):
    self.tfm = tfm

  def change_tt(self,tt):
    self.tt = tt
    
  def __get_current_data__(self):
    return self.data[self.idx]
    
  def __len__(self):
    return len(self.data)


def collate_fn(batch):
  imgs,labels,lengths = [],[],[]
  for dp in batch:
    imgs.append(dp[0])
    labels.append(dp[1])
    lengths.append(dp[2])
  labels = pad_sequence(labels, batch_first=True, padding_value=t2i['<BLANK>'])
  return torch.stack(imgs), labels,torch.stack(lengths).reshape(-1, 1)
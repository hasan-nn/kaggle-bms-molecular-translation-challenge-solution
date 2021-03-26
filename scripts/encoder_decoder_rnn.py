#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:33:27 2021

@author: hasan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoders import get_encoder
from lstm_decoder import LSTMDecoder
from gru_decoder import GRUDecoder
from encoder_attention import SCSE
#Encoder-Decoder

class EncoderDecoder(nn.Module):
  def __init__(self,
               encoder_name,
               pretrained_encoder,
               vocab_size ,
               use_encoder_attention = True,
               decoder_dim = 320,
               decoder_dropout = 0.2,
               device = 'cpu',
               decoder_type = 'lstm',
               **kwargs
               ):
    super().__init__()
    self.encoder = get_encoder(name=encoder_name,
                               pretrained = pretrained_encoder,
                               **kwargs)
    self.attention = SCSE(self.encoder.out_channels,reduction = 8) if(use_encoder_attention) else nn.Identity()
    _decoder = LSTMDecoder if(decoder_type == 'lstm') else GRUDecoder
    self.decoder = _decoder(vocabulary_size = vocab_size,
                               encoder_dim = self.encoder.out_channels,
                               decoder_dim=decoder_dim,
                               dropout = decoder_dropout,
                               device = device)

  def forward(self,x,captions,caption_lengths):
    feats = self.encoder(x)
    feats = self.attention(feats)
    preds,alphas,lengths = self.decoder(feats,captions,caption_lengths)
    return preds,alphas,lengths

  @torch.no_grad()
  def predict(self,x,feats = None,max_length = 300):
    if(feats is None):
      feats = self.encoder(x)
      feats = self.attention(feats)
    preds,stop_idx= self.decoder.get_preds(feats,max_length)
    return preds,stop_idx
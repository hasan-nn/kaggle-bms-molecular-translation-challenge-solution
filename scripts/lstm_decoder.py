#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:31:20 2021

@author: hasan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

#Decoder LSTM

class LSTMAttention(nn.Module):
    def __init__(self, encoder_dim,decoder_dim=512):
        super().__init__()
        self.U = nn.Linear(decoder_dim, decoder_dim)
        self.W = nn.Linear(encoder_dim, decoder_dim)
        self.v = nn.Linear(decoder_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, img_features, hidden_state):
        U_h = self.U(hidden_state).unsqueeze(1)
        W_s = self.W(img_features)
        att = self.tanh(W_s + U_h)
        e = self.v(att).squeeze(2)
        alpha = self.softmax(e)
        context = (img_features * alpha.unsqueeze(2)).sum(1)
        return context, alpha
        
class LSTMDecoder(nn.Module):
    def __init__(self, vocabulary_size, encoder_dim,decoder_dim = 512, dropout = 0,device = 'cpu'):
        super().__init__()

        self.vocab_size = vocabulary_size
        self.encoder_channels = encoder_dim
        self.decoder_channels = decoder_dim
        self.dropout_ratio = dropout
        self.device = device

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.tanh = nn.Tanh()
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(decoder_dim, vocabulary_size)
        self.dropout = nn.Dropout(p = dropout)
        self.attention = LSTMAttention(encoder_dim,decoder_dim)
        self.embedding = nn.Embedding(vocabulary_size, decoder_dim)
        self.lstm = nn.LSTMCell(decoder_dim + encoder_dim, decoder_dim,bias = True)

        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, flat_feats):
        mean_flat_feats = flat_feats.mean(dim=1)
        h = self.init_h(mean_flat_feats)  # (batch_size, decoder_dim)
        c = self.init_c(mean_flat_feats)
        return h, c

    def forward(self,feats,orig_captions,orig_lengths):

        flat_feats = feats.flatten(2).permute(0,2,1)
        batch,pixels,encoder_channels = flat_feats.size()

        lengths, sort_ind = orig_lengths.squeeze(1).sort(dim=0, descending=True)
        sort_d = {v:i for i,v in enumerate(sort_ind)}
        reverse_sort_ind = [sort_d[i] for i in sorted(sort_d.keys())]

        decode_lengths = (lengths - 1).tolist()
        orig_decode_lengths = (orig_lengths[:,0] -1).tolist()
        max_length = max(decode_lengths)

        flat_feats,captions = flat_feats[sort_ind],orig_captions[sort_ind]
        embeddings = self.embedding(captions)
        h,c = self.init_hidden_state(flat_feats)

        preds  = torch.zeros(batch, max_length, self.vocab_size).to(self.device)
        alphas = torch.zeros(batch, max_length, pixels).to(self.device)

        for t in range(max_length):

          t_batch = sum([l > t for l in decode_lengths])

          att_encoding,alpha = self.attention(flat_feats[:t_batch],h[:t_batch])
          gate = self.sigmoid(self.f_beta(h[:t_batch]))
          att_encoding = gate * att_encoding

          cat_att_embeds = torch.cat([embeddings[:t_batch, t, :],att_encoding],dim=1)
          h,c = self.lstm(cat_att_embeds, (h[:t_batch], c[:t_batch]))

          pred = self.fc(self.dropout(h))
          preds[:t_batch,t,:] = pred
          alphas[:t_batch,t,:] = alpha

        preds,alphas = preds[reverse_sort_ind],alphas[reverse_sort_ind]
        return preds,alphas,orig_decode_lengths

    def get_preds(self,feats,max_length):
        flat_feats = feats.flatten(2).permute(0,2,1)
        batch,pixels,encoder_channels = flat_feats.size()

        start_tokens = torch.ones(batch,dtype = torch.long).to(self.device)* t2i['<END>']
        embeddings = self.embedding(start_tokens)
        h,c = self.init_hidden_state(flat_feats)
        
        preds = torch.zeros(batch, max_length, self.vocab_size).to(self.device) 
        stop_inds = torch.ones(batch,dtype = torch.long).to(self.device) * -1
        completed = []
        for t in range(max_length):
          att_encoding,alpha = self.attention(flat_feats,h)
          gate = self.sigmoid(self.f_beta(h))
          att_encoding = gate * att_encoding

          cat_att_embeds = torch.cat([embeddings,att_encoding],dim=1)
          h,c = self.lstm(cat_att_embeds, (h, c))

          pred = self.fc(self.dropout(h))
          preds[:,t,:] = pred

          max_args = torch.argmax(pred,-1) 

          for ind,max_arg in enumerate(max_args):
            if(max_arg.item() == t2i['<END>'] and ind not in completed):
              stop_inds[ind] = t
              completed.append(ind)
          if(-1 not in stop_inds):break 
          embeddings = self.embedding(max_args)
        return preds,stop_inds.tolist()
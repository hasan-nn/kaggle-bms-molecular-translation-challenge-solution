#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:26:24 2021

@author: hasan
"""

import re
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

all_atoms = ['C','Br', 'N', 'Si', 'H', 'O', 'B', 'F', 'S', 'I', 'P', 'Cl']
prefix = 'InChI=1S/'
token_atoms = ['C','br', 'N', 'Si', 'H', 'O', 'B', 'F', 'S', 'I', 'P', 'cl']
joined_atoms = '|'.join(token_atoms)
tail_tokens = ['(', ')', '+', ',', '-', '/', '0', '1', '10', '11', '12', '13', '14', '15', '16', '17',
               '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30',
               '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44',
               '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58',
               '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72',
               '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87',
               '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', 'D', 'H', 'T', 'b',
               'h', 'i', 'm', 's', 't']
def get_atoms(formula):
  formula = formula.replace('Cl','cl').replace('Br','br')
  matches =  re.findall(r"(?=((("+joined_atoms+r")(\d+))|("+joined_atoms+r")))", formula)
  atoms = []
  for match in matches:
    if(match[2] != ''):
      atom,nmbr = match[2],int(match[3])
    else:
      atom,nmbr = match[0],1
    atom = atom.replace('cl','Cl').replace('br','Br')
    atoms.append((atom,nmbr))
  return atoms

def atoms2string(atoms):
    s = ''
    for atom,number in atoms:
        s+= f'{atom} '
        if(number>1):
            s+=f'{number} '
    return s#.rstrip(' ')#.split(' ') 

def split_tail(tail):
    matches=['/']
    num_s = ''
    for w in tail:   
        try:
            elem = int(w)
            num_s+=w
            if(len(num_s)==2):
                matches.append(num_s)
                num_s=''
        except:
            if(len(num_s)>0):
                matches.append(num_s)
            matches.append(w)
            num_s=''
    if(len(num_s)>0):
        matches.append(num_s)
    return ' '.join(matches)

def extract_tokens(ikey):
    splits = ikey.split('/')
    head = splits[1]
    tail = '/'.join(splits[2:])
    
    head_split = atoms2string(get_atoms(head))
    tail_split = split_tail(tail)
    tokens = head_split + tail_split

    return tokens.rstrip(' ')

def get_path(iid,pth):
  return str(pth) + '/{}/{}/{}/{}.png'.format(*iid[:3],iid)

def get_dicts(tokens):
  token_converter = {}
  idx_converter = {}
  for i,token in enumerate([*tokens,'<BEGIN>','<END>','<BLANK>']):
    token_converter[token] = i
  for k,v in token_converter.items():
    idx_converter[v] = k
  return token_converter,idx_converter

def encode(tokens,t2i):
  f1 = lambda x : t2i[x]
  return [t2i['<BEGIN>'],*list(map(f1,tokens)),t2i['<END>']]


def decode(caption,i2t):
  f2 = lambda x : i2t[x]
  return list(map(f2,caption))

def create_folds(df,folds = 10):
  to_frame = []
  for i,row in tqdm(df.iterrows()):
    iid = row['image_id']
    selection = iid[:3]
    to_frame.append({'image_id': iid,'selection' : selection})
  df1 = pd.DataFrame(to_frame,columns = ['image_id','selection'])
  X = df1.groupby('image_id')['selection'].first().index.values
  y = df1.groupby('image_id')['selection'].first().values
  skf = StratifiedKFold(n_splits = folds, random_state = 911, shuffle=True) 
  for i,(tfold,vfold) in enumerate(skf.split(X,y)):
    df1.loc[df1['image_id'].isin(X[vfold]),'fold']=int(i)
  _folds=[int(fold) for fold in df1.groupby('fold').first().index.values]
  for fold in _folds:
    print(f'fold:\t{fold}')
    print(df1.loc[df1['fold']==fold].set_index(['fold','selection']).count(level='selection'))
  return df1
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:44:16 2021

@author: hasan
"""

import numpy as np

#meters
class Meter(object):
    def reset(self):
        pass
    def update(self,value):
        pass
    def get_update(self):
        pass
    def set_name(self,name):
      self.name = f'{name}_{self.kind}'

class AverageMeter(Meter):
    
    def __init__(self):
        super(AverageMeter,self).__init__()
        self.reset()
        self.kind = 'avg'
    def reset(self):
        self.value = 0
        self.average = 0
        self.count = 0
    def update(self,value):
        self.count += 1
        self.value = value
        self.average = ((self.average * (self.count - 1)) + self.value)/float(self.count)
    def update_all(self,values):
        l = len(values)
        self.sum = np.sum(values)
        self.count += l
        self.average = ((self.average * (self.count - l)) + self.sum)/float(self.count)
    def get_update(self):
        return self.average

class InstantMeter(Meter):
    
    def __init__(self):
        super(InstantMeter,self).__init__()
        self.reset()
        self.kind = 'instant'
    def reset(self):
        self.value = 0
        
    def update(self,value):
        self.value = value
        
    def get_update(self):
        return self.value
    
class ShortTermMemoryMeter(Meter):
    def __init__(self,memory_length):
        super(ShortTermMemoryMeter,self).__init__()
        self.reset()
        self.memory_length = memory_length
        self.kind = f'short_term_{memory_length}'
        assert(self.memory_length >1)
    def reset(self):
        self.value = 0
        self.length = 0
        self.in_memory = []
        self.average = 0
    def update(self,value):
        self.value = value 
        if (self.length >= self.memory_length):
            self.in_memory = self.in_memory[1:]
        self.in_memory.append(self.value)
        self.average = np.average(np.array(self.in_memory))
        self.length = len(self.in_memory)
    def get_update(self):
        return self.average
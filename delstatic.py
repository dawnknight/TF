# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 22:53:36 2017

@author: Dawnknight
"""

import cPickle
import numpy as np
from random import shuffle as sf

def conti_num(array,val = 0,len_th = 1):
    length = []
    fnum   = []
    start = 0
    cnt = 0    
    for i in range(len(array)):
        if array[i] == val:
            cnt += 1
        else:
            if cnt >len_th:
                length.append(cnt)
                fnum.append([start,i-1])
            start = i+1
            cnt = 0
    return fnum,length
    
    
Motion_th = 10
    
K = cPickle.load(file('./Concatenate_Data/limb_K_ex4.pkl','r'))
M = cPickle.load(file('./Concatenate_Data/limb_M_ex4.pkl','r'))

vec_K = (K-np.roll(K,1,axis = 1))[:,1:]**2
vec_M = (M-np.roll(M,1,axis = 1))[:,1:]**2

Mo_K = np.sum((vec_K[::3,:]+vec_K[1::3,:]+vec_K[2::3,:])**0.5,0)
Mo_K = np.sum((vec_M[::3,:]+vec_M[1::3,:]+vec_M[2::3,:])**0.5,0)


fnum,length = conti_num((Mo_K>Motion_th)*1,val = 0,len_th = 19)


index = range(len(K[1]))

del_idx = []

for i in  fnum:
    tmp = range(i[0],i[1]+1)
    sf(tmp)
    del_idx += tmp[:-10]

index = np.setxor1d(index,del_idx)
    
    
















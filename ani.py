# -*- coding: utf-8 -*-
"""
Created on Thu Feb 02 22:55:51 2017

@author: Dawnknight
"""

import cPickle,h5py
import numpy as np

W1   = h5py.File('model.h5','r')['W1'][:]
W2   = h5py.File('model.h5','r')['W2'][:]
Wp1  = h5py.File('model.h5','r')['Wp1'][:]
Wp2  = h5py.File('model.h5','r')['Wp2'][:]
b1   = h5py.File('model.h5','r')['b1'][:]
b2   = h5py.File('model.h5','r')['b2'][:]
bp1  = h5py.File('model.h5','r')['bp1'][:]
bp2  = h5py.File('model.h5','r')['bp2'][:]
[KMIN,KMAX,MMIN,MMAX]  = h5py.File('model.h5','r')['minmax'][:]
path  = 'C:/Users/Dawnknight/Documents/GitHub/Kproject/data/Motion and Kinect'

data  = cPickle.load(file(path+'/Unified_MData/Andy_2016-12-15 04.15.27 PM_FPS30_motion_unified_ex4.pkl'))
kdata = cPickle.load(file(path+'/Unified_KData/Andy_12151615_Kinect_unified_ex4.pkl'))


for i in kdata.keys():
    if i == 0:
        joints = kdata[i]
    else:
        joints = np.vstack([joints,kdata[i]])

joints = joints.T

#====== denoise =============

NJ = (joints*2-KMIN-KMAX)/(KMAX-KMIN)
L1 = np.maximum(np.dot(NJ,W1)+b1,0)
L2 = np.maximum(np.dot(L1,W2)+b2,0)
bL1 = np.maximum(np.dot(L2,Wp1)+bp1,0)
bL2 = np.maximum(np.dot(bL1,Wp2)+bp2,0)

KJ_mod = (bL2*(KMAX-KMIN)+(KMAX+KMIN))/2




















# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 01:54:13 2017

@author: Dawnknight
"""

import h5py,cPickle
import numpy as np
import glob,os

#src_path  = 'D:/Project/K_project/data/'
src_path   = 'I:/AllData_0327/'
Trgfolder  = 'unified data array/Unified_MData/'
Infolder   = 'GPRresult/K2M_800/'

Err = 0
TotalLen = 0
for Infile,Trgfile in zip(glob.glob(os.path.join(src_path+Infolder,'*.h5')),\
                          glob.glob(os.path.join(src_path+Trgfolder,'*.pkl'))): 
    print(Infile)
    print(Trgfile)
    Indata = h5py.File(Infile,'r')['data'][:]
    Trgdata  = cPickle.load(file(Trgfile,'rb'))[12:30,:]
    
    Len = min(Trgdata.shape[1],Indata.shape[1])
    Trgdata = Trgdata[:,:Len]
    Indata  = Indata[:,:Len]
    
    Err += np.sum(np.sum(((Trgdata-Indata).reshape(-1,3,Len))**2,axis = 1)**0.5)
    TotalLen += Len
    
print(Err/TotalLen/6)
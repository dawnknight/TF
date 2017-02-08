# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 17:16:17 2017

@author: medialab
"""
import os , glob , cPickle,copy
import numpy as np



path = 'D:/Project/PyKinect2-master/Kproject/data/Motion and Kinect/'

Kfolder = 'Unified_KData/'
Mfolder = 'Unified_MData/'

batchsize = 30 # group of joints
jnum = 33

for kinfile,minfile in zip(glob.glob(os.path.join(path+Kfolder,'*ex4.pkl')),glob.glob(os.path.join(path+Mfolder,'*FPS30_motion_unified_ex4.pkl'))):
#    print  kinfile
#    print  minfile  
    kbatch = np.zeros([jnum,batchsize])
    mbatch = np.zeros([jnum,batchsize])
    kdata = cPickle.load(file(kinfile,'rb'))
    mdata = cPickle.load(file(minfile,'rb'))
    length = min(kdata[0].shape[1],kdata[0].shape[1])
    for i in kdata.keys():
        if i == 0:
            Kjoints = kdata[i]
            Mjoints = mdata[i]
        else:
            Kjoints = np.vstack([Kjoints,kdata[i]])
            Mjoints = np.vstack([Mjoints,mdata[i]])
    for idx,i in enumerate(xrange(batchsize-1,length)):
        
        if idx == 0:
           kbatch[:] = copy.copy(Kjoints[:,idx:batchsize+idx]) 
           mbatch[:] = copy.copy(Mjoints[:,idx:batchsize+idx]) 

            
            
            
            
            
            
            
            
         
         
     
     
     
    
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 17:16:17 2017

@author: medialab
"""
import os , glob , cPickle , h5py
import numpy as np
from random import shuffle as sf


#path = 'C:/Users/Dawnknight/Documents/GitHub/Kproject/data/Motion and Kinect/'
path  = 'D:\Project\PyKinect2-master\Kproject\data\Motion and Kinect/'


Kfolder = 'Unified_KData/'
Mfolder = 'Unified_MData/'

batchsize = 30 # group of joints
jnum = 18
index = 0

#st = time.clock()

#for kinfile,minfile in zip(glob.glob(os.path.join(path+Kfolder,'*ex4.pkl')),glob.glob(os.path.join(path+Mfolder,'*FPS30_motion_unified_ex4.pkl'))):
#    print('group_'+str(index+1)+'......')
#    print  kinfile
#    print  minfile  
#    print('==================================\n\n\n\n\n')
#
#    kdata = cPickle.load(file(kinfile,'rb'))
#    mdata = cPickle.load(file(minfile,'rb'))
#    length = min(kdata[0].shape[1],mdata[0].shape[1])
#    for i in kdata.keys():
#        if i == 0:
#            Kjoints = kdata[i]
#            Mjoints = mdata[i]
#        else:
#            Kjoints = np.vstack([Kjoints,kdata[i]])
#            Mjoints = np.vstack([Mjoints,mdata[i]])
##    pdb.set_trace()        
#    for idx,i in enumerate(xrange(batchsize-1,length)):
#        
#        if idx == 0:
#            Ksubtensor = copy.copy(Kjoints[:,idx:batchsize+idx])
#            Msubtensor = copy.copy(Mjoints[:,idx:batchsize+idx])
#        else: 
#            Ksubtensor = np.dstack([Ksubtensor,Kjoints[:,idx:batchsize+idx]])
#            Msubtensor = np.dstack([Msubtensor,Mjoints[:,idx:batchsize+idx]])
##    pdb.set_trace()
#    if index == 0:
#        Ktensor = Ksubtensor
#        Mtensor = Msubtensor
#    else:
#        Ktensor = np.dstack([Ktensor,Ksubtensor])
#        Mtensor = np.dstack([Mtensor,Msubtensor]) 
#    index += 1
#    
#print time.clock()-st

LEN = []
Ksubtensor = {}
Msubtensor = {}    
for kinfile,minfile in zip(glob.glob(os.path.join(path+Kfolder,'*ex4.pkl')),glob.glob(os.path.join(path+Mfolder,'*FPS30_motion_unified_ex4.pkl'))):
    print('group_'+str(index+1)+'......')
    print  kinfile
    print  minfile  
    print('==================================\n\n\n')

    kdata = cPickle.load(file(kinfile,'rb'))
    mdata = cPickle.load(file(minfile,'rb'))
    length = min(kdata[0].shape[1],mdata[0].shape[1])
    LEN.append(length-batchsize+1) 
    Ksubtensor[index] = np.zeros([jnum,batchsize,length-batchsize+1])
    Msubtensor[index] = np.zeros([jnum,batchsize,length-batchsize+1])
    
    for i in [4,5,6,8,9,10]:#kdata.keys():
        if i == 4:
            Kjoints = kdata[i]
            Mjoints = mdata[i]
        else:
            Kjoints = np.vstack([Kjoints,kdata[i]])
            Mjoints = np.vstack([Mjoints,mdata[i]])
        
    for idx,i in enumerate(xrange(batchsize-1,length)):
        
            Ksubtensor[index][:,:,idx] = Kjoints[:,idx:batchsize+idx]
            Msubtensor[index][:,:,idx] = Mjoints[:,idx:batchsize+idx]
    index += 1
    
Ktensor = np.zeros([jnum,batchsize,sum(LEN)])
Mtensor = np.zeros([jnum,batchsize,sum(LEN)])
start = 0
end   = 0
for i in range(index-1):
        end += LEN[i]
        Ktensor[:,:,start:end] = Ksubtensor[i]
        Mtensor[:,:,start:end] = Msubtensor[i]
        start = end


idx = range(sum(LEN))
testrate = 0.2
sf(idx)

MAX = np.max([Ktensor.max(),Mtensor.max()])
MIN = np.min([Ktensor.min(),Mtensor.min()])

#normalized to 0 to 1
NK = (Ktensor-MIN)/(MAX-MIN)
NM = (Mtensor-MIN)/(MAX-MIN)

teX   = NK[:,:,:int(0.2*sum(LEN))]
teL   = NM[:,:,:int(0.2*sum(LEN))]
trX   = NK[:,:,int(0.2*sum(LEN)):]
trL   = NM[:,:,int(0.2*sum(LEN)):] 

f = h5py.File("NLdata_batch.h5", "w")
f.create_dataset('train_data' , data = trX) 
f.create_dataset('train_label', data = trL) 
f.create_dataset('test_data'  , data = teX) 
f.create_dataset('test_label' , data = teL) 
f.create_dataset('idx'        , data = idx) 
f.create_dataset('minmax'     , data =[MIN,MAX]) 
f.close()             
        
f = h5py.File("batch.h5", "w")
f.create_dataset('Kdata' , data = Ktensor)
f.create_dataset('Mdata' , data = Mtensor)  

f.close()              
            

 
            
            
            
            
            
         
         
     
     
     
    
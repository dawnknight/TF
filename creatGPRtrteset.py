# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 14:56:47 2017

@author: medialab
"""

'''
create training testing set 

'''

import cPickle,glob,os
import numpy as np
from random import shuffle as sf

#src_path  = 'D:/Project/K_project/data/'
src_path  = 'I:/AllData_0327/'
Mfolder   = 'unified data array/Unified_MData/'
Kfolder  = 'unified data array/Unified_KData/'
Rfolder   = 'unified data array/reliability_mod/'

Rel_th = 0.7

dataset={}
totallen = 0
for idx,(Kfile,Mfile,Rfile) in enumerate(zip(glob.glob(os.path.join(src_path+Kfolder,'*.pkl')),\
                                              glob.glob(os.path.join(src_path+Mfolder,'*ex4_FPS30_motion_unified.pkl')),\
                                              glob.glob(os.path.join(src_path+Rfolder,'*ex4.pkl')))):
    
    mdata   = cPickle.load(file(Mfile,'rb'))
    rdata   = cPickle.load(file(Rfile,'rb'))
    kdata   = cPickle.load(file(Kfile,'rb')) 

    Len     = min(kdata.shape[1],mdata.shape[1])
    totallen += Len 

    dataset[idx]={}
    dataset[idx]['Kname'] = Kfile
    dataset[idx]['Mname'] = Mfile
    dataset[idx]['Rname'] = Rfile
    dataset[idx]['length'] = Len
    dataset[idx]['totallen'] = totallen
    
    
    if idx == 0:
        
        M  = mdata[:,:Len]
        K  = kdata[:,:Len]
        R  = rdata[: ,:Len]
        
    else:
        M  = np.hstack([M , mdata[:,:Len]])
        K  = np.hstack([K , kdata[:,:Len]])
        R  = np.hstack([R , rdata[:,:Len]])
        
        
        
        
        
relidx = np.where(np.sum((R<Rel_th)*1,0)==0)[0]   # frames which have all joints reliable
unrelidx = np.where(np.sum((R>Rel_th)*1,0)!=6)[0] 



Rel_M = M[:,relidx]
Rel_K = K[:,relidx]
Rel_R = R[:,relidx]
unRel_M = M[:,unrelidx]
unRel_K = K[:,unrelidx] 
unRel_R = R[:,unrelidx]

Rel_sidx = range(len(relidx))
unRel_sidx = range(len(unrelidx))

sf(Rel_sidx)
sf(unRel_sidx)

rate = 0.8

#dataset['Rel_train_M'] = Rel_M[:,:int(len(Rel_sidx)*0.8) ]
#dataset['Rel_test_M' ] = Rel_M[:, int(len(Rel_sidx)*0.8):]
#dataset['Rel_train_K'] = Rel_K[:,:int(len(Rel_sidx)*0.8) ]
#dataset['Rel_test_K' ] = Rel_K[:, int(len(Rel_sidx)*0.8):]
#dataset['Rel_train_R'] = Rel_R[:,:int(len(Rel_sidx)*0.8) ]
#dataset['Rel_test_R' ] = Rel_R[:, int(len(Rel_sidx)*0.8):]
#
#
#dataset['unRel_train_M'] = unRel_M[:,:int(len(unRel_sidx)*0.8) ]
#dataset['unRel_test_M' ] = unRel_M[:, int(len(unRel_sidx)*0.8):]
#dataset['unRel_train_K'] = unRel_K[:,:int(len(unRel_sidx)*0.8) ]
#dataset['unRel_test_K' ] = unRel_K[:, int(len(unRel_sidx)*0.8):]
#dataset['unRel_train_R'] = unRel_R[:,:int(len(unRel_sidx)*0.8) ]
#dataset['unRel_test_R' ] = unRel_R[:, int(len(unRel_sidx)*0.8):]


dataset['Rel_train_M'] = Rel_M[:,Rel_sidx[:int(len(Rel_sidx)*0.8) ]]
dataset['Rel_test_M' ] = Rel_M[:,Rel_sidx[ int(len(Rel_sidx)*0.8):]]
dataset['Rel_train_K'] = Rel_K[:,Rel_sidx[:int(len(Rel_sidx)*0.8) ]]
dataset['Rel_test_K' ] = Rel_K[:,Rel_sidx[ int(len(Rel_sidx)*0.8):]]
dataset['Rel_train_R'] = Rel_R[:,Rel_sidx[:int(len(Rel_sidx)*0.8) ]]
dataset['Rel_test_R' ] = Rel_R[:,Rel_sidx[ int(len(Rel_sidx)*0.8):]]


dataset['unRel_train_M'] = unRel_M[:,unRel_sidx[:int(len(unRel_sidx)*0.8) ]]
dataset['unRel_test_M' ] = unRel_M[:,unRel_sidx[ int(len(unRel_sidx)*0.8):]]
dataset['unRel_train_K'] = unRel_K[:,unRel_sidx[:int(len(unRel_sidx)*0.8) ]]
dataset['unRel_test_K' ] = unRel_K[:,unRel_sidx[ int(len(unRel_sidx)*0.8):]]
dataset['unRel_train_R'] = unRel_R[:,unRel_sidx[:int(len(unRel_sidx)*0.8) ]]
dataset['unRel_test_R' ] = unRel_R[:,unRel_sidx[ int(len(unRel_sidx)*0.8):]]

dataset['Rel_sidx']   = relidx
dataset['unRel_sidx'] = unrelidx

dataset['Mdata'] = M
dataset['Kdata'] = K
dataset['Rdata'] = R

cPickle.dump(dataset,file('GPR_training_testing_RANDset33.pkl','wb'))
















        
        
        
        
        
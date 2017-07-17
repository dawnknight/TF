# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 16:58:47 2017

@author: medialab
"""



import cPickle,glob,os
import numpy as np
from random import shuffle as sf

#src_path  = 'D:/Project/K_project/data/'
src_path  = 'F:/AllData_0327/'
exeno     = 'ex4'
Mfolder   = 'unified data array/Unified_MData/'
Kfolder   = 'unified data array/Unified_KData/'
Kraw      = 'Motion and Kinect raw data/3D_kinect_joint/'
Mraw      = 'Motion and Kinect raw data/Not_unified_Mdata/'
Rfolder   = 'unified data array/reliability_mod/'
#Rfolder   = 'unified data array/reliability/'

Rel_th = 0.7

dataset={}
totallen = 0
for idx,(Kfile,Mfile,Rfile,rKfile,rMfile) in enumerate(zip(glob.glob(os.path.join(src_path+Kfolder+exeno+'/' ,'*'+exeno+'.pkl')),\
                                                            glob.glob(os.path.join(src_path+Mfolder+exeno+'/' ,'*'+exeno+'_FPS30_motion_unified.pkl')),\
                                                            glob.glob(os.path.join(src_path+Rfolder+exeno+'/' ,'*'+exeno+'.pkl')),\
                                                            glob.glob(os.path.join(src_path+Kraw   +exeno+'/' ,'*'+exeno+'.pkl')),\
                                                            glob.glob(os.path.join(src_path+Mraw   +exeno+'/' ,'*'+exeno+'_FPS30_motion.pkl')))):
    
    mdata   = cPickle.load(file(Mfile,'rb'))
    rdata   = cPickle.load(file(Rfile,'rb'))
    kdata   = cPickle.load(file(Kfile,'rb')) 
    rkdata  = cPickle.load(file(rKfile,'rb')) 
    rmdata  = cPickle.load(file(rMfile,'rb'))
    

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
        rK = rkdata[:,:Len]
        rM = rmdata[:,:Len]
    else:
        M  = np.hstack([M , mdata[:,:Len]])
        K  = np.hstack([K , kdata[:,:Len]])
        R  = np.hstack([R , rdata[:,:Len]])
        rK = np.hstack([rK, rkdata[:,:Len]])
        rM = np.hstack([rM , rmdata[:,:Len]])
        
        
        
relidx   = np.where(np.sum((R<Rel_th)*1,0)==0)[0]   # frames which have all joints reliable
unrelidx = np.where(np.sum((R<Rel_th)*1,0)!=0)[0]



Rel_M  = M[:,relidx]
Rel_K  = K[:,relidx]
Rel_R  = R[:,relidx]
Rel_rK = rK[:,relidx]
Rel_rM = rM[:,relidx]

unRel_M  = M[:,unrelidx]
unRel_K  = K[:,unrelidx] 
unRel_R  = R[:,unrelidx]
unRel_rK = rK[:,unrelidx]
unRel_rM  = rM[:,unrelidx]


Rel_sidx = range(len(relidx))
unRel_sidx = range(len(unrelidx))

sf(Rel_sidx)
sf(unRel_sidx)

rate = 0.8



dataset['Rel_train_M']  = Rel_M[:,Rel_sidx[:int(len(Rel_sidx)*0.8) ]]
dataset['Rel_test_M' ]  = Rel_M[:,Rel_sidx[ int(len(Rel_sidx)*0.8):]]
dataset['Rel_train_K']  = Rel_K[:,Rel_sidx[:int(len(Rel_sidx)*0.8) ]]
dataset['Rel_test_K' ]  = Rel_K[:,Rel_sidx[ int(len(Rel_sidx)*0.8):]]
dataset['Rel_train_R']  = Rel_R[:,Rel_sidx[:int(len(Rel_sidx)*0.8) ]]
dataset['Rel_test_R' ]  = Rel_R[:,Rel_sidx[ int(len(Rel_sidx)*0.8):]]
dataset['Rel_train_rK'] = Rel_rK[:,Rel_sidx[:int(len(Rel_sidx)*0.8) ]]
dataset['Rel_test_rK' ] = Rel_rK[:,Rel_sidx[ int(len(Rel_sidx)*0.8):]]
dataset['Rel_train_rM'] = Rel_rM[:,Rel_sidx[:int(len(Rel_sidx)*0.8) ]]
dataset['Rel_test_rM' ] = Rel_rM[:,Rel_sidx[ int(len(Rel_sidx)*0.8):]]


dataset['unRel_train_M']  = unRel_M[:,unRel_sidx[:int(len(unRel_sidx)*0.8) ]]
dataset['unRel_test_M' ]  = unRel_M[:,unRel_sidx[ int(len(unRel_sidx)*0.8):]]
dataset['unRel_train_K']  = unRel_K[:,unRel_sidx[:int(len(unRel_sidx)*0.8) ]]
dataset['unRel_test_K' ]  = unRel_K[:,unRel_sidx[ int(len(unRel_sidx)*0.8):]]
dataset['unRel_train_R']  = unRel_R[:,unRel_sidx[:int(len(unRel_sidx)*0.8) ]]
dataset['unRel_test_R' ]  = unRel_R[:,unRel_sidx[ int(len(unRel_sidx)*0.8):]]
dataset['unRel_train_rK'] = unRel_rK[:,unRel_sidx[:int(len(unRel_sidx)*0.8) ]]
dataset['unRel_test_rK' ] = unRel_rK[:,unRel_sidx[ int(len(unRel_sidx)*0.8):]]
dataset['unRel_train_rM'] = unRel_rM[:,unRel_sidx[:int(len(unRel_sidx)*0.8) ]]
dataset['unRel_test_rM' ] = unRel_rM[:,unRel_sidx[ int(len(unRel_sidx)*0.8):]]

dataset['Rel_sidx']   = Rel_sidx
dataset['unRel_sidx'] = unRel_sidx

dataset['Mdata']  = M
dataset['Kdata']  = K
dataset['Rdata']  = R
dataset['rKdata'] = rK
dataset['rMdata'] = rM

cPickle.dump(dataset,file('GPR_training_testing_RANDset33_w_raw_'+exeno+'.pkl','wb'))
















        
        
        
        
        
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 22:57:23 2017

@author: Dawnknight
"""

import h5py,glob,os
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from sklearn.cluster import KMeans

try :
    import cPickle
except:
    import _pickle as cPickle

from sklearn.externals import joblib
from scipy.spatial.distance import  cdist







#src_path  = 'D:/Project/K_project/data/'
src_path   = 'I:/AllData_0327/'
Trgfolder  = 'unified data array/Unified_MData/'
Infolder   = 'GPRresult/K2Kprime_800/'
Rfolder    = 'unified data array/reliability_mod/'
#Rfolder   = 'unified data array/reliability/'

Rel_th = 0.7

dataset={}
totallen = 0

InType  = 'h5'
TrgType = 'pkl'
RType   = 'pkl'

for idx,(Infile,Trgfile,Rfile) in enumerate(zip(glob.glob(os.path.join(src_path+Infolder,'*.'+InType)),\
                                                glob.glob(os.path.join(src_path+Trgfolder,'*.'+TrgType)),\
                                                glob.glob(os.path.join(src_path+Rfolder,'*ex4.'+RType)))):

    if InType == 'h5':
        Indata  = h5py.File(Infile,'r')['data'][:]
    else:    
        Indata   = cPickle.load(file(Infile,'rb'))
        
    if TrgType == 'h5':
        Trgdata  = h5py.File(Trgfile,'r')['data'][:]
    else:    
        Trgdata   = cPickle.load(file(Trgfile,'rb')) 
        
    if RType == 'h5':
        rdata  = h5py.File(Rfile,'r')['data'][:]
    else:    
        rdata   = cPickle.load(file(Rfile,'rb'))         

     

    Len     = min(Indata.shape[1],Trgdata.shape[1])
    totallen += Len 

    dataset[idx]={}
    dataset[idx]['Kname'] = Infile
    dataset[idx]['Trgname'] = Trgfile
    dataset[idx]['Rname'] = Rfile
    dataset[idx]['length'] = Len
    dataset[idx]['totallen'] = totallen
    
    
    if idx == 0:
        
        Trg  = Trgdata[:,:Len]
        In   = Indata[:,:Len]
        R    = rdata[: ,:Len]
        
    else:
        Trg  = np.hstack([Trg, Trgdata[:,:Len]])
        In   = np.hstack([In , Indata[:,:Len]])
        R    = np.hstack([R  , rdata[:,:Len]])
        
        
        
        
        
relidx = cPickle.load(file('GPR_training_testing_RANDset33.pkl','rb'))['Rel_sidx']
unrelidx = cPickle.load(file('GPR_training_testing_RANDset33.pkl','rb'))['unRel_sidx']


Rel_Trg = Trg[:,relidx]
Rel_In  = In[:,relidx]
Rel_R   = R[:,relidx]

unRel_Trg = Trg[:,unrelidx]
unRel_In  = In[:,unrelidx] 
unRel_R   = R[:,unrelidx]

Rel_sidx   = relidx
unRel_sidx = unrelidx

dataset['Rel_train_Trg'] = Rel_Trg[:,Rel_sidx[:int(len(Rel_sidx)*0.8) ]]
dataset['Rel_test_Trg' ] = Rel_Trg[:,Rel_sidx[ int(len(Rel_sidx)*0.8):]]
dataset['Rel_train_In']  = Rel_In[ :,Rel_sidx[:int(len(Rel_sidx)*0.8) ]]
dataset['Rel_test_In' ]  = Rel_In[ :,Rel_sidx[ int(len(Rel_sidx)*0.8):]]
dataset['Rel_train_R']   = Rel_R[  :,Rel_sidx[:int(len(Rel_sidx)*0.8) ]]
dataset['Rel_test_R' ]   = Rel_R[  :,Rel_sidx[ int(len(Rel_sidx)*0.8):]]


dataset['unRel_train_Trg'] = unRel_Trg[:,unRel_sidx[:int(len(unRel_sidx)*0.8) ]]
dataset['unRel_test_Trg' ] = unRel_Trg[:,unRel_sidx[ int(len(unRel_sidx)*0.8):]]
dataset['unRel_train_In']  = unRel_In[ :,unRel_sidx[:int(len(unRel_sidx)*0.8) ]]
dataset['unRel_test_In' ]  = unRel_In[ :,unRel_sidx[ int(len(unRel_sidx)*0.8):]]
dataset['unRel_train_R']   = unRel_R[  :,unRel_sidx[:int(len(unRel_sidx)*0.8) ]]
dataset['unRel_test_R' ]   = unRel_R[  :,unRel_sidx[ int(len(unRel_sidx)*0.8):]]

dataset['Rel_sidx']   = Rel_sidx
dataset['unRel_sidx'] = unRel_sidx

dataset['Trgdata'] = Trg
dataset['Indata']  = In
dataset['Rdata']   = R

cPickle.dump(dataset,file('GPR_Kprime2M_Randset.pkl','wb'))


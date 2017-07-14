# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:11:42 2017

@author: medialab
"""

import h5py,glob,os
import numpy as np

try :
    import cPickle
except:
    import _pickle as cPickle

from sklearn.externals import joblib
from scipy.spatial.distance import  cdist


src_path  = 'I:/AllData_0327/'
#src_path  = 'D:/Project/K_project/data/'
#Mfolder   = 'unified data array/Unified_MData/'
#Infolder  = 'unified data array/Unified_KData/'
Infolder  = 'GPRresult/K2Kprime_800/'
gprfolder = 'GPR_Kernel/'
dstfolder = 'GPRresult/Kprime2M(K2M)_800/'

Rel_th    = 0.7
factor    = 5

[MIN,MAX] = h5py.File('./data/CNN/model_CNN_0521_K2M_rel.h5','r')['minmax'][:]


def uni_vec(Body):
    vec = np.roll(Body,-3,axis = 0)-Body

    tmp = ((vec**2).reshape(-1,3,vec.shape[1]).sum(axis=1))**.5
    vlen = np.insert(np.insert(tmp,np.arange(6),tmp,0),np.arange(0,12,2),tmp,0)

    return vec/vlen

def cov(sita0,sita1,W1,W2,noise_level,x1,x2):
    
    dists1 = cdist(x1 / W1, x2 / W1,metric='sqeuclidean')
    dists2 = cdist(x1 / W2, x2 / W2,metric='sqeuclidean')
       
        
    k1=np.exp(-.5 * dists1) 
    k2=np.exp(-.5 * dists2)
    k_return=sita0*k1+sita1*k2
    if np.array_equal(x1,x2):
        k_return=k_return+noise_level
    return k_return



def gp_pred(testdata,gp):
    
    #===== find the parameter of gp =====
    parameter=gp.kernel_.get_params(deep=True)
    sita0 = parameter["k1__k1__k1__constant_value"]
    W1    = parameter["k1__k1__k2__length_scale"]
    sita1 = parameter["k1__k2__k1__constant_value"]
    W2    = parameter["k1__k2__k2__length_scale"]
    noise_level = parameter["k2__noise_level"]    
    traindata = gp.X_train_
    
#    L_    = gp.L_
#    L_inv = np.linalg.inv(L_)
    alpha_= gp.alpha_
    y_train_mean = gp.y_train_mean
    
    #===== Prediction ======
    K_trans = cov(sita0,sita1,W1,W2,noise_level,testdata,traindata)    
#    k1      = cov(sita0,sita1,W1,W2,noise_level,testdata,testdata)    
#    v       = np.dot(L_inv,K_trans.T)             
#    y_cov   = k1-K_trans.dot(v)
    y_mean  = K_trans.dot(alpha_)  
    
    return y_train_mean + y_mean 







gp = joblib.load(src_path+gprfolder+'GPR_K2M_cluster_800.pkl')


#Type = 'pkl'
Type = 'h5'


    
for Infile in glob.glob(os.path.join(src_path+Infolder,'*.'+Type)):
    print(Infile) 
    
    if Type =='pkl':
        Indata  = (cPickle.load(file(Infile,'rb'))[12:30,:]-MIN)/(MAX-MIN)
        fname = src_path+dstfolder+Infile.split('\\')[-1][:-4]+'.h5'
    else:       
        Indata  = (h5py.File(Infile,'r')['data'][:] -MIN)/(MAX-MIN)
        fname = src_path+dstfolder+Infile.split('\\')[-1][:-3]+'.h5'
        
#    GPR_result = gp.predict(Indata.T)
    GPR_result = gp_pred(Indata.T,gp)
                  
    GPR_result     = (GPR_result*(MAX-MIN)+MIN).T
    uni_GPR_result = np.zeros(GPR_result.shape)
    univec_test_rel   = uni_vec(GPR_result)
    
    uni_GPR_result[0:3  ,:] = GPR_result[0:3  ,:]
    uni_GPR_result[9:12 ,:] = GPR_result[9:12 ,:]

    uni_GPR_result[3:6  ,:] = GPR_result[0:3  ,:]+univec_test_rel[0:3  ,:]*33.2*factor
    uni_GPR_result[6:9  ,:] = GPR_result[3:6  ,:]+univec_test_rel[3:6  ,:]*27.1*factor
    uni_GPR_result[12:15,:] = GPR_result[9:12 ,:]+univec_test_rel[9:12 ,:]*33.2*factor
    uni_GPR_result[15:  ,:] = GPR_result[12:15,:]+univec_test_rel[12:15,:]*27.1*factor
    
    f = h5py.File(fname,'w')
    f.create_dataset('data',data = uni_GPR_result)
    f.close()


















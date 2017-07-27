# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 22:05:57 2017

@author: medialab
"""
import h5py,cPickle,pdb
import numpy as np
#import project_function
from   sklearn.gaussian_process import GaussianProcessRegressor
from   sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from   sklearn.cluster import KMeans
from   sklearn.externals import joblib
from scipy.spatial.distance import  cdist
#import Gaussian_predict_function

def uni_vec(Body):
    
    spsh = Body[18:,:]        #spine shoulder
    Lsh  = Body[:3 ,:]        #left shoulder
    Rsh  = Body[9:12 ,:]      #right shoulder
    
    vec = np.roll(Body[:18,:],-3,axis = 0)-Body[:18,:]
    tmp = ((vec**2).reshape(-1,3,vec.shape[1]).sum(axis=1))**.5
    vlen = np.insert(np.insert(tmp,np.arange(6),tmp,0),np.arange(0,12,2),tmp,0)

    vec_L2s =  Lsh - spsh 
    vec_R2s =  Rsh - spsh
    
    tmp_L2s = ((vec_L2s**2).reshape(-1,3,vec.shape[1]).sum(axis=1))**.5
    tmp_R2s = ((vec_R2s**2).reshape(-1,3,vec.shape[1]).sum(axis=1))**.5
    
    vlen_L2s = np.insert(np.insert(tmp_L2s,np.arange(1),tmp_L2s,0),np.arange(0,2,2),tmp_L2s,0)
    vlen_R2s = np.insert(np.insert(tmp_R2s,np.arange(1),tmp_R2s,0),np.arange(0,2,2),tmp_R2s,0)
    
    

    return vec/vlen,vec_L2s/vlen_L2s,vec_R2s/vlen_R2s

def BoneLen(Body):
    
    spsh = Body[18:,:]        #spine shoulder
    Lsh  = Body[:3 ,:]        #left shoulder
    Rsh  = Body[9:12 ,:]      #right shoulder 

    vec = np.roll(Body[:18,:],-3,axis = 0)-Body[:18,:]
    tmp = ((vec**2).reshape(-1,3,vec.shape[1]).sum(axis=1))**.5
    
    vlen  = np.mean(tmp,axis = 1)
    
    vec_L2s =  Lsh - spsh 
    vec_R2s =  Rsh - spsh
    
    tmp_L2s = ((vec_L2s**2).reshape(-1,3,vec.shape[1]).sum(axis=1))**.5
    tmp_R2s = ((vec_R2s**2).reshape(-1,3,vec.shape[1]).sum(axis=1))**.5

    vlen_L2s = np.mean(tmp_L2s)    
    vlen_R2s = np.mean(tmp_R2s) 
    
    return vlen,vlen_L2s,vlen_R2s
    
def data2real(data,refK,refM):

    refdata  = refK.T
    
    real_data = np.zeros(data.shape)     
    univec, univec_L, univec_R  = uni_vec(data)
    bonelen, bonelen_L, bonelen_R = BoneLen(refM.T)

    real_data[18:  ,:] = refdata[18:  ,:]
       
    real_data[0:3  ,:] = real_data[18:  ,:] + univec_L*bonelen_L
    real_data[9:12 ,:] = real_data[18:  ,:] + univec_R*bonelen_R

    real_data[3:6  ,:] = real_data[0:3  ,:] +univec[0:3  ,:]*bonelen[0]
    real_data[6:9  ,:] = real_data[3:6  ,:] +univec[3:6  ,:]*bonelen[1]
    real_data[12:15,:] = real_data[9:12 ,:] +univec[9:12 ,:]*bonelen[3]
    real_data[15:18,:] = real_data[12:15,:] +univec[12:15,:]*bonelen[4]

    return real_data

exeno     = '_ex4'
src_path  = 'F:/AllData_0327/'
Rfolder   = 'unified data array/reliability/'
gprfolder = 'GPR_Kernel/'
Errfolder = 'GPR_cluster_err/'

n_cluster = 800
Rel_th    = 0.7
feature   = '_meter_shum_full'

kernel_gpml = 66.0**2 * RBF(length_scale=67.0)+0.18**2 * RBF(length_scale=0.134)\
               + WhiteKernel(noise_level=0.19**2)
kernel_sep  = 1.0*RBF(length_scale=1.0)+ConstantKernel()+WhiteKernel()

[MIN,MAX]   = h5py.File('./data/CNN/model_CNN_0521_K2M_rel.h5','r')['minmax'][:]

File          = cPickle.load(file('GPR_training_testing_RANDset33_w_raw'+exeno+'.pkl','rb'))

M_train_rel   = File['Rel_train_M'][12:,:].T
K_train_rel   = File['Rel_train_K'][12:,:].T # normalize later 
rK_train_rel  = File['Rel_train_rK'][12:,:].T
rM_train_rel  = File['Rel_train_rM'][12:,:].T

K_test_rel    = (File['Rel_test_K'][12:,:].T-MIN)/(MAX-MIN) 
M_test_rel    =  File['Rel_test_M'][12:,:].T
rK_test_rel   =  File['Rel_test_rK'][12:,:].T 
rM_test_rel   =  File['Rel_test_rM'][12:,:].T

K_test_unrel  = (File['unRel_test_K'][12:,:].T-MIN)/(MAX-MIN) 
M_test_unrel  =  File['unRel_test_M'][12:,:].T 
R_test_unrel  =  File['unRel_test_R'][4:,:]
rK_test_unrel =  File['unRel_test_rK'][12:,:].T
rM_test_unrel =  File['unRel_test_rM'][12:,:].T

M             =  File['Mdata'][12:,:].T 
K             = (File['Kdata'][12:,:].T-MIN)/(MAX-MIN) 
R             =  File['Rdata'][4:,:] 
rK            = File['rKdata'][12:,:].T
rM            = File['rMdata'][12:,:].T 


Rmtx = np.insert(np.insert(R,np.arange(7),R,0),np.arange(0,14,2),R,0)

Rmtx_test_unrel =np.insert(np.insert(R_test_unrel,np.arange(7),R_test_unrel,0),np.arange(0,14,2),R_test_unrel,0)


# === K_test_unrel ===

#y_test_unrel    = sep_kernel_pred(K_test_unrel,GP,n_cluster,K_center)
#data_test_unrel = ((K_test_unrel + y_test_unrel)*(MAX-MIN)+MIN).T
    
uni_K_test_unrel    = data2real((K_test_unrel*(MAX-MIN)+MIN).T,rK_test_unrel,rM_test_unrel)
uni_M_test_unrel    = data2real(M_test_unrel.T ,rK_test_unrel,rM_test_unrel )
    



# === K_test ===

K_test   = np.vstack([K_test_rel ,K_test_unrel])
M_test   = np.vstack([M_test_rel ,M_test_unrel])
rK_test  = np.vstack([rK_test_rel,rK_test_unrel])
rM_test  = np.vstack([rM_test_rel,rM_test_unrel])



uni_K_test = data2real((K_test*(MAX-MIN)+MIN).T,rK_test,rM_test)
uni_M_test    = data2real(M_test.T ,rK_test,rM_test) 


err_test_unrel = np.sum(np.sum( (((uni_M_test_unrel-uni_K_test_unrel)*(Rmtx_test_unrel<Rel_th))\
                                .reshape(-1,3,uni_M_test_unrel.shape[1]))**2,axis=1)**0.5)/np.sum(R_test_unrel<Rel_th)

err_test       = np.sum(np.sum(((uni_M_test-uni_K_test).reshape(-1,3,uni_M_test.shape[1]))**2,axis=1)**0.5)/K_test.shape[0]/6 
 

raw_err_test_unrel = np.sum(np.sum( (((uni_M_test_unrel-rK_test_unrel.T)*(Rmtx_test_unrel<Rel_th))\
                                .reshape(-1,3,uni_M_test_unrel.shape[1]))**2,axis=1)**0.5)/np.sum(R_test_unrel<Rel_th)

raw_err_test       = np.sum(np.sum(((uni_M_test-rK_test.T).reshape(-1,3,uni_M_test.shape[1]))**2,axis=1)**0.5)/K_test.shape[0]/6 

print exeno 
print('Err_test_unrel=',err_test_unrel)
print('Err_test ='     ,err_test)
print('raw_Err_test_unrel=',raw_err_test_unrel)
print('raw_Err_test ='     ,raw_err_test)


err_test_unrel = np.sum(np.sum( (((rM_test_unrel.T-rK_test_unrel.T)*(Rmtx_test_unrel<Rel_th))\
                                .reshape(-1,3,uni_M_test_unrel.shape[1]))**2,axis=1)**0.5)/np.sum(R_test_unrel<Rel_th)

#
#bias = (rM[:,18:]-rK[:,18:]).T.reshape(-1,3,50247)
#
#Bias = np.vstack([bias,bias,bias,bias,bias,bias,bias])
#
#MM = (rM.T.reshape(-1,3,50247)+Bias)
#
#KK = rK.T.reshape(-1,3,50247)
#
#
#np.mean(np.sum((MM-KK)**2,axis=1)**0.5,axis=1)
#
#
#MM[6,:,0]-KK[6,:,0]



bias = (rM[:,18:]-rK[:,18:]).T

a = (rM.reshape(-1,3,rM.shape[0])-bias)

b = rK[0,:].reshape(-1,3)

np.sum((a-b)**2,axis=1)**0.5



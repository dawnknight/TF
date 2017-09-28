# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 14:37:40 2017

@author: medialab

using pre-trianed GPR kernel to generate the K2M data from unified Kinect data


"""



import h5py,glob,os,pdb
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.datasets import fetch_mldata
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
try :
    import cPickle
except:
    import _pickle as cPickle

from sklearn.externals import joblib
from time import time
from scipy.spatial.distance import  cdist

def cov(sita0,sita1,W1,W2,noise_level,x1,x2):
    
    dists1 = cdist(x1 / W1, x2 / W1,metric='sqeuclidean')
    dists2 = cdist(x1 / W2, x2 / W2,metric='sqeuclidean')
       
        
    k1=np.exp(-.5 * dists1) 
    k2=np.exp(-.5 * dists2)
    k_return=sita0*k1+sita1*k2
    if np.array_equal(x1,x2):
        k_return=k_return+noise_level
    return k_return

def gp_pred(testdata,traindata,gp):
    
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

def uni_mod(data,refk,factor=5):
    refdata = refk.T
    uni_data  = np.zeros(data.shape)
    univec, univec_L, univec_R = uni_vec(data)
    
    uni_data[18:  ,:] = refdata[18:  ,:]
       
    uni_data[0:3  ,:] = uni_data[18:  ,:] + univec_L*16.65*factor
    uni_data[9:12 ,:] = uni_data[18:  ,:] + univec_R*16.65*factor

    uni_data[3:6  ,:] = uni_data[0:3  ,:] +univec[0:3  ,:]*33.2*factor
    uni_data[6:9  ,:] = uni_data[3:6  ,:] +univec[3:6  ,:]*27.1*factor
    uni_data[12:15,:] = uni_data[9:12 ,:] +univec[9:12 ,:]*33.2*factor
    uni_data[15:18,:] = uni_data[12:15,:] +univec[12:15,:]*27.1*factor
     
    return uni_data

[MIN,MAX] = h5py.File('./data/CNN/model_CNN_0521_K2M_rel.h5','r')['minmax'][:]

#src_path  = 'F:/AllData_0327/'
#src_path  = 'C:/Users/Dawnknight/Documents/GitHub/K_project/data/'
src_path   = 'D:/Project/K_project/data/'
dst_folder = 'K2M_800_0928/' 
Kfolder    = 'unified data array/Unified_KData/'
gprfolder  = 'GPR_Kernel/'


Rel_th    =  0.7
factor    =  5

ncluster  = 800
exeno     = '_ex4'
feature   = '_meter_fix'   

gp = joblib.load(src_path+gprfolder+'0928/'+'GPR_cluster_'+repr(ncluster)+feature+exeno+'.pkl')
centroids_K = cPickle.load(file(src_path+gprfolder+'0928/'+'centroids_K_'+repr(ncluster)+exeno+'.pkl','rb'))

for infile in glob.glob(os.path.join(src_path+Kfolder,'*.pkl')):
    print infile
    
    K         = (cPickle.load(file(infile,'rb')).T[:,12:]-MIN)/(MAX-MIN)
    K_pred    = gp_pred(K,centroids_K,gp)
    K_rescale = (K_pred*(MAX-MIN)+MIN).T 
    K_uni     = uni_mod(K_rescale,K)
    
    fname     = src_path+gprfolder+dst_folder+exeno[1:]+'/'++infile.split('\\')[-1] 
    cPickle.dump(K_uni,file(fname,'wb'))
    
    

















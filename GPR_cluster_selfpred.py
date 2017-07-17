# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:44:17 2017

@author: medialab
"""


import h5py
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from sklearn.cluster import KMeans

try :
    import cPickle
except:
    import _pickle as cPickle

from sklearn.externals import joblib
from time import time
from scipy.spatial.distance import  cdist

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



def gp_pred(testdata,traindata,gp):
    
    #===== find the parameter of gp =====
    parameter=gp.kernel_.get_params(deep=True)
    sita0 = parameter["k1__k1__k1__constant_value"]
    W1    = parameter["k1__k1__k2__length_scale"]
    sita1 = parameter["k1__k2__k1__constant_value"]
    W2    = parameter["k1__k2__k2__length_scale"]
    noise_level = parameter["k2__noise_level"]    

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

[MIN,MAX] = h5py.File('./data/CNN/model_CNN_0521_K2M_rel.h5','r')['minmax'][:]
#
src_path  = 'F:/AllData_0327/'
#src_path  = 'D:/Project/K_project/data/'

exeno = 'ex3'
gprfolder = 'GPR_Kernel/'




kernel_gpml = 66.0**2 * RBF(length_scale=67.0)+ 0.18**2 * RBF(length_scale=0.134) + WhiteKernel(noise_level=0.19**2)



M_train_rel   = cPickle.load(file('GPR_training_testing_RANDset33_'+exeno+'.pkl','rb'))['Rel_train_M'][12:30,:].T
K_train_rel   = cPickle.load(file('GPR_training_testing_RANDset33_'+exeno+'.pkl','rb'))['Rel_train_K'][12:30,:].T # normalize later 



       
M_rel = (M_train_rel -MIN)/(MAX-MIN) 
K_rel = (K_train_rel -MIN)/(MAX-MIN) 

#M_rel  =  M.T[relidx ,:]
#Mp_rel =  Mp.T[relidx,:]


Err={}
Err['all']={}
Err['unrel']={}
Err['test_rel']={}
Err['test_unrel']={}
Err['test_err'] ={}

ncluster =800

# Cluster of Mocap Data
print('Mocap Clustering('+repr(ncluster)+')')
t0=time()

print('start Kmeans clustering')
kmeans = KMeans(n_clusters=ncluster, random_state=None,init='k-means++',n_init=10).fit(M_rel)
labels_M = kmeans.predict(M_rel)
print('Kmeans clustering finish')

print(time()-t0)

# Align centroids
centroids_M  = np.zeros((ncluster,18),dtype=np.float64)
centroids_K  = np.zeros((ncluster,18),dtype=np.float64)

for i in range(0,ncluster):
    centroids_M[i,:]=np.mean(M_rel[labels_M==i,:],axis=0)
    centroids_K[i,:]=np.mean(K_rel[labels_M==i,:],axis=0)

# Gaussian Regression

gp = GaussianProcessRegressor(kernel=kernel_gpml)

print('Training')
gp.fit(centroids_K, centroids_M)
#gp.fit(centroids_M, centroids_K)

joblib.dump(gp,src_path+gprfolder+exeno+'/GPR_K2M_cluster_'+repr(ncluster)+'_'+exeno+'.pkl')

















    

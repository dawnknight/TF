# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 15:39:45 2017

@author: medialab
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


def uni_vec(Body):
    
#    pdb.set_trace()
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
#    vlen = np.insert(np.insert(tmp,np.arange(6),tmp,0),np.arange(0,12,2),tmp,0)    
    vlen  = np.mean(tmp,axis = 1)
    
    vec_L2s =  Lsh - spsh 
    vec_R2s =  Rsh - spsh
    
    tmp_L2s = ((vec_L2s**2).reshape(-1,3,vec.shape[1]).sum(axis=1))**.5
    tmp_R2s = ((vec_R2s**2).reshape(-1,3,vec.shape[1]).sum(axis=1))**.5
    
#    vlen_L2s = np.insert(np.insert(tmp_L2s,np.arange(1),tmp_L2s,0),np.arange(0,2,2),tmp_L2s,0)
#    vlen_R2s = np.insert(np.insert(tmp_R2s,np.arange(1),tmp_R2s,0),np.arange(0,2,2),tmp_R2s,0)

    vlen_L2s = np.mean(tmp_L2s)    
    vlen_R2s = np.mean(tmp_R2s) 
    
#    pdb.set_trace()
    return vlen,vlen_L2s,vlen_R2s
    


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


#def data2real(data,refK,refM):
#
#    refdata  = refK.T
#    
#    real_data = np.zeros(data.shape)     
#    univec, univec_L, univec_R  = uni_vec(data)
#    bonelen, bonelen_L, bonelen_R = BoneLen(refM.T)
#
#    real_data[18:  ,:] = refdata[18:  ,:]
#       
#    real_data[0:3  ,:] = real_data[18:  ,:] + univec_L*bonelen_L
#    real_data[9:12 ,:] = real_data[18:  ,:] + univec_R*bonelen_R
#
#    real_data[3:6  ,:] = real_data[0:3  ,:] +univec[0:3  ,:]*bonelen[0:3  ,:]
#    real_data[6:9  ,:] = real_data[3:6  ,:] +univec[3:6  ,:]*bonelen[3:6  ,:]
#    real_data[12:15,:] = real_data[9:12 ,:] +univec[9:12 ,:]*bonelen[9:12 ,:]
#    real_data[15:18,:] = real_data[12:15,:] +univec[12:15,:]*bonelen[12:15,:]
#
#    return real_data
def data2real(data,refK,refM):
    # mapping data from pixel to cm
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
[MIN,MAX] = h5py.File('./data/CNN/model_CNN_0521_K2M_rel.h5','r')['minmax'][:]

#src_path  = 'F:/AllData_0327/'
#src_path  = 'C:/Users/Dawnknight/Documents/GitHub/K_project/data/'
src_path  = 'D:/AllData_0327(0220)/AllData_0327/'
Mfolder   = 'unified data array/Unified_MData/'
Kfolder  = 'unified data array/Unified_KData/'
# Rfolder   = 'unified data array/reliability/'
gprfolder = 'GPR_Kernel/'
Errfolder = 'GPR_cluster_err/'

Rel_th    =  0.7
factor    =  5

exeno     = '_ex3'
feature   = '_meter_fix'   


kernel_gpml = 66.0**2 * RBF(length_scale=67.0)+ 0.18**2 * RBF(length_scale=0.134) + WhiteKernel(noise_level=0.19**2)

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

jErr={}
jErr['all']={}
jErr['unrel']={}
jErr['test_rel']={}
jErr['test_unrel']={}
jErr['test_err'] ={}

rawErr={}
rawErr['all']={}
rawErr['unrel']={}
rawErr['test_rel']={}
rawErr['test_unrel']={}
rawErr['test_err'] ={}


for ncluster in range(800,900,100):

    # Cluster of Mocap Data
    print('Mocap Clustering(',ncluster,')')
    t0=time()

    print('start Kmeans clustering')
    kmeans = KMeans(n_clusters=ncluster, random_state=None,init='k-means++',n_init=10).fit(M_rel)
    labels_M = kmeans.predict(M_rel)
    print('Kmeans clustering finish')

    print(time()-t0)

    # Align centroids
    centroids_M  = np.zeros((ncluster,21),dtype=np.float64)
    centroids_K  = np.zeros((ncluster,21),dtype=np.float64)

    for i in range(0,ncluster):
        centroids_M[i,:]=np.mean(M_rel[labels_M==i,:],axis=0)
        centroids_K[i,:]=np.mean(K_rel[labels_M==i,:],axis=0)

    # Gaussian Regression

    # pdb.set_trace()
    gp = GaussianProcessRegressor(kernel=kernel_gpml, n_restarts_optimizer=0)

    print('Training')
    gp.fit(centroids_K, centroids_M)


    joblib.dump(gp,src_path+gprfolder+'kmean/'+'GPR_cluster_'+repr(ncluster)+feature+exeno+'_opt.pkl')


    print('Predicting')
    
#    y_pred  = gp.predict(K)


    y_pred  = gp_pred(K,centroids_K,gp)
      
    data     = (y_pred*(MAX-MIN)+MIN).T 
    
    uni_data = data2real(data,rK,rM)
    uni_M    = data2real(M.T,rK,rM)

    
    # === K_test_rel ===
    
#    y_test_rel        = gp.predict(K_test_rel)
    y_test_rel        = gp_pred(K_test_rel,centroids_K,gp)
                    
    data_test_rel     = (y_test_rel*(MAX-MIN)+MIN).T
    
    uni_data_test_rel  = data2real(data_test_rel,rK_test_rel,rM_test_rel)
    uni_M_test_rel = data2real(M_test_rel.T ,rK_test_rel,rM_test_rel)    
    

    # === K_test_unrel ===
    
#    y_test_unrel        = gp.predict(K_test_unrel)
    y_test_unrel        = gp_pred(K_test_unrel,centroids_K,gp)
                           
    data_test_unrel     = (y_test_unrel*(MAX-MIN)+MIN).T
    
    uni_data_test_unrel = data2real(data_test_unrel,rK_test_unrel,rM_test_unrel)
    uni_M_test_unrel    = data2real(M_test_unrel.T ,rK_test_unrel,rM_test_unrel )

    
    # === K_test ===

    K_test        = np.vstack([K_test_rel ,K_test_unrel])
    M_test        = np.vstack([M_test_rel ,M_test_unrel])
    rK_test       = np.vstack([rK_test_rel,rK_test_unrel])
    rM_test       = np.vstack([rM_test_rel,rM_test_unrel])
    
    y_test        = gp_pred(K_test,centroids_K,gp)
                           
    data_test     = (y_test*(MAX-MIN)+MIN).T
    
    uni_data_test = data2real(data_test,rK_test,rM_test)
    uni_M_test    = data2real(M_test.T ,rK_test,rM_test)    
    



    # unified data err

    err            = np.sum(np.sum(((uni_M-uni_data).reshape(-1,3,uni_M.shape[1]))**2,axis=1)**0.5) /rK.shape[0]/6
    err_unrel      = np.sum(np.sum((((uni_M-uni_data)*(Rmtx<Rel_th)).reshape(-1,3,uni_M.shape[1]))**2,axis=1)**0.5)/np.sum(R<Rel_th)

    err_test_rel   = np.sum(np.sum(((uni_M_test_rel-uni_data_test_rel).reshape(-1,3,uni_M_test_rel.shape[1]))**2,axis=1)**0.5)/K_test_rel.shape[0]/6 
    err_test_unrel = np.sum(np.sum( (((uni_M_test_unrel-uni_data_test_unrel)*(Rmtx_test_unrel<Rel_th))\
                                    .reshape(-1,3,uni_M_test_unrel.shape[1]))**2,axis=1)**0.5)/np.sum(R_test_unrel<Rel_th)

    err_test       = np.sum(np.sum(((uni_M_test-uni_data_test).reshape(-1,3,uni_M_test.shape[1]))**2,axis=1)**0.5)/K_test.shape[0]/6 

    # unified joints err

    jerr            = np.sum(np.sum(((uni_M-uni_data).reshape(-1,3,uni_M.shape[1]))**2,axis=1)**0.5,axis = 1) /rK.shape[0]
    jerr_unrel      = np.sum(np.sum((((uni_M-uni_data)*(Rmtx<Rel_th)).reshape(-1,3,uni_M.shape[1]))**2,axis=1)**0.5,axis = 1)/np.sum(R<Rel_th,axis=1)

    jerr_test_rel   = np.sum(np.sum(((uni_M_test_rel-uni_data_test_rel).reshape(-1,3,uni_M_test_rel.shape[1]))**2,axis=1)**0.5,axis = 1)/K_test_rel.shape[0] 
    jerr_test_unrel = np.sum(np.sum( (((uni_M_test_unrel-uni_data_test_unrel)*(Rmtx_test_unrel<Rel_th))\
                                    .reshape(-1,3,uni_M_test_unrel.shape[1]))**2,axis=1)**0.5,axis = 1)/np.sum(R_test_unrel<Rel_th,axis = 1)

    jerr_test       = np.sum(np.sum(((uni_M_test-uni_data_test).reshape(-1,3,uni_M_test.shape[1]))**2,axis=1)**0.5,axis = 1)/K_test.shape[0] 

    
    # raw data err
    raw_err            = np.sum(np.sum(((M.T- data).reshape(-1,3,M.shape[0]))**2,axis=1)**0.5)/rK.shape[0]/6
    raw_err_unrel      = np.sum(np.sum((((M.T-data)*(Rmtx<Rel_th)).reshape(-1,3,M.shape[0]))**2,axis=1)**0.5)/np.sum(R<Rel_th)
    raw_err_test_rel   = np.sum(np.sum(((M_test_rel.T- data_test_rel).reshape(-1,3,M_test_rel.shape[0]))**2,axis=1)**0.5)/K_test_rel.shape[0]/6 
    raw_err_test_unrel = np.sum(np.sum( (((M_test_unrel.T- data_test_unrel)*(Rmtx_test_unrel<Rel_th))\
                                   .reshape(-1,3,M_test_unrel.shape[0]))**2,axis=1)**0.5)/np.sum(R_test_unrel<Rel_th)

    raw_err_test       = np.sum(np.sum(((M_test.T- data_test).reshape(-1,3,M_test.shape[0]))**2,axis=1)**0.5)/K_test.shape[0]/6     
    
    

    
    Err['all'][ncluster]        = err
    Err['unrel'][ncluster]      = err_unrel
    Err['test_rel'][ncluster]   = err_test_rel
    Err['test_unrel'][ncluster] = err_test_unrel    
    Err['test_err'][ncluster]   = err_test
    
    jErr['all'][ncluster]        = jerr
    jErr['unrel'][ncluster]      = jerr_unrel
    jErr['test_rel'][ncluster]   = jerr_test_rel
    jErr['test_unrel'][ncluster] = jerr_test_unrel    
    jErr['test_err'][ncluster]   = jerr_test

    rawErr['all'][ncluster]        = raw_err
    rawErr['unrel'][ncluster]      = raw_err_unrel
    rawErr['test_rel'][ncluster]   = raw_err_test_rel
    rawErr['test_unrel'][ncluster] = raw_err_test_unrel    
    rawErr['test_err'][ncluster]   = raw_err_test
    
    print('Err='           ,err)
    print('Err_unrel='     ,err_unrel)
    print('Err_test_rel='  ,err_test_rel)
    print('Err_test_unrel=',err_test_unrel)
    print('Err_test ='     ,err_test)
    
    print('jErr='           ,jerr)
    print('jErr_unrel='     ,jerr_unrel)
    print('jErr_test_rel='  ,jerr_test_rel)
    print('jErr_test_unrel=',jerr_test_unrel)
    print('jErr_test ='     ,jerr_test) 

    print('raw_Err='           ,raw_err)
    print('raw_Err_unrel='     ,raw_err_unrel)
    print('raw_Err_test_rel='  ,raw_err_test_rel)
    print('raw_Err_test_unrel=',raw_err_test_unrel)
    print('raw_Err_test ='     ,raw_err_test)    

    
    fname    = src_path+Errfolder+'Err_'+repr(ncluster).zfill(5)+feature+exeno+'.pkl'
    jfname = src_path+Errfolder+'joint_Err'+repr(ncluster).zfill(5)+feature+exeno+'.pkl'
    rfname = src_path+Errfolder+'raw_Err'+repr(ncluster).zfill(5)+feature+exeno+'.pkl'

    cPickle.dump(Err,open(fname,'wb'))
    cPickle.dump(jErr,open(jfname,'wb'))
    cPickle.dump(rawErr,open(rfname,'wb'))

print exeno
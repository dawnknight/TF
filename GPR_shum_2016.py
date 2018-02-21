# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:04:26 2017

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

def cov(sita0,sita1,W,beta_1,x1,x2):
    
    dists    = cdist(x1 / W, x2 / W, metric='sqeuclidean')
    k1       = np.exp(-.5 * dists)   
    k_return = sita0*k1+sita1
    if np.array_equal(x1,x2):
        k_return = k_return + beta_1        
    return k_return

def gp_pred(gp,testdata,n_cluster,K_center):
    
    #===== find the parameter of gp ===== 
    
    
    W      = np.zeros(n_cluster)
    y_mean = np.zeros((testdata.shape[0],testdata.shape[1],n_cluster))
    S      = np.zeros((testdata.shape[0],n_cluster))
    y_pred = np.zeros(testdata.shape)
#    pdb.set_trace()
    
    for i in range(n_cluster):
        parameter=gp[i].kernel_.get_params(deep=True)
        sita0    = parameter["k1__k1__k1__constant_value"]
        W[i]     = parameter["k1__k1__k2__length_scale"]
        sita1    = parameter["k1__k2__constant_value"]
        beta_1   = parameter["k2__noise_level"]
        alpha_   = gp[i].alpha_
        
        y_train_mean = gp[i].y_train_mean
        traindata    = gp[i].X_train_
        # the process of predict    
        K_trans      = cov(sita0,sita1,W[i],beta_1,testdata,traindata) 
        S[:,i]       = cdist(np.atleast_2d(testdata / W[i]), np.atleast_2d(K_center[i,:]/ W[i]), metric='sqeuclidean').flatten()
        y_mean[:,:,i]    = K_trans.dot(alpha_) + y_train_mean
        
#    y_pred = np.sum((S/np.sum(S,axis = 1).reshape(testdata.shape[0],-1)) *y_mean,axis = 1)
    weight = S/np.sum(S,axis = 1).reshape(testdata.shape[0],-1)
    for i in range(n_cluster):
        y_pred += y_mean[:,:,i]* weight[:,i].reshape(testdata.shape[0],-1)  
  
    return  y_pred    

def sep_kernel_pred(data,GP,n_cluster,K_center):
    SL = np.hstack([data[:,18:21],data[:,0:6]])
    EL = data[:,0:9]
    WL = data[:,3:9]
    SR = np.hstack([data[:,18:21],data[:,9:15]])
    ER = data[:,9:18]
    WR = data[:,12:18]
    SP = np.hstack([data[:,0:3],data[:,18:21],data[:,9:12]])
    
    # predict function
    print('pred SL')
    y_pred_SL = gp_pred(GP[0], SL,n_cluster,K_center[0])
    print('pred EL')
    y_pred_EL = gp_pred(GP[1], EL,n_cluster,K_center[1])
    print('pred WL')
    y_pred_WL = gp_pred(GP[2], WL,n_cluster,K_center[2])
    print('pred SR')
    y_pred_SR = gp_pred(GP[3], SR,n_cluster,K_center[3])
    print('pred ER')
    y_pred_ER = gp_pred(GP[4], ER,n_cluster,K_center[4])
    print('pred WR')
    y_pred_WR = gp_pred(GP[5], WR,n_cluster,K_center[5])
    print('pred SP')
    y_pred_SP = gp_pred(GP[6], SP,n_cluster,K_center[6])
        
    # combine data
        
    y_pred = np.hstack([y_pred_SL[:,3:6],y_pred_EL[:,3:6],y_pred_WL[:,3:6],y_pred_SR[:,3:6],y_pred_ER[:,3:6],y_pred_WR[:,3:6],y_pred_SP[:,3:6]])
    
    return y_pred

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

def Kmean_cluster(M_rel,K_rel,ncluster,n_cood):
    # Cluster of Mocap Data
    print('Mocap Clustering(',ncluster,')')
    kmeans = KMeans(n_clusters=ncluster, random_state=None,init='k-means++',n_init=10).fit(K_rel)
    labels_K = kmeans.predict(K_rel)
    # Align centroids
    centroids_M=np.zeros((ncluster,n_cood),dtype=np.float64)
    centroids_K=np.zeros((ncluster,n_cood),dtype=np.float64)
    for i in range(0,ncluster):
        centroids_M[i,:]=np.mean(M_rel[labels_K==i,:],axis=0)
        centroids_K[i,:]=np.mean(K_rel[labels_K==i,:],axis=0)
        
    return centroids_M,centroids_K,labels_K

exeno     = '_ex5'
pre       = 'old_'
src_path  = 'D:/AllData_0327(0220)/AllData_0327/'
# Rfolder   = 'unified data array/reliability/'
gprfolder = 'GPR_Kernel/'
Errfolder = 'GPR_cluster_err/'

n_cluster = 30
Rel_th    = 0.7
feature   = '_meter_shum_full'

kernel_gpml = 66.0**2 * RBF(length_scale=67.0)+0.18**2 * RBF(length_scale=0.134)\
               + WhiteKernel(noise_level=0.19**2)
kernel_sep  = 1.0*RBF(length_scale=1.0)+ConstantKernel()+WhiteKernel()

[MIN,MAX]   = h5py.File('./data/CNN/model_CNN_0521_K2M_rel.h5','r')['minmax'][:]

File          = cPickle.load(file(pre+'GPR_training_testing_RANDset33_w_raw'+exeno+'.pkl','rb'))

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


M_rel = (M_train_rel -MIN)/(MAX-MIN) 
K_rel = (K_train_rel -MIN)/(MAX-MIN) 

Rmtx = np.insert(np.insert(R,np.arange(7),R,0),np.arange(0,14,2),R,0)

Rmtx_test_unrel =np.insert(np.insert(R_test_unrel,np.arange(7),R_test_unrel,0),np.arange(0,14,2),R_test_unrel,0)

K_rel_SL = np.hstack([K_rel[:,18:21],K_rel[:,0:6]])
K_rel_EL = K_rel[:,0:9]
K_rel_WL = K_rel[:,3:9]
K_rel_SR = np.hstack([K_rel[:,18:21],K_rel[:,9:15]])
K_rel_ER = K_rel[:,9:18]
K_rel_WR = K_rel[:,12:18]
K_rel_SP = np.hstack([K_rel[:,0:3],K_rel[:,18:21],K_rel[:,9:12]])

M_rel_SL = np.hstack([M_rel[:,18:21],M_rel[:,0:6]])
M_rel_EL = M_rel[:,0:9]
M_rel_WL = M_rel[:,3:9]
M_rel_SR = np.hstack([M_rel[:,18:21],M_rel[:,9:15]])
M_rel_ER = M_rel[:,9:18]
M_rel_WR = M_rel[:,12:18]
M_rel_SP = np.hstack([M_rel[:,0:3],M_rel[:,18:21],M_rel[:,9:12]])

# Kmeans
centroids_M_SL,centroids_K_SL,label_SL = Kmean_cluster(M_rel_SL,K_rel_SL,n_cluster,9)
centroids_M_EL,centroids_K_EL,label_EL = Kmean_cluster(M_rel_EL,K_rel_EL,n_cluster,9)
centroids_M_WL,centroids_K_WL,label_WL = Kmean_cluster(M_rel_WL,K_rel_WL,n_cluster,6)
centroids_M_SR,centroids_K_SR,label_SR = Kmean_cluster(M_rel_SR,K_rel_SR,n_cluster,9)
centroids_M_ER,centroids_K_ER,label_ER = Kmean_cluster(M_rel_ER,K_rel_ER,n_cluster,9)
centroids_M_WR,centroids_K_WR,label_WR = Kmean_cluster(M_rel_WR,K_rel_WR,n_cluster,6)
centroids_M_SP,centroids_K_SP,label_SP = Kmean_cluster(M_rel_SP,K_rel_SP,n_cluster,9)

K_SL = centroids_K_SL
K_EL = centroids_K_EL
K_WL = centroids_K_WL
K_SR = centroids_K_SR
K_ER = centroids_K_ER
K_WR = centroids_K_WR
K_SP = centroids_K_SP

K_center = [K_SL,K_EL,K_WL,K_SR,K_ER,K_WR,K_SP]

# GPR process

gp_SL = {}
gp_EL = {}
gp_WL = {}
gp_SR = {}
gp_ER = {}
gp_WR = {}
gp_SP = {}
GP    = {}


for i in xrange(n_cluster):
    print('cluster'+repr(i))
    gp_SL[i] = GaussianProcessRegressor(kernel = kernel_sep)
    gp_EL[i] = GaussianProcessRegressor(kernel = kernel_sep)
    gp_WL[i] = GaussianProcessRegressor(kernel = kernel_sep)
    gp_SR[i] = GaussianProcessRegressor(kernel = kernel_sep)
    gp_ER[i] = GaussianProcessRegressor(kernel = kernel_sep)
    gp_WR[i] = GaussianProcessRegressor(kernel = kernel_sep)
    gp_SP[i] = GaussianProcessRegressor(kernel = kernel_sep)
    
    print('fit SL')
    gp_SL[i].fit(K_rel_SL[label_SL==i,:],(M_rel_SL-K_rel_SL)[label_SL==i,:])
    print('fit EL')
    gp_EL[i].fit(K_rel_EL[label_EL==i,:],(M_rel_EL-K_rel_EL)[label_EL==i,:])
    print('fit WL')
    gp_WL[i].fit(K_rel_WL[label_WL==i,:],(M_rel_WL-K_rel_WL)[label_WL==i,:])
    print('fit SR')
    gp_SR[i].fit(K_rel_SR[label_SR==i,:],(M_rel_SR-K_rel_SR)[label_SR==i,:])
    print('fit ER')
    gp_ER[i].fit(K_rel_ER[label_ER==i,:],(M_rel_ER-K_rel_ER)[label_ER==i,:])
    print('fit WR')
    gp_WR[i].fit(K_rel_WR[label_WR==i,:],(M_rel_WR-K_rel_WR)[label_WR==i,:])
    print('fit SP')
    gp_SP[i].fit(K_rel_SP[label_SP==i,:],(M_rel_SP-K_rel_SP)[label_SP==i,:])
    

    if i == 0:
       GP[0] = [gp_SL[i]] 
       GP[1] = [gp_EL[i]] 
       GP[2] = [gp_WL[i]]
       GP[3] = [gp_SR[i]] 
       GP[4] = [gp_ER[i]]
       GP[5] = [gp_WR[i]]
       GP[6] = [gp_SP[i]] 
    else:
       GP[0].append(gp_SL[i]) 
       GP[1].append(gp_EL[i]) 
       GP[2].append(gp_WL[i]) 
       GP[3].append(gp_SR[i]) 
       GP[4].append(gp_ER[i]) 
       GP[5].append(gp_WR[i]) 
       GP[6].append(gp_SP[i])  
        
#    GP[i] = [gp_SL[i],gp_EL[i],gp_WL[i],gp_SR[i],gp_ER[i],gp_WR[i],gp_SP[i]]

joblib.dump(GP,src_path+gprfolder+'kmean/'+pre+'GPR_cluster_'+repr(n_cluster)+feature+exeno+'.pkl')

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


#sep_kernel_pred(data,GP)

y            = sep_kernel_pred(K,GP,n_cluster,K_center)
data         = ((K+y)*(MAX-MIN)+MIN).T

uni_data     = data2real(data,rK,rM)
uni_M        = data2real(M.T,rK,rM)

# === K_test_rel ===

y_test_rel     = sep_kernel_pred(K_test_rel,GP,n_cluster,K_center)
data_test_rel = ((K_test_rel + y_test_rel)*(MAX-MIN)+MIN).T

uni_data_test_rel  = data2real(data_test_rel,rK_test_rel,rM_test_rel)
uni_M_test_rel = data2real(M_test_rel.T ,rK_test_rel,rM_test_rel) 

# === K_test_unrel ===

y_test_unrel    = sep_kernel_pred(K_test_unrel,GP,n_cluster,K_center)
data_test_unrel = ((K_test_unrel + y_test_unrel)*(MAX-MIN)+MIN).T
    
uni_data_test_unrel = data2real(data_test_unrel,rK_test_unrel,rM_test_unrel)
uni_M_test_unrel    = data2real(M_test_unrel.T ,rK_test_unrel,rM_test_unrel )
    
# === K_test ===

K_test   = np.vstack([K_test_rel ,K_test_unrel])
M_test   = np.vstack([M_test_rel ,M_test_unrel])
rK_test  = np.vstack([rK_test_rel,rK_test_unrel])
rM_test  = np.vstack([rM_test_rel,rM_test_unrel])

y_test    = sep_kernel_pred(K_test,GP,n_cluster,K_center)
data_test = ((K_test + y_test)*(MAX-MIN)+MIN).T

uni_data_test = data2real(data_test,rK_test,rM_test)
uni_M_test    = data2real(M_test.T ,rK_test,rM_test)   

# unified data err

err            = np.sum(np.sum(((uni_M-uni_data).reshape(-1,3,uni_M.shape[1]))**2,axis=1)**0.5) /K.shape[0]/6
err_unrel      = np.sum(np.sum((((uni_M-uni_data)*(Rmtx<Rel_th)).reshape(-1,3,uni_M.shape[1]))**2,axis=1)**0.5)/np.sum(R<Rel_th)

err_test_rel   = np.sum(np.sum(((uni_M_test_rel-uni_data_test_rel).reshape(-1,3,uni_M_test_rel.shape[1]))**2,axis=1)**0.5)/K_test_rel.shape[0]/6 
err_test_unrel = np.sum(np.sum( (((uni_M_test_unrel-uni_data_test_unrel)*(Rmtx_test_unrel<Rel_th))\
                                .reshape(-1,3,uni_M_test_unrel.shape[1]))**2,axis=1)**0.5)/np.sum(R_test_unrel<Rel_th)

err_test       = np.sum(np.sum(((uni_M_test-uni_data_test).reshape(-1,3,uni_M_test.shape[1]))**2,axis=1)**0.5)/K_test.shape[0]/6 

# unified joints err

jerr            = np.sum(np.sum(((uni_M-uni_data).reshape(-1,3,uni_M.shape[1]))**2,axis=1)**0.5,axis = 1) /K.shape[0]
jerr_unrel      = np.sum(np.sum((((uni_M-uni_data)*(Rmtx<Rel_th)).reshape(-1,3,uni_M.shape[1]))**2,axis=1)**0.5,axis = 1)/np.sum(R<Rel_th,axis=1)

jerr_test_rel   = np.sum(np.sum(((uni_M_test_rel-uni_data_test_rel).reshape(-1,3,uni_M_test_rel.shape[1]))**2,axis=1)**0.5,axis = 1)/K_test_rel.shape[0] 
jerr_test_unrel = np.sum(np.sum( (((uni_M_test_unrel-uni_data_test_unrel)*(Rmtx_test_unrel<Rel_th))\
                                .reshape(-1,3,uni_M_test_unrel.shape[1]))**2,axis=1)**0.5,axis = 1)/np.sum(R_test_unrel<Rel_th,axis = 1)

jerr_test       = np.sum(np.sum(((uni_M_test-uni_data_test).reshape(-1,3,uni_M_test.shape[1]))**2,axis=1)**0.5,axis = 1)/K_test.shape[0] 



Err['all'][n_cluster]        = err
Err['unrel'][n_cluster]      = err_unrel
Err['test_rel'][n_cluster]   = err_test_rel
Err['test_unrel'][n_cluster] = err_test_unrel    
Err['test_err'][n_cluster]   = err_test

jErr['all'][n_cluster]        = jerr
jErr['unrel'][n_cluster]      = jerr_unrel
jErr['test_rel'][n_cluster]   = jerr_test_rel
jErr['test_unrel'][n_cluster] = jerr_test_unrel    
jErr['test_err'][n_cluster]   = jerr_test

print('Err='           ,err)
print('Err_unrel='     ,err_unrel)
print('Err_test_rel='  ,err_test_rel)
print('Err_test_unrel=',err_test_unrel)
print('Err_test ='     ,err_test)

print('Err='           ,jerr)
print('Err_unrel='     ,jerr_unrel)
print('Err_test_rel='  ,jerr_test_rel)
print('Err_test_unrel=',jerr_test_unrel)
print('Err_test ='     ,jerr_test)    


fname    = src_path+Errfolder+pre+'Err_'+repr(n_cluster).zfill(5)+feature+exeno+'.pkl'
jfname   = src_path+Errfolder+pre+'joint_Err'+repr(n_cluster).zfill(5)+feature+exeno+'.pkl'

cPickle.dump(Err,open(fname,'wb'))
cPickle.dump(jErr,open(jfname,'wb'))

print 'pre :' + pre 




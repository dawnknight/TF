# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:04:59 2017

@author: Dawnknight
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
    vec = np.roll(Body,-3,axis = 0)-Body

    tmp = ((vec**2).reshape(-1,3,vec.shape[1]).sum(axis=1))**.5
    vlen = np.insert(np.insert(tmp,np.arange(6),tmp,0),np.arange(0,12,2),tmp,0)

    return vec/vlen

def cov(sita0,sita1,W1,W2,noise_level,x1,x2,Rel=0):
    if type(Rel) == int:
        dists1 = cdist(x1 / W1, x2 / W1,metric='sqeuclidean')
        dists2 = cdist(x1 / W2, x2 / W2,metric='sqeuclidean')
    else:
        dists1 = np.zeros((x1.shape[0],x2.shape[0]))
        dists2 = np.zeros((x1.shape[0],x2.shape[0]))
#        for i in range(x1.shape[0]):
#            for j in range(x2.shape[0]):
#                dists1[i,j] = np.sum(((x1[i,:]-x2[j,:])*Rel[i,:]/W1)**2)
#                dists2[i,j] = np.sum(((x1[i,:]-x2[j,:])*Rel[i,:]/W2)**2)
                
        for i in range(x1.shape[0]):
#            if sum(Rel[i,:])!=18:
#                pdb.set_trace()
            factor = 18/sum(Rel[i,:])
            dists1[i,:] = np.sum(((x2-x1[i,:])*Rel[i,:]*factor/W1)**2,axis = 1)
            dists2[i,:] = np.sum(((x2-x1[i,:])*Rel[i,:]*factor/W2)**2,axis = 1)    
                
                
        
    k1=np.exp(-.5 * dists1) 
    k2=np.exp(-.5 * dists2)
    k_return=sita0*k1+sita1*k2
    if np.array_equal(x1,x2):
        k_return=k_return+noise_level
    return k_return



def gp_pred(testdata,traindata,gp,Rel=0):
    
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
    K_trans = cov(sita0,sita1,W1,W2,noise_level,testdata,traindata,Rel)    
#    k1      = cov(sita0,sita1,W1,W2,noise_level,testdata,testdata)    
#    v       = np.dot(L_inv,K_trans.T)             
#    y_cov   = k1-K_trans.dot(v)
    y_mean  = K_trans.dot(alpha_)  
    
    return y_train_mean + y_mean 

[MIN,MAX] = h5py.File('./data/CNN/model_CNN_0521_K2M_rel.h5','r')['minmax'][:]

#src_path  = 'I:/AllData_0327/'
#src_path  = 'C:/Users/Dawnknight/Documents/GitHub/K_project/data/'
src_path  = 'D:/Project/K_project/data/'
Mfolder   = 'unified data array/Unified_MData/'
Kfolder  = 'unified data array/Unified_KData/'
Rfolder   = 'unified data array/reliability/'
gprfolder = 'GPR_Kernel/'
Errfolder = 'GPR_cluster_err/'

Rel_th    =  0.7
factor    =  5

#k1 = 66.0**2 * RBF(length_scale=67.0)  # long term smooth rising trend
#
#k2 = 0.18**2 * RBF(length_scale=0.134) \
#    + WhiteKernel(noise_level=0.19**2)  # noise terms

kernel_gpml = 66.0**2 * RBF(length_scale=67.0)+ 0.18**2 * RBF(length_scale=0.134) + WhiteKernel(noise_level=0.19**2)



M_train_rel   = cPickle.load(file('GPR_training_testing_RANDset33.pkl','rb'))['Rel_train_M'][12:30,:].T
K_train_rel   = cPickle.load(file('GPR_training_testing_RANDset33.pkl','rb'))['Rel_train_K'][12:30,:].T

K_test_rel    = (cPickle.load(file('GPR_training_testing_RANDset33.pkl','rb'))['Rel_test_K'][12:30,:].T-MIN)/(MAX-MIN) 
M_test_rel    =  cPickle.load(file('GPR_training_testing_RANDset33.pkl','rb'))['Rel_test_M'][12:30,:].T
K_test_unrel  = (cPickle.load(file('GPR_training_testing_RANDset33.pkl','rb'))['unRel_test_K'][12:30,:].T-MIN)/(MAX-MIN) 
M_test_unrel  =  cPickle.load(file('GPR_training_testing_RANDset33.pkl','rb'))['unRel_test_M'][12:30,:].T 
R_test_unrel  =  cPickle.load(file('GPR_training_testing_RANDset33.pkl','rb'))['unRel_test_R'][4:10,:]

M             =  cPickle.load(file('GPR_training_testing_RANDset33.pkl','rb'))['Mdata'][12:30,:].T
K             = (cPickle.load(file('GPR_training_testing_RANDset33.pkl','rb'))['Kdata'][12:30,:].T-MIN)/(MAX-MIN) 
R             =  cPickle.load(file('GPR_training_testing_RANDset33.pkl','rb'))['Rdata'][4:10,:] 

Rmtx = np.insert(np.insert(R,np.arange(6),R,0),np.arange(0,12,2),R,0)
Rel_mtx = np.zeros(Rmtx.shape)
Rel_mtx[:] = Rmtx
Rel_mtx[Rmtx>Rel_th] = 1
tmpidx = np.where(np.sum(Rel_mtx,0)==0)[0]
Rel_mtx[:,tmpidx]=1



Rmtx_test_unrel =np.insert(np.insert(R_test_unrel,np.arange(6),R_test_unrel,0),np.arange(0,12,2),R_test_unrel,0)
Rel_mtx_test_unrel = np.zeros(Rmtx_test_unrel.shape)
Rel_mtx_test_unrel[:] = Rmtx_test_unrel
Rel_mtx_test_unrel[Rmtx_test_unrel>Rel_th] = 1
tmpidx = np.where(np.sum(Rel_mtx_test_unrel,0)==0)[0]
Rel_mtx_test_unrel[:,tmpidx]=1

       
M_rel = (M_train_rel[:15000,:] -MIN)/(MAX-MIN) 
K_rel = (K_train_rel[:15000,:] -MIN)/(MAX-MIN) 

#M_rel  =  M.T[relidx ,:]
#Mp_rel =  Mp.T[relidx,:]


Err={}
Err['all']={}
Err['unrel']={}
Err['test_rel']={}
Err['test_unrel']={}
Err['test_err']={}

for ncluster in range(200,1100,100):

    # Cluster of Mocap Data
    print('Mocap Clustering(',ncluster,')')
    t0=time()

    print('start Kmeans clustering')
    kmeans = KMeans(n_clusters=ncluster, random_state=None,init='k-means++',n_init=10).fit(M_rel)
    labels_M = kmeans.predict(M_rel)
    print('Kmeans clustering finish')

    print(time()-t0)

    # Align centroids
    centroids_M  = np.zeros((ncluster,18),dtype=np.float64)
    centroids_K = np.zeros((ncluster,18),dtype=np.float64)

    for i in range(0,ncluster):
        centroids_M[i,:]=np.mean(M_rel[labels_M==i,:],axis=0)
        centroids_K[i,:]=np.mean(K_rel[labels_M==i,:],axis=0)

    # Gaussian Regression

    gp = GaussianProcessRegressor(kernel=kernel_gpml)

    print('Training')
    gp.fit(centroids_K, centroids_M)


#    joblib.dump(gp,src_path+gprfolder+'GPR_cluster_'+repr(ncluster)+'.pkl')


    print('Predicting')
    
#    y_pred  = gp.predict(K)


    y_pred  = gp_pred(K,centroids_K,gp,Rel_mtx.T)



        
    data     = (y_pred*(MAX-MIN)+MIN).T  
    uni_data = np.zeros(data.shape)   
    univec   = uni_vec(data)
       
    uni_data[0:3  ,:] = data[0:3  ,:]
    uni_data[9:12 ,:] = data[9:12 ,:]

    uni_data[3:6  ,:] = data[0:3  ,:]+univec[0:3  ,:]*33.2*factor
    uni_data[6:9  ,:] = data[3:6  ,:]+univec[3:6  ,:]*27.1*factor
    uni_data[12:15,:] = data[9:12 ,:]+univec[9:12 ,:]*33.2*factor
    uni_data[15:  ,:] = data[12:15,:]+univec[12:15,:]*27.1*factor





    
    # === K_test_rel ===
    
##    y_test_rel        = gp.predict(K_test_rel)
#    y_test_rel        = gp_pred(K_test_rel,centroids_K,gp)
#                  
#    data_test_rel     = (y_test_rel*(MAX-MIN)+MIN).T
#    uni_data_test_rel = np.zeros(data_test_rel.shape)
#    univec_test_rel   = uni_vec(data_test_rel)
#    
#    uni_data_test_rel[0:3  ,:] = data_test_rel[0:3  ,:]
#    uni_data_test_rel[9:12 ,:] = data_test_rel[9:12 ,:]
#
#    uni_data_test_rel[3:6  ,:] = data_test_rel[0:3  ,:]+univec_test_rel[0:3  ,:]*33.2*factor
#    uni_data_test_rel[6:9  ,:] = data_test_rel[3:6  ,:]+univec_test_rel[3:6  ,:]*27.1*factor
#    uni_data_test_rel[12:15,:] = data_test_rel[9:12 ,:]+univec_test_rel[9:12 ,:]*33.2*factor
#    uni_data_test_rel[15:  ,:] = data_test_rel[12:15,:]+univec_test_rel[12:15,:]*27.1*factor

    # === K_test_unrel ===
    
#    y_test_unrel        = gp.predict(K_test_unrel)
    y_test_unrel        = gp_pred(K_test_unrel,centroids_K,gp,Rel_mtx_test_unrel.T)
                           
    data_test_unrel     = (y_test_unrel*(MAX-MIN)+MIN).T
    uni_data_test_unrel = np.zeros(data_test_unrel.shape)
    univec_test_unrel   = uni_vec(data_test_unrel)
    
    uni_data_test_unrel[0:3  ,:] = data_test_unrel[0:3  ,:]
    uni_data_test_unrel[9:12 ,:] = data_test_unrel[9:12 ,:]

    uni_data_test_unrel[3:6  ,:] = data_test_unrel[0:3  ,:]+univec_test_unrel[0:3  ,:]*33.2*factor
    uni_data_test_unrel[6:9  ,:] = data_test_unrel[3:6  ,:]+univec_test_unrel[3:6  ,:]*27.1*factor
    uni_data_test_unrel[12:15,:] = data_test_unrel[9:12 ,:]+univec_test_unrel[9:12 ,:]*33.2*factor
    uni_data_test_unrel[15:  ,:] = data_test_unrel[12:15,:]+univec_test_unrel[12:15,:]*27.1*factor

    # === K_test ===
    
#    y_test_unrel        = gp.predict(K_test_unrel)

    K_test        = np.vstack([K_test_rel,K_test_unrel])
    M_test        = np.vstack([M_test_rel,M_test_unrel])
    R_test        = np.vstack([np.ones(M_test_rel.shape),Rel_mtx_test_unrel.T])
    y_test        = gp_pred(K_test,centroids_K,gp,R_test)
                           
    data_test     = (y_test*(MAX-MIN)+MIN).T
    uni_data_test = np.zeros(data_test.shape)
    univec_test   = uni_vec(data_test)
    
    uni_data_test[0:3  ,:] = data_test[0:3  ,:]
    uni_data_test[9:12 ,:] = data_test[9:12 ,:]

    uni_data_test[3:6  ,:] = data_test[0:3  ,:]+univec_test[0:3  ,:]*33.2*factor
    uni_data_test[6:9  ,:] = data_test[3:6  ,:]+univec_test[3:6  ,:]*27.1*factor
    uni_data_test[12:15,:] = data_test[9:12 ,:]+univec_test[9:12 ,:]*33.2*factor
    uni_data_test[15:  ,:] = data_test[12:15,:]+univec_test[12:15,:]*27.1*factor


    err = np.sum(np.sum(((M.T-uni_data).reshape(-1,3,M.shape[0]))**2,axis=1)**0.5)/50247/6
    err_unrel = np.sum(np.sum((((M.T-uni_data)*(Rmtx<Rel_th)).reshape(-1,3,M.shape[0]))**2,axis=1)**0.5)/np.sum(R<Rel_th)

#    err_test_rel = np.sum(np.sum(((M_test_rel.T-uni_data_test_rel).reshape(-1,3,M_test_rel.shape[0]))**2,axis=1)**0.5)/5651/6 
    err_test_unrel = np.sum(np.sum( (((M_test_unrel.T-uni_data_test_unrel)*(Rmtx_test_unrel<Rel_th))\
                                    .reshape(-1,3,M_test_unrel.shape[0]))**2,axis=1)**0.5)/np.sum(R_test_unrel<Rel_th)

    err_test       = np.sum(np.sum(((M_test.T-uni_data_test).reshape(-1,3,M_test.shape[0]))**2,axis=1)**0.5)/K_test.shape[0]/6 

    
    Err['all'][ncluster] = err
    Err['unrel'][ncluster] = err_unrel
#    Err['test_rel'][ncluster] = err_test_rel
    Err['test_unrel'][ncluster] = err_test_unrel    
    Err['test_err'][ncluster]   = err_test
    
    print('Err=',err)
    print('Err_unrel=',err_unrel)
#    print('Err_test_rel=',err_test_rel)
    print('Err_test_unrel=',err_test_unrel)
    print('Err_test ='     ,err_test)

    
    fname = src_path+Errfolder+'Err'+repr(ncluster).zfill(5)+'_w_comb_Rel_norm.pkl'


    cPickle.dump(Err,open(fname,'wb'))

##=====================
import matplotlib.pyplot as plt

#src_path  = 'I:/AllData_0327/'
#src_path  = 'D:/Project/K_project/data/'

#Err         = cPickle.load(file(src_path +'GPR_cluster_err/Err01000_Rand.pkl','rb'))
Err_brel    = cPickle.load(file(src_path +'GPR_cluster_err/Err01000_w_b_Rel_norm.pkl','rb'))
Err_orirel  = cPickle.load(file(src_path +'GPR_cluster_err/Err01000_w_ori_Rel_norm.pkl','rb'))
Err_combrel = cPickle.load(file(src_path +'GPR_cluster_err/Err01000_w_comb_Rel_norm.pkl','rb'))



for idx,Key in enumerate(['all','test_unrel','unrel','test_err']):
#    err         = []
    err_brel    = []
    err_orirel  = []
    err_combrel = []
    for i in range(200,1100,100):
#        err.append(Err[Key][i])
        err_brel.append(Err_brel[Key][i])
        err_orirel.append(Err_orirel[Key][i]) 
        err_combrel.append(Err_combrel[Key][i])
    
    plt.figure(idx+1)    
    plt.title('GPR cluster('+Key+')')
    plt.xlabel('cluster number')
    plt.ylabel('err (pixel per joint)')   
#    plt.plot(range(200,1100,100),err        ,color = 'blue'  , label = 'no weighted')  
    plt.plot(range(200,1100,100),err_brel   ,color = 'green' , label = 'binary weighted')
    plt.plot(range(200,1100,100),err_orirel ,color = 'red'   , label = 'original weighted')
    plt.plot(range(200,1100,100),err_combrel,color = 'black' , label = 'combine weighted')
    
    plt.legend( loc=1)
    plt.draw()
    plt.show()





    

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 15:10:26 2017

@author: medialab
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:29:40 2017

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

def uni_vec(Body):
    vec = np.roll(Body,-3,axis = 0)-Body

    tmp = ((vec**2).reshape(-1,3,vec.shape[1]).sum(axis=1))**.5
    vlen = np.insert(np.insert(tmp,np.arange(6),tmp,0),np.arange(0,12,2),tmp,0)

    return vec/vlen

#[MIN,MAX] = h5py.File('./data/CNN/model_CNN_0521_K2M_rel.h5','r')['minmax'][:]

#src_path  = 'I:/AllData_0327/'
src_path  = 'D:/Project/K_project/data/'
Mfolder   = 'unified data array/Unified_MData/'
Mpfolder  = 'unified data array/Unified_KData/'
Rfolder   = 'unified data array/reliability/'
gprfolder = 'GPR_K2M/'
Errfolder = 'GPR_cluster_err/'

Rel_th    =  0.7
factor    =  5

k1 = 66.0**2 * RBF(length_scale=67.0)  # long term smooth rising trend

k2 = 2.4**2 * RBF(length_scale=90.0) \
    * ExpSineSquared(length_scale=1.3, periodicity=1.0)  # seasonal component
# medium term irregularity
k3 = 0.66**2 \
    * RationalQuadratic(length_scale=1.2, alpha=0.78)

k4 = 0.18**2 * RBF(length_scale=0.134) \
    + WhiteKernel(noise_level=0.19**2)  # noise terms

kernel_gpml = k1 + k4




for idx,(Mpfile,Mfile,Rfile) in enumerate(zip(glob.glob(os.path.join(src_path+Mpfolder,'*.pkl')),\
                                              glob.glob(os.path.join(src_path+Mfolder,'*ex4_FPS30_motion_unified.pkl')),\
                                              glob.glob(os.path.join(src_path+Rfolder,'*ex4.pkl')))):

    print(Mpfile)
    print(Rfile)
    print(Mfile)
    print('==================================\n\n\n')

    mdata   = cPickle.load(file(Mfile,'rb'))
    rdata   = cPickle.load(file(Rfile,'rb'))
    mpdata  = cPickle.load(file(Mpfile,'rb'))
#    mdata   = cPickle.load(open(Mfile,'rb') ,encoding = 'latin1')
#    rdata   = cPickle.load(open(Rfile,'rb') ,encoding = 'latin1')
#    mpdata  = cPickle.load(open(Mpfile,'rb'),encoding = 'latin1')

    Len     = min(mpdata.shape[1],mdata.shape[1])


    if idx == 0:

        M  = mdata[12:30,:Len]
        Mp = mpdata[12:30,:Len]
        R  = rdata[4:10 ,:Len]
    else:
        M  = np.hstack([M , mdata[12:30,:Len]])
        Mp = np.hstack([Mp, mpdata[12:30,:Len]])
        R  = np.hstack([R , rdata[4:10 ,:Len]])

relidx = np.where(np.sum((R<Rel_th)*1,0)==0)[0]

#M=M.T
#Mp=Mp.T
#R=R.T


M_rel  =  M.T[relidx ,:]
Mp_rel =  Mp.T[relidx,:]

M_mean = np.mean(M_rel,0)
Mp_mean = np.mean(Mp_rel,0)

M_rel = M_rel-M_mean
Mp_rel = Mp_rel-Mp_mean

Err={}

for ncluster in range(200,1100,100):

    # Cluster of Mocap Data
    print('Mocap Clustering(',ncluster,')')
    t0=time()
#    image_array_sample = shuffle(M_rel, random_state=0)[:3000]

    print('start Kmeans clustering')
    kmeans = KMeans(n_clusters=ncluster, random_state=None,init='k-means++',n_init=10).fit(M_rel)
    labels_M = kmeans.predict(M_rel)
    print('Kmeans clustering finish')
    #recreate_m=kmeans.cluster_centers_[labels_m]
    #centroids_m=kmeans.cluster_centers_
    print(time()-t0)

    # Align centroids
    centroids_M  = np.zeros((ncluster,18),dtype=np.float64)
    centroids_Mp = np.zeros((ncluster,18),dtype=np.float64)

    for i in range(0,ncluster):
        centroids_M[i,:]=np.mean(M_rel[labels_M==i,:],axis=0)
        centroids_Mp[i,:]=np.mean(Mp_rel[labels_M==i,:],axis=0)

    # Gaussian Regression

    gp = GaussianProcessRegressor(kernel=kernel_gpml)

    print('Training')
    gp.fit(centroids_Mp, centroids_M)

#    print("\nLearned kernel: %s" % gp.kernel_)
#    print("Log-marginal-likelihood: %.3f"
#          % gp.log_marginal_likelihood(gp.kernel_.theta))
#    learned_kernel=gp.kernel_

    # Prediction
    print('Predicting')
    y_pred, y_std = gp.predict(Mp.T, return_std=True)
    data=y_pred.T + M_mean.reshape(18,-1)
    uni_data = np.zeros(data.shape)
    univec = uni_vec(data)

    uni_data[0:3  ,:] = data[0:3  ,:]
    uni_data[9:12 ,:] = data[9:12 ,:]

    uni_data[3:6  ,:] = data[0:3  ,:]+univec[0:3  ,:]*33.2*factor
    uni_data[6:9  ,:] = data[3:6  ,:]+univec[3:6  ,:]*27.1*factor
    uni_data[12:15,:] = data[9:12 ,:]+univec[9:12 ,:]*33.2*factor
    uni_data[15:  ,:] = data[12:15,:]+univec[12:15,:]*27.1*factor

    diff = M-uni_data
    diffT = M.T-uni_data.T
    err = np.sum(np.sum(((M-uni_data).reshape(-1,3,M.shape[1]))**2,axis=1)**0.5)/50247/6
    
    Err[ncluster] = err
    print('Err=',err)
    fname = src_path+Errfolder+'Err'+repr(ncluster).zfill(5)+'_rmmean.pkl'
#    f = h5py.File(fname,'w')
#    f.create_dataset('data',data = Err)
#    f.close()
    cPickle.dump(Err,open(fname,'wb'))
    
#=====================
import matplotlib.pyplot as plt

Err     = cPickle.load(open('Err00200.pkl','rb')        ,encoding = 'latin1')
Err_opt = cPickle.load(open(src_path+Errfolder+'Err01000.pkl','rb')        ,encoding = 'latin1')
Err_rm  = cPickle.load(open(src_path+Errfolder+'Err01000_rmmean.pkl','rb') ,encoding = 'latin1')

err    = []
erropt = []
errrm  = []

for i in Err_opt.keys():
    err.append(Err[i]/50247/6)
    erropt.append(Err_opt[i]/50247/6)
    errrm.append(Err_rm[i]) 
    
plt.title('GPR cluster')
plt.xlabel('cluster number')
plt.ylabel('err (pixel per joint)')   
plt.plot(range(200,1100,100),err,color = 'blue' , label = 'no opt')  
plt.plot(range(200,1100,100),erropt,color = 'green' , label = 'with opt')
plt.plot(range(200,1100,100),errrm,color = 'red' , label = 'mean remove with opt')
plt.legend( loc=1)
plt.draw()
plt.show()





    

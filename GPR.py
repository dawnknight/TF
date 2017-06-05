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
try :
    import cPickle
except:
    import _pickle as cPickle

from sklearn.externals import joblib    
    
[MIN,MAX] = h5py.File('./data/CNN/model_CNN_0521_K2M_rel.h5','r')['minmax'][:]

#src_path  = 'I:/AllData_0327/'
src_path  = 'D:/Project/K_project/data/'
Mfolder   = 'unified data array/Unified_MData/'
Mpfolder  = 'unified data array/Unified_KData/'
Rfolder   = 'unified data array/reliability/'
gprfolder = 'GPR_K2M/'

Rel_th    =  0.7

k1 = 66.0**2 * RBF(length_scale=67.0)  # long term smooth rising trend

k2 = 2.4**2 * RBF(length_scale=90.0) \
    * ExpSineSquared(length_scale=1.3, periodicity=1.0)  # seasonal component
# medium term irregularity
k3 = 0.66**2 \
    * RationalQuadratic(length_scale=1.2, alpha=0.78)
    
k4 = 0.18**2 * RBF(length_scale=0.134) \
    + WhiteKernel(noise_level=0.19**2)  # noise terms
    
kernel_gpml = k1 + k4


gp = GaussianProcessRegressor(kernel=kernel_gpml, alpha=0,
                              optimizer=None, normalize_y=True)


for idx,(Mpfile,Mfile,Rfile) in enumerate(zip(glob.glob(os.path.join(src_path+Mpfolder,'*.pkl')),\
                                              glob.glob(os.path.join(src_path+Mfolder,'*ex4_FPS30_motion_unified.pkl')),\
                                              glob.glob(os.path.join(src_path+Rfolder,'*ex4.pkl')))):
    
    print(Mpfile)
    print(Rfile)
    print(Mfile)  
    print('==================================\n\n\n')    
    
#    mdata   = cPickle.load(file(Mfile,'rb'))
#    rdata   = cPickle.load(file(Rfile,'rb'))
#    mpdata  = cPickle.load(file(Mpfile,'rb'))
    mdata   = cPickle.load(open(Mfile,'rb') ,encoding = 'latin1')
    rdata   = cPickle.load(open(Rfile,'rb') ,encoding = 'latin1')
    mpdata  = cPickle.load(open(Mpfile,'rb'),encoding = 'latin1')
#    mpdata  = h5py.File(Mpfile,'r')['data'][:]  
    Len     = min(mpdata.shape[1],mdata.shape[1])


    if idx == 0:
        
        M  = mdata[12:30,:Len]
        Mp = mpdata[12:30,:Len]
        R  = rdata[4:10 ,:Len]
    else:
        M  = np.hstack([M , mdata[12:30,:Len]])
        Mp = np.hstack([Mp, mpdata[12:30,:Len]])
        R  = np.hstack([R , rdata[4:10 ,:Len]])
        

relidx = np.where(np.sum((R<Rel_th)*1,0)==0)[0]   # frames which have all joints reliable
        
M  = (M.T[relidx,:] -MIN)/(MAX-MIN) 
Mp = (Mp.T[relidx,:]-MIN)/(MAX-MIN) 

print('training ....')
gp.fit(Mp, M)

print('training finish....')
#cPickle.dump(gp,file(src_path+gprfolder+'GP_model_0521.pkl','wb'))
joblib.dump(gp,src_path+gprfolder+'GP_model_0524.pkl')

print('model saved....')


# =======================================


for Mpfile in glob.glob(os.path.join(src_path+Mpfolder,'*.h5')):
     
#    mpdata  = (h5py.File(Mpfile,'r')['data'][:] -MIN)/(MAX-MIN)
    mpdata  = (cPickle.load(open(Mpfile,'rb'),encoding = 'latin1')[12:30,:Len]-MIN)/(MAX-MIN)
    Len     = mpdata.shape[1]
    Mgpr    = np.zeros((18,Len))
    print(Mpfile)
    for ii in range(Len):
        
        Mpred = gp.predict(mpdata[:,ii].reshape((-1,18)))
        Mgpr[:,ii] = Mpred[0,:]
    
    fname = src_path+gprfolder+Mpfile.split('\\')[-1][:-3]+'.h5'
    
#    cPickle.dump(Mgpr*(MAX-MIN)+MIN,open(fname,'wb'))
    f = h5py.File(fname,'w')
    f.create_dataset('data',data = Mgpr*(MAX-MIN)+MIN)
    f.close()




      
        
    

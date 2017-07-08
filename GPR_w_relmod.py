# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 15:48:10 2017

@author: medialab
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
import glob,os,h5py   
    
[MIN,MAX] = h5py.File('./data/CNN/model_CNN_0521_K2M_rel.h5','r')['minmax'][:]

#src_path  = 'I:/AllData_0327/'
src_path  = 'D:/Project/K_project/data/'
outfolder   = 'unified data array/Unified_MData/'
Infolder  = 'unified data array/Unified_KData/'
Rfolder   = 'unified data array/reliability_mod/'
gprfolder = 'GPR_K2M_mod_rel/'

Rel_th    =  0.7
factor = 5

def uni_vec(Body):
    vec = np.roll(Body,-3,axis = 0)-Body

    tmp = ((vec**2).reshape(-1,3,vec.shape[1]).sum(axis=1))**.5
    vlen = np.insert(np.insert(tmp,np.arange(6),tmp,0),np.arange(0,12,2),tmp,0)

    return vec/vlen

#
#k1 = 66.0**2 * RBF(length_scale=67.0)  # long term smooth rising trend
#
#    
#k4 = 0.18**2 * RBF(length_scale=0.134) \
#    + WhiteKernel(noise_level=0.19**2)  # noise terms
#    
#kernel_gpml = k1 + k4
#
#
#gp = GaussianProcessRegressor(kernel=kernel_gpml)
#
#
#M_train_rel = cPickle.load(file('GPR_training_testing_set33.pkl','rb'))['Rel_train_M'][12:30].T
#K_train_rel = cPickle.load(file('GPR_training_testing_set33.pkl','rb'))['Rel_train_K'][12:30].T
#
#        
#M  = (M_train_rel[:15000,:]-MIN)/(MAX-MIN) 
#K  = (K_train_rel[:15000,:]-MIN)/(MAX-MIN) 
#
#print('training ....')
#gp.fit(K, M)
#
#print('training finish....')
#
#joblib.dump(gp,src_path+gprfolder+'GP_model_0707_18j.pkl')
#
#print('model saved....')

##gp = joblib.load(src_path+gprfolder+'GP_model_0625.pkl')
## =======================================


Err = 0
Err_unrel = 0

for Infile,outfile,Rfile in zip(glob.glob(os.path.join(src_path+Infolder,'*.pkl')),\
                                glob.glob(os.path.join(src_path+outfolder,'*ex4_FPS30_motion_unified.pkl')),\
                                 glob.glob(os.path.join(src_path+Rfolder,'*ex4.pkl'))):
     
    Indata  = (cPickle.load(file(Infile,'rb'))[12:30,:] -MIN)/(MAX-MIN)
    outdata = cPickle.load(file(outfile,'rb'))[12:30,:]
    rdata   = cPickle.load(file(Rfile,'rb'))
        
#    Indata  = (cPickle.load(open(Infile,'rb'),encoding = 'latin1')[12:30,:]-MIN)/(MAX-MIN)

    Len     = min(Indata.shape[1],outdata.shape[1])
    R    = rdata[4:10 ,:Len]
    Rmtx = np.insert(np.insert(R,np.arange(6),R,0),np.arange(0,12,2),R,0)
    unrel_idx = Rmtx<Rel_th
    
    Mgpr    = np.zeros((18,Indata.shape[1]))
    print(Infile)
    
#    for ii in range(Len):
#        
#        Mpred = gp.predict(Indata[:,ii].reshape((-1,18)))
#        Mgpr[:,ii] = Mpred[0,:]*(MAX-MIN)+MIN

        
    Mpred = gp.predict(Indata.T)
    Mgpr[:] = Mpred.T*(MAX-MIN)+MIN
    uni_data = np.zeros(Mgpr.shape) 
    univec   = uni_vec(Mgpr)
    
    uni_data[0:3  ,:] = Mgpr[0:3  ,:]
    uni_data[9:12 ,:] = Mgpr[9:12 ,:]

    uni_data[3:6  ,:] = Mgpr[0:3  ,:]+univec[0:3  ,:]*33.2*factor
    uni_data[6:9  ,:] = Mgpr[3:6  ,:]+univec[3:6  ,:]*27.1*factor
    uni_data[12:15,:] = Mgpr[9:12 ,:]+univec[9:12 ,:]*33.2*factor
    uni_data[15:  ,:] = Mgpr[12:15,:]+univec[12:15,:]*27.1*factor    
    

    
#    Err = Err + np.sum(abs(Mgpr[:,:Len]-outdata[:,:Len]))    
#    Err_unrel + np.sum(abs((Mgpr[:,:Len]-outdata[:,:Len])[unrel_idx]))

    



    Err = Err + np.sum(np.sum(((uni_data[:,:Len]-outdata[:,:Len]).reshape(-1,3,Len))**2,axis=1)**0.5)    
    Err_unrel + np.sum(np.sum((((uni_data[:,:Len]-outdata[:,:Len])[unrel_idx]).reshape(-1,3,Len))**2,axis=1)**0.5)


    fname = src_path+gprfolder+Infile.split('\\')[-1][:-3]+'h5'
    
    f = h5py.File(fname,'w')
    f.create_dataset('data',data = Mgpr)
    f.close()

R           = cPickle.load(file('GPR_training_testing_set33.pkl','rb'))['Rdata'][4:10,:] 
   
print Err/50247/6
print Err_unrel/np.sum(R<Rel_th)
      
        
    

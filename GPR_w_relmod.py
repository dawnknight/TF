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
    
[MIN,MAX] = h5py.File('./data/CNN/model_CNN_0521_K2M_rel.h5','r')['minmax'][:]

#src_path  = 'I:/AllData_0327/'
src_path  = 'D:/Project/K_project/data/'
outfolder   = 'unified data array/Unified_MData/'
Infolder  = 'unified data array/Unified_KData/'
Rfolder   = 'unified data array/reliability_mod/'
gprfolder = 'GPR_K2M_mod_rel/'

Rel_th    =  0.7

k1 = 66.0**2 * RBF(length_scale=67.0)  # long term smooth rising trend

    
k4 = 0.18**2 * RBF(length_scale=0.134) \
    + WhiteKernel(noise_level=0.19**2)  # noise terms
    
kernel_gpml = k1 + k4


gp = GaussianProcessRegressor(kernel=kernel_gpml)


M_train_rel = cPickle.load(file('GPR_training_testing_set.pkl','rb'))['Rel_train_M'].T
K_train_rel = cPickle.load(file('GPR_training_testing_set.pkl','rb'))['Rel_train_K'].T

        
M  = (M_train_rel[:15000,:] -MIN)/(MAX-MIN) 
K  = (K_train_rel[:15000,:]-MIN)/(MAX-MIN) 

print('training ....')
gp.fit(K, M)

print('training finish....')

joblib.dump(gp,src_path+gprfolder+'GP_model_0706.pkl')

print('model saved....')

#gp = joblib.load(src_path+gprfolder+'GP_model_0625.pkl')
# =======================================

#gp = cPickle.load(file(src_path+gprfolder+'GP_model_0524.pkl','rb'))

Err = 0
Err_unrel = 0

for Infile,outfile,Rfile in zip(glob.glob(os.path.join(src_path+Infolder,'*.pkl')),\
                                glob.glob(os.path.join(src_path+outfolder,'*ex4_FPS30_motion_unified.pkl')),\
                                 glob.glob(os.path.join(src_path+Rfolder,'*ex4.pkl'))):
     
    Indata  = (h5py.File(Infile,'r')['data'][:] -MIN)/(MAX-MIN)
    outdata = h5py.File(outfile,'r')['data'][:]
    rdata   = cPickle.load(file(Rfile,'rb'))
        
#    Indata  = (cPickle.load(open(Infile,'rb'),encoding = 'latin1')[12:30,:]-MIN)/(MAX-MIN)

    Len     = min(Indata.shape[1],outdata.shape[1])
    R    = rdata[4:10 ,:Len]
    unrel_idx = R<Rel_th
    
    Mgpr    = np.zeros((18,Indata.shape[1]))
    print(Infile)
    
    for ii in range(Len):
        
        Mpred = gp.predict(Indata[:,ii].reshape((-1,18)))
        Mgpr[:,ii] = Mpred[0,:]*(MAX-MIN)+MIN
    
    Err = Err + np.sum(abs(Mgpr[:,:Len]-outdata[:,:Len]))
    Err_unrel + np.sum(abs((Mgpr[:,:Len]-outdata[:,:Len])[unrel_idx]))

    fname = src_path+gprfolder+Infile.split('\\')[-1][:-3]+'h5'
    
    f = h5py.File(fname,'w')
    f.create_dataset('data',data = Mgpr)
    f.close()

   
print Err/50247/6
print Err_unrel/50247/6
      
        
    

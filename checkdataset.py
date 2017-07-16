# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 03:57:13 2017

@author: medialab
"""

import h5py
import cPickle

[MIN,MAX] = h5py.File('./data/CNN/model_CNN_0521_K2M_rel.h5','r')['minmax'][:]

exeno     = '_ex4'   

File          = cPickle.load(file('GPR_training_testing_RANDset33'+exeno+'.pkl','rb'))

M_train_rel   = File['Rel_train_M'][12:30,:].T
K_train_rel   = File['Rel_train_K'][12:30,:].T # normalize later 

K_test_rel    = (File['Rel_test_K'][12:30,:].T-MIN)/(MAX-MIN) 
M_test_rel    =  File['Rel_test_M'][12:30,:].T

K_test_unrel  = (File['unRel_test_K'][12:30,:].T-MIN)/(MAX-MIN) 
M_test_unrel  =  File['unRel_test_M'][12:30,:].T 
R_test_unrel  =  File['unRel_test_R'][4:10,:]

M             =  File['Mdata'][12:30,:].T 
K             = (File['Kdata'][12:30,:].T-MIN)/(MAX-MIN) 
R             =  File['Rdata'][4:10,:] 

print 'M :'+repr(M.shape)
print 'K :'+repr(K.shape)

print 'M_train_rel :'+repr(M_train_rel.shape)
print 'K_train_rel :'+repr(K_train_rel.shape)

print 'M_test_rel :'+repr(M_test_rel.shape)
print 'K_test_rel :'+repr(K_test_rel.shape)

print 'M_test_unrel :'+repr(M_test_unrel.shape)
print 'K_test_unrel :'+repr(K_test_unrel.shape)
print 'R_test_unrel :'+repr(R_test_unrel.shape)

print 'M_test :'+repr(M_test_unrel.shape[0]+M_test_rel.shape[0])
print 'K_test :'+repr(K_test_unrel.shape[0]+K_test_rel.shape[0])





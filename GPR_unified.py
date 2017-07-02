# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:47:36 2017

@author: medialab
"""

import h5py
import numpy as np
import glob,os,pdb

'''
Jlen = {}

Jlen['0809'] = 33.2  #Rshoulder2Relbow
Jlen['0910'] = 27.1  #Relbow2Rwrist
Jlen['0405'] = 33.2  #Lshoulder2Lelbow
Jlen['0506'] = 27.1  #Lelbow2Lwrist
'''
factor = 5
src_path = 'D:/Project/K_project/data/'
folder  = 'GPR_M2K/'
dst_path = 'D:/Project/K_project/data/unified GPR_M2K/test/'

def uni_vec(Body):
    vec = np.roll(Body,-3,axis = 0)-Body
    
    tmp = ((vec**2).reshape(-1,3,vec.shape[1]).sum(axis=1))**.5
    vlen = np.insert(np.insert(tmp,np.arange(6),tmp,0),np.arange(0,12,2),tmp,0)
    vecLR  = Body[9:12,:]-Body[0:3,:]
    L2Rshlder = vecLR / ((vecLR**2).reshape(-1,3,vecLR.shape[1]).sum(axis=1))**.5
       
    return vec/vlen,L2Rshlder



#for infile in glob.glob(os.path.join(src_path+folder,'*ex4.pkl')):
for infile in glob.glob(os.path.join(src_path+folder,'*.h5')):
    print(infile)
#    data = cPickle.load(file(infile,'rb'))
    data = h5py.File(infile,'r')['data'][:]
    
    uni_data = np.zeros(data.shape)
    univec,L2Rshlder = uni_vec(data)
    
    uni_data[0:3  ,:] = data[0:3  ,:]
#    uni_data[9:12 ,:] = data[9:12 ,:]
    uni_data[9:12 ,:] = data[0:3  ,:]+L2Rshlder*16.65*2*factor
    
    uni_data[3:6  ,:] = data[0:3  ,:]+univec[0:3  ,:]*33.2*factor
    uni_data[6:9  ,:] = data[3:6  ,:]+univec[3:6  ,:]*27.1*factor
    uni_data[12:15,:] = data[9:12 ,:]+univec[9:12 ,:]*33.2*factor
    uni_data[15:  ,:] = data[12:15,:]+univec[12:15,:]*27.1*factor

    fname = dst_path + infile.split('\\')[-1][:-3]+'h5'

    f = h5py.File(fname,'w')
    f.create_dataset('data',data = uni_data)
    f.close()







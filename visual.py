# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 23:16:41 2017

@author: Dawnknight
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
[MIN,MAX]  = h5py.File('model0213.h5','r')['minmax'][:]
Kmod = h5py.File('batch0213.h5','r')['Kdata_mod'][:]
Mcam = h5py.File('batch0213.h5','r')['Mdata'][:]
Kcam = h5py.File('batch0213.h5','r')['Kdata'][:]

def visual(Kmod,Mcam,Kcam,idx=0,vtype = 0):
    
    
    tmp = np.zeros([18,30])
    tmp[:] = Kmod.T[0,:,:,idx]
    
    Mtmp = np.zeros([18,30])
    Mtmp[:] = Kcam.T[:,:,idx]
    
    ktmp = np.zeros([18,30])
    ktmp[:] = Kmod.T[0,:,:,idx]

    
    KJ_mod = tmp*(MAX-MIN)+MIN
    Mjoints = Mtmp*(MAX-MIN)+MIN
    joints  = ktmp*(MAX-MIN)+MIN    


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Z axis')
    ax.set_ylabel('X axis')
    ax.set_zlabel('Y axis')
    
    for i in range(30):
        plt.cla()
        kxm = KJ_mod.T[i,::3]
        kym = KJ_mod.T[i,1::3]
        kzm = KJ_mod.T[i,2::3] 
    
        mx = Mjoints.T[i,::3]
        my = Mjoints.T[i,1::3]
        mz = Mjoints.T[i,2::3]     
 
        kx = joints.T[i,::3]
        ky = joints.T[i,1::3]
        kz = joints.T[i,2::3]  
   
        if vtype == 0 or vtype == 2 or vtype == 3:
            ax.scatter(kzm, kxm, kym,c = 'red',s = 50,alpha=.4,label='K modified')
        if vtype == 1 or vtype == 2 or vtype == 3:
            ax.scatter(mz, mx, my,c = 'green',s = 50,alpha=.4,label='MoCam Joints')
        if vtype == 3:
            ax.scatter(kz, kx, ky, c = 'blue', s = 100,label='Kinect Joints') 
        
        ax.set_title(i)
        ax.set_xlabel('Z axis')
        ax.set_ylabel('X axis')
        ax.set_zlabel('Y axis')
        ax.set_xlim(-300,300)
        ax.set_ylim(-200,400)
        ax.set_zlim(50,600)
        plt.legend( loc=1)
        plt.draw()
        plt.pause(1.0/10)
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 02 22:55:51 2017

@author: Dawnknight
"""

import cPickle,h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import expit

W1   = h5py.File('model.h5','r')['W1'][:]
W2   = h5py.File('model.h5','r')['W2'][:]
Wp1  = h5py.File('model.h5','r')['Wp1'][:]
Wp2  = h5py.File('model.h5','r')['Wp2'][:]
b1   = h5py.File('model.h5','r')['b1'][:]
b2   = h5py.File('model.h5','r')['b2'][:]
bp1  = h5py.File('model.h5','r')['bp1'][:]
bp2  = h5py.File('model.h5','r')['bp2'][:]
[MIN,MAX]  = h5py.File('model.h5','r')['minmax'][:]
path  = 'D:\Project\PyKinect2-master\Kproject\data\Motion and Kinect'

data  = cPickle.load(file(path+'/Unified_MData/Andy_2016-12-15 04.15.27 PM_FPS30_motion_unified_ex4.pkl','r'))
kdata = cPickle.load(file(path+'/Unified_KData/Andy_12151615_Kinect_unified_ex4.pkl','r'))

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Z axis')
ax.set_ylabel('X axis')
ax.set_zlabel('Y axis')

for i in kdata.keys():
    if i == 0:
        joints = kdata[i]
        Mjoints = data[i]
    else:
        joints = np.vstack([joints,kdata[i]])
        Mjoints = np.vstack([Mjoints,data[i]])
        
joints  = joints.T
Mjoints = Mjoints.T
#====== denoise =============

NJ = (joints-MIN)/(MAX-MIN)

eL1 = np.maximum(np.dot(NJ,W1)+b1,0)
eL2 = np.maximum(np.dot(eL1,W2)+b2,0)
dL1 = np.maximum(np.dot(eL2,Wp1)+bp1,0)
#dL2 = np.maximum(np.dot(dL1,Wp2)+bp2,0)
dL2 = expit(np.dot(dL1,Wp2)+bp2)

KJ_mod = dL2*(MAX-MIN)+MIN



for i in range(300,500):
    plt.cla()
    kxm = KJ_mod[i,::3]
    kym = KJ_mod[i,1::3]
    kzm = KJ_mod[i,2::3]

    mx = Mjoints[i,::3]
    my = Mjoints[i,1::3]
    mz = Mjoints[i,2::3]    

    kx = joints[i,::3]
    ky = joints[i,1::3]
    kz = joints[i,2::3]    

#    ax.scatter(kz, kx, ky, c = 'blue', s = 100,label='Kinect Joints')    
#    ax.scatter(mz, mx, my,c = 'green',s = 50,alpha=.4,label='MoCam Joints')
    ax.scatter(kzm, kxm, kym,c = 'red',s = 50,alpha=.4,label='K modified')
#    ax.set_xlim(-300,300)
#    ax.set_ylim(-200,400)
#    ax.set_zlim(50,600)
    ax.set_title(i)
    ax.set_xlabel('Z axis')
    ax.set_ylabel('X axis')
    ax.set_zlabel('Y axis')
    plt.legend( loc=1)
    plt.draw()
    plt.pause(1.0/120)














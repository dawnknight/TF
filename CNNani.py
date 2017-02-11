# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 17:19:50 2017

@author: medialab
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import expit
import tensorflow as tf


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding = 'SAME')

def deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, 2, 2, 1], padding = 'SAME')

We1   = h5py.File('model0210.h5','r')['We1'][:]
We2   = h5py.File('model0210.h5','r')['We2'][:]
Wd1  = h5py.File('model0210.h5','r')['Wd1'][:]
Wd2  = h5py.File('model0210.h5','r')['Wd2'][:]
be1   = h5py.File('model0210.h5','r')['be1'][:]
be2   = h5py.File('model0210.h5','r')['be2'][:]
bd1  = h5py.File('model0210.h5','r')['bd1'][:]
bd2  = h5py.File('model0210.h5','r')['bd2'][:]
[MIN,MAX]  = h5py.File('model0210.h5','r')['minmax'][:]




#path  = 'D:\Project\PyKinect2-master\Kproject\data\Motion and Kinect'
#
#data  = cPickle.load(open(path+'/Unified_MData/Andy_2016-12-15 04.15.27 PM_FPS30_motion_unified_ex4.pkl','rb'),encoding = 'latin1')
#kdata = cPickle.load(open(path+'/Unified_KData/Andy_12151615_Kinect_unified_ex4.pkl','rb'),encoding = 'latin1')

KT   = h5py.File('batch.h5','r')['Kdata'][:]
KT = (KT-MIN)/(MAX-MIN)
MT   = h5py.File('batch.h5','r')['Mdata'][:]
MT = (MT-MIN)/(MAX-MIN)


tmp = np.ones([33,850])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Z axis')
ax.set_ylabel('X axis')
ax.set_zlabel('Y axis')

#for i in kdata.keys():
#    if i == 0:
#        joints = kdata[i]
#        Mjoints = data[i]
#    else:
#        joints = np.vstack([joints,kdata[i]])
#        Mjoints = np.vstack([Mjoints,data[i]])
#        
#joints  = joints.T
#Mjoints = Mjoints.T
#====== denoise =============

#NJ = (joints-MIN)/(MAX-MIN)
x = tf.placeholder(tf.float32, shape = [None, 30,33])
x_origin = tf.reshape(x, [-1, 30, 33, 1])
he1 = tf.nn.relu(tf.add(conv2d(x_origin, We1), be1))
he2 = tf.nn.relu(tf.add(conv2d(he1, We2), be2))

output_shape_d1 = tf.pack([16, 15, 17, 16])
output_shape_d2 = tf.pack([16, 30, 33, 1])
hd1 = tf.nn.relu(deconv2d(he2+bd1, Wd1,output_shape_d1))
hd2 = tf.nn.relu(deconv2d(hd1+bd2, Wd2,output_shape_d2))

sess = tf.InteractiveSession()
batch_size = 16
counter = 0



for i in range(50):
                
    batch_raw = KT[:,:,i*batch_size:(i+1)*batch_size].T
    a = sess.run(hd2,feed_dict={x:batch_raw}).T[0,:,:,:] 
              
    if i == 0:
        tmp[:,:30] = a[:,:,0]
        for j in range(1,16):
            tmp[:,29+j] = a[:,:,j][:,-1]
    else:
        for j in range(16):
            tmp[:,29+j+16*i] = a[:,:,j][:,-1]





KJ_mod = tmp*(MAX-MIN)+MIN



for i in range(300,500):
    plt.cla()
    kxm = KJ_mod.T[i,::3]
    kym = KJ_mod.T[i,1::3]
    kzm = KJ_mod.T[i,2::3]

#    mx = Mjoints[i,::3]
#    my = Mjoints[i,1::3]
#    mz = Mjoints[i,2::3]    
#
#    kx = joints[i,::3]
#    ky = joints[i,1::3]
#    kz = joints[i,2::3]    

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














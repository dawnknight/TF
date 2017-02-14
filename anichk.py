# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 13:34:47 2017

@author: medialab
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import expit
import tensorflow as tf


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')

def deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, 1, 1, 1], padding = 'SAME')
idx = 1
start = 200
#def CNNanichk(idx =1,start = 200):  
    
NK = h5py.File('KandM.h5','r')['N_Kinect'][:]
K = h5py.File('KandM.h5','r')['Kinect'][:]
M = h5py.File('KandM.h5','r')['Mcam'][:]   
    
 

   
We1   = h5py.File('We1.h5','r')['w_e_conv1_'+str(idx)][:]
We2   = h5py.File('We2.h5','r')['w_e_conv2_'+str(idx)][:]
Wd1   = h5py.File('Wd1.h5','r')['w_d_conv1_'+str(idx)][:]
Wd2   = h5py.File('Wd2.h5','r')['w_d_conv2_'+str(idx)][:]
be1   = h5py.File('be1.h5','r')['b_e_conv1_'+str(idx)][:]
be2   = h5py.File('be2.h5','r')['b_e_conv2_'+str(idx)][:]
bd1   = h5py.File('bd1.h5','r')['b_d_conv1_'+str(idx)][:]
bd2   = h5py.File('bd2.h5','r')['b_d_conv2_'+str(idx)][:]

[MIN,MAX]  = h5py.File('model0210.h5','r')['minmax'][:]





fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Z axis')
ax.set_ylabel('X axis')
ax.set_zlabel('Y axis')

x = tf.placeholder(tf.float32, shape = [None, 30,18])
x_origin = tf.reshape(x, [-1, 30, 18, 1])
he1 = tf.nn.relu(tf.add(conv2d(x_origin, We1), be1))
he2 = tf.nn.relu(tf.add(conv2d(he1, We2), be2))

output_shape_d1 = tf.pack([16, 30, 18, 16])
output_shape_d2 = tf.pack([16, 30, 18, 1])
hd1 = tf.nn.relu(deconv2d(he2+bd1, Wd1,output_shape_d1))
hd2 = tf.nn.relu(deconv2d(hd1+bd2, Wd2,output_shape_d2))

sess = tf.InteractiveSession()
batch_size = 16
counter = 0
#    start = 200

tmp = np.ones([18,90])


for i in range(2):
    batch_raw = np.zeros([18,30,16])
    for j in range(batch_size) :
        batch_raw[:,:,j] = NK[:,start+j+i*(batch_size-1):start+j+i*(batch_size-1)+30]
                
           
#    batch_raw = NK[:,:,i*batch_size:(i+1)*batch_size].T
    a = sess.run(hd2,feed_dict={x:batch_raw}).T[0,:,:,:] 
              
    if i == 0:
        tmp[:,:30] = a[:,:,0]
        for j in range(1,16):
            tmp[:,29+j] = a[:,:,j][:,-1]
    else:
        tmp[:,45:75] = a[:,:,0]
        for j in range(1,16):
            tmp[:,74+j] = a[:,:,j][:,-1]





KJ_mod = tmp*(MAX-MIN)+MIN



for i in range(90):
    plt.cla()
    kxm = KJ_mod.T[i,::3]
    kym = KJ_mod.T[i,1::3]
    kzm = KJ_mod.T[i,2::3]

    mx = M[start+i,::3]
    my = M[start+i,1::3]
    mz = M[start+i,2::3]    
#
    kx = K[start+i,::3]
    ky = K[start+i,1::3]
    kz = K[start+i,2::3]    

    ax.scatter(kz, kx, ky, c = 'blue', s = 100,label='Kinect Joints')    
    ax.scatter(mz, mx, my,c = 'green',s = 50,alpha=.4,label='MoCam Joints')
    ax.scatter(kzm, kxm, kym,c = 'red',s = 50,alpha=.4,label='K modified')
    ax.set_xlim(-300,300)
    ax.set_ylim(-200,400)
    ax.set_zlim(50,600)
    ax.set_title(i)
    ax.set_xlabel('Z axis')
    ax.set_ylabel('X axis')
    ax.set_zlabel('Y axis')
    plt.legend( loc=1)
    plt.draw()
    plt.pause(1.0/120)














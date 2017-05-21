# -*- coding: utf-8 -*-
"""
Created on Sun May 21 14:55:47 2017

@author: Dawnknight
"""

import os , glob  , h5py
import _pickle as cPickle
import numpy as np
import tensorflow as tf

#src_path = 'I:/AllData_0327/unified data/'
src_path = 'D:/Project/K_project/data/Motion and Kinect unified array/'
Kfolder  = 'Unified_KData/'
Mfolder  = 'Unified_MData/'


dst_path = './Concatenate_Data/CNN/'

group_size = 30 # sample number per group
jnum = 11      # joint number *3 (xyz) per sample 
index = 0

apdlen = group_size-1 #append length





date_ext     = '_CNN_0521'
data_ext     = '_K2M_rel'

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')

def deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, 1, 1, 1], padding = 'SAME')

CNNparam  = h5py.File('./data/CNN/model'+date_ext+data_ext+'.h5','r')
We1       = CNNparam ['We1'][:]
We2       = CNNparam ['We2'][:]
Wd1       = CNNparam ['Wd1'][:]
Wd2       = CNNparam ['Wd2'][:]
be1       = CNNparam ['be1'][:]
be2       = CNNparam ['be2'][:]
bd1       = CNNparam ['bd1'][:]
bd2       = CNNparam ['bd2'][:]
[MIN,MAX] = CNNparam ['minmax'][:]
[joints_num,group_size,batch_size,conv_ker_L1,conv_ker_L2] = CNNparam['parm'][:]    



x = tf.placeholder(tf.float32, shape = [None, group_size,joints_num*3])
x_origin = tf.reshape(x, [-1, group_size,joints_num*3, 1])
he1 = tf.nn.relu(tf.add(conv2d(x_origin, We1), be1))
he2 = tf.nn.relu(tf.add(conv2d(he1, We2), be2))

output_shape_d1 = tf.pack([batch_size, group_size, joints_num*3, conv_ker_L1])
output_shape_d2 = tf.pack([batch_size, group_size, joints_num*3, 1])
hd1 = tf.nn.relu(deconv2d(he2, Wd1,output_shape_d1)+bd1)
hd2 = tf.nn.relu(deconv2d(hd1, Wd2,output_shape_d2)+bd2)

sess = tf.InteractiveSession()












for idx,(kinfile,minfile)  in enumerate(zip(glob.glob(os.path.join(src_path+Kfolder,'*ex4.pkl'))[0],\
                                            glob.glob(os.path.join(src_path+Mfolder,'*ex4_FPS30_motion_unified.pkl')))[0]):
    
    print('group_'+str(idx+1)+'......')
    print(kinfile)
    print(minfile)  
    print('==================================\n\n\n')    
    kdata = cPickle.load(file(kinfile,'r'))
    mdata = cPickle.load(file(minfile,'r'))
    
    Len = min(kdata[0].shape[1],mdata[0].shape[1])
    
    
    apdK = (np.hstack([np.tile(kdata[:,0],(apdlen,1)).T,kdata])-MIN)/(MAX-MIN)
#    apdM = np.hstack([np.tile(mdata[:,0],(apdlen,1)).T,mdata])
    
    for i in [0]:#range(Len):
        
        rawdata = apdK[:,i:i+group_size].T
        a = sess.run(hd2,feed_dict={x:rawdata}).T
    
    
    
    
    
    
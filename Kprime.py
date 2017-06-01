# -*- coding: utf-8 -*-
"""
Created on Wed May 24 17:12:54 2017

@author: medialab
"""


import os , glob  , h5py
import _pickle as cPickle
import numpy as np
import tensorflow as tf

import pdb

#src_path = 'I:/AllData_0327/unified data array/'
src_path = 'D:/Project/K_project/data/unified data array/'
#infolder  = '../unified GPR/'
infolder  = '/Unified_KData/'


dst_path = 'D:/Project/K_project/data/result/'

group_size = 30 # sample number per group
index = 0

apdlen = group_size-1 #append length





date_ext     = '_CNN_gpr_0531_smo'
data_ext     = '_Mgpr2K_rel'

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

output_shape_d1 = tf.pack([1, group_size, joints_num*3, conv_ker_L1])
output_shape_d2 = tf.pack([1, group_size, joints_num*3, 1])
hd1 = tf.nn.relu(deconv2d(he2, Wd1,output_shape_d1)+bd1)
hd2 = tf.nn.relu(deconv2d(hd1, Wd2,output_shape_d2)+bd2)

sess = tf.InteractiveSession()


for infile in glob.glob(os.path.join(src_path+infolder,'*ex4.pkl')):
    

    print(infile)
 
    print('=============== Kprime ==================\n\n\n')    
    data = cPickle.load(open(infile,'rb'),encoding = 'latin1')
#    data = h5py.File(infile,'r')['data'][:]
    
    Len = data.shape[1]
    
    Kprime = np.zeros((joints_num*3,Len))

    fname  = dst_path+infile.split('\\')[-1][:-3]+'h5'
    apdK = (np.hstack([np.tile(data[:,0],(apdlen,1)).T,data])-MIN)/(MAX-MIN)  # repeat first col for another apdlen times (in this case 30-1 = 29)

    
    for i in range(Len):
        
        rawdata = apdK[12:30,i:i+group_size].T.reshape((-1,30,18))

        a = sess.run(hd2,feed_dict={x:rawdata}).T[0,:,:,0]   # 18*30 array

        Kprime[:,i] = a[:,-1]

#    cPickle.dump(Kprime*(MAX-MIN)+MIN,file(fname,'wb'))
    f = h5py.File(fname, "w")
    f.create_dataset('data', data = Kprime*(MAX-MIN)+MIN)
    f.close()
    
    
    
    
    
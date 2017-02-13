# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 23:31:40 2017

@author: Dawnknight
"""
import tensorflow as tf
import numpy as np
import h5py
from random import shuffle as sf


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')

def deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, 1, 1, 1], padding = 'SAME')

    

# set a placeholder for future input
x = tf.placeholder(tf.float32, shape = [None, 30,18])
x_noise = tf.placeholder(tf.float32, shape = [None, 30,18])    

x_origin = tf.reshape(x, [-1, 30, 18, 1])
x_origin_noise = tf.reshape(x_noise, [-1, 30, 18, 1])
    
    
# initialize weight
W_e_init1 = tf.truncated_normal([3, 18, 1, 16], mean=0.0, stddev=0.01)
W_e_init2 = tf.truncated_normal([3, 18, 16,32], mean=0.0, stddev=0.01)
W_d_init1 = tf.truncated_normal([3, 18, 16,32], mean=0.0, stddev=0.01)
W_d_init2 = tf.truncated_normal([3, 18,  1,16], mean=0.0, stddev=0.01)
    

#encoder

W_e_conv1 = tf.Variable(W_e_init1 , name = 'W_e_1')
b_e_conv1 = tf.Variable(tf.constant(0.1,shape = [16]), name='b_e_1')
h_e_conv1 = tf.nn.relu(tf.add(conv2d(x_origin_noise, W_e_conv1), b_e_conv1))

W_e_conv2 = tf.Variable(W_e_init2 , name = 'W_e_2')
b_e_conv2 = tf.Variable(tf.constant(0.1,shape = [32]), name='b_e_2')
h_e_conv2 = tf.nn.relu(tf.add(conv2d(h_e_conv1, W_e_conv2), b_e_conv2))

code_layer = h_e_conv2
print("code layer shape : %s" % h_e_conv2.get_shape())


#decoder

W_d_conv1 = tf.Variable(W_d_init1 , name = 'W_d_1')
b_d_conv1 = tf.Variable(tf.constant(0.1,shape = [16]), name='b_d_1')
output_shape_d_conv1 = tf.pack([tf.shape(x)[0], 30, 18,16])
h_d_conv1 = tf.nn.relu(tf.add(deconv2d(h_e_conv2, W_d_conv1,output_shape_d_conv1),b_d_conv1))

#h_d_conv1 = tf.nn.relu(tf.add(conv2d((h_e_conv2, W_d_conv1,output_shape_d_conv1), b_d_conv1)))

W_d_conv2 = tf.Variable(W_d_init2 , name = 'W_d_2')
b_d_conv2 = tf.Variable(tf.constant(0.1,shape = [1]), name='b_d_2')
output_shape_d_conv2 = tf.pack([tf.shape(x)[0], 30, 18,1])
h_d_conv2 = tf.nn.relu(tf.add(deconv2d(h_d_conv1, W_d_conv2,output_shape_d_conv2),b_d_conv2))
#h_d_conv2 = tf.nn.relu(tf.add(deconv2d((h_d_conv1, W_d_conv2,output_shape_d_conv1), b_d_conv2)))

x_reconstruct = h_d_conv2[:,:,:,0]
print("reconstruct layer shape : %s" % x_reconstruct.get_shape())

cost = tf.reduce_mean(tf.pow(x_reconstruct - x, 2))
optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)


sess = tf.InteractiveSession()
batch_size = 16
counter = 0
init_op = tf.global_variables_initializer()
sess.run(init_op)

f_test   = h5py.File('NLdata_batch.h5','r')['test_data'][:].T
f_telab  = h5py.File('NLdata_batch.h5','r')['test_label'][:].T
f_train  = h5py.File('NLdata_batch.h5','r')['train_data'][:].T
f_trlab  = h5py.File('NLdata_batch.h5','r')['train_label'][:].T
minmax   = h5py.File('NLdata_batch.h5','r')['minmax'][:]

idx = np.arange(f_train.shape[0])
sf(idx)

for epoch in range(10000):
        
    if (counter+1)*batch_size >f_train.shape[0]:
        counter = 0
        idx = np.arange(f_train.shape[0])
        sf(idx)
        
    batch_raw = f_trlab[idx[(counter)*batch_size:(counter+1)*batch_size],:,:]
    batch_noise =f_train[idx[(counter)*batch_size:(counter+1)*batch_size],:,:]
                             
    if epoch < 1500:
        if epoch%100 == 0:
            print("step %d, loss %g"%(epoch, cost.eval(feed_dict={x:batch_raw, x_noise: batch_noise})))
    else:
        if epoch%1000 == 0: 
            print("step %d, loss %g"%(epoch, cost.eval(feed_dict={x:batch_raw, x_noise: batch_noise})))
    
    optimizer.run(feed_dict={x:batch_raw, x_noise: batch_noise})

#print(sess.run(cost, feed_dict={x: f_telab, x_noise: f_test}))
KKK = sess.run(h_d_conv2, feed_dict={x: batch_raw, x_noise: batch_noise})


f = h5py.File("model0213.h5", "w")
f.create_dataset('We1'  , data = W_e_init1.eval()) 
f.create_dataset('We2'  , data = W_e_init2.eval()) 
f.create_dataset('Wd1'  , data = W_d_init1.eval()) 
f.create_dataset('Wd2'  , data = W_d_init2.eval()) 
f.create_dataset('be1'  , data = b_e_conv1.eval()) 
f.create_dataset('be2'  , data = b_e_conv2.eval()) 
f.create_dataset('bd1'  , data = b_d_conv1.eval()) 
f.create_dataset('bd2'  , data = b_d_conv2.eval()) 
f.create_dataset('minmax' , data = minmax) 
f.close() 



#f = h5py.File("btest.h5", "w")
#f.create_dataset('Kdata' , data = batch_noise)
#f.create_dataset('Kdata_mod' , data = KKK)
#f.create_dataset('Mdata' , data = batch_raw)  
#
#f.close()    












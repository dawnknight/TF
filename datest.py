# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 13:57:19 2017

@author: medialab
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from utils import weight_variable, bias_variable
from tensorflow.examples.tutorials.mnist import input_data
import h5py

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding = 'SAME')

def deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, 2, 2, 1], padding = 'SAME')
    
def build_graph():
    x_origin = tf.reshape(x, [-1, 3, 11, 1])
    x_origin_noise = tf.reshape(x_noise, [-1, 3, 11, 1])

    W_e_conv1 = weight_variable([5, 5, 1, 16], "w_e_conv1")
    b_e_conv1 = bias_variable([16], "b_e_conv1")
    print(conv2d(x_origin_noise, W_e_conv1).get_shape())
    h_e_conv1 = tf.nn.relu(tf.add(conv2d(x_origin_noise, W_e_conv1), b_e_conv1))

    W_e_conv2 = weight_variable([5, 5, 16, 32], "w_e_conv2")
    b_e_conv2 = bias_variable([32], "b_e_conv2")
    h_e_conv2 = tf.nn.relu(tf.add(conv2d(h_e_conv1, W_e_conv2), b_e_conv2))

    code_layer = h_e_conv2
    print("code layer shape : %s" % h_e_conv2.get_shape())

    W_d_conv1 = weight_variable([5, 5, 16, 32], "w_d_conv1")
#    output_shape_d_conv1 = tf.pack([tf.shape(x)[0], 14, 14, 16])
    output_shape_d_conv1 = tf.pack([tf.shape(x)[0], 1, 3, 32])
    h_d_conv1 = tf.nn.relu(deconv2d(h_e_conv2, W_d_conv1, output_shape_d_conv1))

    W_d_conv2 = weight_variable([5, 5, 1, 16], "w_d_conv2")
    b_d_conv2 = bias_variable([16], "b_d_conv2")
#    output_shape_d_conv2 = tf.pack([tf.shape(x)[0], 3, 11, 1])
    output_shape_d_conv2 = tf.pack([tf.shape(x)[0], 2, 6, 16])
    h_d_conv2 = tf.nn.relu(deconv2d(h_d_conv1, W_d_conv2, output_shape_d_conv2))

    x_reconstruct = h_d_conv2
    print("reconstruct layer shape : %s" % x_reconstruct.get_shape())
    
    return x_origin, code_layer, x_reconstruct

tf.reset_default_graph()

# set a placeholder for future input
x = tf.placeholder(tf.float32, shape = [None, 33])
x_noise = tf.placeholder(tf.float32, shape = [None, 33])


x_origin, code_layer, x_reconstruct = build_graph()

cost = tf.reduce_mean(tf.pow(x_reconstruct - x_origin, 2))
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

sess = tf.InteractiveSession()
batch_size = 50
counter = 0
init_op = tf.global_variables_initializer()
sess.run(init_op)



f_test   = h5py.File('data.h5','r')['test_data'][:].T
f_telab  = h5py.File('data.h5','r')['test_label'][:].T
f_train  = h5py.File('data.h5','r')['train_data'][:].T
f_trlab  = h5py.File('data.h5','r')['train_label'][:].T



for epoch in range(10000):

    if (counter+1)*batch_size >f_train.shape[0]:
        counter = 0
 
    batch_raw = f_trlab[(counter)*batch_size:(counter+1)*batch_size,:].reshape(batch_size,-1)
    batch_noise =f_train[(counter)*batch_size:(counter+1)*batch_size,:].reshape(batch_size,-1)
    counter += 1
     
        
    if epoch < 1500:
        if epoch%100 == 0:
            print("step %d, loss %g"%(epoch, cost.eval(feed_dict={x:batch_raw, x_noise: batch_noise})))
    else:
        if epoch%1000 == 0: 
            print("step %d, loss %g"%(epoch, cost.eval(feed_dict={x:batch_raw, x_noise: batch_noise})))
    
    optimizer.run(feed_dict={x:batch_raw, x_noise: batch_noise})
    
#print("final loss %g" % cost.eval(feed_dict={x: mnist.test.images, x_noise: mnist.test.images}))










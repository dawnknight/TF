# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 13:58:47 2017

@author: medialab
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import h5py


n_hidden1 = 512
n_hidden2 = 1024

# set a placeholder for future input
x = tf.placeholder(tf.float32, shape = [None, 33])
x_noise = tf.placeholder(tf.float32, shape = [None, 33])



W_init1 = tf.random_uniform(shape=[33, n_hidden1])
W_init2 = tf.random_uniform(shape=[n_hidden1,n_hidden1])

# encoder
W1= tf.Variable(W_init1, name='W1')#shape:33*512
b1 = tf.Variable(tf.zeros([n_hidden1]), name='b1')#bias
W2= tf.Variable(W_init2, name='W2')#shape:33*512
b2 = tf.Variable(tf.zeros([n_hidden2]), name='b2')#bias

#decoder
W_prime1 = tf.transpose(W2)  
b_prime1 = tf.Variable(tf.zeros([n_hidden1]), name='b1_prime')
W_prime2 = tf.transpose(W1)  
b_prime2 = tf.Variable(tf.zeros([33]), name='b2_prime')


h_d_1 = tf.nn.relu(tf.matmul(x,W1)+b1)
h_d_2 = tf.nn.relu(tf.matmul(h_d_1,W2)+b2)
b_d_1 = tf.nn.relu(tf.matmul(h_d_2,W_prime1)+b_prime1)
b_d_2 = tf.nn.relu(tf.matmul(h_d_1,W_prime2)+b_prime2)

output = b_d_2

cost = tf.reduce_mean(tf.pow(output - x_noise, 2))
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





















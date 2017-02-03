# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 13:58:47 2017

@author: medialab
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import h5py
from random import shuffle as sf

n_hidden1 = 512
n_hidden2 = 1024

# set a placeholder for future input
x = tf.placeholder(tf.float32, shape = [None, 33])
x_noise = tf.placeholder(tf.float32, shape = [None, 33])



W_init1 = tf.random_uniform(shape=[33, n_hidden1])
W_init2 = tf.random_uniform(shape=[n_hidden1,n_hidden2])
Wp_init1 = tf.random_uniform(shape=[n_hidden2, n_hidden1])
Wp_init2 = tf.random_uniform(shape=[n_hidden1,33])

# encoder
W1= tf.Variable(W_init1, name='W1')#shape:33*512
b1 = tf.Variable(tf.zeros([n_hidden1]), name='b1')#bias
W2= tf.Variable(W_init2, name='W2')#shape:33*512
b2 = tf.Variable(tf.zeros([n_hidden2]), name='b2')#bias

#decoder
#W_prime1 = tf.transpose(W2)  
W_prime1 = tf.Variable(Wp_init1, name='Wp1')
b_prime1 = tf.Variable(tf.zeros([n_hidden1]), name='b1_prime')
#W_prime2 = tf.transpose(W1)  
W_prime2 = tf.Variable(Wp_init2, name='Wp2')
b_prime2 = tf.Variable(tf.zeros([33]), name='b2_prime')


h_d_1 = tf.nn.relu(tf.matmul(x_noise,W1)+b1)
h_d_2 = tf.nn.relu(tf.matmul(h_d_1,W2)+b2)
b_d_1 = tf.nn.relu(tf.matmul(h_d_2,W_prime1)+b_prime1)
b_d_2 = tf.nn.relu(tf.matmul(h_d_1,W_prime2)+b_prime2)

output = b_d_2

cost = tf.reduce_mean(tf.pow(output - x, 2))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

sess = tf.InteractiveSession()
batch_size = 50
counter = 0
init_op = tf.global_variables_initializer()
sess.run(init_op)

f_test   = h5py.File('Ndata.h5','r')['test_data'][:].T
f_telab  = h5py.File('Ndata.h5','r')['test_label'][:].T
f_train  = h5py.File('Ndata.h5','r')['train_data'][:].T
f_trlab  = h5py.File('Ndata.h5','r')['train_label'][:].T
minmax   = h5py.File('Ndata.h5','r')['minmax'][:]

idx = np.arange(f_train.shape[0])
sf(idx)

for epoch in range(30000):
        
    if (counter+1)*batch_size >f_train.shape[0]:
        counter = 0
        idx = np.arange(f_train.shape[0])
        sf(idx)
        
        
    batch_raw = f_trlab[idx[(counter)*batch_size:(counter+1)*batch_size],:].reshape(batch_size,-1)
    batch_noise =f_train[idx[(counter)*batch_size:(counter+1)*batch_size],:].reshape(batch_size,-1)
    counter += 1
     
        
    if epoch < 1500:
        if epoch%100 == 0:
            print("step %d, loss %g"%(epoch, cost.eval(feed_dict={x:batch_raw, x_noise: batch_noise})))
    else:
        if epoch%1000 == 0: 
            print("step %d, loss %g"%(epoch, cost.eval(feed_dict={x:batch_raw, x_noise: batch_noise})))
    
    optimizer.run(feed_dict={x:batch_raw, x_noise: batch_noise})


print(sess.run(cost, feed_dict={x: f_telab, x_noise: f_test}))



f = h5py.File("model.h5", "w")
f.create_dataset('W1'  , data = W1.eval()) 
f.create_dataset('W2'  , data = W2.eval()) 
f.create_dataset('Wp1' , data = W_prime1.eval()) 
f.create_dataset('Wp2' , data = W_prime2.eval()) 
f.create_dataset('b1'  , data = b1.eval()) 
f.create_dataset('b2'  , data = b2.eval()) 
f.create_dataset('bp1' , data = b_prime1.eval()) 
f.create_dataset('bp2' , data = b_prime2.eval()) 
f.create_dataset('minmax' , data = minmax) 
f.close() 














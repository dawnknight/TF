# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:28:45 2017

@author: medialab
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import h5py
from random import shuffle as sf

n_hidden1 = 8
n_hidden2 = 16
spts = 3 #sample points per frame (joints number *3)

# NLdata.h5 made by concatenate.py
f_test   = h5py.File('./data/NLdata.h5','r')['test_data'][:].T
f_telab  = h5py.File('./data/NLdata.h5','r')['test_label'][:].T
f_train  = h5py.File('./data/NLdata.h5','r')['train_data'][:].T
f_trlab  = h5py.File('./data/NLdata.h5','r')['train_label'][:].T
minmax   = h5py.File('./data/NLdata.h5','r')['minmax'][:]
    

for i in range(6):

    print('Joint Group ' +str(i+1))
    
    # set a placeholder for future input
    x = tf.placeholder(tf.float32, shape = [None, spts])
    x_noise = tf.placeholder(tf.float32, shape = [None, spts])
    
    
    
    W_init1  = tf.truncated_normal(shape=[spts, n_hidden1],stddev=0.01)
    W_init2  = tf.truncated_normal(shape=[n_hidden1,n_hidden2],stddev=0.01)
    Wp_init1 = tf.truncated_normal(shape=[n_hidden2, n_hidden1],stddev=0.01)
    Wp_init2 = tf.truncated_normal(shape=[n_hidden1,spts],stddev=0.01)
    
    # encoder
    W1= tf.Variable(W_init1, name='W1')#shape:spts*512
    b1 = tf.Variable(tf.constant(0.1,shape = [n_hidden1]), name='b1')#bias
    W2= tf.Variable(W_init2, name='W2')#shape:spts*512
    b2 = tf.Variable(tf.constant(0.1,shape = [n_hidden2]), name='b2')#bias
    
    #decoder
    W_prime1 = tf.transpose(W2)  
    #W_prime1 = tf.Variable(Wp_init1, name='Wp1')
    b_prime1 = tf.Variable(tf.constant(0.1,shape = [n_hidden1]), name='b1_prime')
    W_prime2 = tf.transpose(W1)  
    #W_prime2 = tf.Variable(Wp_init2, name='Wp2')
    b_prime2 = tf.Variable(tf.constant(0.1,shape = [spts]), name='b2_prime')
    
    
    h_e_1 = tf.nn.relu(tf.matmul(x_noise,W1)+b1)
    h_e_2 = tf.nn.relu(tf.matmul(h_e_1,W2)+b2)
    h_d_1 = tf.nn.relu(tf.matmul(h_e_2,W_prime1)+b_prime1)
    h_d_2 = tf.sigmoid(tf.matmul(h_d_1,W_prime2)+b_prime2)
    
    output = h_d_2
    
    cost = tf.reduce_mean(tf.pow(output - x, 2))
    optimizer = tf.train.AdamOptimizer(0.00001).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    
    
    sess = tf.InteractiveSession()
    batch_size = 8
    counter = 0
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    

    idx = np.arange(f_train.shape[0])
    sf(idx)
    
    for epoch in range(30000):
            
        if (counter+1)*batch_size >f_train.shape[0]:
            counter = 0
            idx = np.arange(f_train.shape[0])
            sf(idx)
                        
        batch_raw   = f_trlab[idx[(counter)*batch_size:(counter+1)*batch_size],i*3:(i+1)*3].reshape(batch_size,-1)
        batch_noise = f_train[idx[(counter)*batch_size:(counter+1)*batch_size],i*3:(i+1)*3].reshape(batch_size,-1)
        counter += 1
         
        if epoch%1000 == 0: 
            print("step %d, loss %g"%(epoch, cost.eval(feed_dict={x:batch_raw, x_noise: batch_noise})))
        
        optimizer.run(feed_dict={x:batch_raw, x_noise: batch_noise})
    
    filename = 'single_' + str(i) +'.h5'    
    f = h5py.File(filename, "w")
    f.create_dataset('W1'  , data = W1.eval()) 
    f.create_dataset('W2'  , data = W2.eval()) 
    f.create_dataset('b1'  , data = b1.eval()) 
    f.create_dataset('b2'  , data = b2.eval()) 
    f.create_dataset('bp1' , data = b_prime1.eval()) 
    f.create_dataset('bp2' , data = b_prime2.eval()) 
    f.create_dataset('minmax' , data = minmax) 
    f.close() 
        
        
        


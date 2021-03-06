# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 17:33:31 2017

@author: medialab
"""



import tensorflow as tf

import numpy as np
import h5py
from random import shuffle as sf

src_path = './Concatenate_Data/'
dst_path = './data/FC/'
date_ext = '_REL0504'
data_ext = ''



n_hidden1    = 64
n_hidden2    = 18
x_size       = 18
Rel_size     = 18
x_noise_size = 18


We1 = h5py.File(dst_path + 'We1'+date_ext+data_ext+'.h5', "w")  
We2 = h5py.File(dst_path + 'We2'+date_ext+data_ext+'.h5', "w")
Wd1 = h5py.File(dst_path + 'Wd1'+date_ext+data_ext+'.h5', "w")  
Wd2 = h5py.File(dst_path + 'Wd2'+date_ext+data_ext+'.h5', "w")
be1 = h5py.File(dst_path + 'be1'+date_ext+data_ext+'.h5', "w")  
be2 = h5py.File(dst_path + 'be2'+date_ext+data_ext+'.h5', "w")
bd1 = h5py.File(dst_path + 'bd1'+date_ext+data_ext+'.h5', "w")  
bd2 = h5py.File(dst_path + 'bd2'+date_ext+data_ext+'.h5', "w")

# set a placeholder for future input
x = tf.placeholder(tf.float32, shape = [None, x_size])
x_noise = tf.placeholder(tf.float32, shape = [None, x_noise_size])
Rel     = tf.placeholder(tf.float32, shape = [None, Rel_size])

keep_prob = tf.placeholder(tf.float32)


W_init1  = tf.truncated_normal(shape=[x_noise_size, n_hidden1],stddev=0.01)
W_init2  = tf.truncated_normal(shape=[n_hidden1,n_hidden2],stddev=0.01)
Wp_init1 = tf.truncated_normal(shape=[n_hidden2, n_hidden1],stddev=0.01)
Wp_init2 = tf.truncated_normal(shape=[n_hidden1,x_size],stddev=0.01)

# encoder
W1= tf.Variable(W_init1, name='W1')#shape:18*512
b1 = tf.Variable(tf.constant(0.1,shape = [n_hidden1]), name='b1')#bias
W2= tf.Variable(W_init2, name='W2')#shape:18*512
b2 = tf.Variable(tf.constant(0.1,shape = [n_hidden2]), name='b2')#bias

#decoder
#W_prime1 = tf.transpose(W2)  
W_prime1 = tf.Variable(Wp_init1, name='Wp1')
b_prime1 = tf.Variable(tf.constant(0.1,shape = [n_hidden1]), name='b1_prime')
#W_prime2 = tf.transpose(W1)  
W_prime2 = tf.Variable(Wp_init2, name='Wp2')
b_prime2 = tf.Variable(tf.constant(0.1,shape = [x_size]), name='b2_prime')


h_e_1 = tf.nn.relu(tf.matmul(x_noise,W1)+b1)
h_e_2 = tf.nn.relu(tf.matmul(h_e_1,W2)+b2)

h_d_1 = tf.nn.relu(tf.matmul(h_e_2,W_prime1)+b_prime1)
h_d_2 = tf.sigmoid(tf.matmul(h_d_1,W_prime2)+b_prime2)

#h_e_1 = tf.nn.relu(tf.matmul(x_noise,W1)+b1)
#h_e_1_drop = tf.nn.dropout(h_e_1, keep_prob)
#h_e_2 = tf.nn.relu(tf.matmul(h_e_1,W2)+b2)
#h_e_2_drop = tf.nn.dropout(h_e_2, keep_prob)
#h_d_1 = tf.nn.relu(tf.matmul(h_e_2,W_prime1)+b_prime1)
#h_d_1_drop = tf.nn.dropout(h_d_1, keep_prob)
#h_d_2 = tf.sigmoid(tf.matmul(h_d_1,W_prime2)+b_prime2)


output = h_d_2


cost = tf.reduce_mean(tf.pow((output - x), 2))
optimizer = tf.train.AdamOptimizer(0.00001).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)


sess = tf.InteractiveSession()
batch_size = 32
counter = 0
init_op = tf.global_variables_initializer()
sess.run(init_op)

f_test   = h5py.File(src_path+'NLdata'+date_ext+'.h5','r')['test_data'][:].T
f_telab  = h5py.File(src_path+'NLdata'+date_ext+'.h5','r')['test_label'][:].T
f_teRel  = h5py.File(src_path+'NLdata'+date_ext+'.h5','r')['test_data_rel'][:].T
f_train  = h5py.File(src_path+'NLdata'+date_ext+'.h5','r')['train_data'][:].T
f_trlab  = h5py.File(src_path+'NLdata'+date_ext+'.h5','r')['train_label'][:].T
f_trRel  = h5py.File(src_path+'NLdata'+date_ext+'.h5','r')['train_data_rel'][:].T

minmax   = h5py.File(src_path+'NLdata'+date_ext+'.h5','r')['minmax'][:]

idx = np.arange(f_train.shape[0])
sf(idx)

for epoch in range(3000000):
        
    if (counter+1)*batch_size >f_train.shape[0]:
        counter = 0
        idx = np.arange(f_train.shape[0])
        sf(idx)
        
        
    batch_raw   = f_trlab[idx[(counter)*batch_size:(counter+1)*batch_size],:].reshape(batch_size,-1)
    batch_noise = f_train[idx[(counter)*batch_size:(counter+1)*batch_size],:].reshape(batch_size,-1)
    batch_rel   = f_trRel[idx[(counter)*batch_size:(counter+1)*batch_size],:].reshape(batch_size,-1)
    counter += 1
     
        
    if epoch%1000 == 0: 
        print("step %d, loss %g"%(epoch, cost.eval(feed_dict={x:batch_raw, x_noise: batch_noise, Rel: batch_rel})))
        ewname1 = 'w_e_1_' + str(epoch//1000)
        ewname2 = 'w_e_2_' + str(epoch//1000)
        dwname1 = 'w_d_1_' + str(epoch//1000)
        dwname2 = 'w_d_2_' + str(epoch//1000)        
        ebname1 = 'b_e_1_' + str(epoch//1000)
        ebname2 = 'b_e_2_' + str(epoch//1000)
        dbname1 = 'b_d_1_' + str(epoch//1000)
        dbname2 = 'b_d_2_' + str(epoch//1000)
        
        We1.create_dataset(ewname1 , data = W1.eval())
        We2.create_dataset(ewname2 , data = W2.eval())
        Wd1.create_dataset(dwname1 , data = W_prime1.eval())
        Wd2.create_dataset(dwname2 , data = W_prime2.eval())
        be1.create_dataset(ebname1 , data = b1.eval())
        be2.create_dataset(ebname2 , data = b2.eval())
        bd1.create_dataset(dbname1 , data = b_prime1.eval())
        bd2.create_dataset(dbname2 , data = b_prime2.eval())
    
    optimizer.run(feed_dict={x:batch_raw, x_noise: batch_noise, Rel: batch_rel})


print(sess.run(cost, feed_dict={x: f_telab, x_noise: f_test, Rel: f_teRel}))

We1.close()
We2.close()
Wd1.close()
Wd2.close()
be1.close()
be2.close()
bd1.close()
bd2.close()


f = h5py.File(dst_path+'model'+date_ext+data_ext+'.h5', "w")
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


print(date_ext)













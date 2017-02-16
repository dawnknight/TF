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
#idx = 1
#start = 200
def CNNanichk(idx =1,start = 200, taro = False , Type = 'CNN'):  
    
    NK = h5py.File('./data/KandM.h5','r')['N_Kinect'][:]
    K  = h5py.File('./data/KandM.h5','r')['Kinect'][:]
    M  = h5py.File('./data/KandM.h5','r')['Mcam'][:]   
    [MIN,MAX]  = h5py.File('./data/model0210.h5','r')['minmax'][:]        

    if taro:
        K_taro = h5py.File('./data/KM_taro.h5','r')['Ktaro'][:]
        M_taro = h5py.File('./data/KM_taro.h5','r')['Mtaro'][:]
     
    if Type == 'CNN' :   
       
#        We1   = h5py.File('./data/We1.h5','r')['w_e_conv1_'+str(idx)][:]
#        We2   = h5py.File('./data/We2.h5','r')['w_e_conv2_'+str(idx)][:]
#        Wd1   = h5py.File('./data/Wd1.h5','r')['w_d_conv1_'+str(idx)][:]
#        Wd2   = h5py.File('./data/Wd2.h5','r')['w_d_conv2_'+str(idx)][:]
#        be1   = h5py.File('./data/be1.h5','r')['b_e_conv1_'+str(idx)][:]
#        be2   = h5py.File('./data/be2.h5','r')['b_e_conv2_'+str(idx)][:]
#        bd1   = h5py.File('./data/bd1.h5','r')['b_d_conv1_'+str(idx)][:]
#        bd2   = h5py.File('./data/bd2.h5','r')['b_d_conv2_'+str(idx)][:]
    
        We1   = h5py.File('We1.h5','r')['w_e_conv1_'+str(idx)][:]
        We2   = h5py.File('We2.h5','r')['w_e_conv2_'+str(idx)][:]
        Wd1   = h5py.File('Wd1.h5','r')['w_d_conv1_'+str(idx)][:]
        Wd2   = h5py.File('Wd2.h5','r')['w_d_conv2_'+str(idx)][:]
        be1   = h5py.File('be1.h5','r')['b_e_conv1_'+str(idx)][:]
        be2   = h5py.File('be2.h5','r')['b_e_conv2_'+str(idx)][:]
        bd1   = h5py.File('bd1.h5','r')['b_d_conv1_'+str(idx)][:]
        bd2   = h5py.File('bd2.h5','r')['b_d_conv2_'+str(idx)][:]
    
    
        x = tf.placeholder(tf.float32, shape = [None, 30,18])
        x_origin = tf.reshape(x, [-1, 30, 18, 1])
        he1 = tf.nn.relu(tf.add(conv2d(x_origin, We1), be1))
        he2 = tf.nn.relu(tf.add(conv2d(he1, We2), be2))
        
        output_shape_d1 = tf.pack([16, 30, 18, 16])
        output_shape_d2 = tf.pack([16, 30, 18, 1])
        hd1 = tf.nn.relu(deconv2d(he2, Wd1,output_shape_d1)+bd1)
        hd2 = tf.nn.relu(deconv2d(hd1, Wd2,output_shape_d2)+bd2)
        
        sess = tf.InteractiveSession()
        batch_size = 16
        loop = 5
        #    start = 200
        
        tmp = np.ones([18,90])
        
        
        for i in range(2):
            batch_raw = np.zeros([16,30,18])
            for j in range(batch_size) :
                batch_raw[j,:,:] = NK[:,start+j+i*(batch_size-1):start+j+i*(batch_size-1)+30].T
                        
                   
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



    elif Type == 'FC': #fully connected

        spts = 18 #sample points per frame (joints number *3)
        
        W1  = h5py.File('./data/FC/We1.h5','r')['w_e_1_'+str(idx)][:]
        W2  = h5py.File('./data/FC/We2.h5','r')['w_e_2_'+str(idx)][:]
        be1 = h5py.File('./data/FC/be1.h5','r')['b_e_1_'+str(idx)][:]
        be2 = h5py.File('./data/FC/be2.h5','r')['b_e_2_'+str(idx)][:]
        bd1 = h5py.File('./data/FC/bd1.h5','r')['b_d_1_'+str(idx)][:]
        bd2 = h5py.File('./data/FC/bd2.h5','r')['b_d_2_'+str(idx)][:] 
        W_prime1 = tf.transpose(W2)
        W_prime2 = tf.transpose(W1)
        

        x = tf.placeholder(tf.float32, shape = [None, spts])
        
        
        h_e_1 = tf.nn.relu(tf.matmul(x,W1)+be1)
        h_e_2 = tf.nn.relu(tf.matmul(h_e_1,W2)+be2)
        h_d_1 = tf.nn.relu(tf.matmul(h_e_2,W_prime1)+bd1)
        h_d_2 = tf.sigmoid(tf.matmul(h_d_1,W_prime2)+bd2)        
        
        sess = tf.InteractiveSession()
        
        batch_size = 32
        loop = 5
        tmp = np.ones([18,batch_size*loop])
        
        for i in range(loop):
            batch_raw = np.zeros([batch_size,spts])
            batch_raw[:] = NK[:,start+i*batch_size:start+(i+1)*batch_size].T

            a = sess.run(h_d_2,feed_dict={x:batch_raw}).T
            tmp[:,i*batch_size:(i+1)*batch_size] = a

    
    else:        
        print('can not understand')

    
    
    KJ_mod = tmp*(MAX-MIN)+MIN


    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Z axis')
    ax.set_ylabel('X axis')
    ax.set_zlabel('Y axis')    
    
    for i in range(batch_size*loop):
        plt.cla()
        kxm = KJ_mod.T[i,::3]
        kym = KJ_mod.T[i,1::3]
        kzm = KJ_mod.T[i,2::3]

    
        mx = M.T[start+i,::3]
        my = M.T[start+i,1::3]
        mz = M.T[start+i,2::3]    
    #
        kx = K.T[start+i,::3]
        ky = K.T[start+i,1::3]
        kz = K.T[start+i,2::3]  

        if taro:
            kx_taro = K_taro.T[start+i,::3]
            ky_taro = K_taro.T[start+i,1::3]
            kz_taro = K_taro.T[start+i,2::3]
    
            mx_taro = M_taro.T[start+i,::3]
            my_taro = M_taro.T[start+i,1::3]
            mz_taro = M_taro.T[start+i,2::3]
                
            ax.scatter(kz_taro, kx_taro, ky_taro, c = 'blue', s = 100)    
            ax.scatter(mz_taro, mx_taro, my_taro,c = 'green',s = 50,alpha=.4)
            ax.scatter(kz_taro, kx_taro, ky_taro,c = 'red',s = 50,alpha=.4)
            
        
        ax.scatter(kz, kx, ky, c = 'blue', s = 100,label='Kinect Joints')    
        ax.scatter(mz, mx, my,c = 'green',s = 50,alpha=.4,label='MoCam Joints')
        ax.scatter(kzm, kxm, kym,c = 'red',s = 50,alpha=.4,label='K modified')
        ax.set_xlim(-300,300)
        ax.set_ylim(-200,400)
        ax.set_zlim(50,600)
        ax.set_title('frame no : '+str(i+start)+'  training '+ str(idx*1000) +' times ' )
        ax.set_xlabel('Z axis')
        ax.set_ylabel('X axis')
        ax.set_zlabel('Y axis')
        plt.legend( loc=1)
        plt.draw()
        plt.pause(1.0/60)
    
    
    










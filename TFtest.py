# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 18:26:24 2017

@author: Dawnknight
"""

import tensorflow as tf 
import numpy as np


x = tf.placeholder(tf.float32,shape = [3, 5])
#y = x*2
colsum = tf.reduce_sum(x, 0)

with tf.control_dependencies([colsum]):
    
    return  x = x/colsum   
    
sess = tf.Session()
a = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])


print(sess.run(colsum,feed_dict = {x:a }))

#x = tf.constant([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
#
#y = x*2
#
#sess = tf.Session()
#
#print(sess.run(y))


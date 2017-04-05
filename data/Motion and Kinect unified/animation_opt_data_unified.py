# -*- coding: utf-8 -*-
"""
Created on Fri Dec 02 13:04:16 2016

@author: liuqi
"""
JointType_SpineBase = 0
JointType_SpineMid = 1
JointType_Neck = 2
JointType_Head = 3
JointType_ShoulderLeft = 4
JointType_ElbowLeft = 5
JointType_WristLeft = 6
JointType_HandLeft = 7
JointType_ShoulderRight = 8
JointType_ElbowRight = 9
JointType_WristRight = 10
JointType_HandRight = 11
JointType_HipLeft = 12
JointType_KneeLeft = 13
JointType_AnkleLeft = 14
JointType_FootLeft = 15
JointType_HipRight = 16
JointType_KneeRight = 17
JointType_AnkleRight = 18
JointType_FootRight = 19
JointType_SpineShoulder = 20
JointType_HandTipLeft = 21
JointType_ThumbLeft = 22
JointType_HandTipRight = 23
JointType_ThumbRight = 24

import cPickle as pk
import matplotlib.pyplot as plt

############################
# Motion capture
###########################
input_root = 'Unified_MData/'

input_file_name = 'qingyuan_2016-12-15 03.57.00 PM_FPS30_motion_unified_ex7'

input_file_tail = '.pkl'

'''
###########################
# Kinect
###########################
input_root = 'Unified_KData/'

input_file_name = 'Andy_12151611_Kinect_unified_ex1'

input_file_tail = '.pkl'
'''

#data_all = pk.load(file(input_root + input_file_name + input_file_tail,'rb'))

data_all = pk.load(file('/data/Motion and Kinect/Unified_MData/Andy_2016-12-15 04.15.27 PM_FPS30_motion_unified_ex4.pkl'))
kdata_all = pk.load(file('/data/Motion and Kinect/Unified_KData/Andy_12151615_Kinect_unified_ex4.pkl'))


NUM_LABELS = len(data_all)  # total number of the joints
NUM_FRAMES = len(data_all[0][1])   # total number of the frames
kNUM_FRAMES = len(kdata_all[0][1]) 
print 'The total frames: ', NUM_FRAMES

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.set_xlabel('Z axis')
ax.set_ylabel('X axis')
ax.set_zlabel('Y axis')
    
for frame_no in xrange(250,min(kNUM_FRAMES,NUM_FRAMES)):
    plt.cla()
    
    xs = []
    ys = []
    zs = []
    for joint_idx in  data_all.keys() :
        xs.append(data_all[joint_idx][0][frame_no])
        ys.append(data_all[joint_idx][1][frame_no])
        zs.append(data_all[joint_idx][2][frame_no])
        
    ax.scatter(zs, xs, ys)
    ax.set_xlim(-300,300)
    ax.set_ylim(-200,400)
    ax.set_zlim(50,600)
    ax.set_title(frame_no)
    plt.draw()
    plt.pause(1.0/120)
#    ax.set_xlabel('Z axis')
#    ax.set_ylabel('X axis')
#    ax.set_zlabel('Y axis')

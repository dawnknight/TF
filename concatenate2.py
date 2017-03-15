# -*- coding: utf-8 -*-
"""
Created on Wed Feb 01 17:35:50 2017

@author: liuqi
"""

import glob,os
import cPickle,h5py
import numpy as np


#Ksrc_path = 'I:/Kinect Project/Motion and Kinect unified/Unified_KData/'
#Msrc_path = 'I:/Kinect Project/Motion and Kinect unified/Unified_MData/'

Ksrc_path = 'I:/Kinect Project/Motion and Kinect unified/Unified_KData/'
Msrc_path = 'I:/Kinect Project/Motion and Kinect unified/Unified_MData/'

dst_path = './Concatenate_Data/'
date_ext = '_0306'

#exe_list = ['ex1','ex2','ex3','ex4','ex5','ex6','ex7']
exe_list = ['ex4']
K = []
M = []
for file_idx in exe_list:
    
    
    Klist = glob.glob(os.path.join(Ksrc_path, '*.pkl') )
    
    Mlist = glob.glob(os.path.join(Msrc_path, '*.pkl') )
    
    for Kinfile,Minfile in zip(Klist,Mlist):

        
        if 'ex4' in Kinfile:
            
            K.append(Kinfile)
        if 'ex4' in Minfile: 
            
            M.append(Minfile)  
            
#    for KK,MM in zip(K,M):            
#        print  KK.split('\\')[-1]   
#        print  MM.split('\\')[-1] 

    for idx, (Kinfile,Minfile) in enumerate(zip(K,M)):
        
        Kdata = cPickle.load(file(Kinfile,'r'))
        Mdata = cPickle.load(file(Minfile,'r'))
        
        len_stand = min(len(Kdata[0][0]),len(Mdata[0][0]))
        
        # concatenate all joints of one person
        for i in [0,1,2,3,4,5,6,8,9,10,20]:#[4,5,6,8,9,10]:#
            if i == 0:
                Kjoints_1_person = Kdata[i]
                Mjoints_1_person = Mdata[i]
            else:
                Kjoints_1_person = np.vstack([Kjoints_1_person,Kdata[i]])
                Mjoints_1_person = np.vstack([Mjoints_1_person,Mdata[i]])
            
        
        # concatenate all persons of one exercise(ex)
        if idx ==0:
            Kjoints_1_ex = Kjoints_1_person[:,:len_stand]
            Mjoints_1_ex = Mjoints_1_person[:,:len_stand]
        else:
            Kjoints_1_ex = np.hstack([Kjoints_1_ex,Kjoints_1_person[:,:len_stand]])
            Mjoints_1_ex = np.hstack([Mjoints_1_ex,Mjoints_1_person[:,:len_stand]])
            

         
    if Kjoints_1_ex.shape == Mjoints_1_ex.shape:
        print 'Size of ' + file_idx + ' is:'
        print Kjoints_1_ex.shape
    else:
        print 'Data generating process is wrong'
    
    cPickle.dump(Kjoints_1_ex,file(dst_path + 'K_'+file_idx+date_ext+'.pkl','wb'))
    cPickle.dump(Mjoints_1_ex,file(dst_path + 'M_'+file_idx+date_ext+'.pkl','wb'))
    cPickle.dump(Kjoints_1_ex[12:30,:],file(dst_path + 'limb_K_'+file_idx+date_ext+'.pkl','wb'))
    cPickle.dump(Mjoints_1_ex[12:30,:],file(dst_path + 'limb_M_'+file_idx+date_ext+'.pkl','wb'))    
    
from random import shuffle as sf

exeno = 'ex4'
K = cPickle.load(file(dst_path+'limb_K_'+exeno+date_ext+'.pkl','r'))
M = cPickle.load(file(dst_path+'limb_M_'+exeno+date_ext+'.pkl','r'))
taroK = cPickle.load(file(dst_path+'K_'+exeno+date_ext+'.pkl','r'))
taroM = cPickle.load(file(dst_path+'M_'+exeno+date_ext+'.pkl','r'))

MAX = np.max([K.max(),M.max()])
MIN = np.min([K.min(),M.min()])

#OsK = (K-MIN)/(MAX-MIN)
#OsM = (M-MIN)/(MAX-MIN)
#
#f = h5py.File("odata.h5", "w")
#f.create_dataset('data' , data = OsK) 
#f.create_dataset('label', data = OsM) 
#f.create_dataset('minmax'     , data =[MIN,MAX]) 
#f.close() 

test_rate = 0.2


idx = np.arange(K.shape[1])
sf(idx)
sK = K[:,idx]
sM = M[:,idx]

teX = sK[:,:int(0.2*K.shape[1])]
trX = sK[:,int(0.2*K.shape[1]):]

teL = sM[:,:int(0.2*K.shape[1])]
trL = sM[:,int(0.2*K.shape[1]):]


# normalized to -1 to 1
#NsK = (sK*2-KMIN-KMAX)/(KMAX-KMIN)
#NsM = (sM*2-MMIN-MMAX)/(MMAX-MMIN)

#normalized to 0 to 1
NsK = (sK-MIN)/(MAX-MIN)
NsM = (sM-MIN)/(MAX-MIN)

NteX = NsK[:,:int(0.2*K.shape[1])]
NtrX = NsK[:,int(0.2*K.shape[1]):]

NteL = NsM[:,:int(0.2*K.shape[1])]
NtrL = NsM[:,int(0.2*K.shape[1]):]


f = h5py.File(dst_path+'Ldata'+date_ext+'.h5', "w")
f.create_dataset('train_data' , data = trX) 
f.create_dataset('train_label', data = trL) 
f.create_dataset('test_data'  , data = teX) 
f.create_dataset('test_label' , data = teL) 
f.create_dataset('idx'        , data = idx) 
f.create_dataset('minmax'     , data =[MIN,MAX]) 
f.close() 


f = h5py.File(dst_path+'NLdata'+date_ext+'.h5', "w")
f.create_dataset('train_data' , data = NtrX) 
f.create_dataset('train_label', data = NtrL) 
f.create_dataset('test_data'  , data = NteX) 
f.create_dataset('test_label' , data = NteL) 
f.create_dataset('idx'        , data = idx) 
f.create_dataset('minmax'     , data =[MIN,MAX]) 
f.close() 

#build K and M data 
    
f = h5py.File(dst_path+'limb_KandM_'+exeno+date_ext+'.h5', "w")
f.create_dataset('N_Kinect' , data = (K-MIN)/(MAX-MIN))
f.create_dataset('N_Mcam'   , data = (M-MIN)/(MAX-MIN))
f.create_dataset('Kinect'   , data = K)      
f.create_dataset('Mcam'     , data = M)       
f.close()   

  
##build taro data
idx = np.array([0,1,2,3,10]) # joints id 0,1,2,3,20
bias = np.array([0,1,2])

taro_idx = np.tile(idx,(3,1)).T.flatten()*3+np.tile(bias,len(idx))

K_taro = taroK[taro_idx,:]
M_taro = taroM[taro_idx,:]   

f = h5py.File(dst_path+'KM_taro_'+exeno+date_ext+'.h5', "w")
f.create_dataset('Ktaro' , data = K_taro)
f.create_dataset('Mtaro' , data = M_taro )  

f.close()           

   
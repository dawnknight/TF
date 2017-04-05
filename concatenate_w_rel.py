# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 14:59:31 2017

@author: medialab
"""

import glob,os,pdb
import cPickle,h5py
import numpy as np
from random import shuffle as sf


Ksrc_path = './data/Motion and Kinect unified/Unified_KData/'
Msrc_path = './data/Motion and Kinect unified/Unified_MData/'
Rsrc_path = './data/Motion and Kinect unified/reliability/'

dst_path = './Concatenate_Data/'
date_ext = '_REL_b'

#exe_list = ['ex1','ex2','ex3','ex4','ex5','ex6','ex7']
exe_list = ['ex4']

for file_idx in exe_list:
    K = []
    M = [] 
    R = []  
    
    Klist = glob.glob(os.path.join(Ksrc_path, '*.pkl') )
    
    Mlist = glob.glob(os.path.join(Msrc_path, '*.pkl') )
    Rlist = glob.glob(os.path.join(Rsrc_path, '*.pkl') )
    
    if len(Klist)!=len(Mlist):
        print '###############################'
        print 'file number are not the same !!'
        print '###############################'
    
    for Kinfile,Minfile,Rinfile in zip(Klist,Mlist,Rlist):

        
        if 'ex4' in Kinfile:            
            K.append(Kinfile)
        if 'ex4' in Minfile:            
            M.append(Minfile) 
        if 'ex4' in Rinfile:            
            R.append(Rinfile)             
            
            
#    print 'EX : ' + file_idx
#    print 'K : '+repr(len(K))
#    print 'M : '+repr(len(M))+'\n'
            
#    for KK,MM in zip(K,M):            
#        print  KK.split('\\')[-1]   
#        print  MM.split('\\')[-1] 

    for idx, (Kinfile,Minfile,Rinfile) in enumerate(zip(K,M,R)):
        
        Kdata = cPickle.load(file(Kinfile,'r'))
        Mdata = cPickle.load(file(Minfile,'r'))
        Rdata = cPickle.load(file(Rinfile,'r'))
        data_len = min(len(Kdata[0][0]),len(Mdata[0][0]))
        
        # concatenate all joints of one person
        for i in [0,1,2,3,4,5,6,8,9,10,20]:
            if i == 0:
                Kjoints_1_person = Kdata[i]
                Mjoints_1_person = Mdata[i]
                Rjoints_1_person = Rdata[i]
                
            else:
                Kjoints_1_person = np.vstack([Kjoints_1_person,Kdata[i]])
                Mjoints_1_person = np.vstack([Mjoints_1_person,Mdata[i]])
                Rjoints_1_person = np.vstack([Rjoints_1_person,Rdata[i]])
        
        # concatenate all persons of one exercise(ex)
        if idx ==0:
            Kjoints_1_ex = Kjoints_1_person[:,:data_len]
            Mjoints_1_ex = Mjoints_1_person[:,:data_len]
            Rjoints_1_ex = Rjoints_1_person[:,:data_len]
        else:
            Kjoints_1_ex = np.hstack([Kjoints_1_ex,Kjoints_1_person[:,:data_len]])
            Mjoints_1_ex = np.hstack([Mjoints_1_ex,Mjoints_1_person[:,:data_len]])
            Rjoints_1_ex = np.hstack([Rjoints_1_ex,Rjoints_1_person[:,:data_len]])

         
    if Kjoints_1_ex.shape == Mjoints_1_ex.shape:
        print 'Size of ' + file_idx + ' is:'
        print Kjoints_1_ex.shape
    else:
        print 'Data generating process is wrong'
    
    cPickle.dump(Kjoints_1_ex,file(dst_path + 'K_'+file_idx+date_ext+'.pkl','wb'))
    cPickle.dump(Mjoints_1_ex,file(dst_path + 'M_'+file_idx+date_ext+'.pkl','wb'))
    cPickle.dump(Rjoints_1_ex,file(dst_path + 'R_'+file_idx+date_ext+'.pkl','wb'))
    cPickle.dump(Kjoints_1_ex[12:30,:],file(dst_path + 'limb_K_'+file_idx+date_ext+'.pkl','wb'))
    cPickle.dump(Mjoints_1_ex[12:30,:],file(dst_path + 'limb_M_'+file_idx+date_ext+'.pkl','wb'))    
    cPickle.dump(Rjoints_1_ex[4:10,:],file(dst_path + 'limb_R_'+file_idx+date_ext+'.pkl','wb'))
    

# normalize data
exeno = 'ex4'
#K = cPickle.load(file(dst_path+'limb_K_'+exeno+date_ext+'.pkl','r'))
#M = cPickle.load(file(dst_path+'limb_M_'+exeno+date_ext+'.pkl','r'))
#taroK = cPickle.load(file(dst_path+'K_'+exeno+date_ext+'.pkl','r'))
#taroM = cPickle.load(file(dst_path+'M_'+exeno+date_ext+'.pkl','r'))
ins_idx = np.arange(3,19,3)
K     = Kjoints_1_ex[12:30,:]
R     = Rjoints_1_ex[4:10,:]
M     = Mjoints_1_ex[12:30,:]
taroK = Kjoints_1_ex
taroM = Mjoints_1_ex


MAX = np.max([K.max(),M.max()])
MIN = np.min([K.min(),M.min()])

#create testing and training data

test_rate = 0.2


idx = np.arange(K.shape[1])
sf(idx)

# normalized to -1 to 1
#NK = (K*2-MIN-MAX)/(MAX-MIN)
#NM = (M*2-MIN-MAX)/(MAX-MIN)

#normalized to 0 to 1
NK = (K-MIN)/(MAX-MIN)
NK = np.insert(NK,ins_idx,R,0)
NM = (M-MIN)/(MAX-MIN)

tmpR = R
tmpR   = np.insert(tmpR,np.array([0,1,2,3,4,5]),R,0)
tmpR   = np.insert(tmpR,np.array([0,2,4,6,8,10]),R,0)


sNK = NK[:,idx]
sNM = NM[:,idx]
sR  = (tmpR[:,idx]>0.75)*1

NteX = sNK[:,:int(0.2*K.shape[1])]
teR  = sR[:,:int(0.2*K.shape[1])]          
NtrX = sNK[:,int(0.2*K.shape[1]):]
trR  = sR[:,int(0.2*K.shape[1]):]  

NteL = sNM[:,:int(0.2*K.shape[1])]
NtrL = sNM[:,int(0.2*K.shape[1]):]



# Normalized limb data with training and testing set 
f = h5py.File(dst_path+'NLdata'+date_ext+'.h5', "w")
f.create_dataset('train_data' , data = NtrX) 
f.create_dataset('train_label', data = NtrL) 
f.create_dataset('test_data'  , data = NteX) 
f.create_dataset('test_label' , data = NteL) 
f.create_dataset('idx'        , data = idx) 
f.create_dataset('train_data_rel' , data = trR)
f.create_dataset('test_data_rel'  , data = teR)
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

   
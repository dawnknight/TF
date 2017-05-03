# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 17:16:17 2017

@author: medialab
"""
import os , glob , cPickle , h5py
import numpy as np
from random import shuffle as sf


#src_path = 'I:/AllData_0327/unified data/'
src_path = 'D:/Project/K_project/data/Motion and Kinect unified/'
Kfolder  = 'Unified_KData/'
Mfolder  = 'Unified_MData/'
Rfolder  = 'reliability/'

dst_path = './Concatenate_Data/CNN/'
date_ext = '_CNN_0502'
exeno =  'ex4'

group_size = 30 # sample number per group
jnum = 11      # joint number *3 (xyz) per sample 
index = 0


LEN = []
Ksubtensor = {}
Msubtensor = {} 
Rsubtensor = {}    
for kinfile,minfile,rinfile  in zip(glob.glob(os.path.join(src_path+Kfolder,'*ex4.pkl')),\
                                    glob.glob(os.path.join(src_path+Mfolder,'*ex4_FPS30_motion_unified.pkl')),\
                                    glob.glob(os.path.join(src_path+Rfolder,'*Rel_ex4.pkl'))):
    print('group_'+str(index+1)+'......')
    print  kinfile
    print  minfile  
    print  rinfile
    print('==================================\n\n\n')

    kdata = cPickle.load(file(kinfile,'r'))
    mdata = cPickle.load(file(minfile,'r'))
    rdata = cPickle.load(file(rinfile,'r'))
    
    length = min(kdata[0].shape[1],mdata[0].shape[1])
    LEN.append(length-group_size+1) 
    Ksubtensor[index] = np.zeros([jnum*3,group_size,length-group_size+1])
    Msubtensor[index] = np.zeros([jnum*3,group_size,length-group_size+1])
    Rsubtensor[index] = np.zeros([jnum*3,group_size,length-group_size+1])  

    for i in kdata.keys():#[4,5,6,8,9,10]:
        if i == 0:
            Kjoints = kdata[i]
            Mjoints = mdata[i]
            Rjoints = np.vstack([rdata[i],rdata[i],rdata[i]])
        else:
            Kjoints = np.vstack([Kjoints,kdata[i]])
            Mjoints = np.vstack([Mjoints,mdata[i]])
            Rjoints = np.vstack([Rjoints,rdata[i],rdata[i],rdata[i]])
            
 
    for idx,_ in enumerate(xrange(group_size-1,length)):
        
            Ksubtensor[index][:,:,idx] = Kjoints[:,idx:group_size+idx]
            Msubtensor[index][:,:,idx] = Mjoints[:,idx:group_size+idx]
            Rsubtensor[index][:,:,idx] = Rjoints[:,idx:group_size+idx]

    index += 1
    
Ktensor = np.zeros([jnum*3,group_size,sum(LEN)])
Mtensor = np.zeros([jnum*3,group_size,sum(LEN)])
Rtensor = np.zeros([jnum*3,group_size,sum(LEN)])
Klimbtensor = np.zeros([18,group_size,sum(LEN)])
Mlimbtensor = np.zeros([18,group_size,sum(LEN)])
Rlimbtensor = np.zeros([18,group_size,sum(LEN)])

start = 0
end   = 0
for i in xrange(index-1):
        end += LEN[i]
        Ktensor[:,:,start:end] = Ksubtensor[i]
        Mtensor[:,:,start:end] = Msubtensor[i]
        Rtensor[:,:,start:end] = Rsubtensor[i]
#        Klimbtensor[:,:,start:end] = Ksubtensor[i][12:30,:,:]
#        Mlimbtensor[:,:,start:end] = Msubtensor[i][12:30,:,:]
        start = end


# normalize data       
        
idx = range(sum(LEN))
testrate = 0.2
sf(idx)

K = Ktensor[12:30,:,:]
M = Mtensor[12:30,:,:]
R = Rtensor[12:30,:,:]

MAX = np.max([K.max(),M.max()])
MIN = np.min([K.min(),M.min()])

#normalized to 0 to 1
NK = (K-MIN)/(MAX-MIN)
NM = (M-MIN)/(MAX-MIN)

#shuffle
sNK = NK[:,:,idx]
sNM = NK[:,:,idx]
sR  =  R[:,:,idx]

teX   = sNK[:,:,:int(0.2*sum(LEN))]
teL   = sNM[:,:,:int(0.2*sum(LEN))]
teR   =  sR[:,:,:int(0.2*sum(LEN))]            
trX   = sNK[:,:,int(0.2*sum(LEN)):]
trL   = sNM[:,:,int(0.2*sum(LEN)):] 
trR   =  sR[:,:,int(0.2*sum(LEN)):]            
           

f = h5py.File(dst_path+'NLdata'+date_ext+'.h5')
f.create_dataset('train_data'    , data = trX) 
f.create_dataset('train_label'   , data = trL) 
f.create_dataset('train_data_rel', data = trR)
f.create_dataset('test_data'     , data = teX) 
f.create_dataset('test_label'    , data = teL) 
f.create_dataset('test_data_rel' , data = teR)
f.create_dataset('idx'           , data = idx) 
f.create_dataset('minmax'        , data =[MIN,MAX]) 
f.close()             
        
f = h5py.File(dst_path+'batch'+date_ext+'.h5', "w")
f.create_dataset('Kdata' , data = Ktensor)
f.create_dataset('Mdata' , data = Mtensor)  
f.create_dataset('Rdata' , data = Rtensor)
f.close()              
            

 
f = h5py.File(dst_path+'limb_KandM_'+exeno+date_ext+'.h5', "w")
f.create_dataset('N_Kinect' , data = (K-MIN)/(MAX-MIN))
f.create_dataset('N_Mcam'   , data = (M-MIN)/(MAX-MIN))
f.create_dataset('Kinect'   , data = K)      
f.create_dataset('Mcam'     , data = M) 
      
f.close()   
        
            
            
            
            
         
         
     
     
     
    
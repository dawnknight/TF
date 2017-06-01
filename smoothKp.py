# -*- coding: utf-8 -*-
"""
Created on Wed May 31 15:09:27 2017

@author: medialab
"""

from scipy.ndimage.filters import gaussian_filter as gf
import h5py
import glob,os


src_path = 'D:/Project/K_project/data/unified Kprime/'
dst_path = 'D:/Project/K_project/data/unified Kprime smooth/'

for infile in glob.glob(os.path.join(src_path,'*.h5')):

    data = h5py.File(infile,'r')['data'][:]
    for idx in range(18):
        data[idx,:] = gf(data[idx,:],3)
        
    fname = dst_path + infile.split('\\')[-1]
 
    f = h5py.File(fname,'w')
    f.create_dataset('data',data = data)
    f.close()

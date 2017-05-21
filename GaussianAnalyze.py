#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:22:36 2017

@author: shijie
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.datasets import fetch_mldata
import _pickle as cPickle


file  = h5py.File('result_drop_G.h5', 'r')  # Kinect data after dae
return_y=file['kinect'][:]
motion=file['motion'][:]
file.close
file    = h5py.File('NLdata_0306.h5', 'r') # kinect unified data
kdataset = file['train_data'][:].T
dataset=file['train_label'][:].T
file.close

k1 = 66.0**2 * RBF(length_scale=67.0)  # long term smooth rising trend
k2 = 2.4**2 * RBF(length_scale=90.0) \
    * ExpSineSquared(length_scale=1.3, periodicity=1.0)  # seasonal component
# medium term irregularity
k3 = 0.66**2 \
    * RationalQuadratic(length_scale=1.2, alpha=0.78)
k4 = 0.18**2 * RBF(length_scale=0.134) \
    + WhiteKernel(noise_level=0.19**2)  # noise terms
kernel_gpml = k1 + k4

Max=np.max(motion)
Min=np.min(motion)
motion_n=(motion-Min)/(Max-Min)
Max=np.max(return_y)
Min=np.min(return_y)
return_y_n=(return_y-Min)/(Max-Min)

gp = GaussianProcessRegressor(kernel=kernel_gpml, alpha=0,
                              optimizer=None, normalize_y=True)

gp.fit(kdataset[0:3000,:], dataset[0:3000,:])

print("\nLearned kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f"
      % gp.log_marginal_likelihood(gp.kernel_.theta))
learned_kernel=gp.kernel_
y_pred, y_std = gp.predict(return_y_n, return_std=True)
err=y_pred-motion_n
plt.plot(err)


#f = h5py.File("result_drop_G.h5", "w")
#f.create_dataset('kinect'  , data = y_pred)
#f.create_dataset('motion'  , data = motion_n)
#f.close()
#f = h5py.File("model_drop_G.h5","w")
#f.create_dataset('kernel',data=gp)
#f.close()

s = cPickle.dumps(gp)
cPickle.dump(gp,open('model_drop_G.pkl','wb'))
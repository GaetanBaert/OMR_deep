# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 17:57:43 2018

@author: gaeta_000
"""
import glob
import matplotlib.image as mpimg
import numpy as np

folder  = "train"
path='../data/{}'.format(folder)
path=path+'_out_y'

for fpimg in glob.glob(path+'/*'):
    img = mpimg.imread(fpimg)
    write = False
    while img.shape[0]>326 :
        write = True
        if img.shape[0]<360 :
            img= img[:,:326,:]
        else:
            tmp= img[:,327:,:]
            img= img[:,:326,:]
            
    
    
    
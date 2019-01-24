# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 22:45:33 2018

@author: gaeta_000
"""

import glob
import matplotlib.image as mpimg

if __name__=="__main__":
   
    folder  = "validation"
    path='../data/{}'.format(folder)
    path=path+'_out_y'
    
    l = glob.glob(path+'/*')
    dico = {}
    test=[]
    for i in l :
        img = mpimg.imread(i)
        if img.shape[0] in dico:
            dico[img.shape[0]] += 1
        else:
            dico[img.shape[0]]=1
        if img.shape[0]==445:
            print(i)
                


    
    
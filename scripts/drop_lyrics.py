# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 15:56:30 2018

@author: gaeta_000
"""

import glob
import lxml.etree as ET

if __name__=="__main__":
   
    folder  = "evaluation"
    path='../data/{}'.format(folder)
    path=path+'_out_x'
    k= 0
    l = glob.glob(path+'/*')
    total = len(l)
    for i in l:
    #i= path + "/100017_0.xml"
        modified = False
        k=k+1
        tree = ET.parse(i)
        for lyric in tree.xpath('.//lyric'):
            if not modified:
                modified = True
            lyric.getparent().remove(lyric)
        for lyric in tree.xpath('.//harmony'):
            if not modified:
                modified = True
            lyric.getparent().remove(lyric) 
        for lyric in tree.xpath('.//print'):
            if not modified:
                modified = True
            lyric.getparent().remove(lyric)
    
        if modified:
            tree.write(i)
        if k%500 ==0 : 
            print (k/total)

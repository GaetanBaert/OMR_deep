# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 07:51:27 2018

@author: gaeta_000
"""

from music21 import converter
import glob
from music21.converter.subConverters import ConverterMusicXML
from music21.search.lyrics import LyricSearcher
import os


def removeLyrics(s):
    i = 0;
    for el in s.elements:
        try:
            for item in el.elements:
                try:
                    item.lyric=None 
                except Exception:
                    i=i+1
        except:
            i=i+1
        

if __name__=="__main__":
    list_problem = []
    conv_musicxml = ConverterMusicXML()
    folder  = "validation"
    path='../data/{}'.format(folder)
    output_y=path+'_out_y'
    output_x=path+'_out_x'
    k=0
    l=glob.glob(output_x+'/*')
    total = len(l)
    """
    for i in l:
        k=k+1
    """
    for i in ['..//data//train//48426.mxl']:
        try:
            sheet= converter.parse(i)
            m = sheet.makeMeasures()    
            #if len(LyricSearcher(m).index())>0:
            #    removeLyrics(m)
            #for j in range(0,round(len(m.getElementsByClass('Measure'))/4)):
            name = (i.split("//")[-1]).split(".")[0]
            fpimg = ((output_y+'/{}.png').format(name))
            if os.path.exists(fpimg):
                os.remove(fpimg)
            obj =m 
            #fp=((output_x+'/{}_{}.xml').format(name,j))
            #obj.write('musicxml',fp = fp)
            error=conv_musicxml.write(obj,fmt='musicxml',subformats=['png'])
            os.rename(error,fpimg)
            if k%500==0:
                print(k/total)
        except Exception:
            print( "ne possède pas de mesures")
            list_problem.append(i)


    

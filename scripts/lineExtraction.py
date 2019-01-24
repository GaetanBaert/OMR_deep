#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 12:11:34 2018

@author: robin
"""
import music21
from music21 import converter
from music21.converter.subConverters import ConverterMusicXML
import os
import inspect
import sys
import time
import cv2

undesiredTypes = sum([[i[0] for i in inspect.getmembers(classe,inspect.isclass)] for classe in [music21.expressions,
                                                                                            music21.text,
                                                                                            music21.bar,
                                                                                            music21.spanner,
                                                                                            music21.harmony,
                                                                                            music21.tempo,
                                                                                            music21.dynamics,
                                                                                            music21.repeat,
                                                                                            music21.chord]],[])


def dropLyrics(sheetLine):
    for o in sheetLine:
        if type(o).__name__ not in undesiredTypes:
            if type(o)==music21.stream.Part:
                for obj in o:
                    if type(obj).__name__ not in undesiredTypes:
                        if type(obj)==music21.stream.Measure:
                            for obj2 in obj:
                                if type(obj2)==music21.note.Note:
                                    obj2.lyric=None
                                    obj2.expressions = []
                                    obj2.articulations = []
                                elif type(obj2).__name__ in undesiredTypes:
                                    obj.remove(obj2)
                    else:
                        o.remove(obj)
        else:
            sheetLine.remove(o)
    return sheetLine

def labelList(sheetLine):
    labelList = []
    # Append sheetLine per measures
    measures = sheetLine.parts[0].makeMeasures()
    for m in [m for m in measures if type(m)==music21.stream.Measure]:
        measureLabelList = []
        for obj in m:
            # Clef detection
            if type(obj).__name__ in [i[0] for i in inspect.getmembers(music21.clef,inspect.isclass)]:
                measureLabelList.append('Clef '+obj.sign+str(obj.line)+'_'+str(obj.octaveChange))
            # Key detection
            elif type(obj).__name__ in [i[0] for i in inspect.getmembers(music21.key,inspect.isclass)]:
                measureLabelList.append('Key '+obj.getScale().name)
            # Time Signature detection
            elif type(obj).__name__ in [i[0] for i in inspect.getmembers(music21.meter,inspect.isclass)]:
                measureLabelList.append(obj.ratioString)
            # Note detection
            elif type(obj)==music21.note.Note:
                for alter in ['#~','-~','~','-`','`']:
                    if alter in obj.name:
                        return None
                measureLabelList.append(obj.name+'_'+str(obj.octave)+'_'+str(obj.duration.quarterLength))
            # Rest detection
            elif type(obj)==music21.note.Rest:
                measureLabelList.append(obj.name+'_'+str(obj.duration.quarterLength))
        labelList.append(measureLabelList)
    return labelList

def remainingTimeDisplay(i,nbImages,startTime):
    nbSecondes = int((time.time()-startTime)*(nbImages-i-1)/(i+1))
    if nbSecondes>60:
        nbMinutes = nbSecondes//60
        nbSecondes = nbSecondes%60
        if nbMinutes>60:
            nbHeures = nbMinutes//60
            nbMinutes = nbMinutes%60
            return str(nbHeures)+' h '+str(nbMinutes)+' m '+str(nbSecondes)+' s'
        else:
            return str(nbMinutes)+' m '+str(nbSecondes)+' s'
    else:
        return str(nbSecondes)+' s'


def extract_lines(pathLoad,pathSave,labelFileName):
    startTime = time.time()
    os.makedirs(pathSave, exist_ok=True)
    filenames = [f for f in os.listdir(pathLoad) if f.endswith('.mxl')]
    print("Lines extraction")
    errorFiles = []
    with open(labelFileName,'w') as txt:
        for filename,i in zip(filenames,range(len(filenames))):
            try:
                conv_musicxml = ConverterMusicXML()
                sheet = converter.parse(pathLoad+filename)
                measuresLine= []
                lineToKeep = False
                nbImages = 0
                for part in sheet.parts:
                    measures = [m for m in part if type(m)==music21.stream.Measure]
                    for measure in measures:
                        if len([obj for obj in measure if type(obj) == music21.layout.SystemLayout])!=0:
                            if lineToKeep:
                                sheetLine = sheet.measures(measuresLine[0],measuresLine[-1])
                                sheetLine=dropLyrics(sheetLine)
                                label = labelList(sheetLine)
                                if label is not None:
                                    error=conv_musicxml.write(sheetLine,fmt='musicxml',subformats=['png'])
                                    os.rename(error,pathSave+filename.replace('.mxl','')+'_'+str(nbImages)+'.png')
                                    txt.write(filename.replace('.mxl','')+'_'+str(nbImages)+'.png'+' : '+'||'.join(['|'.join(m) for m in labelList(sheetLine)])+'\n')
                                    nbImages+=1
                            measuresLine = []
                            lineToKeep=False
                        if len([obj for obj in measure if type(obj) == music21.note.Note])!=0:
                            lineToKeep = True
                        measuresLine.append(measure.number)
                        if measure==measures[-1]:
                            if lineToKeep:
                                sheetLine = sheet.measures(measuresLine[0],measuresLine[-1])
                                sheetLine=dropLyrics(sheetLine)
                                label = labelList(sheetLine)
                                if label is not None:
                                    error=conv_musicxml.write(sheetLine,fmt='musicxml',subformats=['png'])
                                    os.rename(error,pathSave+filename.replace('.mxl','')+'_'+str(nbImages)+'.png')
                                    txt.write(filename.replace('.mxl','')+'_'+str(nbImages)+'.png'+' : '+'||'.join(['|'.join(m) for m in labelList(sheetLine)])+'\n')
                                    nbImages+=1
            except:
                errorFiles.append(filename)
                print('New error')
            sys.stdout.write('\r'+'{0:.2f}'.format((i+1)*100/len(filenames))+' % | Remaining Time : '+remainingTimeDisplay(i,len(filenames),startTime)+'                         ')
            sys.stdout.flush()

    endTime = time.time()
    print("\nTotal time : "+str(endTime-startTime))
    return errorFiles

def labelAnalysis(filename):
    labels = []
    with open(filename,'r') as txt:
        for line in txt:
            if line is not '\n':
                labels.append(line.replace('\n','').split(' : '))

    notes = []
    clefs = dict()
    for label in labels:
        content = sum([l.split('|') for l in label[1].split('||')],[])
        for c in content:
            if ('_' in c) and ('Clef' not in c):
                notes.append('_'.join(c.split('_')[:-1]))
            elif 'Clef' in c:
                if not c[5:7] in clefs.keys():
                    clefs[c[5:7]]=0
                clefs[c[5:7]]+=1

    notes_occurences = list(set(notes))
    print('Nb occurences notes : '+str(len(notes_occurences)))
    notes_occurences.remove('rest')

    heights = [int(n.split('_')[-1]) for n in notes_occurences]
    heights_occurences = list(set(heights))
    print('Amplitude : from '+str(min(heights_occurences))+' to '+str(max(heights_occurences)))

    types = [n.split('_')[0] for n in notes_occurences]
    types_occurences = list(set(types))
    print('Differents notes : '+str(len(types_occurences)))

    clefs_occurences = list(set(clefs.keys()))
    print('Nb clefs : '+str(len(clefs_occurences)))

    return types_occurences

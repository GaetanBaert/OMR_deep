# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 18:19:13 2019

@author: gaeta
"""
import glob
import os
from neural_network import create_network
import cv2
import numpy as np
from imgaug import augmenters as iaa
from matplotlib import pyplot as plt
from music21 import converter


tinynotation_converter = {}

def load_weights_pred(model, weight_path, epoch):
    models = glob.glob(os.path.join(weight_path ,'*','*'))
    for weights in models:
        saved_epoch = int(weights.split("_")[1])
        
        if saved_epoch == epoch:
            type_model,_ = os.path.split(weights)
            _, type_model = os.path.split(type_model)
            if type_model == "notes":
                print(weights)
                model["notes"].model_pred.load_weights(weights)
                print("notes model loaded !")
            if type_model == "octaves":
                model["octaves"].model_pred.load_weights(weights)
                print("octaves model loaded !")
            if type_model == "rythms":
                model["rythms"].model_pred.load_weights(weights)
                print("rythms model loaded !")

def convert_into_number(y, list_label):
    t = list_label.split('{')[1]
    t = t.split('}')[0]
    t = t.split(',')
    res = list()
    for j in range(len(t)):
        t[j] = t[j].split("'")[1]
    for i in y:
        for j in range(len(t)):
            if i== t[j]:
                res.append(j)
                break
    return res

def convert_numbers_into_labels(model_output, list_labels):
    res = []
    for i in model_output:
        res.append(list_labels[i])
    return res


def build_label_list(string_labels):
    t = string_labels.split('{')[1]
    t = t.split('}')[0]
    t = t.split(',')
    for j in range(len(t)):
        t[j] = t[j].split("'")[1]
    return t

def predict(model, image, notes_list, octaves_list, rythms_list):
    notes_pred = model["notes"].predict_on_batch(image)
    octaves_pred = model["octaves"].predict_on_batch(image)
    rythms_pred = model["rythms"].predict_on_batch(image)
    return {"rythms" : convert_numbers_into_labels(rythms_pred, rythms_list), "octaves": convert_numbers_into_labels(octaves_pred, octaves_list),"notes": convert_numbers_into_labels(notes_pred, notes_list)} 
        
def open_image(image_path, aug_rate = 0):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED )
    if len(img.shape)>2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    img = cv2.resize(img,(0,0),fx=0.5,fy=0.5, interpolation =cv2.INTER_CUBIC)
    img = cv2.resize(img,(0,0),fx=0.5,fy=0.5, interpolation =cv2.INTER_CUBIC).T
    if aug_rate >0:
        augment_image(img, aug_rate)
    return img
        
def augment_image(img, aug_rate):
    # gaussian noise
    if aug_rate > np.random.rand():
        gaussian_noise = iaa.AdditiveGaussianNoise(0.01,np.random.rand()*30)
        img = gaussian_noise.augment_image(img)
    # Elastic Transformation (low sigma)
    if aug_rate> np.random.rand():
        et_low = iaa.ElasticTransformation(alpha=np.random.rand(), sigma=0.2)
        img = et_low.augment_image(img)
    # Elastic Transformation (High Sigma)
    if aug_rate> np.random.rand():
        et_high = iaa.ElasticTransformation(alpha=np.random.rand()*40.0, sigma=10.0)
        img = et_high.augment_image(img)
    # Clouds
    if aug_rate> np.random.rand():
        clouds = iaa.Clouds()
        img = clouds.augment_image(img)
        img = img-np.min(img)
        mulfactor = 255/np.max(img)
        img = img*mulfactor
        img = img.astype(np.uint8)
    return img

def make_note(note_symbol, octave_symbol):
    switcher = {
            -1 : note_symbol[0].capitalize(*5)
            0 :  note_symbol[0].capitalize()*4,
            1: note_symbol[0].capitalize()*3,
            2: note_symbol[0].capitalize()*2,
            3: note_symbol[0].capitalize(),
            4: note_symbol[0].lowercase(),
            5: note_symbol[0].lowercase()+ "'",
            6: note_symbol[0].lowercase()+ "'" * 2,
            7: note_symbol[0].lowercase()+ "'" * 3,
            8: note_symbol[0].lowercase()+ "'" * 4,
            9: note_symbol[0].lowercase()+ "'" * 5,
            10: note_symbol[0].lowercase()+ "'" * 6,
            11: note_symbol[0].lowercase()+ "'" * 7,
            12: note_symbol[0].lowercase()+ "'" * 8,
            }

def make_tiny_notation(prediction):
    res = ''
    # Se base sur les rythmes car c'est là que l'on a le meilleur score 
    parser_octave = 0
    parser_notes = 0
    time_by_measure = []
    last_rythm = 0.
    for i in range (len(prediction["rythms"])):
        if prediction["rythms"][i] == '|':
            if prediction["notes"][parser_notes]!='|':
                if prediction["notes"][parser_notes-1]=='|' or len(prediction["notes"]) <=len(prediction["rythms"]:
                    parser_notes -= 1
                elif prediction["notes"][parser_notes+1]=='|' or len(prediction["notes"]) <=len(prediction["rythms"]:
                    parser_notes += 1
            if prediction["octaves"][parser_octaves]!='|':
                if prediction["octaves"][parser_octaves-1]=='|' or len(prediction["octaves"]) <=len(prediction["rythms"]:
                    parser_octaves -= 1
                elif prediction["octaves"][parser_octaves+1]=='|' or len(prediction["octaves"]) <=len(prediction["rythms"]:
                    parser_octaves += 1
            time_by_measure.append(0)
        else:
            #Append the note
            if prediction["notes"][parser_notes]== "rest" :
                res.append('r')
            elif prediction["notes"][parser_notes]=='|':
                if prediction["octaves"][parser_notes]== '|':
                    
            if last_rythm!= prediction["rythms"][i]:
                res.append(string(int(1/float(prediction["rythms"][i])*4))
                last_rythm = prediction["rythms"][i]
        parser_notes += 1
        parser_octaves += 1
                    


if __name__ =="__main__":
    list_label_notes = build_label_list(open("labels_notes.txt",'r').read())
    list_label_octaves = build_label_list(open("labels_octaves.txt",'r').read())
    list_label_rythms = build_label_list(open("labels_rythms.txt",'r').read())
    img= open_image("D:/Documents/OMR deep/database/data/evaluation_out_x/569_0.png", 0.5)
    plt.imshow(img.T, cmap='gray')
    model = create_network(img.shape[1], 255, lr = 0)
    prediction = predict(model, img, list_label_notes, list_label_octaves, list_label_rythms)
    tiny = make_tiny_notation(predict)
    output = converter.parse('tinynotation:{}'.format(tiny))
    output.show()
        
        
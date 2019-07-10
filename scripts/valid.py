# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 18:27:03 2019

@author: gaeta
"""

from neural_network import create_network
from data_generator import DataGenerator
import numpy as np
import os
import glob

def validate(model, category, ids):
    valid_generator = DataGenerator(ids,"validation", batch_size = 16, aug_rate = 0.25, out=category)
    return model[category].evaluate_generator(valid_generator, verbose=1)

def load_weights_eval(model, weight_path, epoch):
    models = glob.glob(os.path.join(weight_path ,'*','*'))
    for weights in models:
        saved_epoch = int(weights.split("_")[1])
        
        if saved_epoch == epoch:
            type_model,_ = os.path.split(weights)
            _, type_model = os.path.split(type_model)
            if type_model == "notes":
                print(weights)
                model["notes"].model_eval.load_weights(weights)
                print("notes model loaded !")
            if type_model == "octaves":
                model["octaves"].model_eval.load_weights(weights)
                print("octaves model loaded !")
            if type_model == "rythms":
                model["rythms"].model_eval.load_weights(weights)
                print("rythms model loaded !")

    
if __name__ == "__main__":
    ids= os.listdir(os.path.abspath("../data/validation_out_x/"))
    valid_gen = DataGenerator(ids,"validation", batch_size = 2, aug_rate = 0.25)
    x_valid = valid_gen[0]
    nb_features = int(x_valid[1][0][0].shape[2])
    padding_value = 255
    model = create_network(nb_features, padding_value, lr = 0.0001)
    weights_path = "../models/checkout/test2"
    load_weights_eval(model,weights_path, 27)
    ler = {}
    ser = {}
#    for cat in model.keys():
#        ler[cat], ser[cat] = validate(model, cat, ids)
#        print("{} : ler - {}, ser - {}".format(cat, np.mean(ler[cat]), ser[cat]))
    ler, ser = validate(model, "notes", ids)
    print("{} : ler - {}, ser - {}".format("notes", np.mean(ler), ser))
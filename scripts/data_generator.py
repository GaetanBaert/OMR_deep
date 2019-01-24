


import os
import cv2
import numpy as np
import mmap
import keras

import tensorflow as tf

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, directory, category, category_function, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.directory = directory
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.label_file = open("../data/{}_labels.txt".format(directory),'r')
        self.category_function = category_function
        self.list_label_file = open("labels_{}.txt".format(category),'r')
        self.list_label = self.list_label_file.read()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if(index+1)*self.batch_size<len(self.indexes):
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        else:
            indexes = self.indexes[index*self.batch_size:-1]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)
        y = np.zeros(len(X[2]))

        return (X,y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        images_path = os.path.abspath("../data/{}_out_x/".format(self.directory))
        X = list()
        y = list()
        X_len = list()
        y_len = list()
        fail =0
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            i = i-fail
            y_temp = np.asarray(list(self.category_function(self.label_file , ID)),dtype="str")
            if y_temp.shape[0] != 0:
                image_temp = cv2.imread(images_path +'/'+ ID)
                if len(image_temp.shape)>2:
                    	image_temp = cv2.cvtColor(image_temp, cv2.COLOR_BGR2GRAY)
                image_temp = cv2.resize(image_temp,(0,0),fx=0.5,fy=0.5, interpolation =cv2.INTER_CUBIC)
                image_temp = cv2.resize(image_temp,(0,0),fx=0.5,fy=0.5, interpolation =cv2.INTER_CUBIC).T
                X_len.append(image_temp.shape[0])

                y.append(self.convert_into_number(y_temp))
                y_len.append(y_temp.shape[0])
                # Store sample
                X.append(image_temp)

        y_len = np.asarray(y_len)
        X_len = np.asarray(X_len)

        pad_value = max(X_len)
#        y_pad_value = max(y_len)
        for i in range(len(X)):
            if X[i].shape[0]!=pad_value:
                X[i] = np.concatenate((X[i] , np.ones((pad_value - X[i].shape[0], X[i].shape[1]))*127),axis=0)
#        for i in range(len(y)):
#            if len(y[i])!=y_pad_value:
#                y[i] = np.concatenate((y[i] , np.floor(np.random.rand(y_pad_value-len(y[i]))*4)+self.n_classes))
        X = keras.preprocessing.sequence.pad_sequences(X, value= float(127),dtype="float32", padding="post")
        y = keras.preprocessing.sequence.pad_sequences(y, value=self.n_classes , dtype="int32", padding="post")
            # Store class
        n,length, height = X.shape

        return [X,y,X_len,y_len]
#        return [np.reshape(X, [n,length,height, 1]), y, X_len, y_len]

    def convert_into_number(self, y):
        t = self.list_label.split('{')[1]
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




    def generate_data(self,directory,category_function):
        images_path = os.path.abspath("../data/{}_out_x/".format(directory))
        f = open("../data/{}_labels.txt".format(directory),'r')
        for imagename in os.listdir(images_path) :
            image_path = images_path+'\\'+imagename
            image = cv2.imread(image_path,0)
            labels = category_function(f, imagename)
            yield(image,list(labels))

    def labels_for_image(self, f,imagename):
        s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        byte_number = s.find(imagename.encode('utf-8'))
        image_labels = f.read()[byte_number:].split('\n')[0]
        return (image_labels.split('|')[1:])



    def notes_label(self, f, imagename):
        image_labels = self.labels_for_image(f,imagename)
        list_notes_labels=["rest","A","A-","A#","B","B-","B#","C","C-","C#","D","D-","D#","E","E-","E#","F","F-","F#","G","G-","G#"]
        for res in image_labels[2:] :
            temp = res.split('_')
            if temp[0] in list_notes_labels:
                yield temp[0]
            elif temp[0] == '':
                yield '|'


    def octaves_label(self, f,imagename):
        image_labels = self.labels_for_image(f,imagename)
        for res in image_labels[2:]:
            temp = res.split('_')
            if len(temp)>2:
                yield temp[1]
            elif temp[0]=='':
                yield '|'
            elif temp[0]=='rest' :
                yield temp[0]


    def rythms_label(self, f, imagename):
        image_labels = self.labels_for_image(f,imagename)

        for res in image_labels[2:]:
            temp = res.split('_')
            if len(temp)>1:
                yield(temp[-1])
            elif temp[0]=='':
                yield '|'

        
        return(res.split('_')[-1] if res.split('_')[-1]!=''  else '|' for res in image_labels[2:] )

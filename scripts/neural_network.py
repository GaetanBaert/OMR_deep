
# coding: utf-8

#%%


from keras.layers import TimeDistributed, Activation, Dense, Input, Bidirectional, LSTM, Masking, GaussianNoise
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from CTCModel import CTCModel
import data_generator as datas
import numpy as np
import os
import mmap
import pickle


#%%


def labels_for_image(f,imagename):
    s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    byte_number = s.find(imagename.encode('utf-8'))
    image_labels = f.read()[byte_number:].split('\n')[0]
    return (image_labels.split('|')[1:])



def notes_label(f, imagename):
    image_labels = labels_for_image(f,imagename)
    list_notes_labels=["rest","A","A-","A#","B","B-","B#","C","C-","C#","D","D-","D#","E","E-","E#","F","F-","F#","G","G-","G#"]
    for res in image_labels[2:] :
        temp = res.split('_')
        if temp[0] in list_notes_labels:
            yield temp[0]
        elif temp[0] == '':
            yield '|'


def octaves_label(f,imagename):
    image_labels = labels_for_image(f,imagename)
    for res in image_labels[2:]:
        temp = res.split('_')
        if len(temp)>2:
            yield temp[1]
        elif temp[0]=='':
            yield '|'
        elif temp[0]=='rest' :
            yield temp[0]


def rythms_label(f, imagename):
    image_labels = labels_for_image(f,imagename)
    
    for res in image_labels[2:]:
        temp = res.split('_')
        if len(temp)>1:
            yield(temp[-1])
        elif temp[0]=='':
            yield '|'

    
    return(res.split('_')[-1] if res.split('_')[-1]!=''  else '|' for res in image_labels[2:] )
    


def create_network(nb_features, nb_labels, padding_value, name):

    # Define the network architecture
    input_data = Input(name='input', shape=(None, nb_features)) # nb_features = image height
    masking = Masking(mask_value=padding_value)(input_data)
    noise = GaussianNoise(0.01)(masking)
#    conv = Conv2D(32, kernel_size=(3,3), activation='relu')(noise)
#    pool = MaxPooling2D(pool_size=(2,2))(conv)
#    flatten = TimeDistributed(Flatten())(pool)
    blstm = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3))(noise)
    blstm = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3))(blstm)
    blstm = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3))(blstm)

    dense = TimeDistributed(Dense(nb_labels + 1, name="dense"))(blstm)
    outrnn = Activation('softmax', name=name)(dense)

    network = CTCModel([input_data], [outrnn])
    network.compile(Adam(lr=0.0001))

    return network


# ## Premier réseau : Nom des notes 

#%%



#nb_labels = 23 # 23 symboles pour les notes 
nb_labels = 15 # 15 symboles pour les octaves
nb_epochs = 10
ids=dict()
ids['train']=os.listdir(os.path.abspath("../data/train_out_x/"))
ids['eval']= os.listdir(os.path.abspath("../data/evaluation_out_x/"))[:2000]
train_generator = datas.DataGenerator(ids['train'], "train", "octave", octaves_label, n_classes = nb_labels, batch_size = 24)
eval_generator = datas.DataGenerator(ids['eval'],"evaluation", "octave", octaves_label, n_classes = nb_labels, batch_size = 24)
nb_train = len(ids['train'])
nb_eval = len(ids['eval'])
x_eval = eval_generator[0]
y_eval = np.zeros(len(x_eval[0][2]))
nb_features = int(x_eval[0][0].shape[2]) #Hauteur des images
padding_value = 127
#%%
network = create_network(nb_features, nb_labels, padding_value, "octave")
checkpoint = ModelCheckpoint(filepath="octave_model/notes_model_256-{epoch:02d}-{val_loss:.2f}.h5")
network.save_model("octave_model")
network.load_model("octave_model", Adam(lr=0.0001), "notes_model/notes_model_256_generator.h5",by_name=True )

#%%
train_generator.indexes = pickle.load(open("notes_model/generator.pkl",'rb'))

#%%

network.fit_generator(generator = train_generator, steps_per_epoch= len(train_generator), callbacks = [checkpoint], workers = 1, validation_data=eval_generator, epochs=30, initial_epoch = 12)

network.model_train.save_weights("octave_model/octave_model_256_generator.h5")

#%%
eval = datas.DataGenerator(ids['eval'], "evaluation", "octave", octaves_label, n_classes = nb_labels, batch_size = 240)[0][0]
model_eval = network.evaluate(x = eval, batch_size=24, metrics=['ler', 'ser'])

#%%
f_eval = open("notes_model/eval_256_generator.txt",'a')
f_eval.write(str(model_eval))
f_eval.close()


#%% 
eval = datas.DataGenerator(ids['eval'], "evaluation", "notes", notes_label, n_classes = nb_labels, batch_size = 10)[0][0]
def convert_into_notes(list_label, y):
    t = list_label.split('{')[1]
    t = t.split('}')[0]
    t = t.split(',')
    res = []
    for i in y:
        if i < len(t) and i>=0:
            res.append(t[int(i)])
    return res

x_test, y_test, x_test_len, y_test_len= eval

pred = network.predict([x_test, x_test_len],batch_size=32, verbose=1, max_value = padding_value)

for i in range(10):  # print the 10 first predictions
        print("Prediction n°",i," : ", convert_into_notes(train_generator.list_label,pred[i]))
        print("Label : ", convert_into_notes(train_generator.list_label,y_test[i])) 


            




# coding: utf-8

#%%


from keras.layers import TimeDistributed, Activation, Dense, Input, Bidirectional, LSTM, GaussianNoise, Conv2D, MaxPooling2D, Lambda, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from CTCModel import CTCModel
import data_generator as datas
import numpy as np
import os
from tqdm import trange
import time
import glob

#%%




def train_multitask(train_generator, valid_generator, epochs, models,batch_size_eval, checkout_path = None, log_path = None, start_epoch = 0, start_batch = 0):
    for i in range(start_epoch+1,epochs+1):
        losses = {"notes": 100, "octaves":100, "rythms":100}
        nb_batches_per_model={"notes":0, "octaves": 0, "rythms": 0}
        toc = time.clock()
        print("epoch {}/{} : ".format(i, epochs))
        with trange(start_batch,len(train_generator)) as t:
            for j in t:
                t.set_postfix(loss_notes = losses["notes"], loss_octaves = losses["octaves"], loss_rythms = losses["rythms"])
                [specific_labels,batch] = train_generator[j]
                nb_batches_per_model[specific_labels] = nb_batches_per_model[specific_labels] +1
                if nb_batches_per_model[specific_labels] == 1:
                    losses[specific_labels] = models[specific_labels].train_on_batch(batch[0], batch[1])
                else:
                    losses[specific_labels] =(losses[specific_labels]* 
                          (nb_batches_per_model[specific_labels]-1)+ 
                          models[specific_labels].train_on_batch(batch[0], batch[1]))/nb_batches_per_model[specific_labels]
        start_batch = 0
        train_generator.on_epoch_end()
        valid_losses = {"notes": 100, "octaves":100, "rythms":100}
        nb_valid_batches_per_model = {"notes":0, "octaves": 0, "rythms": 0}
        with trange(len(valid_generator)) as t:
            for j in t:
                t.set_postfix(loss_notes = valid_losses["notes"], loss_octaves = valid_losses["octaves"], loss_rythms = valid_losses["rythms"])
                [specific_labels, batch] = valid_generator[j]
                nb_valid_batches_per_model[specific_labels] = nb_valid_batches_per_model[specific_labels]+1
                if nb_valid_batches_per_model[specific_labels] == 1:
                    valid_losses[specific_labels]= models[specific_labels].get_loss_on_batch(batch[0])[0]/batch_size_eval
                else:
                    valid_losses[specific_labels] =(valid_losses[specific_labels] *
                                (nb_valid_batches_per_model[specific_labels]-1) + 
                                (models[specific_labels].get_loss_on_batch(batch[0]))[0]/batch_size_eval)/nb_valid_batches_per_model[specific_labels]
        print("time: {:.0f}s".format(time.clock()-toc))
        if checkout_path != None:
            os.makedirs(os.path.join(checkout_path, "notes"), exist_ok=True)
            os.makedirs(os.path.join(checkout_path, "octaves"), exist_ok=True)
            os.makedirs(os.path.join(checkout_path, "rythms"), exist_ok=True)
            models["notes"].model_train.save_weights("{}/notes/notes_{:03d}_train-{:.2f}_valid-{:.2f}.h5"
                  .format(checkout_path, i, losses["notes"], valid_losses["notes"] ))
            models["octaves"].model_train.save_weights("{}/octaves/octaves_{:03d}_train-{:.2f}_valid-{:.2f}.h5"
                  .format(checkout_path, i, losses["octaves"], valid_losses["octaves"]))
            models["rythms"].model_train.save_weights("{}/rythms/rythms_{:03d}_train-{:.2f}_valid-{:.2f}.h5"
                  .format(checkout_path, i, losses["rythms"], valid_losses["rythms"]))
        if log_path != None :
            os.makedirs(log_path, exist_ok=True)
            writer = tf.summary.FileWriter(log_path)
            summary = tf.Summary(value=[tf.Summary.Value(tag="notes train loss", simple_value = losses["notes"] ), ])
            writer.add_summary(summary,i)
            summary = tf.Summary(value=[tf.Summary.Value(tag="notes valid loss", simple_value = valid_losses["notes"] ), ])
            writer.add_summary(summary,i)
            summary = tf.Summary(value=[tf.Summary.Value(tag="octaves train loss", simple_value = losses["octaves"] ), ])
            writer.add_summary(summary,i)
            summary = tf.Summary(value=[tf.Summary.Value(tag="octaves valid loss", simple_value = valid_losses["octaves"] ), ])
            writer.add_summary(summary,i)
            summary = tf.Summary(value=[tf.Summary.Value(tag="rythms train loss", simple_value = losses["rythms"] ), ])
            writer.add_summary(summary,i)
            summary = tf.Summary(value=[tf.Summary.Value(tag="rythms valid loss", simple_value = valid_losses["rythms"] ), ])
            writer.add_summary(summary,i)

    
def build_head(input_data, network, nb_labels, name, lr=0.0001):
    blstm = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3))(network)
    blstm = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3))(blstm)
    
    dense = TimeDistributed(Dense(nb_labels + 1, name="dense"))(blstm)
    outrnn = Activation('softmax', name=name)(dense)
    
    model = CTCModel([input_data], [outrnn])
    if lr > 0:
        model.compile(Adam(lr=lr))
    
    return model

def cnn_base(input_data):
    cnn = Conv2D(64, (3, 3), padding='same')(input_data)
    cnn = Activation("relu")(cnn)
    #cnn = BatchNormalization()(cnn)
    cnn = Conv2D(64, (3, 3), padding='same')(cnn)
    cnn = Activation("relu")(cnn)
    #cnn = BatchNormalization()(cnn)
    cnn = Conv2D(32, (3, 3), padding='same')(cnn)
    cnn = Activation("relu")(cnn)
    #cnn = BatchNormalization()(cnn)
    cnn = Conv2D(32, (3, 3), padding='same')(cnn)
    cnn = Activation("relu")(cnn)
    #cnn = BatchNormalization()(cnn)
    cnn = Conv2D(16, (3, 3), padding='same')(cnn)
    cnn = Activation("relu")(cnn)
    #cnn = BatchNormalization()(cnn)
    cnn = Conv2D(16, (3, 3), padding='same')(cnn)
    cnn = Activation("relu")(cnn)
    #cnn = BatchNormalization()(cnn)
    return cnn

def create_network(nb_features, padding_value, lr= 0.00001):
    
    # Define the network architecture
    input_data = Input(name='input', shape=(None,nb_features, 1)) # nb_features = image height
    noise = GaussianNoise(0.01)(input_data)
    cnn = cnn_base(noise)
    
    out_cnn = MaxPooling2D(pool_size = (1,nb_features))(cnn)
    
    input_blstm = Lambda(lambda x: K.squeeze(x, axis=2))(out_cnn)
    
    blstm = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3))(input_blstm)
    
    notes_model = build_head(input_data, blstm, 23, "notes", lr)
    octaves_model = build_head(input_data, blstm, 15, "octaves", lr)
    rythms_model = build_head(input_data, blstm, 60, "rythms", lr)

    return {"notes": notes_model, "octaves": octaves_model, "rythms": rythms_model}

def load_weights(network, epoch, weights_folder):
    models = glob.glob(os.path.join(weights_folder ,'*','*'))
    for model in models:
        saved_epoch = int(model.split("_")[1])
        
        if saved_epoch == epoch:
            type_model,_ = os.path.split(model)
            _, type_model = os.path.split(type_model)
            if type_model == "notes":
                network["notes"].model_train.load_weights(model)
                print("notes model loaded !")
            if type_model == "octaves":
                network["octaves"].model_train.load_weights(model)
                print("octaves model loaded !")
            if type_model == "rythms":
                network["rythms"].model_train.load_weights(model)
                print("rythms model loaded !")

# ## Premier réseau : Nom des notes 

#%%


if __name__ =="__main__":
    #nb_labels = 23 # 23 symboles pour les notes 
    nb_labels = 15 # 15 symboles pour les octaves
    nb_epochs = 50
    ids=dict()
    ids['train']=os.listdir(os.path.abspath("../data/train_out_x/"))
    ids['valid']= os.listdir(os.path.abspath("../data/validation_out_x/"))
    batch_size_eval = 16
    train_generator = datas.DataGenerator(ids['train'], "train", batch_size =12, aug_rate = 0.25)
    valid_generator = datas.DataGenerator(ids['valid'],"validation", batch_size = batch_size_eval, aug_rate = 0.25)
    nb_train = len(ids['train'])
    nb_eval = len(ids['valid'])
    x_valid = valid_generator[0]
    y_valid = np.zeros(len(x_valid[1][0][2]))
    nb_features = int(x_valid[1][0][0].shape[2]) #Hauteur des images
    padding_value = 255
    #%%
    network = create_network(nb_features, padding_value, lr = 0.0001)
    #%%
    checkout_path = "../models/checkout/test2"
    start_epoch = 16
    if start_epoch != 0:
        load_weights(network, start_epoch, checkout_path)
    #%%
    #train_multitask(train_generator,valid_generator, nb_epochs, network,batch_size_eval,  checkout_path,"logs/test2", start_batch = 0, start_epoch =start_epoch )
    
    
    # Entraînement du réseau de notes seulement
    checkpoint_name="notes_{epoch:03d}_train-{loss:.2f}_valid-{val_loss:.2f}.h5"
    checkpoint = ModelCheckpoint(os.path.join(checkout_path, "octaves",checkpoint_name))
    tensorboard = TensorBoard(log_dir=os.path.join("logs","test2", "notes"))
    
    train_generator = datas.DataGenerator(ids['train'], "train", batch_size =12, aug_rate = 0.25, out="octaves" )
    valid_generator = datas.DataGenerator(ids['valid'],"validation", batch_size = batch_size_eval, aug_rate = 0.25, out="octaves")
    network["octaves"].fit_generator(train_generator, len(train_generator), epochs = 30, initial_epoch = start_epoch,
           validation_data=valid_generator, validation_steps=len(valid_generator), callbacks=[checkpoint, tensorboard])
    #%% 
    
    
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
    
    
                
    

import time
time_start = time.time()

import scipy.io as scio
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from keras import callbacks
from keras.callbacks import ReduceLROnPlateau

def read_data(directory, key): 
    data_dict = scio.loadmat(directory)
    data = data_dict[key]
    return data

train_input_directory = './geostat/2D/input_train_16_PIML_2.mat'
train_input_key = 'input_train'
train_input_data = read_data(train_input_directory, train_input_key)
print(train_input_data.shape)

train_output_directory = './geostat/2D/output_train_16_PIML_2.mat'
train_output_key = 'output_train'
train_output_data = read_data(train_output_directory, train_output_key)
print(train_output_data.shape)

inputs = tf.keras.layers.Input(shape=(32,))
Dense1 = tf.keras.layers.Dense(50)(inputs)
leaky1 = tf.keras.layers.LeakyReLU(alpha=0.01)(Dense1)
Dense2 = tf.keras.layers.Dense(50)(leaky1)
leaky2 = tf.keras.layers.LeakyReLU(alpha=0.01)(Dense2)
Dense3 = tf.keras.layers.Dense(50)(leaky2)
leaky3 = tf.keras.layers.LeakyReLU(alpha=0.01)(Dense3)
Dense4 = tf.keras.layers.Dense(50)(leaky3)
leaky4 = tf.keras.layers.LeakyReLU(alpha=0.01)(Dense4)
Dense5 = tf.keras.layers.Dense(17)(leaky4)
predictions = tf.keras.layers.LeakyReLU(alpha=0.01)(Dense5)
model = tf.keras.Model(inputs=inputs, outputs=predictions)
model.summary()

def custom_loss(layer):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred[:,:16]-y_true[:,:16])) + K.mean(K.square(y_pred[:,-1]-y_true[:,-1])) - K.square(K.mean(y_pred[:,-1]-y_true[:,-1])) + K.mean(K.square(1 - K.sum(y_pred[:,:16],axis=1))) + K.mean(K.square(K.batch_dot(K.repeat_elements(K.expand_dims(y_pred[:,-1],1),16,1)-layer[:,16:32],y_pred[:,:16],axes=(1,1))))
    return loss

rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=50)
adam = optimizers.Adam(lr = 0.001)
model.compile(optimizer = adam, loss = custom_loss(inputs))
callbacks = callbacks.EarlyStopping(monitor='val_loss', patience=20)
history = model.fit(train_input_data, train_output_data, batch_size = 64, epochs = 800, verbose = 2, validation_split = 0.2, shuffle = True, callbacks = [rlrop,callbacks])
model.save('./geostat/2D/PIML_model_16_2-1.h5')
#plt.figure()
#plt.plot(history.history['loss'], 'b')
#plt.plot(history.history['val_loss'], 'k')
#plt.legend(('training','validation'))
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.show()

train_input_directory = './geostat/2D/input_test_16_PIML_2.mat'
train_input_key = 'input_test'
train_input_data = read_data(train_input_directory, train_input_key)
print(train_input_data.shape)

prediction = model.predict(train_input_data)
time_end = time.time()
print('time_cost', time_end-time_start, 's')

scio.savemat('./geostat/2D/prediction_2D_16_2.mat', {'prediction_2D': prediction})
scio.savemat('./geostat/2D/train_loss_16_2.mat', {'train_loss_16': history.history['loss']})
scio.savemat('./geostat/2D/val_loss_16_2.mat', {'val_loss_16': history.history['val_loss']})


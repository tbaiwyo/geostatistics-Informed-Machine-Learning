import time
time_start = time.time()

import scipy.io as scio
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, Reshape, Dense
from keras import optimizers
import matplotlib.pyplot as plt
from keras import callbacks

def read_data(directory, key): 
    data_dict = scio.loadmat(directory)
    data = data_dict[key]
    return data

train_input_directory = './geostat/cov_train_36_z_2.mat'
train_input_key = 'cov_train'
train_input_data = read_data(train_input_directory, train_input_key)
print(train_input_data.shape)

train_output_directory = './geostat/lambda_train_36_z_2.mat'
train_output_key = 'lambda_train'
train_output_data = read_data(train_output_directory, train_output_key)
print(train_output_data.shape)

train_input = np.zeros((8192,37,36,1))
train_input[:,:,:,0] = train_input_data 

train_output = np.zeros((8192,37,1,1))
train_output[:,:,0,0] = train_output_data 

inputs = tf.keras.layers.Input(shape=(37,36,1))
convo1 = tf.keras.layers.Conv2D(filters=2, kernel_size=(3,3), strides=(1,3),padding='same')(inputs)
convo2 = tf.keras.layers.Conv2D(filters=4, kernel_size=(2,2), strides=(1,3),padding='same')(convo1)
convo3 = tf.keras.layers.Conv2D(filters=1, kernel_size=(2,2), strides=(1,2),padding='same')(convo2)
reshape1 = tf.keras.layers.Reshape((74,))(convo3)
Dense1 = tf.keras.layers.Dense(56)(reshape1)
leaky1 = tf.keras.layers.LeakyReLU(alpha=0.01)(Dense1)
Dense2 = tf.keras.layers.Dense(56)(leaky1)
leaky2 = tf.keras.layers.LeakyReLU(alpha=0.01)(Dense2)
Dense3 = tf.keras.layers.Dense(37)(leaky2)
leaky3 = tf.keras.layers.LeakyReLU(alpha=0.01)(Dense3)
predictions = tf.keras.layers.Reshape((37,1,1))(leaky3)
model = tf.keras.Model(inputs=inputs, outputs=predictions)
model.summary()

def custom_loss(layer):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred[:,:36,:,:]-y_true[:,:36,:,:])) + K.mean(K.square(y_pred[:,-1,:,:]-y_true[:,-1,:,:])) - K.square(K.mean(y_pred[:,-1,:,:]-y_true[:,-1,:,:])) + K.mean(K.square(1 - K.sum(y_pred[:,:36,:,:],axis=1))) + K.mean(K.square(K.batch_dot(y_pred[:,-1,:,:]-layer[:,-1,:,:],y_pred[:,:36,:,:],axes=(1,1))))
    return loss

rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=50)
model.compile(optimizer = 'adam', loss = custom_loss(inputs))
callbacks = callbacks.EarlyStopping(monitor='val_loss',patience=20)
history = model.fit(train_input, train_output, batch_size = 64, epochs = 800, verbose = 2, validation_split = 0.2, shuffle = True,callbacks=[rlrop,callbacks])
model.save('./geostat/cnn/PICNN-model-36-8.h5')
#plt.figure()
#plt.plot(history.history['loss'], 'b')
#plt.plot(history.history['val_loss'], 'k')
#plt.legend(('training','validation'))
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.show()

train_input_directory = './geostat/cov_test_36_z.mat'
train_input_key = 'cov_test'
train_input_data = read_data(train_input_directory, train_input_key)
print(train_input_data.shape)

train_input = np.zeros((124500,37,36,1))
train_input[:,:,:,0] = train_input_data

prediction = model.predict(train_input)
time_end = time.time()
print('time_cost', time_end-time_start, 's')
scio.savemat('./geostat/cnn/pre_lambda_36_z_4_1.mat', {'pre_lambda_36_z': prediction})

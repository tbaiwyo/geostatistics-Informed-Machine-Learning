import scipy.io as scio
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import callbacks
import matplotlib.pyplot as plt

def read_data(directory, key): 
    data_dict = scio.loadmat(directory)
    data = data_dict[key]
    return data

train_input_directory = '/home/taobai/Downloads/geostat/2D/location_train_ANN.mat'
train_input_key = 'location_train'
train_input_data = read_data(train_input_directory, train_input_key)
print(train_input_data.shape)

train_output_directory = '/home/taobai/Downloads/geostat/2D/value_train_ANN.mat'
train_output_key = 'value_train'
train_output_data = read_data(train_output_directory, train_output_key)
print(train_output_data.shape)

model = Sequential()
model.add(Dense(30, activation='relu',input_shape=(2,)))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='relu'))
adam = optimizers.Adam(lr = 0.0001)
model.compile(optimizer=adam,loss='mse')
model.summary()

callbacks = callbacks.EarlyStopping(monitor='val_loss',patience=20)
history = model.fit(train_input_data, train_output_data, batch_size = 64, epochs = 800, verbose = 2, validation_split = 0.2, shuffle = True,callbacks=[callbacks])
#history = model.fit(train_input_data, train_output_data, batch_size = 64, epochs = 800, verbose = 2, validation_split = 0.2, shuffle = True)
model.save('/home/taobai/Downloads/geostat/2D/regular_ANN_model-1.h5')
#plt.figure()
#plt.plot(history.history['loss'], 'b')
#plt.plot(history.history['val_loss'], 'k')
#plt.legend(('training','validation'))
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.show()


train_input_directory = '/home/taobai/Downloads/geostat/2D/location_test_ANN.mat'
train_input_key = 'location_test'
train_input_data = read_data(train_input_directory, train_input_key)
print(train_input_data.shape)

prediction = model.predict(train_input_data)
scio.savemat('/home/taobai/Downloads/geostat/2D/prediction_2D_ANN.mat', {'prediction_2D': prediction})

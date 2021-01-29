# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 20:44:01 2021

@author: akpo2
"""

import os
import pickle
import math
import random
import csv
from PIL import Image

import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
#import tensorflow.contrib.layers as layers
from sklearn.utils import shuffle

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD


print('All modules imported.')

# Reload the processed data
pickle_file = './traffic-signs-data_new/processed_data.p'
with open(pickle_file, 'rb') as f:
    pickle_data = pickle.load(f)
    X_train = pickle_data['train_features']
    y_train = pickle_data['train_labels']
    X_valid = pickle_data['valid_features']
    y_valid = pickle_data['valid_labels']
    X_test = pickle_data['test_features']
    y_test = pickle_data['test_labels']
    signnames = pickle_data['signnames']
    
    
# Shuffle the data set
X_train, y_train = shuffle(X_train, y_train)
X_valid, y_valid = shuffle(X_valid, y_valid)
X_test, y_test = shuffle(X_test,y_test)

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)
print(len(signnames))
print('Data loaded.')

IMAGE_SHAPE = (32, 32, 3)
CLASS_NUM = 43
def inception(x, filters):
        layer1  = Conv2D(filters=filters[0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
        
        layer2 = Conv2D(filters=filters[1][0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
        layer2 = Conv2D(filters=filters[1][1], kernel_size=(3,3), strides=1, padding='same', activation='relu')(layer2)
        
        layer3 = Conv2D(filters=filters[2][0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
        layer3 = Conv2D(filters=filters[2][1], kernel_size=(5,5), strides=1, padding='same', activation='relu')(layer3)
        
        layer4 = MaxPooling2D(pool_size=(3,3), strides=1, padding='same')(x)
        layer4 = Conv2D(filters=filters[3], kernel_size=(1,1), strides=1, padding='same', activation='relu')(layer4)
        
        return Concatenate(axis=-1)([layer1,layer2,layer3,layer4])
        
def define_model():
        IMAGE_SHAPE = (32, 32, 3)
        CLASS_NUM = 43
        layer_in = Input(shape=IMAGE_SHAPE)
        
        #stage-1
        layer = Conv2D(filters=64, kernel_size=(3,3), strides=2, padding='same', activation='relu')(layer_in)
#        layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
#        layer = BatchNormalization()(layer)
        
        # Stage - 2
#        layer = Conv2D(filters=64, kernel_size=(1,1), strides=1, padding='same', activation='relu')(layer)
#        layer = Conv2D(filters=192, kernel_size=(3,3), strides=1, padding='same', activation='relu')(layer)
#        layer = BatchNormalization()(layer)
#        layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
        
        #stage-3
        layer = inception(layer, [64, (96,128), (16,32), 32])
        layer = inception(layer, [128, (128,192), (32,96), 64])
        layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
        
        # stage-4
        layer = inception(layer, [192,  (96,208),  (16,48),  64]) #4a
#        aux1  = auxiliary(layer, name='aux1')
        layer = inception(layer, [160, (112,224),  (24,64),  64]) #4b
        layer = inception(layer, [128, (128,256),  (24,64),  64]) #4c
        layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
#        layer = inception(layer, [112, (144,288),  (32,64),  64]) #4d
#        aux2  = auxiliary(layer, name='aux2')
#        layer = inception(layer, [256, (160,320), (32,128), 128]) #4e
#        layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
        
        # stage-5
        layer = inception(layer, [256, (160,320), (32,128), 128]) #5a
        layer = inception(layer, [384, (192,384), (48,128), 128]) #5b
        layer = AveragePooling2D(pool_size=(3,3), strides=1, padding='valid')(layer)
        
        # stage-6
        layer = Flatten()(layer)
        layer = Dropout(0.4)(layer)
        layer = Dense(units=256, activation='linear')(layer)
        layer = Dense(units=CLASS_NUM, activation='softmax', name='main')(layer)
        
        model = Model(inputs=layer_in, outputs=layer)
        
        return model
    
        
    
        
def auxiliary(x, name=None):
        layer = AveragePooling2D(pool_size=(5,5), strides=3, padding='valid')(x)
        layer = Conv2D(filters=128, kernel_size=(1,1), strides=1, padding='same', activation='relu')(layer)
        layer = Flatten()(layer)
        layer = Dense(units=256, activation='relu')(layer)
        layer = Dropout(0.4)(layer)
        layer = Dense(units=CLASS_NUM, activation='softmax',name=name)(layer)
        return layer

model = define_model()
print("Summary of model")
model.summary()

# Defining the optimiser
optimizer = Adam(lr=2 * 1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#optimizer = SGD(lr=1 * 1e-1, momentum=0.9, nesterov=True)

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

loss = 'sparse_categorical_crossentropy'
BATCH_SIZE = 16
EPOCHS = 50
history_all = {}
MODEL_NAME = './traffic-signs-data_new/googlenet_Traffic.h5'

model.compile(loss=loss, 
                  optimizer=optimizer, metrics=['sparse_categorical_accuracy'])

print("Fit model on training data")
train_history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    shuffle=True,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(X_valid, y_valid),
)

#train_history = model.fit_generator(
#            train_generator,
#            steps_per_epoch=EPOCH_STEPS,
#            epochs=epochs[i],
#            #callbacks=[checkpoint]
#            shuffle=True
#            )
    
# save history    
if len(history_all) == 0:
        history_all = {key: [] for key in train_history.history}
    
for key in history_all:
        history_all[key].extend(train_history.history[key])

model.save(MODEL_NAME)

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(X_test, y_test, batch_size=128)
print("test loss, test acc:", results)
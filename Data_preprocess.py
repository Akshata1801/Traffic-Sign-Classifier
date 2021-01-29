# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 17:19:25 2021

@author: akpo2
"""
#!/usr/bin/env python

import os
import pickle
import csv
from PIL import Image
import math
from scipy import integrate
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np
from sklearn.utils import shuffle

print('All modules imported.')

#Load the data

train_file = "./traffic-signs-data_new/train.p"
test_file = "./traffic-signs-data_new/test.p"
valid_file = "./traffic-signs-data_new/valid.p"
signname_file = "./traffic-signs-data_new/signnames.csv"

with open(train_file, mode = 'rb') as f:
    train = pickle.load(f)
with open(test_file, mode = 'rb') as f:
    test = pickle.load(f)
with open(valid_file, mode = 'rb') as f:
    valid = pickle.load(f)
    
with open(signname_file) as f:
    f.readline() # skip the headers
    signnames = [row[1] for row in csv.reader(f)]
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
X_valid, y_valid = valid['features'], valid['labels']

n_train = len(X_train)
n_valid = len(X_valid)
n_test = len(X_test)

image_shape = (X_train.shape[1], X_train.shape[2])
n_classes = len(signnames)

print("Number of training examples =", n_train)
print("Number of validation examples =", n_valid)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


def normalize_data(image_set):
    return image_set.astype(np.float32)/128.0 - 1.0

def rgb_to_gray(image_set):
    new_set = np.array([])
    for img in image_set:
        np.append(new_set, cv2.cvtColor(img,cv2.COLOR_RGB2GRAY))
    return new_set

X_train = normalize_data(X_train)
X_test = normalize_data(X_test)
X_valid = normalize_data(X_valid)

print(X_train.shape, X_train.dtype)
print(X_valid.shape, X_valid.dtype)
print(X_test.shape, X_test.dtype)

## Save the data

pickle_file = "./traffic-signs-data_new/processed_data.p"
if not os.path.isfile(pickle_file):
    print("Saving the data to pickle file .... ")
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump(
                    {
                            'train_features': X_train,
                            'train_labels' : y_train,
                            'valid_features' : X_valid,
                            'valid_labels' : y_valid,
                            'test_features' : X_test,
                            'test_labels' : y_test,
                            'signnames' : signnames  
                    },
                    pfile, protocol =2 )
    except Exception as e:
        print(' unable to save data to', pickle_file , ':' ,e)
        raise
        
print("Data saved to file")
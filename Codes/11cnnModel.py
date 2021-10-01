#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 09:01:48 2019

@author: selcukkorkmaz
"""
from numpy.random import seed
seed(123)
import tensorflow as tf
tf.random.set_seed(123) 

#import matplotlib
#matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
#from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import matplotlib.pyplot as plt





X_train = pd.read_csv("/Users/selcukkorkmaz/Documents/Studies/DeepCNNandMLP/exercise/numericalData/dataset/AID_485314/afterPreprocess/train/X_train.txt", delimiter="\t")
y_train = pd.read_csv("/Users/selcukkorkmaz/Documents/Studies/DeepCNNandMLP/exercise/numericalData/dataset/AID_485314/afterPreprocess/train/Y_train.txt", delimiter="\t")
X_test = pd.read_csv("/Users/selcukkorkmaz/Documents/Studies/DeepCNNandMLP/exercise/numericalData/dataset/AID_485314/afterPreprocess/test/X_test.txt", delimiter="\t")
y_test = pd.read_csv("/Users/selcukkorkmaz/Documents/Studies/DeepCNNandMLP/exercise/numericalData/dataset/AID_485314/afterPreprocess/test/Y_test.txt", delimiter="\t")

ratio = round((Counter(y_train['PUBCHEM_ACTIVITY_OUTCOME'])[0] / Counter(y_train['PUBCHEM_ACTIVITY_OUTCOME'])[1]))


from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

img_size = [96,96]

# Getting the train images and rescaling
train_image_folder = '/Users/selcukkorkmaz/Documents/Studies/DeepCNNandMLP/exercise/imageData/pubchem/pose/train'
train_image_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        train_image_folder, shuffle=False, class_mode='binary',
        target_size=(img_size[0], img_size[1]), batch_size=X_train.shape[0])
train_images, train_labels = next(train_image_generator)
# Output: Found 20000 images belonging to 2 classes.

# Checking the labels
train_image_generator.class_indices
# Output: {'danger': 0, 'safe': 1}

# Getting the ordered list of filenames for the images
train_image_files = pd.Series(train_image_generator.filenames)
train_image_files = list(train_image_files.str.split("/", expand=True)[1].str[:-11])
train_image_files = np.asarray(train_image_files)
train_image_files = train_image_files.astype(int)

# Sorting the structured data into the same order as the images
X_train_sorted = X_train.reindex(train_image_files)
y_train_sorted = y_train.reindex(train_image_files)


# Getting the test images and rescaling
test_image_folder = '/Users/selcukkorkmaz/Documents/Studies/DeepCNNandMLP/exercise/imageData/pubchem/pose/test'
test_image_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        test_image_folder, shuffle=False, class_mode='binary',
        target_size=(img_size[0], img_size[1]), batch_size=X_test.shape[0])
test_images, test_labels = next(test_image_generator)
# Output: Found 20000 images belonging to 2 classes.

# Checking the labels
test_image_generator.class_indices
# Output: {'danger': 0, 'safe': 1}

# Getting the ordered list of filenames for the images
test_image_files = pd.Series(test_image_generator.filenames)
test_image_files = list(test_image_files.str.split("/", expand=True)[1].str[:-11])
test_image_files = np.asarray(test_image_files)
test_image_files = test_image_files.astype(int)

# Sorting the structured data into the same order as the images
X_test_sorted = X_test.reindex(test_image_files)
y_test_sorted = y_test.reindex(test_image_files)


def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

from keras.models import Sequential
from keras.layers.core import Dense

def create_mlp(dim, dropout=0.50):
    """Creates a simple two-layer MLP with inputs of the given dimension"""
    model = Sequential()
    
    model.add(Dense(2000, input_dim=dim, activation="relu",kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    
    #model.add(Dense(500, activation="relu",kernel_initializer='glorot_uniform'))
    #model.add(BatchNormalization())
    #model.add(Dropout(dropout))
    
    model.add(Dense(500, activation="relu",kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    
    #model.add(Dense(10, activation="relu"))
    #model.add(BatchNormalization())
    #model.add(Dropout(dropout))
    
    return model

from keras.layers import Flatten, Input, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dropout, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.applications import MobileNet
from keras.layers import Dense,GlobalAveragePooling2D

cost_FP = 1
cost_FN = 1
cost_TP = 1
cost_TN = 1


def create_cnn(width, height, depth, filters=(4, 8, 12), dropout=0.50, regularizer=None):
    """
    Creates a CNN with the given input dimension and filter numbers.
    """
    # Initialize the input shape and channel dimension, where the number of channels is the last dimension
    inputShape = (height, width, depth)
    chanDim = -1
 
    # Define the model input
    inputs = Input(shape=inputShape)
    
    #base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

    #x=base_model.output
    #x=GlobalAveragePooling2D()(x)
 
    #x = Conv2D(32, (3, 3), padding="same", input_shape=inputs)
    #x = Activation("relu")(x)
    #x = BatchNormalization(axis=chanDim)(x)
    #x = MaxPooling2D(pool_size=(3, 3))(x)
    #x = Dropout(dropout)(x)
    
    #x = Conv2D(64, (3, 3), padding="same")
    #x = Activation("relu")(x)
    #x = BatchNormalization(axis=chanDim)(x)
    #x = Conv2D(64, (3, 3), padding="same")
    #x = Activation("relu")(x)
    #x = BatchNormalization(axis=chanDim)(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Dropout(dropout)(x)
    
    #x = Conv2D(128, (3, 3), padding="same")
    #x = Activation("relu")(x)
    #x = BatchNormalization(axis=chanDim)(x)
    #x = Conv2D(128, (3, 3), padding="same")
    #x = Activation("relu")(x)
    #x = BatchNormalization(axis=chanDim)(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Dropout(dropout)(x)
    
    #x = Conv2D(256, (3, 3), padding="same")
    #x = Activation("relu")(x)
    #x = BatchNormalization(axis=chanDim)(x)
    #x = Conv2D(256, (3, 3), padding="same")
    #x = Activation("relu")(x)
    #x = BatchNormalization(axis=chanDim)(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Dropout(dropout)(x)
    
    # Loop over the number of filters 
    for (i, f) in enumerate(filters):
        # If this is the first CONV layer then set the input appropriately
        if i == 0:
            x = inputs
 
        # Create loops of CONV => RELU => BN => POOL layers
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
    # Final layers - flatten the volume, then Fully-Connected => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(256, kernel_regularizer=regularizer)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(dropout)(x)
 
    # Apply another fully-connected layer, this one to match the number of nodes coming out of the MLP
    x = Dense(500, kernel_regularizer=regularizer)(x)
    x = Activation("relu")(x)
 
    # Construct the CNN
    #model = Model(base_model.input, x)
    model = Model(inputs, x)

    # Return the CNN
    return model    

from keras.models import Sequential
from keras.optimizers import Adam # Other optimisers are available

# Create the MLP and CNN models
mlp = create_mlp(X_train_sorted.shape[1])
cnn = create_cnn(img_size[1], img_size[0], 3)    

# Create the input to the final set of layers as the output of both the MLP and CNN
combinedInput = concatenate([mlp.output, cnn.output])

# The final fully-connected layer head will have two dense layers (one relu and one sigmoid)
x = Dense(10, activation="relu")(combinedInput)
x = Dense(1, activation="sigmoid")(x)

# The final model accepts numerical data on the MLP input and images on the CNN input, outputting a single value
model1 = Model(inputs=[mlp.input, cnn.input], outputs=x)

# Compile the model 
opt = Adam(lr=0.00001)
model1.compile(loss="binary_crossentropy", metrics=['acc', f1_m], optimizer=opt)

class_weights_dict = {0: 1., 1: 2.}
#def generate_sample_weights(training_data, class_weight_dictionary): 
#    sample_weights = [class_weight_dictionary[np.where(one_hot_row==1)[0][0]] for one_hot_row in training_data]
#    return np.asarray(sample_weights)

#sample_weight = generate_sample_weights(y_train_sorted.values.tolist(), class_weights_dict)



# Train the model
model1_history = model1.fit(
  [X_train_sorted, train_images], 
  y_train_sorted.to_numpy()[:,0], 
  validation_split = 0.10,
  epochs=100, 
  batch_size=2,
  class_weight = class_weights_dict)


# list all data in history
print(model1_history.history.keys())
# summarize history for accuracy
plt.plot(model1_history.history['acc'])
plt.plot(model1_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(model1_history.history['loss'])
plt.plot(model1_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')



y_pred=model1.predict([X_test_sorted, test_images])
y_expected=pd.DataFrame(y_test_sorted.to_numpy()[:,0])

pred = y_pred.round()[:,0]
Y_test = y_expected.values[:,0]


from sklearn.metrics import confusion_matrix

cnf_matrix=confusion_matrix(y_expected,y_pred.round())


tn = cnf_matrix[0,0]
tp = cnf_matrix[1,1]
fn = cnf_matrix[1,0]
fp = cnf_matrix[0,1]

import math
from sklearn.metrics import roc_auc_score

bacc = ((tp/(tp+fn))+(tn/(tn+fp)))/2
pre = tp/(tp+fp)
rec = tp/(tp+fn)
f1 = 2*pre*rec/(pre+rec)
mcc = ((tp*tn) - (fp*fn))/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
auc = roc_auc_score(Y_test, y_pred[:,0])


cnf_matrix




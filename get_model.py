# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:46:51 2019

@author: Zak
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

"""

Returns various models, hardcoded for the XY problem

"""

def enumerate_models():
    return {
            '3xConv2xDropDense',
            '2xConvConvPoolDrop',
            '2xConvPoolDrop',
            '3xConvPoolDrop'
            }

def get_model(model, input_shape):
    n_classes = 2
    
    """
    
    3xConv2xDropDense
    
    Idea: Three-stack of conv layers w/ activations yield highly non-linear transformation
          of input data. If that is needed, it will likely perform well. It is not clear
          that XY state needs this, but to reconstruct vortex map we do use the non-linear
          (but piecewise linear) sawtooth map. 
    
    
    2xConvConvPoolDrop
    
    Idea: Essentially same as 3xConv2xDropDense, but now we regularize much more heavily, 
          with more dropouts and pooling.
    
    
    2xConvPoolDrop
    3xConvPoolDrop
    
    Idea: Now we assume the XY stat can be accurately classified using single conv layers, e.g.
          less non-linearity needed. We also reduce the amount of regularization significantly
          compared to the ConvConvDropPool approach.
    
    """
    
    if model == '3xConv2xDropDense': # Score: .8962
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape,
                         padding = 'same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_classes, activation='softmax'))
        
    elif model == '2xConvConvPoolDrop':
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation = 'relu',
                         input_shape = input_shape, padding = 'same'))
        model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
        model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(512, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_classes, activation = 'softmax'))
        
    elif model == '2xConvPoolDrop':
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation = 'relu',
                         input_shape = input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.15))
        
        model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.15))
        
        model.add(Flatten())
        model.add(Dense(512, activation = 'relu'))
        model.add(Dropout(0.1))
        model.add(Dense(n_classes, activation = 'softmax'))
        
    elif model == '3xConvPoolDrop': 
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation = 'relu',
                         input_shape = input_shape,
                         padding = 'same'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
        model.add(Dropout(0.15))
        
        model.add(Conv2D(128, (3, 3), padding='same', activation = 'relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
        model.add(Dropout(0.15))
        
        model.add(Flatten())
        model.add(Dense(512, activation = 'relu'))
        model.add(Dropout(0.1))
        model.add(Dense(n_classes, activation = 'softmax'))
        
    return model
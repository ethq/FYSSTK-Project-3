# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:46:51 2019

@author: Zak
"""

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201

import keras

"""

Various metafunctions for the nets we use.

get_model() constructs all Keras models and compiles them for use.
the rest are utility functions that can be queried for behaviour of any given model

"""


"""
Returns whether or not the model requires 3 channels depthwise
"""
def requires_rgb(model):
    if 'resnet' in model.lower() or 'densenet' in model.lower():
        return True
    return False

# Returns whether or not the model is a convolutional neural net
def is_cnn(model):
    return model in enumerate_models()[0]

# Returns whether or not the model is a (regular) neural net
def is_nn(model):
    return model in enumerate_models()[1]

# Lists all models that can be constructed
def enumerate_models():
    cnns = [
            '3xConv2xDropDense',
            '2xConvConvPoolDrop',
            '2xConvPoolDrop',
            '3xConvPoolDrop',
            'ResNet50',
            'DenseNet121',
            'DenseNet169',
            'DenseNet201'
            ]
    nns = [
            'nn_simple'
            ]
    
    return  cnns, nns


# Returns a compiled model, must be one of the models enumerated by enumerate_models()
def get_model(model, input_shape):
    n_classes = 2 # we're doing xy problem only, which is a binary classification problem. hence 2 classes
    
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
    
    # Todo add option for various resnets, e.g. resnet152 etc
    if 'resnet' in model.lower():
        # We set include_top to false so that we an add in our own classification layer(s)
        rn = ResNet50(include_top = False, weights = 'imagenet', input_shape = input_shape)
        
        out = rn.layers[-1].output
        out = keras.layers.Flatten()(out)
        
        rn = Model(rn.input, output = out)
        rn.trainable = True
        
        # Below our own layers, we also allow for some modifications of the weights trained on imagenet
        top_layers = ['res5c_branch2b', 'res5c_branch2c', 'activation_97']
        
        # Set them trainable..
        for layer in rn.layers:
            if layer.name in top_layers:
                layer.trainable = True
            else:
                layer.trainable = False
            
        # Add our own classification layers
        model = Sequential()
        model.add(rn)
        model.add(Dense(512, activation = 'relu', input_dim = input_shape))
        model.add(Dropout(.3))
        model.add(Dense(512, activation = 'relu'))
        model.add(Dropout(.3))
        model.add(Dense(n_classes, activation = 'softmax'))
        
    elif 'densenet' in model.lower():
        # Extract depth
        depth = model.lower().lstrip('densenet')
        
        # Must specify _some_ variant of densenet
        assert depth != ''
        
        # Determine model to create
        if depth == '121':
            mod = DenseNet121
        elif depth == '169':
            mod = DenseNet169
        elif depth == '201':
            mod = DenseNet201
        
        # We set include_top to false so that we an add in our own classification layer(s)
        dn = mod(include_top = False, weights = 'imagenet', input_shape = input_shape)
        
        out = dn.layers[-1].output
        out = keras.layers.Flatten()(out)
        
        dn = Model(rn.input, output = out)
        dn.trainable = True
        dn.summary()
        
        # Below our own layers, we also allow for some modifications of the weights trained on imagenet
        top_layers = ['res5c_branch2b', 'res5c_branch2c', 'activation_97']
        
        # Set them trainable..
        for layer in dn.layers:
            if layer.name in top_layers:
                layer.trainable = True
            else:
                layer.trainable = False
            
        # Add our own classification layers
        model = Sequential()
        model.add(dn)
        model.add(Dense(512, activation = 'relu', input_dim = input_shape))
        model.add(Dropout(.3))
        model.add(Dense(512, activation = 'relu'))
        model.add(Dropout(.3))
        model.add(Dense(n_classes, activation = 'softmax'))
    
    
    elif model == '3xConv2xDropDense': # Score: .8962
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
        
        
    elif model == 'nn_simple':
        model = Sequential()
        model.add(Dense(512, activation = 'relu'))
        model.add(Dropout(.25))
        model.add(Dense(256), activation = 'relu')
        model.add(Dropout(.5))
        model.add(Dense(n_classes), activation = 'relu')
    
    # Compile model before returning       
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    return model
# -*- coding: utf-8 -*-
"""
AdaBoosted Neural Net
"""

## KERAS imports
import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier

# Standard imports
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import rc
import pandas as pd
import seaborn as sns

# My stuff
from XY import XY
from Analyze_XY import Analyze_XY
from get_model import get_model, is_cnn, requires_rgb, is_nn, enumerate_models
from fancy_confusion_matrix import galplot, plot_cm

# Utility imports
from itertools import product
import ctypes
import re
from os import walk

# SKLEARN imports
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from imblearn.over_sampling import SMOTE

rc('text', usetex=True)

class Bunch:
    def __init__(self):
        pass


import ctypes

fname = 'L7_M500_N5'
model_s = 'nnsimple'
n_classes = 2
epochs = 10
batch_size = 128
    
axy = Analyze_XY(fname, plotty = False, subsample = 1, boost_nn = False, X_vortex = False)

x_train, x_test, y_train, y_test = train_test_split(axy.X, axy.labels, test_size = .2, shuffle = True)

print('Training balance: %.2f. Testing balance: %.2f' % (np.sum(y_train)/len(y_train), np.sum(y_test)/len(y_test)))

input_shape = None
if is_cnn(model_s):
    grey2rgb = requires_rgb(model_s)
    x_train, input_shape = axy.prepare_X_for_cnn(x_train, grey2rgb)
    x_test, _ = axy.prepare_X_for_cnn(x_test, grey2rgb)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

# Number of classifiers to use
n_clf = 10

# Initialize sample weights
weights = np.ones(x_train.shape[0])/x_train.shape[0]

for _ in np.arange(n_clf):
    model = get_model(model_s, input_shape)
    
    """
    CNN will likely overfit XY states, at least on L = 7 lattice. Hence we need early stopping.
    Patience is set to epochs such that we keep looking for the best model over all epochs.
    """
    es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = epochs, verbose = 1)
    #mc = ModelCheckpoint('Models/Epoch{epoch:02d}_Acc{val_acc:.2f}_V%d_L%d_M%d_N%d_%s_ADA.h5' % (int(axy.X_vortex), axy.L, axy.M, axy.N, model_s) , monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    
    # Fit and record history
    history = model.fit(x_train, y_train,
              batch_size = batch_size,
              epochs = epochs,
              verbose = 1,
              callbacks = [es],
              validation_split = 0.1)
    
    # Get the score on the unseen test set
    score = model.evaluate(x_test, y_test, verbose=1)

print(score)
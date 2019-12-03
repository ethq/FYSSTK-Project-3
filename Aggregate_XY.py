# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:26:39 2019

@author: Zak
"""

## KERAS imports
import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

# Standard imports
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# My stuff
from XY import XY
from get_model import get_model
from Analyze_XY import Analyze_XY

# Utility imports
from itertools import product
import ctypes

# SKLEARN imports
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

### Two simple functions for aggregate analysis of XY model data ###



"""
Helper function. Aggregates tkt predictions on different lattice sizes, 
does a linear fit to (log L)^-2 to find tkt on the infinite lattice
"""    
def plot_tkt_infinite():
    fnames = [
            'xy_data_L7_M500_N5.dat',
            'xy_data_L7_M40_N1.dat',
            'xy_data_L7_M200_N1.dat',
            'xy_data_L7_M201_N1.dat',
            'xy_data_L7_M202_N1.dat',
            'xy_data_L7_M203_N1.dat',
            'xy_data_L7_M204_N1.dat',
            'xy_data_L7_M205_N1.dat',
            'xy_data_L7_M206_N1.dat',
            'xy_data_L7_M208_N1.dat',
            'xy_data_L8_M500_N5.dat',
            'xy_data_L10_M500_N5.dat',
            'xy_data_L12_M204_N1.dat',
            'xy_data_L12_M205_N1.dat',
            'xy_data_L12_M206_N1.dat',
            'xy_data_L12_M207_N1.dat',
            'xy_data_L12_M208_N1.dat',
            'xy_data_L16_M500_N5.dat',
            'xy_data_L24_M10_N5.dat'
            ]
    
    L = [7,7,7,7,7,7,7,7,7,7, 8, 10, 12,12,12,12,12, 16, 24]
         
         
    # Disregard first and last datapoints
#    L = [7, 16] <- with flip gives correct, why?
    Linv = 1/np.log(L)**2
    
    tkt = []
    
    for l, f in zip(L, fnames):
        axy = Analyze_XY(f, l)
        tkt.append(axy.tkt)
#        axy.plot()
        
    print(Linv, tkt)
#    tkt = np.flip(tkt)
        
    model = make_pipeline(PolynomialFeatures(1), LinearRegression())
    model.fit(Linv[:, np.newaxis], tkt)
    
    Linv_new = np.linspace(0, .3, 100)
    Tkt_pred = model.predict(Linv_new[:, np.newaxis])
    
    print('Predicted T_KT = %.2f' % model.steps[1][1].intercept_)
    
    
    plt.plot(Linv_new, Tkt_pred)
    plt.plot(Linv, tkt, 'o')
    plt.show()
    
    return tkt

"""
Plot all energies at given temperature. If we are in equilibrium, should visibly fluctuate around the mean.
"""
def plot_energies():
    fname = 'xy_data_L16_M10000_N1.dat'
    
    with open(fname, 'rb') as f:
        data = pickle.load(f)
        
        xy = XY(L = 16)
        
        print(data['temperature'])
        for tset in data['states']:
            e = []
            for s in tset:
                e.append(xy.get_energy(s))
            
            print(np.mean(e))
#            plt.plot(e)
#            plt.show()
    
    return data
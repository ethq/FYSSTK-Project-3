# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 11:42:40 2019

@author: Z
"""

import numpy as np
import pickle
from XY import XY
#from NeuralNet import NeuralNet
#from GridSearchCV import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd

from itertools import product

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import ctypes
from mpl_toolkits.mplot3d import axes3d, Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle

from decimal import Decimal

fname = 'gridsearch_results.dat'

with open(fname, "rb") as f:
    gs = pickle.load(f)

data = {'params': gs.cv_results_['params'], 'scores': gs.cv_results_['mean_test_score']}
#indices = np.arange(len(data['scores']))
#df = pd.DataFrame(data, indices)
#df.sort_values(by = 'hidden_layer_sizes')

x, y, z1, z2 = [], [], [], []

for i, p in enumerate(data['params']):
    xy = p['hidden_layer_sizes']
    x.append(xy[0])
    y.append(xy[1])
    z1.append(p['learning_rate_init'])
    z2.append(data['scores'][i])
    
n_learning_rates = len(np.unique(z1))
n_layer_size = len(np.unique(x))
nbars = int(len(x)/n_learning_rates)

fig, axes = plt.subplots(nrows = 2, ncols = 3)
fig.suptitle('Gridsearch over neuron numbers and learning rate')

x_size = 10*np.ones(nbars)
y_size = 10*np.ones(nbars)
z_size = np.ones(nbars)

for i, ax in enumerate(axes.ravel()):
    xp = np.array(x[i::n_learning_rates]).reshape(n_layer_size, n_layer_size)
    yp = np.array(y[i::n_learning_rates]).reshape(n_layer_size, n_layer_size)
    z2p = np.array(z2[i::n_learning_rates]).reshape(n_layer_size, n_layer_size)
    
    im = ax.contourf(xp, yp, z2p, vmin = np.min(z2p), vmax = np.max(z2p))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
#    cbar.set_ticks([0,1])
#    cbar.ax.set_yticklabels([0,1])
    
    lr = Decimal(np.unique(z1)[i])
    ax.set_title('Learning rate: %.0E' % lr)
    ax.set_xlabel('Neurons in first layer')
    ax.set_ylabel('Neurons in second layer')

#fig.colorbar(im, ax = axes.ravel().tolist() )
    




#ax = plt.axes(projection="3d")
#ax.bar3d(x, y, z2, x_size, y_size, z2, color=(.7, .2, .2, .6))
#plt.contourf(x,y,z2)
plt.tight_layout()
plt.show()















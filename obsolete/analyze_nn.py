# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:08:08 2019

@author: Z
"""

# Analyze dataset - find T_KT - using a regular neural network
# Initially use sklearn due to speed advantage

"""

!!!

NOTE: THIS IS 1) A ROYAL MESS AND 2) OLD PROTOTYPE CODE AND IS USED PRIMARILY FOR ADABOOSTING NEURAL NETS
      OTHER ANALYSIS IS TO BE DONE USING Analyze_XY.py
      
      
NOTE2: AdaBoosting neural nets seems to be a terrible idea. Or it must at least be done by manually 
       implementing the boosting algorithm. No time to do this for the moment.
      
!!!

"""

import numpy as np
import pickle
from Analyze_XY import Analyze_XY
from XY import XY
#from NeuralNet import NeuralNet
#from GridSearchCV import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from itertools import product

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeClassifier

import ctypes

fname = 'Data/xy_data_L7_M500_N5.dat'
indata = None

with open(fname, "rb") as f:
    indata = pickle.load(f)
    
energy = np.array(indata['energy'])
t = np.array(indata['temperature'])

# Murder outliers
mask1 = t > .5
mask2 = t < 1.4

mask = [m1 and m2 for m1,m2 in zip(mask1, mask2)]
t = t[mask]
energy = energy[mask]
    
## Make a fit to E(T) just for smoother Cv
model = make_pipeline(PolynomialFeatures(10), Ridge())
model.fit(t[:, np.newaxis], energy)

tnew = np.linspace(t[0], t[-1], 200)[:, np.newaxis]

energy_pred = model.predict(tnew)
    

# Energy and heat capacity plots
#f = plt.figure()
#ax = f.add_subplot(121)
#ax.plot(tnew, energy_pred)
#ax.plot(t, energy, 'o', markeredgecolor = 'black', markerfacecolor = (0.9, 0.2, 0.2, 0.7 ) )
#
#ax = f.add_subplot(122)
#cv_s = np.diff(energy_pred)/np.diff(tnew.flatten())
#ax.plot(tnew[1:], cv_s)
#
#plt.show()

"""
Honor to: https://stackoverflow.com/questions/55632010/using-scikit-learns-mlpclassifier-in-adaboostclassifier
"""
class customMLPClassifier(MLPClassifier):
    def resample_with_replacement(self, X_train, y_train, sample_weight):
        
        if sample_weight is None:
            sample_weight = np.ones(X_train.shape[0])/X_train.shape[0]
        
        # normalize sample_weights if not already
        sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

        X_train_resampled = np.zeros((len(X_train), len(X_train[0])), dtype=np.float32)
        y_train_resampled = np.zeros((len(y_train)), dtype=np.int)
        for i in range(len(X_train)):
            # draw a number from 0 to len(X_train)-1
            draw = np.random.choice(np.arange(len(X_train)), p=sample_weight)

            # place the X and y at the drawn number into the resampled X and y
            X_train_resampled[i] = X_train[draw]
            y_train_resampled[i] = y_train[draw]

        return X_train_resampled, y_train_resampled


    def fit(self, X, y, sample_weight = None):
#        X, y = self.resample_with_replacement(X, y, sample_weight)

        return self._fit(X, y, incremental=(self.warm_start and
                                            hasattr(self, "classes_")))



# Train a neural net

# Create design matrix given energies, states and temperatures
#X, y = create_reduced_X(indata)
#Xf, yf = create_full_X(indata)
#
#X = Xf
#y = yf

axy = Analyze_XY('L7_M500_N5', plotty = False, subsample = 1, boost_nn = False, X_vortex = True)

X_train, X_test, Y_train, Y_test = train_test_split(axy.X, axy.labels, test_size = .2, shuffle = True)

# for final score, crank up max iter and lower learning rate
# signs that alpha  lower than 1e-2 is detrimental
clf = customMLPClassifier(max_iter = 2000,
                    hidden_layer_sizes = (100, 50),
                    activation = 'relu',
                    solver = 'adam',
                    alpha = 1e-2,
                    learning_rate = 'constant',
                    learning_rate_init = 1e-4
                    )

#clf = RandomForestClassifier(max_depth = 5, n_estimators = 10)
#clf = DecisionTreeClassifier(max_depth = 5)


#clf = AdaBoostClassifier(base_estimator = clf, n_estimators = 500, learning_rate = 1)
#print('Beginning AdaBoost fit')
#clf.fit(X_train, Y_train)
#print('Fit done')
#y_true, y_pred = Y_test, clf.predict(X_test)
    
clf.fit(X_train, Y_train)
y_true, y_pred = Y_test, clf.predict(X_test)

axy.tkt_from_pred(clf, axy.X, 'nnmlp')


print('Results on the test set:')
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print('Accuracy: %.3f' % accuracy_score(y_true, y_pred))


ctypes.windll.user32.FlashWindow(ctypes.windll.kernel32.GetConsoleWindow(), True )
    

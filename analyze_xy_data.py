# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:08:08 2019

@author: Z
"""

# Analyze dataset - find T_KT - using a regular neural network
# Initially use sklearn due to speed advantage

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

fname = 'xy_data_L7_M100_N5.dat'
t_kt = {fname: 1.11234963}
indata = None

with open(fname, "rb") as f:
    indata = pickle.load(f)
    
energy = np.array(indata['energy'])
t = np.array(indata['temperature'])

# Murder outliers
#mask = t > .5
#t = t[mask]
#energy = energy[mask]
    
# Make a fit to E(T) just for smoother Cv
#model = make_pipeline(PolynomialFeatures(10), Ridge())
#model.fit(t[:, np.newaxis], energy)
#
#tnew = np.linspace(t[0], t[-1], 200)[:, np.newaxis]
#
#energy_pred = model.predict(tnew)
#    
#
## Energy and heat capacity plots
#f = plt.figure()
#ax = f.add_subplot(121)
#ax.plot(tnew, energy_pred)
##ax.plot(t[::5], energy[::5], 'o')
#
#ax = f.add_subplot(122)
#cv_s = np.diff(energy_pred)/np.diff(tnew.flatten())
#ax.plot(tnew[1:], cv_s)
#
#plt.show()


# Make sure states are distinct
def test_states(indata):
    s = indata['states']
    s0 = s[0][0]
    tot = 0
    for v1 in s:
        for v2 in v1:
            tot += np.sum(v2-s0)
    # If tot == 0, then all states are equal and we have a serious problem - namely, PYTHON LISTS ALWAYS BEING PASSED BY REFERENCE )"#(¤/")#(¤/)#¤))
    assert tot != 0

"""
Creates design matrix. 

Input:
    data: [Dictionary]
        Assumed to contains keys 'energy', 'states' and 'temperature'
        The corresponding values should be lists or arrays of equal length
        data['states'] should contain lists, each containing all states that were used to calculate the average energies in data['energy']
        
Output:
    Design matrix. Number of features equals the size of a state, number of instances equal to the number of energies/temperatures
"""
def create_reduced_X(data):
    # Compute size of state - it's 2d and square
    L = len(data['states'][0][0][0])
    xy = XY(T = 1, L = L)
    
    states = []
    ts = data['temperature']
    labels = (ts > t_kt[fname]).astype(int)
    
    # Loop over all energies, select a representative state
    for ei, e in enumerate(data['energy']):
        # Smallest energy difference
        smallest_diff = np.Inf
        
        # State with smallest energy difference
        best_state_id = -1
        
        # Find state with energy closest to the mean
        for si, s in enumerate(data['states'][ei]):
            diff = abs(e - xy.get_energy(s))
            
            if diff < smallest_diff:
                best_state_id = si
                smallest_diff = diff
        bs = np.array(data['states'][ei][best_state_id])
        vs = xy.get_vortex_map(bs)
        states.append( vs.flatten() )
        
    X = np.zeros((len(states), L**2))
    
    for i, s in enumerate(states):
        X[i, :] = s.flatten()
    
    return X, labels
    
    
"""
At each temperature T we have sampled N states. Corresp. the design matrix consists of N*T instances, where the features are flattened spin configurations(for a 7x7 grid, 49 features)

Unfortunately each simulation has its own T_KT - likely more accurate the more samples we gather.

xy_data_1.dat: on a 7x7 lattice, T_KT ~ 1.11234963 by interpolation

"""
def create_full_X(data):
    # Compute size of state - it's 2d and square
    L = len(data['states'][0][0][0])
    
    states = []
    ts = data['temperature']
    t_mask = (ts > t_kt[fname]).astype(int)
    labels = []
    
    for ei, e in enumerate(data['states']):
        for s in data['states'][ei]:
            states.append(s)
            labels.append(t_mask[ei])
    
    X = np.zeros((len(states), L**2))
    
    for i, s in enumerate(states):
        X[i, :] = s.flatten()
        
    return X, np.array(labels)
        
    
# Sanity check - are labels roughly correct?
def test_X(X, labels):
    for i, row in enumerate(X):
        l = np.sqrt(len(row))
        s = np.reshape(row, (l,l))
        
        es = XY.get_energy(None, s)
        el = labels[i]
        
        print('State energy: %.2f. Label energy: %.2f.' % (es, el))



# Train a neural net

# Create design matrix given energies, states and temperatures
X, y = create_reduced_X(indata)
Xf, yf = create_full_X(indata)

ss = StandardScaler()
#X = ss.fit_transform(X)
#Xf = ss.fit_transform(Xf)

X = Xf
y = yf


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = .2, shuffle = True)


mlp = MLPClassifier(max_iter = 3000,
                    hidden_layer_sizes = (50,),
                    activation = 'relu',
                    solver = 'adam',
                    alpha = 1e-4,
                    learning_rate = 'constant'
                    )

# Search for coarse parameters + regularization
#parameter_space = {
#    'hidden_layer_sizes': [(50,50,50), (50,50), (50,)],
#    'solver': ['adam'],
#    'alpha': [0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
#    'learning_rate': ['constant','adaptive'],
#}

# Search learning rate as we found constant to be best
#parameter_space = {
#        'learning_rate_init': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
#        'hidden_layer_sizes': [(50,50,50),
#                               (50,50),
#                               (50,),
#                               (100,),
#                               (150,),
#                               (100,100),
#                               (20,),
#                               (20,20),
#                               (30,30),
#                               (40,40),
#                               (10,),
#                               (30,),
#                               (70,),
#                               (85,),
#                               (20,20)
#                               
#                               
#                               
#                               ]


#        }

#mul = [(10*i, 10*j) for i,j in product(np.linspace(1, 20, 20).astype(int), np.flip(np.linspace(1, 20,20).astype(int)))]
#print(mul)


## Note: takes ~20 hrs to run
#parameter_space = {
#        'learning_rate_init': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
#        'hidden_layer_sizes': mul
#        }


clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=5)
clf.fit(X_train, Y_train)

# Best parameter set
print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    
y_true, y_pred = Y_test, clf.predict(X_test)


print('Results on the test set:')
print(classification_report(y_true, y_pred))


ctypes.windll.user32.FlashWindow(ctypes.windll.kernel32.GetConsoleWindow(), True )
    

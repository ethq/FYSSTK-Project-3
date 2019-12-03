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
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import ctypes

fname = 'xy_data_L7_M500_N5.dat'
t_kt = {fname: 1.11234963}
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
f = plt.figure()
ax = f.add_subplot(121)
ax.plot(tnew, energy_pred)
ax.plot(t, energy, 'o', markeredgecolor = 'black', markerfacecolor = (0.9, 0.2, 0.2, 0.7 ) )

ax = f.add_subplot(122)
cv_s = np.diff(energy_pred)/np.diff(tnew.flatten())
ax.plot(tnew[1:], cv_s)

plt.show()



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

# for final score, crank up max iter and lower learning rate
# signs that alpha  lower than 1e-2 is detrimental
mlp = MLPClassifier(max_iter = 5000,
                    hidden_layer_sizes = (170,70),
                    activation = 'relu',
                    solver = 'adam',
                    alpha = 1e-2,
                    learning_rate = 'constant',
                    learning_rate_init = 1e-4
                    )

mlp.fit(X_train, Y_train)
    
y_true, y_pred = Y_test, mlp.predict(X_test)


print('Results on the test set:')
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))


ctypes.windll.user32.FlashWindow(ctypes.windll.kernel32.GetConsoleWindow(), True )
    

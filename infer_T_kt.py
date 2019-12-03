# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 10:29:48 2019

@author: Z
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

fnames = [
        'xy_data_L7_M500_N5.dat',
        'xy_data_L10_M500_N5.dat',
        'xy_data_L16_M500_N5.dat'
        ]

Linv = 1/np.log(np.array([10, 16]))**2
Tkt = []

# Use the heat capacity to predict transition temperature
for fname in fnames:
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
    cv = np.diff(energy_pred)/np.diff(tnew.flatten())
    
    arg_tkt = np.argmax(cv)
    
    Tkt.append(tnew[arg_tkt])

Tkt = np.flip(Tkt)
Tkt = [Tkt[1], Tkt[2]]
# Do a linear plot of tkt vs L^-2. Tkt on the infinite plane should then be the intercept
model = make_pipeline(PolynomialFeatures(1), LinearRegression())
model.fit(Linv[:, np.newaxis], Tkt)

Linv_new = np.linspace(0, .3, 100)
Tkt_pred = model.predict(Linv_new[:, np.newaxis])

print('Predicted T_KT = %.2f' % model.steps[1][1].intercept_)


plt.plot(Linv_new, Tkt_pred)
plt.plot(Linv, Tkt, 'o')
plt.show()
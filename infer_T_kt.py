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
        'Data/xy_data_L7_M500_N5.dat',
        'Data/xy_data_L10_M500_N5.dat',
        'Data/xy_data_L16_M500_N5.dat'
        ]

Linv = 1/np.log(np.array([7, 10, 16]))**2
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

# Do a linear plot of tkt vs L^-2. Tkt on the infinite plane should then be the intercept
model = make_pipeline(PolynomialFeatures(1), LinearRegression())
model.fit(Linv[:, np.newaxis], Tkt)

Linv_new = np.linspace(0, .3, 100)
Tkt_pred = model.predict(Linv_new[:, np.newaxis])

print('Predicted T_KT = %.2f' % model.steps[1][1].intercept_)


plt.plot(Linv_new, Tkt_pred)
plt.plot(Linv, Tkt, 'o')

Tkt_5k = np.array([1.219, 1.198])
L_5k = np.array([8, 16])
Linv_5k = 1/np.log(L_5k)**2

model = make_pipeline(PolynomialFeatures(1), LinearRegression())
model.fit(Linv_5k[:, np.newaxis], Tkt_5k)

Tkt_pred2 = model.predict(Linv_new[:, np.newaxis])

plt.plot(Linv_new, Tkt_pred2)
plt.plot(Linv_5k, Tkt_5k, 'o')

print('Predicted T_KT = %.2f' % model.steps[1][1].intercept_)

# Easier to just do this manually...
# Numbers taken from Plots/Predictions. Could also be obtained automatically by using Analyze_XY, but time and all that
Linv = 1/np.log([4, 8, 12, 32])**2
Tkt = [1.239, 1.124, 1.214, 1.160]
#Linv_new = np.linspace(0.3, np.max(Linv), 100)

model = make_pipeline(PolynomialFeatures(1), LinearRegression())
model.fit(Linv[:, np.newaxis], Tkt)

Tkt_pred2 = model.predict(Linv_new[:, np.newaxis])

plt.plot(Linv_new, Tkt_pred2)
plt.plot(Linv, Tkt, 'o')

print('Predicted T_KT = %.2f' % model.steps[1][1].intercept_)

Linv = 1/np.log([12, 32])**2
Tkt = [1.214, 1.160]
#Linv_new = np.linspace(0.3, np.max(Linv), 100)

model = make_pipeline(PolynomialFeatures(1), LinearRegression())
model.fit(Linv[:, np.newaxis], Tkt)

Tkt_pred2 = model.predict(Linv_new[:, np.newaxis])

plt.plot(Linv_new, Tkt_pred2)

print('Predicted T_KT = %.2f' % model.steps[1][1].intercept_)

plt.legend([
        'Fit to $T_{KT}$ for M = 500',
        '$T_{KT}$ for M = 500',
        'Fit to $T_{KT}$ for M = 5000',
        '$T_{KT}$ for M = 5000',
        'Fit to $T_{KT}$ for M = 1000',
        '$T_{KT}$ for M = 1000',
        'Partial fit to $T_{KT}$ for M = 1000',
        ])
    
plt.title('$T_{KT}$ scaling with lattice dimension')
plt.xlabel('$(log\ L)^{-2}$')
plt.ylabel('Temperature')



plt.show()
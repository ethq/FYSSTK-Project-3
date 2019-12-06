# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 13:18:55 2019

@author: Z
"""

import numpy as np
from imblearn.over_sampling import SMOTE
from XY import XY


"""

Quick script to save a smoted version of a design matrix.
Takes an extreme amount of time, at least on L > 16

"""

# Design matrix filename
x_name = 'Data/Design matrix/X_V0_L16_M5000_N1.npy'

# Regular labels filename
l_name = 'Data/Design matrix/L_V0_L16_M5000_N1.npy' 

# Energy labels filename
el_name = 'Data/Design matrix/EL_V0_L16_M5000_N1.npy'

# Attempt to load from file
try:
    X = np.load(x_name)
    labels = np.load(l_name)
    e_labels = np.load(el_name)
    
    print('Loaded design matrix from file.')
except:
    print('Failure')


#s = SMOTE()
#X, y = s.fit_resample(X, y)
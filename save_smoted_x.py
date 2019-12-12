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
x_name = 'Data/Design matrix/X_V0_L32_M1000_N1.npy'

# Regular labels filename
l_name = 'Data/Design matrix/L_V0_L32_M1000_N1.npy' 

# Energy labels filename
el_name = 'Data/Design matrix/EL_V0_L32_M1000_N1.npy'

L = 32

# Attempt to load from file
try:
    X = np.load(x_name)
    labels = np.load(l_name)
    e_labels = np.load(el_name)
    
    print('Loaded design matrix from file.')
except:
    print('Failure')
    
def restore_energy_labels(self):
    xy = XY(T = 1, L = self.L)
    return np.array([xy.get_energy(np.reshape(s, (L, L))) for s in X])

# Generate new instances to fix any class imbalance(relevant for (16,) set)
sm = SMOTE()
X, labels = sm.fit_resample(X, labels)

# Recalculate energy for SMOTEd instances
e_labels = restore_energy_labels()

x_name = "Data/Design matrix/X_SM_V0_L32_M1000_N1.npy"
l_name = "Data/Design matrix/L_SM_V0_L32_M1000_N1.npy"
el_name = "Data/Design matrix/EL_SM_V0_L32_M1000_N1.npy"

# Save to file
np.save(x_name, X)
np.save(l_name, np.array(labels))
np.save(el_name, np.array(e_labels))
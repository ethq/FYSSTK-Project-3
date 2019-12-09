# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:30:47 2019

@author: Z
"""

from XY import XY
import numpy as np
import pickle

import ctypes

#L = 32
#xy = XY(T = 1, L = L)
#
#t = np.linspace(0.5, 1.5, 50)
#ecb = lambda c = None: xy.get_energy()
#
#energy = []
#states = []
#
#M = 500
#N = 1
#for t_ in t:
#
#    result = xy.measure_observables(t_, N, M, {'energy': ecb})
#    energy.append(result['energy'])
#    states.append(result['states'])
#    
## States is now a len(t) length array of len M arrays, containing states at the given temperature
## To get an ensemble of states at the stated temperature, should these be averaged? Should I pick one with energy closest corresp the E-T average curve?
#
#fname = 'Data/xy_data_L%d_M%d_N%d.dat' % (L, M, N)
#
#data = {
#        'sweeps_between_measurement': N,
#        'number_of_measurements': M,
#        'lattice_dim': L,
#        'temperature': t,
#        'energy': energy,
#        'states': states
#        }
#
#with open(fname, "wb") as f:
#    pickle.dump(data, f)
#
#ctypes.windll.user32.FlashWindow(ctypes.windll.kernel32.GetConsoleWindow(), True )

## Run 2 if Run 1 finishes in time...

L = 8
xy = XY(T = 1, L = L)

t = np.linspace(0.5, 1.5, 50)
ecb = lambda c = None: xy.get_energy()

energy = []
states = []

M = 5000
N = 1
for t_ in t:

    result = xy.measure_observables(t_, N, M, {'energy': ecb})
    energy.append(result['energy'])
    states.append(result['states'])
    
# States is now a len(t) length array of len M arrays, containing states at the given temperature
# To get an ensemble of states at the stated temperature, should these be averaged? Should I pick one with energy closest corresp the E-T average curve?

fname = 'Data/xy_data_L%d_M%d_N%d.dat' % (L, M, N)

data = {
        'sweeps_between_measurement': N,
        'number_of_measurements': M,
        'lattice_dim': L,
        'temperature': t,
        'energy': energy,
        'states': states
        }

with open(fname, "wb") as f:
    pickle.dump(data, f)

ctypes.windll.user32.FlashWindow(ctypes.windll.kernel32.GetConsoleWindow(), True )
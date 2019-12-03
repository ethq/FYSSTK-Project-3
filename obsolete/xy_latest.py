# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:45:58 2019

@author: Zak
"""

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg
from matplotlib.legend_handler import HandlerLine2D
import math
from scipy.optimize import curve_fit
from numpy import pi
from itertools import product
from tqdm import tqdm

class XY:
    def __init__(self, 
                 T = 1,
                 L = 8,
                 tol = 1e-3,
                 max_sweeps = 1e4
                 ):
        self.L = L
        self.T = T
        self.tol = tol
        self.max_sweeps = max_sweeps
        
        # Uniform initialization
        self.spin_config = np.random.rand(L, L)*2*np.pi
        
        self.M = []
        self.Cv = []
        
        self.construct_neighbour_map()
        
        imap = []
        for r, c in product(np.arange(L), np.arange(L)):
            imap.append((r,c))
        self.index_map = imap

        self.energy = self.get_energy()
    
    
    def construct_neighbour_map(self, bdry = 'periodic'):
        n_map = {}
        L = self.L
        
        if bdry == 'periodic':
            for r,c in product(np.arange(L), np.arange(L)):
                # Find neighbour indices
                left = (r - 1) % L
                right = (r + 1) % L
                top = (c + 1) % L
                bot = (c - 1) % L
                
                n_map[r, c] = ((left, c), (right, c), (r, top), (r, bot))
                
        elif bdry == 'fixed':
            for r, c in product(np.arange(L), np.arange(L)):
                left = max(left, 0)
                right = min(right, self.L-1)
                bot = max(bot, 0)
                top = min(top, self.L-1)
        
                n_map[r, c] = ((left, c), (right, c), (r, top), (r, bot))

        self.neighbour_map = n_map
    
    
    """
    Performs one full sweep of the metropolis algorithm, e.g. attempts to "flip" each spin once.
    """
    def sweep(self):
        beta = 1.0 / self.T
        random.shuffle(self.index_map)
        
        # Do note that i and j are tuples
        for i in self.index_map:
            energy_i = -np.sum(np.cos([self.spin_config[i] - self.spin_config[j] for j in self.neighbour_map[i]])) 
            
            dtheta = np.random.uniform(-np.pi, np.pi)
            spin_temp = self.spin_config[i] + dtheta
            energy_f = -np.sum([np.cos(spin_temp - self.spin_config[j]) for j in self.neighbour_map[i]]) 
            delta_E = energy_f - energy_i
            
            if np.random.uniform(0.0, 1.0) < np.exp(-beta * delta_E):
                self.spin_config[i] += dtheta
                self.energy = self.get_energy()
                
    """
    Calculates the energy per spin for the current configuration
    Returns the value and sets the corresponding internal variable
    """
    def get_energy(self, cfg = None):
        H = 0
        
        if cfg == None:
            cfg = self.spin_config
        
        # Might have to optimize this
        for i in self.index_map:
            H += -np.sum([np.cos(cfg - cfg[j]) for j in self.neighbour_map[i]])
        
        return H/self.L**2
        
    ## Let the system evolve to equilibrium state
    def equilibrate(self, T = None, H = None):
        if T != None:
            self.T = T
        
        
        energies = []
        beta = 1.0 / self.T
        energy_temp = 0
        
        min_eq = 500
        for k in np.arange(self.max_sweeps):
            self.sweep()     
            
            energy = self.get_energy()
            energies.append( energy )
            
            if (abs(energy-energy_temp)/abs(energy) < self.tol) and k > min_eq:
                print('Eq reached at T = %.1f, E = %.2f, sweep #%d' % (self.T,energy, k))
                break
            energy_temp = energy
        
        nstates = len(energies)
        energy = np.mean(energies[int(nstates/2):])
        self.energy = energy
        energy2 = np.mean(np.array(energies[int(nstates/2):])**2)
        self.Cv=(energy2-energy**2)*beta**2

    """
    
    Calculates an observable at equilibrium. Thus the function first relaxes the initial spin configuration to equilibrium.
    After (approximate) equilibrium is reached, the observable functions are called every N'th sweep M times. A sufficiently high value of N
    must be chosen to get statistically independent measurements, and larger values of M yield better averages. At the end averages are returned.
    
    Input:
        T: temperature
        n_sweeps: N sweeps at eq per measurement
        m_callback: how many measurements to take
        observables: dictionary of form {label: callback}. Callback must accept a spin-configuration on the lattice specified in __init__()
        
    Output:
        observables: dictionary of form {label: value}
    
    """
    
    def measure_observables(self, T, n_sweeps, m_callback, observables, store_states = True):
        self.equilibrate(T)
        
        results = {}
        states = []
        
        for k in observables.keys():
            results[k] = 0
        
        # Loop over all measurements
        for m in tqdm(np.arange(m_callback)):
            
            # Evolve the system N sweeps to ensure statistically "independent" measurements
            for n in np.arange(n_sweeps):
                self.sweep()
            
            # Store state
            if store_states:
                states.append(self.spin_config)
            
            # Measure observables
            for k in results.keys():
                results[k] += observables[k](self.spin_config)
        
        for k, v in results.items():
            results[k] = v/m_callback
            
        if store_states:
            results['states'] = states
            
        return results
    
    
    def annealing(self, T_init=2.5, T_final=0.1, nsteps = 20):
        dic_thermal = {}
        dic_thermal['temperature'] = list(np.linspace(T_init, T_final, nsteps))
        dic_thermal['energy'] = []
        dic_thermal['Cv'] = []
        for T in dic_thermal['temperature']:
            self.equilibrate(T)
            dic_thermal['energy'] += [self.energy]
            dic_thermal['Cv'] += [self.Cv]
            
        plt.plot(dic_thermal['temperature'], dic_thermal['Cv'], '.')
        plt.ylabel(r'$C_v$')
        plt.xlabel('T')
        plt.show()
        plt.plot(dic_thermal['temperature'], dic_thermal['energy'], '.')
        plt.ylabel(r'$\langle E \rangle$')
        plt.xlabel('T')
        plt.show()
        return dic_thermal


if __name__ == '__main__':
    xy = XY()
#    xy.annealing()
#    # Measure energy as function of temperature
    ecb = lambda c = None: xy.get_energy()
    
    temp = np.linspace(0.1, 2, 20)
    energy = []
    states = []
    
    for t in temp:
        energy.append( xy.measure_observables(t, 5, 1000, {'energy': ecb})['energy'] )
    
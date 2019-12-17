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

"""
Class for generating states of the XY model and measuring observables on them

Input:
    T: Temperature,
    L: Lattice dimensions,
    tol: Equilibration tolerance
    max_sweeps: Max allowed iterations for equilibration
"""

class XY:
    def __init__(self, 
                 T = 1,
                 L = 7,
                 tol = 1e-3,
                 max_sweeps = 1e4
                 ):
        self.L = L
        self.T = T
        self.tol = tol
        self.max_sweeps = max_sweeps
        
        # Uniform initialization
        self.spin_config = np.random.rand(L, L)*2*np.pi - np.pi
        
        self.M = []
        self.Cv = []
        
        self.neighbour_map = self.get_neighbour_map()
        
        imap = []
        for r, c in product(np.arange(L), np.arange(L)):
            imap.append((r,c))
        self.index_map = imap

        self.energy = self.get_energy()
        
        self.saw = np.vectorize(self._saw)
    
    
    """
    Calculates a vorticity configuration given a raw spin configuration
    
    Input:
        cfg: [Array] LxL array of floats
        
        bdry: [String] Currently only 'periodic' supported
        
        convention: [String] Supports 'plaq' and 'loop'. 
                        'plaq' computes vorticity by summing angle differences on a 2x2 plaquette
                        'loop' computes vorticity by summing angle differences on the 8 points surrounding a given point
    """
    def get_vortex_map(self, cfg, bdry = 'periodic', convention = 'plaq'):
        # Get dimensions
        L = np.array(cfg).shape[0]
        
        vmap = np.zeros((L, L))
        
        for i in self.index_map:
            if convention == 'plaq':
                # Get neighbour indices
                i2 = ( (i[0]-1) % L, i[1] )
                i3 = ( i2[0], (i[1]+1) % L )
                i4 = ( i[0], i3[1] )
                
                # Calculate their difference
                a_diff = np.zeros(4)
                a_diff[0] = cfg[i2] - cfg[i]
                a_diff[1] = cfg[i3] - cfg[i2]
                a_diff[2] = cfg[i4] - cfg[i3]
                a_diff[3] = cfg[i] - cfg[i4]
                
                # Saw it
                a_diff = self.saw(a_diff)
                
                # And sum it
                vmap[i] = np.sum(a_diff)
            elif convention == 'loop':
                # Get indices left, right, up and down
                l, r, u, d = (i[0]-1)%L, (i[0]+1)%L, (i[1]+1)%L, (i[1]-1)%L
                # And indices of current location
                cr,cc = i
                
                # Get spins at neighbour locations
                a_diff = np.array([
                        cfg[l, u],
                        cfg[cr, u],
                        cfg[r, u],
                        cfg[r, cc],
                        cfg[r, d],
                        cfg[cr, d],
                        cfg[l, d],
                        cfg[l, cc]
                        ])
                
                # Calculate their difference
                a_diff = np.diff(a_diff)
                
                # Manually close the loop
                np.append(a_diff, cfg[l, u] - cfg[l, cc])
                
                # Saw it
                a_diff = self.saw(a_diff)
                
                # And sum it
                vmap[i] = np.sum(a_diff)
                        
        return vmap
    
    
    def _saw(self, x):
        if x <= -np.pi:
            return x + 2*np.pi
        elif x <= np.pi:
            return x
        else:
            return x - 2*np.pi
    
    def get_neighbour_map(self, bdry = 'periodic'):
        if hasattr(self, 'neighbour_map'):
            return self.neighbour_map
        
        n_map = {}
        L = self.L
        
        if bdry == 'periodic':
            for r, c in product(np.arange(L), np.arange(L)):
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

        return n_map
    
    
    """
    Performs one full sweep of the metropolis algorithm, e.g. attempts to "flip" each spin once.
    """
    def sweep(self):
        beta = 1.0 / self.T
        random.shuffle(self.index_map)
        
        # Do note that i and j are tuples
        for i in self.index_map:
            # Energy of lattice site prior to spinflip
            energy_i = -np.sum([1 - np.cos(self.spin_config[i] - self.spin_config[j]) for j in self.neighbour_map[i]]) 
            
            # Angle/phase change
            dtheta = np.random.uniform(-np.pi, np.pi)
            
            # New spin value
            spin_temp = (self.spin_config[i] + dtheta) % 2*np.pi
            
            # Energy of lattice site after spinflip
            energy_f = -np.sum([1 - np.cos(spin_temp - self.spin_config[j]) for j in self.neighbour_map[i]]) 
            
            # Energy change
            delta_E = energy_f - energy_i
            
            # Rejection sampling. Note that if delta_E <= 0, exp >= 1 and we always accept
            if np.random.uniform(0.0, 1.0) < np.exp(-beta * delta_E):
                self.spin_config[i] = (self.spin_config[i] + dtheta) % 2*np.pi
                self.energy = self.get_energy()
                
    """
    Calculates the energy per spin for the current configuration
    Returns the value and sets the corresponding internal variable
    """
    def get_energy(self, cfg = []):
        H = 0
        
        if not len(cfg):
            cfg = self.spin_config
        
        # Might have to optimize this
        for i in self.index_map:
            H += -np.sum([1 - np.cos(cfg[i] - cfg[j]) for j in self.neighbour_map[i]])
        
        return H/self.L**2
        
    ## Let the system evolve to equilibrium state
    def equilibrate(self, T = None, H = None):
        if T != None:
            self.T = T
        
        
        energies = []
        energy_temp = 0
        
        # Minimum steps for equilibration
        min_eq = 200
        for k in np.arange(self.max_sweeps):
            self.sweep()     
            
            energy = self.get_energy()
            energies.append( energy )
            
            # Reached equilibrium?
            if (abs(energy-energy_temp)/abs(energy) < self.tol) and k > min_eq:
                tqdm.write('Eq reached at T = %.1f, E = %.2f, sweep #%d' % (self.T,energy, k))
                break
            energy_temp = energy
        
        nstates = len(energies)
        energy = np.mean(energies[int(nstates/2):])

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
                states.append(self.spin_config.copy())
            
            # Measure observables
            for k in results.keys():
                results[k] += observables[k](self.spin_config)
        
        # Average
        for k, v in results.items():
            results[k] = v/m_callback
            
        if store_states:
            results['states'] = states
            
        return results


if __name__ == '__main__':
    xy = XY()
#    xy.annealing()
#    # Measure energy as function of temperature
    ecb = lambda c = None: xy.get_energy()
    
    print('ALERT: Running XY.py.')
    temp = np.linspace(0.1, 2, 2)
    energy = []
    states = []
    
    for t in temp:
        energy.append( xy.measure_observables(t, 5, 1000, {'energy': ecb})['energy'] )
    
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 09:15:16 2019

@author: Zak
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
XY model.

Init:
    L:      [Integer] system size
    T:      [Float]   temperature
"""
class XY:
    
    def __init__(self, L, T):
        self.L = L
        self.T = T
        self.J = 1
        self.spins = np.random.rand(L, L)*2*np.pi
        self.max_iter = 1e4
        self.tol = 1e-4
        
        self.construct_neighbour_map()
        
        self.energy = self.get_energy()
        
        self.energies = []
    
    def construct_neighbour_map(self):
        n_map = {}
        for r in np.arange(self.L):
            for c in np.arange(self.L):
                # Find neighbour indices
                left = r - 1
                right = r + 1
                top = c + 1
                bot = c - 1
                
                # Take boundaries into account
                left = max(left, 0)
                right = min(right, self.L-1)
                bot = max(bot, 0)
                top = min(top, self.L-1)
                
                n_map[r, c] = ((r, left), (r, right), (top, c), (bot, c))
                
        self.neighbour_map = n_map
                
                
    
    # Attempts to equilibrate the model at temperature T
    def equilibrate(self):
        
        
        # Step until convergence or failure
        for i in tqdm(np.arange(self.max_iter)):
            oe = self.energy
            for r in np.arange(self.L):
                for c in np.arange(self.L):
                    self.xy_step(r, c)
            
            # New energy if any
            ne = self.energy
            self.energies.append(ne)
            
            if np.abs(ne-oe) < self.tol:
                tqdm.write('Converged at epoch %d' % i)
                break
        
    # Steps the xy model using metropolis algorithm
    def xy_step(self, r, c):
        # Pick a spin
#        r, c = np.random.randint(0, self.L, 2)
        
        # How much angle to add
        delta = 1
        
        # Give it a new value
        nphi = self.spins[r, c] + delta*(np.random.rand()*2 - 1)*np.pi
        
        # Pick an acceptance number in [0, 1]
        acc = np.random.rand()
        
        # Get the energy differential
        dE = self.energy_diff(nphi, r, c)
        
        # Calculate relative Boltzman factors to decide if we should flip
        dp = np.exp(-1/self.T*dE)
        
        # Accept spinflip?
        if acc < dp or dE < 0:
            self.spins[r, c] = nphi
            self.energy = self.get_energy()
    
    # Calculates the difference in energy given a new angle [phi] at site [r, c]
    def energy_diff(self, phi, r, c):        
        # Current angle
        cphi = np.ones(4)*self.spins[r, c]
        
        # New angle
        phi = np.ones(4)*phi
        
        # Neighbour angles
        nphi = [self.spins[idx] for idx in self.neighbour_map[r, c]]
        
        e_before = -self.J*np.cos(cphi-nphi)
        e_after = -self.J*np.cos(phi-nphi)
        
        # Return energy diff
        return np.sum( e_after-e_before )

        
    # Calculates the total energy of the system given the spin config
    def get_energy(self):
        H = 0
        # Loop over rows
        for r in np.arange(self.L):
            # Loop over columns
            for c in np.arange(self.L):                
                # Current position angle
                cphi = np.ones(4)*self.spins[r, c]
                
                # Neighbour angles
                nphi = [self.spins[idx] for idx in self.neighbour_map[r, c]]
                
                # Contribution to energy
                dH = np.sum( self.J*np.cos(cphi-nphi) )
                
                # Update energy
                H = H + dH
                
        return H
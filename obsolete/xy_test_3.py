import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from tqdm import tqdm

def initialstate(L):   
    ''' generates a random spin configuration for initial condition'''
    state = np.random.rand(L, L)*2*np.pi
    return state


def mcmove(config, beta):
    '''Monte Carlo move using Metropolis algorithm '''
    for i in range(N):
        for j in range(N):
                a = np.random.randint(0, N)
                b = np.random.randint(0, N)
                s =  np.ones(4)*config[a, b] + (np.random.rand()*2 - 1)*np.pi
                nb = np.array([config[(a+1)%N,b], config[a,(b+1)%N], config[(a-1)%N,b], config[a,(b-1)%N]])
                cost = -np.sum(np.cos(s-nb))
                if cost < 0 or np.random.rand() < np.exp(-cost*beta):
                    config[a, b] = s[0]
    return config


def calcEnergy(config):
    '''Energy of a given configuration'''
    energy = 0
    for i in range(len(config)):
        for j in range(len(config)):
            S = np.ones(4)*config[i,j]
            nb = [config[(i+1)%N, j], config[i,(j+1)%N], config[(i-1)%N, j], config[i,(j-1)%N]]
            energy = energy - np.sum(np.cos(S - nb))
    return energy/4


def calcMag(config):
    '''Magnetization of a given configuration'''
    mag = np.sum(config)
    return mag

## change these parameters for a smaller (faster) simulation 
nt      = 10         #  number of temperature points
N       = 16         #  size of the lattice, N x N
eqSteps = 4096       #  number of MC sweeps for equilibration
mcSteps = 1024       #  number of MC sweeps for calculation

T       = np.linspace(0.6, 1.2, nt); 
E,M,C,X = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N) 
# divide by number of samples, and by system size to get intensive values


for tt in tqdm(range(nt)):
    E1 = M1 = E2 = M2 = 0
    config = initialstate(N)
    iT=1.0/T[tt]; iT2=iT*iT;
    
    for i in range(eqSteps):         # equilibrate
        mcmove(config, iT)           # Monte Carlo moves
    
    for i in range(mcSteps):
        mcmove(config, iT)           
        Ene = calcEnergy(config)     # calculate the energy
        Mag = calcMag(config)        # calculate the magnetisation

        E1 = E1 + Ene
        M1 = M1 + Mag
        M2 = M2 + Mag*Mag 
        E2 = E2 + Ene*Ene
        
    E[tt] = n1*E1
    M[tt] = n1*M1
    C[tt] = (n1*E2 - n2*E1*E1)*iT2
    X[tt] = (n1*M2 - n2*M1*M1)*iT
    
f = plt.figure(figsize=(18, 10)); # plot the calculated values    

sp =  f.add_subplot(2, 2, 1 );
plt.scatter(T, E, s=50, marker='o', color='IndianRed')
plt.xlabel("Temperature (T)", fontsize=20);
plt.ylabel("Energy ", fontsize=20);         plt.axis('tight');

sp =  f.add_subplot(2, 2, 2 );
plt.scatter(T, C, s=50, marker='o', color='IndianRed')
plt.xlabel("Temperature (T)", fontsize=20);  
plt.ylabel("Specific Heat ", fontsize=20);   plt.axis('tight');   


plt.show()
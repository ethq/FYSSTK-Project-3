# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 12:07:34 2019

@author: Z
"""

## KERAS imports
import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

# Standard imports
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd

# My stuff
from XY import XY
from get_model import get_model

# Utility imports
from itertools import product
import ctypes
import re
from os import walk

# SKLEARN imports
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

rc('text', usetex=True)

"""
Should be put in a package but I cannot be bothered due to the fun times that is Python packages.
Some aggregate analysis put in aggregate_xy.py for now


Class to perform analysis on an XY dataset both using regular calculus methods and (C)NN's
Input:
    fname: [String]   An identifier that labels the state. Is of form L%_M%_N%,
                      where L = lattice size, M = number of measurements per temperature 
                      and N = number of sweeps between each measurement
                      
    X_vortex: [Bool]  Whether to use the raw spin states or the vorticity as features in design matrix.
                      
    X_full:   [Bool]  Whether to use all states at a given temperature or just one in design matrix
    
    plot_data_spacing: [Integer] When plotting interpolated energy curve, how many datapoints to cut between
    
    verbose:  [Bool]  Talkative or not
"""
class Analyze_XY:
    def __init__(self, 
                 fname,
                 X_vortex = False,
                 X_full = True,
                 plot_data_spacing = 1,
                 verbose = True):
        data = None
        
        fname = 'Data/xy_data_' + fname + '.dat'
        with open(fname, "rb") as f:
            data = pickle.load(f)
            
        self.fname = fname.lstrip('Data/xy_data_').rstrip('.dat')
        self.data = data
        
        self.X_vortex = X_vortex
        self.X_full = X_full
        self.verbose = verbose
        self.L = self.get_L_from_id(self.fname)
        self.energy = np.array(data['energy'])
        self.t = np.array(data['temperature'])
        self.states = np.array(data['states'])
        
        # Get M - the number of measurements at fixed temperature
        # and N - the number of sweeps between measurements
        # N cannot be obtained unless it is either in the filename or stored in data
        
        if 'sweeps_between_measurement' in data.keys():
            self.N = data['sweeps_between_measurement']
        else:
            # Time for some str manipulation. 
            # We know the filename is of type 'xy_data_L%_M%_N%.dat'
            f = fname.lstrip('xy_data_').rstrip('.dat')
            
            # Now it is of form L%_M%_N%. We need a regexp to remove the first part
            s = 'L[0-9]+_M[0-9]+_N'
            match = re.search(s, f).group(0)
            
            # And finally get N
            self.N = int(f.lstrip(match))
            
            print('From filename: %s, regexp detects N = %d' % (fname, self.N))
            
        if 'number_of_measurements'  in data.keys():
            self.M = data['number_of_measurements']
        else:
            # Infer it
            self.M = len(self.states.flatten())/len(self.t)
        
        
        self.energy_interp = []
        self.t_interp = []
        self.cv_interp = []
        
        self.plot_data_spacing = plot_data_spacing
        
        self.tkt = None
        self.X = None
        self.labels = None
        self.e_labels = None
        
        self.interpolate()
        self.get_cv()
        self.get_tkt()
        
        self.create_design_matrix()
        
        print('On lattice L = %d, T = %.2f' % (self.L, self.tkt))
    
    
    # From an ID of type L%_M%_N%, extracts L
    def get_L_from_id(self, fid):
        s = '_M[0-9]+_N[0-9]+'
        return int(fid.rstrip(re.search(s, fid).group(0)).lstrip('L'))
    
    """
    Input: a design matrix
    Output: a design matrix, but reshaped for use with a cnn using the appropriate backend
            also returns the input shape, which keras needs
    """
    def prepare_X_for_cnn(self, x):
        rows, cols = self.L, self.L
        
        # Theano vs tensorflow convention check
        if K.image_data_format() == 'channels_first':
            x = x.reshape(x.shape[0], 1, rows, cols)
            input_shape = (1, rows, cols)
        else:
            x = x.reshape(x.shape[0], rows, cols, 1)
            input_shape = (rows, cols, 1)
        
        # Make sure types are right
        x = x.astype('float32')
        
        return x, input_shape
    
    """
    Trains a convolutional neural network on the dataset using Keras.
    Input:
        batch_size: training instances used in each pass
        epochs: number of complete passes
        model_s: a string, choices enumerated in get_model.py
    Output:
        Saves a plot of training vs validation error
        Saves a plot of the model used
    """
    def train_cnn(self,
                  model_s,
                  batch_size = 128,
                  epochs = 20
                  ):
        n_classes = 2
        
#        # Test/train split
#        print(self.X.shape, self.labels.shape)
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.labels, test_size = .2, shuffle = True)
#        
#        # Theano vs tensorflow convention check
#        if K.image_data_format() == 'channels_first':
#            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#            input_shape = (1, img_rows, img_cols)
#        else:
#            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#            input_shape = (img_rows, img_cols, 1)
#        
#        # Make sure types are right
#        x_train = x_train.astype('float32')
#        x_test = x_test.astype('float32')
#        
#        # Normalize
#        x_train /= (2*np.pi)
#        x_test /= (2*np.pi)
        
        x_train, input_shape = self.prepare_X_for_cnn(x_train)
        x_test, _ = self.prepare_X_for_cnn(x_test)
        
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, n_classes)
        y_test = keras.utils.to_categorical(y_test, n_classes)
        
        # Squawk if desired
        if self.verbose:
            print('x_train shape:', x_train.shape)
            print(x_train.shape[0], 'train samples')
            print(x_test.shape[0], 'test samples')
            
        # Pick a model
        model_s = '3xConv2xDropDense'
        
        # Get it
        model = get_model(model_s, input_shape)
        
        # Compile it        
        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        
        """
        CNN will likely overfit XY states, at least on L = 7 lattice. Hence we need early stopping.
        Patience is set to epochs such that we keep looking for the best model over all epochs.
        """
        es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = epochs, verbose = 1)
        
        # We also want to store our best model, as judged by accuracy
        mc = ModelCheckpoint('Models/Epoch{epoch:02d}_Acc{val_acc:.2f}_V%d_L%d_M%d_N%d_%s.h5' % (int(self.X_vortex), self.L, self.M, self.N, model_s) , monitor='val_acc', mode='max', verbose=1, save_best_only=True)
        
        # Fit and record history
        history = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks = [es, mc],
                  validation_split = 0.1)
        
        # Get the score on the unseen test set
        score = model.evaluate(x_test, y_test, verbose=0)
        
        # Squawk if desired
        if self.verbose:
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
        
        # Plot training history
        plt.plot(history.history['loss'], label = 'train')
        plt.plot(history.history['val_loss'], label = 'val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model: %s, Score: %.3f' % (model_s, score[1]))
        plt.legend()
        
        # Save the plot to file
        plt.savefig('Plots/Train Test Scores/L%d_M%d_N%d_%s.png' % (self.L, self.M, self.N, model_s) )
        
        # Save a graph of the model
        plot_model(model, to_file = 'Plots/Model Graphs/%s.png' % (model_s)  )
        
        # And show plot if desired
        if self.verbose:
            plt.show()
            
    """
    Given a trained CNN, uses the decision boundary to locate Tkt. 
    Should coincide with that found using the heat capacity.
    
    Input:
        model:      [String] Just the name of the model used, as enumerated in get_model.py
    """
    def locate_tkt_from_cnn(self, model_s):
        # Load best model. 
        
        # First re-create filename based on instance parameters        
        fid = 'V%d_L%d_M%d_N%d_%s.h5' % (int(self.X_vortex), self.L, self.M, self.N, model_s)
        
        # Collect all filenames in Models directory
        f = []
        for (_, _, fn) in walk('Models/'):
            f.extend(fn)
            break
        
        # Look for all filenames matching instance parameters
        # We may have several fits with differing accuracy at differing epochs
        path = ''
        best_acc = -np.Inf
        
        # Loop over files in directory
        for fn in f:
            # Match to instance parameters
            if re.search(fid, fn):                
                # Extract accuracy
                pat = '_Acc0.[0-9]+_'
                match = re.search(pat, fn).group(0).rstrip('_').lstrip('_Acc')
                acc = np.float32(match)
                
                # Best model so far? Select it
                if acc > best_acc:
                    best_acc = acc
                    path = 'Models/' + fn
        
        # Make sure we have a match before we proceed
        if path == '':
            print('No pre-trained model corresponding to instance parameters was found. Returning')
            return
        
        # Load model
        model = keras.models.load_model(path)
        
        # Matching design matrix has been created in __init__(), but it must be prepped for cnn
        x, input_shape = self.prepare_X_for_cnn(self.X)
        # But we'd like to sort it so that also our predictions are sorted by energy
        i = self.e_labels.argsort()
        energy_sorted = self.e_labels[i]
        X_sorted = x[i]
        
        # Predict probability (keras model does have predict proba func)
        prob = model.predict_proba(X_sorted)
        print('Class balance: ', sum(self.labels)/len(self.labels))
        
        assert type(prob) == np.ndarray
        
        # The moment of truth
        plt.plot(energy_sorted, prob[:, 0], label = 'P < TKT')
        plt.plot(energy_sorted, prob[:, 1], label = 'P > TKT')
        plt.legend()
        plt.show()
        
        pred_class = model.predict(X_sorted)
        print(confusion_matrix(self.labels[i], pred_class))
    
    """     
    Returns design matrix for the given temperature/energy. 
    Expects that T_KT has been found from the heat capacity before this is called
    
    Input:
        full: [Boolean] whether to return all states or just one at a given temperature
        vortex: [Boolean] whether to return the raw spin configuration or the vortex representation
    Output:
        Design matrix. Number of features equals the size of a state, 
                       number of instances equal to the number of energies/temperatures
    """ 
    def create_design_matrix(self):   
        # Make sure the class is properly initialized
        assert self.tkt != None
        
        # Notational shortcut
        data = self.data
        
        # Compute size of state - it's 2d and square. Note that data['states'] is of form N x M x (L,L)
        # where N = number of temperatures, M = number of measurements and L is the lattice dimension
        L = len(data['states'][0][0][0])
        
        # Instantiate an XY model, so that we can grab the vortex rep of the state
        xy = XY(T = 1, L = L)
        
        # By default, we label states by whether they are above or below T_KT
        labels = []
        
        # We also label each state by its energy. This is useful when (C)NN 
        # is to determine T_TK
        e_labels = []
        
        states = []
        ts = data['temperature']
        t_mask = (ts > self.tkt).astype(int)
        
        # If we want _all_ states at the given temperature ranges, then return that
        if self.X_full:
            # Loop over temperatures
            for ei, e in enumerate(data['states']):
                # Loop over all states at a given temperature
                for s in data['states'][ei]:
                    # Add the raw state if desired
                    if not self.X_vortex:
                        states.append(s)
                    # Otherwise add the vortex representation
                    else:
                        vs = xy.get_vortex_map(s)
                        states.append(vs)
                    # Add the label - since the number of states now far exceeds the length of t_mask,
                    # it is not usable as a label set anymore. Could possibly duplicate entries etc but w/e.
                    labels.append(t_mask[ei])
                    
                    # Add energy label
                    e_labels.append(xy.get_energy(s))
            
            # Design matrix
            X = np.zeros((len(states), L**2))
            
            # Add raveled states
            for i, s in enumerate(states):
                X[i, :] = s.flatten()
            
            # And done
            self.X = X/(2*np.pi)
            self.labels = np.array(labels)
            self.e_labels = np.array(e_labels)
            
            # Return to avoid executing code for a sparse design matrix
            return
        
        # Otherwise, we just return a single state per temperature, namely the one with energy closest to the mean
        # Then t_mask is 1-1 with number of states, so we use it as labels
        labels = t_mask
        
        # Loop over all energies, select a representative state
        for ei, e in enumerate(data['energy']):
            # Smallest energy difference
            smallest_diff = np.Inf
            
            # State with smallest energy difference
            best_state_id = -1
            
            # Find state with energy closest to the mean
            for si, s in enumerate(data['states'][ei]):
                diff = abs(e - xy.get_energy(s))
                
                # If it is smaller, update best state
                if diff < smallest_diff:
                    best_state_id = si
                    smallest_diff = diff
                    
            # Add best state
            bs = np.array(data['states'][ei][best_state_id])
            
            # Label energy
            e_labels.append(xy.get_energy(bs))
                
            # Add the raw spin configuration if desired
            if not self.X_vortex:
                states.append(bs)
            # Otherwise add the vortex representation
            else:
                vs = xy.get_vortex_map(bs)
                states.append(vs)
        
        # Design matrix
        X = np.zeros((len(states), L**2))
        
        # Add states
        for i, s in enumerate(states):
            X[i, :] = s.flatten()
        
        # And done
        self.X = X/(2*np.pi)
        self.labels = np.array(labels)
        self.e_labels = np.array(e_labels)
    
    """
    Calculates the heat capacity given a fitted Energy(Temperature) curve
    """
    def get_cv(self):
        assert len(self.energy_interp) and len(self.t_interp)
        
        t = np.array(self.t_interp)
        e = np.array(self.energy_interp)
        
        self.cv_interp = np.array(np.diff(e)/np.diff(t.flatten()))
        
    """
    Fits a curve to the Energy(Temperature) data. Gives better estimate of heat capacity, and hence better labeling of states
    """
    def interpolate(self):
        model = make_pipeline(PolynomialFeatures(10), Ridge())
        model.fit(self.t[:, np.newaxis], self.energy)
        
        tnew = np.linspace(self.t[0], self.t[-1], 200)[:, np.newaxis]
        energy_pred = model.predict(tnew)
        cv = np.diff(energy_pred)/np.diff(tnew.flatten())
        
        self.t_interp = tnew
        self.energy_interp = energy_pred
        self.cv_interp = cv
        
    """
    Plots E(T) and C(T), and if interpolation has been done, also the fitted curve
    """
    def plot(self):
        # Energy and heat capacity plots
        f = plt.figure()
        ax = f.add_subplot(121)
        
        if len(self.t_interp):
            ax.plot(self.t_interp, self.energy_interp)
            ax.set_xlabel('Temperature')
            ax.set_ylabel('Energy')
            ax.set_title('Energy vs Temperature!')
        
        ax.plot(self.t[::self.plot_data_spacing], self.energy[::self.plot_data_spacing], 'o', markeredgecolor ='black', markerfacecolor = (0.7, .2, .2, .5))
        
        ax = f.add_subplot(122)
        
        if len(self.cv_interp):
            ax.plot(self.t_interp[1:], self.cv_interp)
            ax.set_title(r'Estimated $T_{KT}$: %.3f' % self.tkt)
            ax.set_xlabel('Temperature')
            ax.set_ylabel('Heat capacity')
        
        plt.show()
    
    def get_tkt(self):
        t = np.array(self.t_interp[1:]).flatten()
        
        # The high order fits cause endpoint issues, so just look in the expected region for tc(it is always less than 1.5 for grid sizes L > 6)
        mask_e = t < 1.5
        
        # Similarly we know tc > 0.3 at the very least
        mask_s = t > 0.3
        
        mask = [m1 and m2 for m1, m2 in zip(mask_e, mask_s)]
        cvm = self.cv_interp[mask]
        tm = t[mask]
        
        self.tkt = tm[np.argmax(cvm)]
    

        
if __name__ == '__main__':
#    tkt = plot_tkt_infinite()
#    d = plot_energies()
    
    # Do a complete analysis
    fname = 'L7_M500_N5'
    # Finds TKT by itself, but worth inspecting manually as well
    axy = Analyze_XY(fname)
    print(axy.tkt)
#    axy.plot()
    
    # Ok, now we train a CNN
#    model = '3xConv2xDropDense'
#    axy.train_cnn(model)
    
    # Load a model and locate decision boundary
    model = '3xConv2xDropDense'
    axy.locate_tkt_from_cnn(model)
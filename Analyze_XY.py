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
from keras.wrappers.scikit_learn import KerasClassifier

# Standard imports
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import rc
import pandas as pd
import seaborn as sns

# My stuff
from XY import XY
from get_model import get_model, is_cnn, requires_rgb, is_nn, enumerate_models
from fancy_confusion_matrix import galplot, plot_cm

# Utility imports
from itertools import product
import ctypes
import re
from os import walk

# SKLEARN imports
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from imblearn.over_sampling import SMOTE

rc('text', usetex=True)

class Bunch:
    def __init__(self):
        pass


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
    
    subsample: [Integer] Pick every N'th instance for use in design matrix instead of all. Handy for big datasets(e.g. 5k measurements on L > 16)
    
    verbose:  [Bool]  Talkative or not
"""
class Analyze_XY:
    def __init__(self, 
                 fname,
                 X_vortex = False,
                 X_full = True,
                 plot_data_spacing = 3,
                 verbose = True,
                 plotty = True, 
                 subsample = 1,
                 boost_nn = False):
        data = None
        
        fname = 'Data/xy_data_' + fname + '.dat'
        with open(fname, "rb") as f:
            data = pickle.load(f)
            
        self.fname = fname.lstrip('Data/xy_data_').rstrip('.dat')
        self.data = data
        
        # Whether to supply vortex config or spin config
        self.X_vortex = X_vortex
        
        # Whether to compute the design matrix using _all_ datapoints or just a subset
        self.X_full = X_full
        
        # Whether or not to AdaBoost a regular neural network, if trained
        self.boost_nn = boost_nn
        
        # Whether or not to plot stuff
        self.plotty = plotty
        
        # Talkative?
        self.verbose = verbose
        
        # Lattice dimension
        self.L = self.get_L_from_id(self.fname)
        
        # Energy
        self.energy = np.array(data['energy'])
        
        # Temperature
        self.t = np.array(data['temperature'])
        
        # States
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
        
        self.tkt = {}
        self.X = None
        self.labels = None
        self.e_labels = None
        
        self.interpolate()
        self.get_cv()
        self.get_tkt()
        self.plot(only_save = True)
        
        self.create_design_matrix(subsample = subsample)
        
        print('On lattice L = %d, T = %.2f' % (self.L, self.tkt['MCMC']))
    
    
    # From an ID of type L%_M%_N%, extracts L
    def get_L_from_id(self, fid):
        s = '_M[0-9]+_N[0-9]+'
        return int(fid.rstrip(re.search(s, fid).group(0)).lstrip('L'))
    
    
    """
    Input: a design matrix
    Output: a design matrix, but reshaped for use with a cnn using the appropriate backend
            also returns the input shape, which keras needs
    """
    def prepare_X_for_cnn(self, x, grey2rgb = False):
        rows, cols = self.L, self.L
        depth = 1
        
        if grey2rgb:
            depth = 3
        
        # Theano vs tensorflow convention check
        if K.image_data_format() == 'channels_first':
            x = x.reshape(x.shape[0], 1, rows, cols)
            if grey2rgb:
                x = np.repeat(x, 3, 1)
            input_shape = (depth, rows, cols)
        else:
            x = x.reshape(x.shape[0], rows, cols, 1)
            if grey2rgb:
                x = np.repeat(x, 3, 3)
            input_shape = (rows, cols, depth)
        
        # Make sure types are right
        x = x.astype('float32')
        
        return x, input_shape
        
    def restore_energy_labels(self):
        xy = XY(T = 1, L = self.L)
        self.e_labels = np.array([xy.get_energy(np.reshape(s, (self.L, self.L))) for s in self.X])
    
    def reduce_data(self, nskip):
        # We want to skip nskip rows, pick one, skip again etc
        n_samples = self.X.shape[0] // nskip
        ind = np.random.randint(self.X.shape[0], size = n_samples)
        
        # Create smaller dataset
        self.X = self.X[ind]
        self.labels = self.labels[ind]
        self.e_labels = self.e_labels[ind]
        
    """
    Trains a (convolutional) neural network on the dataset using Keras.
    Input:
        batch_size: training instances used in each pass
        epochs: number of complete passes
        model_s: a string, choices enumerated in get_model.py
    Output:
        Saves a plot of training vs validation error
        Saves a plot of the model used
    """
    def train_net(self,
                  model_s,
                  batch_size = 128,
                  epochs = 20
                  ):
        n_classes = 2
        
#        self.reduce_data(20)
#
#        # Generate new instances to fix any class imbalance(relevant for (16,) set)
#        sm = SMOTE()
#        self.X, self.labels = sm.fit_resample(self.X, self.labels)
        
        # Recalculate energy for SMOTEd instances
#        self.restore_energy_labels()
        
#        if self.verbose:
#            print('Done SMOTEing')
            
        # Test/train split
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.labels, test_size = .2, shuffle = True)

        if self.verbose:
            print('Training balance: %.2f. Testing balance: %.2f' % (np.sum(y_train)/len(y_train), np.sum(y_test)/len(y_test)))
        
        input_shape = None
        if is_cnn(model_s):
            grey2rgb = requires_rgb(model_s)
            x_train, input_shape = self.prepare_X_for_cnn(x_train, grey2rgb)
            x_test, _ = self.prepare_X_for_cnn(x_test, grey2rgb)
        
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, n_classes)
        y_test = keras.utils.to_categorical(y_test, n_classes)
        
        # Squawk if desired
        if self.verbose:
            print('x_train shape:', x_train.shape)
            print(x_train.shape[0], 'train samples')
            print(x_test.shape[0], 'test samples')
        
        # Get it
        model = get_model(model_s, input_shape)
        
        """
        CNN will likely overfit XY states, at least on L = 7 lattice. Hence we need early stopping.
        Patience is set to epochs such that we keep looking for the best model over all epochs.
        """
        es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = epochs, verbose = 1)
        
        # We also want to store our best model, as judged by accuracy
        mc = ModelCheckpoint('Models/Epoch{epoch:02d}_Acc{val_acc:.2f}_V%d_L%d_M%d_N%d_%s.h5' % (int(self.X_vortex), self.L, self.M, self.N, model_s) , monitor='val_acc', mode='max', verbose=1, save_best_only=True)
        
        # Check for boosting
        if self.boost_nn and is_nn(model_s):
            # Different convention for labels. AdaBoostClassifier expects Y to be of form (nsamples,)
            # This in turn means models in get_model must be modified _WHEN_ used in conjuction with AdaBoostClf
            y_test = y_test[:, 0] + y_test[:, 1]*-1
            y_train = y_train[:, 0] + y_train[:, 1]*-1
            
            y_test = (y_test+1)/2
            y_train = (y_train+1)/2
            
            build = lambda: get_model(model_s, input_shape)
            est = KerasClassifier(build_fn = build, epochs = epochs, batch_size = batch_size, verbose = 0)
            
            model = AdaBoostClassifier(base_estimator = est, n_estimators = 1)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = .1)
            print(x_train.shape, y_train.shape)
            model.fit(x_train, y_train)
            self.MODEL = model
            self.XTE = x_test
            # Need to construct our own history manually
            pred_val = model.staged_predict(x_val)
            pred_tr = model.staged_predict(x_train)
            
            accs_val = []
            accs_train = []
            
            for predv, predr in zip(pred_val, pred_tr):
                accs_val.append(accuracy_score(predv, y_val))
                accs_train.append(accuracy_score(predr, y_train))
            
            # Bit lazy, but using accuracy is less hassle. But then we need to trick ourselves:
            history = Bunch()
            history.history = {'loss': accs_train,
                                'val_loss': accs_val
                                }
            score = (-1, accuracy_score(model.predict(x_test), y_test))
            
            # If it's an AdaBoosted neural net, we won't do early stopping or save/load. 
            # It's hackish, but we just store it in instance. Why? Because we already know
            # it'll perform worse than a CNN, so it's not worth the effort at the moment.
            self.model_adaboost = model
        else:
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
            

        y_true = y_test[:, 1].astype(int)
        y_pred = np.round(model.predict(x_test)[:, 1]).astype(int)
        
        self.AA = y_true
        self.BB = y_pred
        
        print(classification_report(y_true, y_pred))
        self.f1 = f1_score(y_true, y_pred)
        print('F1-score: %.3f' % self.f1)
        print(confusion_matrix(y_true, y_pred))
        self.rocauc = roc_auc_score(y_true, y_pred)
        self.accuracy = accuracy_score(y_true, y_pred)
        
        # Plot training history
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.plot(history.history['loss'], label = 'train')
        ax.plot(history.history['val_loss'], label = 'val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        if is_nn(model_s) and self.boost_nn:
            ax.set_ylabel('Accuracy')
        ax.set_title('Model: %s, Test score: %.3f' % (model_s, score[1]))
        ax.legend()
        
        # Save the plot to file
        plt.savefig('Plots/TrainTestScores/V%d_L%d_M%d_N%d_%s.png' % (int(self.X_vortex), self.L, self.M, self.N, model_s) )
        
        # Save a graph of the model
        plot_model(model, to_file = 'Plots/Model Graphs/%s.png' % (model_s)  )
        
        # And show plot if desired
        if self.plotty:
            plt.show()
      
    # Deos exactly what it says on the tin, converts an energy to a temperature
    # as determined by the interpolated E(T) curve
    def energy2temp(self, e):
        idx = np.argmin(np.abs(self.energy_interp - e))
        t = self.t_interp[idx]
        
        print('Energy %.2f --> Temp %.2f' % (e,t))
        
        return t
    
    """
    
    Locates T_KT from a predictor and a design matrix. 
    
    Input:
        pred:  [Predictor] Must implement .predict() and .predict_proba()
        X:     [Array] Design matrix, must be appropriately adjusted for use with the Predictor
        model_s: [String] Name of model used by Predictor. Used to label determined data.
                       
    Output:
        Returns nothing. Sets self.tkt[model_s] to the determined value of T_KT. Saves a single plot
                         containing probability curves/interpolations/determined value of T_TK
    
    """
    def tkt_from_pred(self, model, X, model_s):
        i = self.e_labels.argsort()
        energy_sorted = self.e_labels[i]
        X_sorted = X[i]
        
        # Predict probability (keras model does have predict proba func)
        prob = model.predict_proba(X_sorted)
        if self.verbose:
            print('Class balance: ', sum(self.labels)/len(self.labels))
        
        assert type(prob) == np.ndarray
        
        # Fit a curve to each probability, we try with simply a 10'th order poly
        mod = make_pipeline(PolynomialFeatures(5), LinearRegression())
        
        mod.fit(energy_sorted[:, np.newaxis], prob[:, 0])
        prob0_pred = mod.predict(energy_sorted[:, np.newaxis])
        
        mod.fit(energy_sorted[:, np.newaxis], prob[:, 1])
        prob1_pred = mod.predict(energy_sorted[:, np.newaxis])
        
        # Finite polyfits will necessarily have some unpleasant edge effects. So we'll constrain plots
        # to the middle area of the energy range, which is where we are interested in them anyway
        l = len(prob1_pred) // 4
        
        # Locate where the probability curves overlap, e.g. where P0 = P1 = .5
        most_uncertain_energy = energy_sorted[np.argmin(np.abs(prob0_pred - prob1_pred))]
        if self.verbose:
            print('%s is most uncertain about how to classify states of energy E = %.2f' % (model_s, most_uncertain_energy))
        
        # Get the corresponding temperature
        most_uncertain_temperature = self.energy2temp(most_uncertain_energy)
        
        self.tkt[model_s] = most_uncertain_temperature
        if self.verbose:
            print('CNN determined $T_{KT} = %.2f. MCMC determined T_{KT} = %.2f' % (self.tkt[model_s], self.tkt['MCMC']))
        
        ### !!! Method 2 !!! ### s
        ########################
        
        # Here we aim to grab all states which the network is highly uncertain about, e.g. p ~ .5
        # We then calculate their energy, average and get the corresponding temperature
        
        
        ########################
        ### !!! Method 2 !!! ### e
        
        # Figures, plots and oranges. Myyy preciousssss orangesssss. 
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
        
        # Add some extra space for the second axis at the bottom
        fig.subplots_adjust(bottom=0.2)
        
        net_type = 'ConvNet'
        if is_nn(model_s):
            net_type = 'NeuralNet'
        
        ax1.set_title(r"%s prediction: $T_{KT}$ = %.3f" % (net_type, self.tkt[model_s]))
        ax1.set_xlabel('Energy')
        ax1.set_ylabel('Probability')
        
        ax1.plot(energy_sorted, prob[:, 0], label = r'$P < T_{KT}$', color = (.2, .7, .2, .6), marker = 'o')
        ax1.plot(energy_sorted, prob[:, 1], label = r'$P > T_{KT}$', color = (.7, .2, .2, .6), marker = 'o')
        
        ax1.plot(energy_sorted[l:-l], prob0_pred[l:-l], label = r'$P_{pred} < T_{KT}$')
        ax1.plot(energy_sorted[l:-l], prob1_pred[l:-l], label = r'$P_{pred} > T_{KT}$')
        
        ax1.axvline(x = most_uncertain_energy, linestyle = '--', color = 'black', ymin = -10)
        
        ax1.legend()
        fig.canvas.draw()
        
        # Now a load of shit to get a temperature axis
        pattern = "'\$-?[0-9]+.[0-9]+\$'"
        matches = [re.search(pattern, str(p)).group(0) for p in ax1.get_xticklabels()]
        
        tick_labels = [np.round(self.energy2temp(np.float32(m.rstrip("$'").lstrip("'$"))), 2) for m in matches]
        
        # Move twinned axis ticks and label from top to bottom
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")
        
        # Offset the twin axis below the host
        ax2.spines["bottom"].set_position(("axes", -0.15))
        
        # Turn on the frame for the twin axis, but then hide all 
        # but the bottom spine
        ax2.set_frame_on(True)
        ax2.patch.set_visible(False)
        for k,sp in ax2.spines.items():
            sp.set_visible(False)
        ax2.spines["bottom"].set_visible(True)
        
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks( ax1.get_xticks() )
        ax2.set_xticklabels(tick_labels)
        ax2.set_xlabel("Temperature")
        
        plt.tight_layout()
        
        plt.savefig('Plots/Predictions/CNN_V%d_L%d_M%d_N%d_%s.png' % (int(self.X_vortex), self.L, self.M, self.N, model_s) )
        
        if self.plotty:
            plt.show()
        
        # Plot/save confusion matrix
        y_pred = np.round(model.predict(X_sorted)[:, 0]).astype(int).flatten()
        y_true = self.labels.flatten()
        
        cm = np.array(confusion_matrix(y_true, y_pred))
        
        # seaborn heatmap broken in last update apparently. fix manually or text goes poof
        plt.figure(figsize=(10,7))
        ax = sns.heatmap(cm, annot=True, linewidths=1, fmt = 'd')
        ax.set(yticks=[-.5, 1.5], 
               xticks=[0, 1])
        ax.set_ylim([-0, 2])

        ax.yaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0.5))
        ax.xaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0.5))
        
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.savefig('Plots/Confusion Matrix/V%d_L%d_M%d_N%d_%s.png' % (int(self.X_vortex), self.L, self.M, self.N, model_s) )
        
        if self.plotty:
            plt.show()

        if self.verbose:
            print(cm)
    
    """
    Given a trained CNN, uses two methods to locate T_KT.
    Method 1)
            Orders all states by energy ascending, and predicts class probabilities.
            Where class probabilities are 0.5, the network is most uncertain and should corresp T_KT
            Method is problematic because E = T only in the thermodynamic limit. Hence the network
            should not classify exclusively on E - and indeed it does not.
    Method 2) Take the mean energy of all states with probability = 0.5, then corresp temp should be T_KT
            Really this is what we did above by also interpolating on the probabilities.
    Should coincide with that found using the heat capacity.
    
    Input:
        model:      [String] Just the name of the model used, as enumerated in get_model.py
    """
    def tkt_from_nn(self, model_s):
        # Load best model. 
        # If it's neural net and adaboosted, it's stored in class instance
        if self.boost_nn and is_nn(model_s):
            self.tkt_from_pred(self.model_adaboost, self.X, model_s)
            return
        
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
                pat = '_Acc[01].[0-9]+_'
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
        
        input_shape = None
        # Matching design matrix has been created in __init__(), but it must be prepped for cnn
        if is_cnn(model_s):
            x, input_shape = self.prepare_X_for_cnn(self.X)
        else:
            x = self.X
        
        # Model constructed and design matrix prepared, pass to more general function
        self.tkt_from_pred(model, x, model_s)
    
    
    
    """     
    Returns design matrix for the given temperature/energy. 
    Expects that T_KT has been found from the heat capacity before this is called
    
    Input:
        full: [Boolean] whether to return all states or just one at a given temperature
        vortex: [Boolean] whether to return the raw spin configuration or the vortex representation
        subsample: [Integer] Skips subsample-1 instances before adding one to the design matrix
    Output:
        Design matrix. Number of features equals the size of a state, 
                       number of instances equal to the number of energies/temperatures
    """ 
    def create_design_matrix(self, subsample = 1):   
        # Make sure the class is properly initialized
        assert self.tkt['MCMC'] != None
        
        # Design matrix filename
        x_name = 'Data/Design matrix/X_V%d_L%d_M%d_N%d.npy' % (int(self.X_vortex), self.L, self.M, self.N)
        
        # Regular labels filename
        l_name = 'Data/Design matrix/L_V%d_L%d_M%d_N%d.npy' % (int(self.X_vortex), self.L, self.M, self.N)
        
        # Energy labels filename
        el_name = 'Data/Design matrix/EL_V%d_L%d_M%d_N%d.npy' % (int(self.X_vortex), self.L, self.M, self.N)
        
        # Attempt to load from file
        try:
            X = np.load(x_name)
            labels = np.load(l_name)
            e_labels = np.load(el_name)
            
            print('Loaded design matrix from file.')
            
            self.X = X
            self.labels = labels
            self.e_labels = e_labels
            
            return
        except IOError:
            print('Could not load design matrix, creating...')
        
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
        t_mask = (ts > self.tkt['MCMC']).astype(int)
        
        ctr = 0
        # If we want _all_ states at the given temperature ranges, then return that
        if self.X_full:
            # Loop over temperatures
            for ei, e in enumerate(data['states']):
                # Loop over all states at a given temperature
                for s in data['states'][ei]:
                    # Subsample first
                    ctr = ctr + 1
                    if ctr % subsample:
                        continue
                    
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
                
            # Normalize and clean
            X = X/(2*np.pi)
            X[abs(X) < 1e-6] = 0
            
            # Save to file
            np.save(x_name, X)
            np.save(l_name, np.array(labels))
            np.save(el_name, np.array(e_labels))
            
            # And set instance variables
            self.X = X
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
        
        self.t_interp = tnew.flatten()
        self.energy_interp = energy_pred
        self.cv_interp = cv
        
    """
    Plots E(T) and C(T), and if interpolation has been done, also the fitted curve
    """
    def plot(self, only_save = False):
        # Energy and heat capacity plots
        f = plt.figure()
        ax = f.add_subplot(121)
        
        if len(self.t_interp):
            ax.plot(self.t_interp, self.energy_interp, zorder = 500, color = (.2, .2, .5, 1), lw = 2)
            ax.set_xlabel('Temperature')
            ax.set_ylabel('Energy')
            ax.set_title('Energy vs Temperature')
        
        ax.plot(self.t[::self.plot_data_spacing], self.energy[::self.plot_data_spacing],
                'o', markeredgecolor ='black', markerfacecolor = (0.7, .2, .2, .5),
                zorder = 0)
        
        ax = f.add_subplot(122)
        
        if len(self.cv_interp):
            ax.plot(self.t_interp[1:], self.cv_interp, zorder = 500, color = (.2, .2, .5, 1), lw = 2)
            ax.axvline(x = self.tkt['MCMC'], color = 'black', linestyle = '--')
            ax.set_title(r'Estimated $T_{KT}$: %.3f' % self.tkt['MCMC'])
            ax.set_xlabel('Temperature')
            ax.set_ylabel('Heat capacity')
        
        plt.savefig('Plots/Predictions/MCMC_V%d_L%d_M%d_N%d.png' % (int(self.X_vortex), self.L, self.M, self.N) )
        
        if self.plotty and not only_save:
            plt.show()
            
        f.clf()
    
    def get_tkt(self):
        t = np.array(self.t_interp[1:]).flatten()
        
        # The high order fits cause endpoint issues, so just look in the expected region for tc(it is always less than 1.5 for grid sizes L > 6)
        mask_e = t < 1.5
        
        # Similarly we know tc > 0.3 at the very least
        mask_s = t > 0.3
        
        mask = [m1 and m2 for m1, m2 in zip(mask_e, mask_s)]
        cvm = self.cv_interp[mask]
        tm = t[mask]
        
        self.tkt['MCMC'] = tm[np.argmax(cvm)]
    

        
if __name__ == '__main__':
#    tkt = plot_tkt_infinite()
#    d = plot_energies()
    
    # Do a complete analysis
    fname = 'L7_M500_N5'
#    fname = 'L16_M5000_N1'
    # Finds TKT by itself, but worth inspecting manually as well
    axy = Analyze_XY(fname, plotty = False, subsample = 1, boost_nn = False, X_vortex = True)
#    print(axy.tkt)
#    axy.plot()
    
    # Ok, now we train a CNN
#    model = '3xConvPoolDrop'
    model = '2xConvConvPoolDrop'
#    model = '3xConv2xDropDense'
#    model = '2xConvPoolDrop'
#    model = 'ResNet50'
#    model = 'nnsimple'
#    model = 'nndeep'
    axy.train_net(model)
    
    # Load a model and locate decision boundary
    axy.tkt_from_nn(model)
    
#    get_model('DenseNet121', (7,7,3))
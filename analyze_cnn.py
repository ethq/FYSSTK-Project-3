## KERAS imports
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

# Standard imports
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# My stuff
from XY import XY
from get_model import get_model

# Utility imports
from itertools import product
import ctypes

# SKLEARN imports
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


fname = 'xy_data_L7_M500_N5.dat'
t_kt = {fname: 1.11234963}
indata = None

with open(fname, "rb") as f:
    indata = pickle.load(f)
    
energy = np.array(indata['energy'])
t = np.array(indata['temperature'])

# Murder outliers
#mask = t > .5
#t = t[mask]
#energy = energy[mask]


"""
Creates design matrix. 

Input:
    data: [Dictionary]
        Assumed to contains keys 'energy', 'states' and 'temperature'
        The corresponding values should be lists or arrays of equal length
        data['states'] should contain lists, each containing all states that were used to calculate the average energies in data['energy']
        
Output:
    Design matrix. Number of features equals the size of a state, number of instances equal to the number of energies/temperatures
"""
def create_reduced_X(data):
    # Compute size of state - it's 2d and square
    L = len(data['states'][0][0][0])
    xy = XY(T = 1, L = L)
    
    states = []
    ts = data['temperature']
    labels = (ts > t_kt[fname]).astype(int)
    
    # Loop over all energies, select a representative state
    for ei, e in enumerate(data['energy']):
        # Smallest energy difference
        smallest_diff = np.Inf
        
        # State with smallest energy difference
        best_state_id = -1
        
        # Find state with energy closest to the mean
        for si, s in enumerate(data['states'][ei]):
            diff = abs(e - xy.get_energy(s))
            
            if diff < smallest_diff:
                best_state_id = si
                smallest_diff = diff
        bs = np.array(data['states'][ei][best_state_id])
        vs = xy.get_vortex_map(bs)
        states.append( vs.flatten() )
        
    X = np.zeros((len(states), L**2))
    
    for i, s in enumerate(states):
        X[i, :] = s.flatten()
    
    return X, labels
    
    
"""
At each temperature T we have sampled N states. Corresp. the design matrix consists of N*T instances, where the features are flattened spin configurations(for a 7x7 grid, 49 features)

Unfortunately each simulation has its own T_KT - likely more accurate the more samples we gather.

xy_data_1.dat: on a 7x7 lattice, T_KT ~ 1.11234963 by interpolation

"""
def create_full_X(data):
    # Compute size of state - it's 2d and square
    L = len(data['states'][0][0][0])
    
    states = []
    ts = data['temperature']
    t_mask = (ts > t_kt[fname]).astype(int)
    labels = []
    
    for ei, e in enumerate(data['states']):
        for s in data['states'][ei]:
            states.append(s)
            labels.append(t_mask[ei])
    
    X = np.zeros((len(states), L**2))
    
    for i, s in enumerate(states):
        X[i, :] = s.flatten()
        
    return X, np.array(labels)

batch_size = 128
num_classes = 2
epochs = 20

# input image dimensions
img_rows, img_cols = 7, 7

# the data, split between train and test sets
# Create design matrix given energies, states and temperatures
X, y = create_reduced_X(indata)
Xf, yf = create_full_X(indata)

#ss = StandardScaler()
#X = ss.fit_transform(X)
#Xf = ss.fit_transform(Xf)

X = Xf
y = yf

print(X.shape)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = .2, shuffle = True)

print(x_train.shape)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= (2*np.pi)
x_test /= (2*np.pi)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices # this I have already taken care of, remove
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#model_s = '2xConvDropPool'
#model = get_model(model_s, input_shape)
#
#model.summary()
#
#model.compile(loss=keras.losses.binary_crossentropy,
#              optimizer=keras.optimizers.Adadelta(),
#              metrics=['accuracy'])
#
#"""
#CNN will likely overfit XY states, at least on L = 7 lattice. Hence we need early stopping.
#Patience is set to epochs such that we keep looking for the best model over all epochs.
#"""
#es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = epochs, verbose = 1)
#
## We also want to store our best model, as judged by accuracy
#mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
#
#history = model.fit(x_train, y_train,
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=1,
#          callbacks = [es, mc],
#          validation_split = 0.1)
#
#score = model.evaluate(x_test, y_test, verbose=0)
#
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
#
## Plot training history
#plt.plot(history.history['loss'], label = 'train')
#plt.plot(history.history['val_loss'], label = 'val')
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.title('Model: %s, Score: %.3f' % (model_s, score[1]))
#plt.legend()
#plt.savefig('TrainTestError_%s.png' % model_s)
#plot_model(model, to_file = 'KerasModel_%s.png' % model_s)
#plt.show()


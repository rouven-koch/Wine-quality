#------------------------------------------------------------------------------
# SOM for good or bad wine (are the good ones really good?)
#------------------------------------------------------------------------------
# General Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Sklearn libraries
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import GridSearchCV

# Keras libraries and packages
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
#from keras.layers import Dropout
#from keras.wrappers.scikit_learn import KerasClassifier
#import keras.models as km

# For SOMs
from pylab import bone, pcolor, colorbar, plot, show
from minisom import MiniSom

#------------------------------------------------------------------------------
# Preprocessing
#------------------------------------------------------------------------------
# Loading data
data = pd.read_csv('winequality-white.csv', delimiter =';')
X = data.iloc[:, 0:11].values
y = data.iloc[:, 11].values

# Good wine with rating > 5
for i in range(len(y)): 
    if y[i] > 5:
        y[i] = 1
    else:
        y[i] = 0

# Test & training set split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling
sc = StandardScaler()
X = sc.fit_transform(X)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#------------------------------------------------------------------------------
# SOM
#------------------------------------------------------------------------------
# Training the SOM
som = MiniSom(x = 10, y = 10, input_len = 11, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(1,2)], mappings[(2,3)]), axis = 0)
frauds = sc.inverse_transform(frauds)




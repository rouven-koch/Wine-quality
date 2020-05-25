#------------------------------------------------------------------------------
# Good or Bad Wine
#------------------------------------------------------------------------------
# General Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Sklearn libraries
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

# Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import keras.models as km

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
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#------------------------------------------------------------------------------
# Building the ANN
#------------------------------------------------------------------------------
# Function for ANNs
def make_my_classifier(optimizer, neurons, n_layer):
    classifier = Sequential()
    classifier.add(Dense(units = neurons, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    if n_layer > 1: 
        for i in range(n_layer - 1): #überprüfugen
            classifier.add(Dense(units = neurons, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    #print('Build ANN with '+str(neurons)+' neurons / '+str(n_layer)+' layers')
    return classifier

# Training of the ANN
classifier = make_my_classifier('rmsprop', 6, 3)
classifier.fit(X_train, y_train, batch_size = 32, epochs = 250)

#------------------------------------------------------------------------------
# Result tuning
#------------------------------------------------------------------------------
# Grid search
classifier = KerasClassifier(build_fn = make_my_classifier)
parameters = {'batch_size': [32],
              'epochs': [100],
              'optimizer': ['rmsprop'],
              'neurons': [5,6,7],
              'n_layer': [3]}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

# evaluation (k-fold crossvalidation included in grid search)
#classifier = KerasClassifier(build_fn = make_my_classifier, batch_size = 32, epochs = 250)
#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
#mean = accuracies.mean()
#variance = accuracies.std()

#------------------------------------------------------------------------------
# Predictions
#------------------------------------------------------------------------------
prediction = classifier.predict(X_test)
prediction = (prediction > 0.5)
cm = confusion_matrix(y_test, prediction)

#------------------------------------------------------------------------------
# Save and/or load model
#------------------------------------------------------------------------------
classifier.save('wine_good_or_nah.h5', overwrite=True)
km.load_model('wine_good_or_nah.h5')




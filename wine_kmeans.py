#------------------------------------------------------------------------------
# Wine classification (Random Forest)
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

# Keras libraries and packages
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
#from keras.layers import Dropout
#from keras.wrappers.scikit_learn import KerasClassifier
#import keras.models as km

#------------------------------------------------------------------------------
# Preprocessing
#------------------------------------------------------------------------------
# Loading data
data = pd.read_csv('winequality-white.csv', delimiter =';')

# Free vs. total sulfur dioxide
X = data.iloc[:, [5,6]].values

#------------------------------------------------------------------------------
# K-means(++) clustering
#------------------------------------------------------------------------------
# Using the elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)    
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of wines')
plt.xlabel('Sugar [g]')
plt.ylabel('Alcohol [%]')
plt.legend()
plt.show()
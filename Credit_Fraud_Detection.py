#####           SELF-ORGANISING_MAPS (SOM)

## Importing libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

## Importing dataset
df = pd.read_csv('Credit_Card_Applications.csv')
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

## Feature Scaling (Normalization)
from sklearn.preprocessing import MinMaxScaler
scale_X = MinMaxScaler(feature_range=(0,1))
X = scale_X.fit_transform(X)

## Training the SOM
# "MINISOM.PY" is code developed by someone that has SOM. SOM is not available in any library and hence we need to build it from scratch.
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
# x, y = dimension of the 2-D Neuron Lattice. Choice is arbitrary. 
# input_len = no. of feature in dataset-X. X contains 14+1=15.
# sigma = radius of different neighbourhoods in the grid. Its default value is 1.0
som.random_weights_init(X)  # randomly initialising the weights.
som.train_random(data=X, num_iteration=100)

## Visualising the Result
from pylab import bone, pcolor, colorbar, plot, show
bone()    
pcolor(som.distance_map().T)  
colorbar() 
markers = ['o', 's']    # '0' - circle      's'-square
colors=['r', 'g']

for i, x in enumerate(X):    
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
    
show()

## Finding the Frauds
mappings = som.win_map(X)  
frauds = mappings[(4,3)]
frauds = scale_X.inverse_transform(frauds)


    
    


































####                            HYBRID DEEP LEARNING MODEL WITH SOM

### PART-1 Identifying Frauds with Self-Organising Maps

## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset
df = pd.read_csv('Credit_Card_Applications.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

## Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

## Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

## Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
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

## Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(5,4)], mappings[(7,5)]), axis = 0)
frauds = mappings[(4,3)]
frauds = sc.inverse_transform(frauds)


### PART-2 Going from unsupervised to supervised deep learning TO PREDICT THE FRAUDs.

## Creating the matrix of features
customers = df.iloc[:,1:].values

## Creating the dependent variable 
# 0 - if there is no fraud
# 1 - if there was fraud
is_fraud = np.zeros(len(df))    # initialises a vector of the length of dataset.
for i in range(len(df)):
    # finding the customer ID in fraud array and assigning '1' to corresponding row in the is_fraud vector. 
    # this will act as label column for our dataset.
    if df.iloc[i,0] in frauds:
         is_fraud[i] = 1 

## Training our ANN
         
## Feature Scaling
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
customers = scale.fit_transform(customers)

## building the ANN
from keras.models import Sequential
from keras.layers import Dense

# initialising ANN
classifier = Sequential()
# adding first input layer and first hidden layer
classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu', input_dim=15))   
# adding output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))   
# compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# fitting the ANN to training set
classifier.fit(customers, is_fraud, batch_size=1, epochs=2)
# since the dataset is small, we would get optimum epochs after 2 or 3 epochs. but if dataset is large, we nned to train for more epochs.

# predicting the probabilities of frauds
y_pred = classifier.predict(customers)
y_pred = np.concatenate((df.iloc[:,0:1].values, y_pred), axis=1)
y_pred = y_pred[y_pred[:, 1].argsort()]  # this will sort the array based on column 1.






























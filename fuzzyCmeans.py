# Imports
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin

def FCM(data, k, r, m):
    # Run algorithm r times and pick the least MSE
    mseArray = []
    labelsArray = []
    centroidsArray = []
    newCentroids = np.zeros((k, data.shape[1]))
    # Initialize random membership weights
    weights = np.random.uniform(low=0, high=1, size=(data.shape[0], k))
    weights = np.array([[0.4, 0.6], [0.7, 0.3]])
    # Initialize with random centroids from the dataset
    # to test for convergence in the first run
    centroids = (data.sample(k))
    centroids = centroids.values
    
    data = data.values
    # Compute new centroids
    for i in range(k):
        summation = 0
        for x in range(data.shape[0]):
            summation += ((weights[x][i] ** m) * data[x])
        newCentroids[i] = summation / weights[:,i].sum()
    print("")
    
        

# Import dataset
data = pd.read_csv('cluster_dataset.txt', sep="  ", header=None)
data = np.array([[1, 2],[0, -1]])
data = pd.DataFrame(data=data)
# Run kmeans and plot
FCM(data, k=2, r=5, m=1)
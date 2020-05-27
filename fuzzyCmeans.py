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
    weights = np.zeros((data.shape[0], k))

    # Initialize with random centroids
    # centroids = (data.sample(k))
    # centroids = centroids.values
    centroids = np.random.rand(k, data.shape[1])
    df = data
    data = data.values

    while(1):

        # Compute new membership weights
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                summation = 0
                for a in range(k):
                    # If point is the centroid
                    if np.all(data[i] == centroids[j]) or np.all(data[i] == centroids[a]):
                        summation += 0.0001
                    else:
                        summation+= (np.linalg.norm(data[i]-centroids[j]) / np.linalg.norm(data[i]-centroids[a])) ** (2/ (m - 1))
                weights[i][j] = 1/summation

        # Compute new centroids
        for i in range(k):
            summation1 = np.array([0.0, 0.0])
            for x in range(data.shape[0]):
                summation1 += ((weights[x][i] ** m) * data[x])
            newCentroids[i] = summation1 / weights[:,i].sum()

        # Check for convergence
        if np.all(centroids == newCentroids):
            break
        centroids = newCentroids

    # Put labels
    labels = []
    for i in range(data.shape[0]):
        weightedDistaces = []
        for j in range(k):
            weightedDistaces.append(weights[i][j]*np.linalg.norm(data[i]-centroids[j]))
        labels.append(np.argmin(weightedDistaces))
    print("")
    # Plot data clustered and centroids in red
    #plt.title(mse)
    plt.scatter(df[0],df[1],c=labels)
    plt.scatter(centroids[:, 0],centroids[:, 1],c='red')
    plt.xlabel('Col0')
    plt.ylabel('Col1')
    plt.show()
        

# Import dataset
data = pd.read_csv('cluster_dataset.txt', sep="  ", header=None)

# Run kmeans and plot
FCM(data, k=5, r=5, m=2)
# Imports
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin

def MSE(data, labels, centroids, k):

    # Calculate mean squared error
    summation = 0
    for i in range(k):
        diff = (data[i] - centroids[labels[i]])**2
        summation += diff
    return summation

def k_means(data, k):
    # Initialize with random centroids from the dataset
    centroids = (data.sample(k))
    centroids = centroids.values

    while(1):
        # Assign points to cluster based on L2 distance
        labels = pairwise_distances_argmin(data, centroids, metric='euclidean')

        # Calculate new centroids for the data
        newCentroids = np.array([data[labels == i].mean(0)
                                    for i in range(k)])

        # Check if converged
        if np.all(centroids == newCentroids):
            break
        centroids = newCentroids
    
    # Calculate mean squared errors
    mse = "MSE: " + str(MSE(data.values, labels, centroids, k))    

    # Plot data clustered and centroids in red
    plt.title(mse)
    plt.scatter(data[0],data[1],c=labels)
    plt.scatter(centroids[:, 0],centroids[:, 1],c='red')
    plt.xlabel('Col0')
    plt.ylabel('Col1')
    plt.show()

# Import dataset
data = pd.read_csv('cluster_dataset.txt', sep="  ", header=None)

k_means(data, k=3)
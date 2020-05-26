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
    return (summation[0] + summation[1])**2

def k_means(data, k, r):
    # Run algorithm r times and pick the least MSE
    mseArray = []
    labelsArray = []
    centroidsArray = []
    for a in range(r):
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
        # Push data of the rth run into the arrays
        mseArray.append(MSE(data.values, labels, centroids, k))
        labelsArray.append(labels)
        centroidsArray.append(centroids)

    # Get index of the minimum MSE
    minIndex = mseArray.index(min(mseArray))

    # Use labels and centroids of minimum MSE
    mse = "MSE: " + str(mseArray[minIndex])    
    centroids = centroidsArray[minIndex]
    labels = labelsArray[minIndex]

    # Plot data clustered and centroids in red
    plt.title(mse)
    plt.scatter(data[0],data[1],c=labels)
    plt.scatter(centroids[:, 0],centroids[:, 1],c='red')
    plt.xlabel('Col0')
    plt.ylabel('Col1')
    plt.show()

# Import dataset
data = pd.read_csv('cluster_dataset.txt', sep="  ", header=None)

k_means(data, k=5, r=5)
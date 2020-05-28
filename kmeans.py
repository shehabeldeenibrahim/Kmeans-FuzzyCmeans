# Imports
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin
import math

def SSE(data, labels, centroids, k):

    # Calculate mean squared error
    summation = 0
    for i in range(k):
        diff = (data[i] - centroids[labels[i]])**2
        summation += diff
    return math.sqrt((summation[0]**2 + summation[1]**2))

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
        mseArray.append(SSE(data.values, labels, centroids, k))
        labelsArray.append(labels)
        centroidsArray.append(centroids)

    # Get index of the minimum MSE
    minIndex = mseArray.index(min(mseArray))

    # Use labels and centroids of minimum MSE
    mse = "SSE: " + str(mseArray[minIndex])    
    centroids = centroidsArray[minIndex]
    labels = labelsArray[minIndex]

    # Plot data clustered and centroids in red
    plt.title(mse)
    plt.scatter(data[0],data[1],c=labels)
    plt.scatter(centroids[:, 0],centroids[:, 1],c='red')
    plt.xlabel('Col0')
    plt.ylabel('Col1')
    plt.savefig('KM_Results/'+ str(k) + '.png')
    #plt.show()

# Import dataset
data = pd.read_csv('cluster_dataset.txt', sep="  ", header=None)

# Run kmeans and plot
i = 2
while(i < 6):
    k_means(data, k=i, r=5)
    i+=1
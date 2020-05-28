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


def FCM(data, k, r, m):
    # Run algorithm r times and pick the least MSE
    mseArray = []
    labelsArray = []
    centroidsArray = []
    newCentroids = np.zeros((k, data.shape[1]))
    weights = np.zeros((data.shape[0], k))

    # Initialize with random centroids
    centroids = np.random.rand(k, data.shape[1])
    df = data
    data = data.values
    
    for a in range(r):
        while(1):

            # Compute new membership weights
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    summation = 0
                    for a in range(k):
                        summation+= (np.linalg.norm(data[i]-centroids[j]) / np.linalg.norm(data[i]-centroids[a])) ** (2/ (m - 1))
                    weights[i][j] = 1/summation

            # Compute new centroids
            for i in range(k):
                summation1 = 0
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
                weightedDistaces.append(np.linalg.norm(data[i]-centroids[j]))
            labels.append(np.argmin(weightedDistaces))

        # Calculate mean squared errors
        # Push data of the rth run into the arrays
        mseArray.append(SSE(data, labels, centroids, k))
        labelsArray.append(labels)
        centroidsArray.append(centroids)

    # Get index of the minimum MSE
    minIndex = mseArray.index(min(mseArray))

    # Use labels and centroids of minimum MSE
    mse = "SSE: " + str(mseArray[minIndex])    
    centroids = centroidsArray[minIndex]
    labels = labelsArray[minIndex]

    print("SSE: " + str(mse))

    # Plot data clustered and centroids in red
    plt.title(mse)
    plt.scatter(df[0],df[1],c=labels)
    plt.scatter(centroids[:, 0],centroids[:, 1],c='red', marker='x')
    plt.xlabel('Col0')
    plt.ylabel('Col1')
    plt.savefig('FCM_Results/'+ str(k) + '.png')
    #plt.show()
        

# Import dataset
data = pd.read_csv('cluster_dataset.txt', sep="  ", header=None)

# Run kmeans and plot
i = 2

print("###FCM Clustering started###")
while(i < 6):
    print("k = " + str(i))
    FCM(data, k=i, r=5, m=1.001)
    i+=1


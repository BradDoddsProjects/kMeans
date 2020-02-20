import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# INITIALIZATION number of klusters and initial centroids
K = 3
convergeTotal = 10
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

df = pd.read_csv('file.csv', delimiter=',', header=None, names=['A', 'B'])

X = np.array(df)
#print('Printing data in X...')
# print(X)

print('Dimensions of X', X.shape)

num_points = len(X)
print("Total number of points in dataset, ie. X:", num_points)

# PLOT: original data set with initial centroids
plt.plot(X[:, 0], X[:, 1], 'go')
plt.plot(initial_centroids[:, 0], initial_centroids[:, 1], 'rx')
plt.show()

# FIRST STEP: CLUSTER ASSIGNMENT
dist = np.zeros((K, num_points))
distanceValues = np.zeros((num_points))


# gives x value X[:, 0]
# gives y value X[:, 1]

def distanceBetPointsAndCentroids(centroids):
    for i in range(K):
        for j in range(num_points):
            dist[i, j] = np.linalg.norm(centroids[i] - X[j])


def shortestDistance():
    a = []
    b = []
    c = []

    for i in range(num_points):

        num1 = dist[0, i]
        num2 = dist[1, i]
        num3 = dist[2, i]

        if (num1 < num2) and (num1 < num3):
            smallest_num = num1
            a.append(X[i])

        elif (num2 < num1) and (num2 < num3):
            smallest_num = num2
            b.append(X[i])
        else:
            smallest_num = num3
            c.append(X[i])
        # print("The smallest of the 3 numbers is : ", smallest_num)

    # point x.y values for shortest distance
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    plt.scatter(a[:, 0], a[:, 1], color='c')
    plt.scatter(b[:, 0], b[:, 1], color='y')
    plt.scatter(c[:, 0], c[:, 1], color='b')

    print("New Centroid Values")
    newCentroids = []
    newCentroids.append(centroids(a))
    newCentroids.append(centroids(b))
    newCentroids.append(centroids(c))
    print('\n')

    centroidArray = np.array(newCentroids)



    # Comment out for no plots
    plt.plot(centroidArray[:, 0], centroidArray[:, 1], 'rx')
    plt.show()

    return centroidArray


def centroids(points):

        n = len(points)

        x = points[:, 0]
        y = points[:, 1]

        x = sum(x) / n
        y = sum(y) / n

        centroid = np.zeros(2)
        centroid[0] = x
        centroid[1] = y

        print(centroid)

        return centroid



distanceBetPointsAndCentroids(initial_centroids)
new_centroid = shortestDistance()

for i in range(10):
    distanceBetPointsAndCentroids(new_centroid)
    new_centroid = shortestDistance()

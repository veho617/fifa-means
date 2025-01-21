#K-means clustering from scratch
#dataset is about fifa players

#1)import data
#2)scale it
#3)initialize centroids
#4)assign data to centroids
#5)update centroids
#6)repeat 3 and 4 until no change


import numpy as np
import pandas as pd

#importing the data
master_data = pd.read_csv("Clustering/fifa_players.csv")

features = ["age","overall_rating","potential","value_euro","wage_euro",]
master_data = master_data.dropna(subset = features)

data = master_data[features].copy()

#scaling the data on a 1-10 scale MinMaxScale
data = ((data-data.min())/(data.max()-data.min()))*9+1



#intializing random centroids

def random_centroids(data,k):
    centroids = []
    for i in range(k):
        centroid = data.apply(lambda x: float(x.sample())) #genrates a random sample row for your centroid based on the data
        centroids.append(centroid)
    return pd.concat(centroids, axis=1) #makes the panda series centroids into a dataframe for consistency when comparing

def assign_labels(data,centroids):
    distances = centroids.apply(lambda x: np.sqrt(((data - x)**2).sum(axis = 1)))
    return distances.idxmin(axis=1)


def update_centroids(data,labels):
    return data.groupby(labels).apply(lambda x:np.exp(np.log(x).mean())).T


#for plotting and visualizing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA #to turn our 5D data into 2D data for better visualization

def plot_clusters(data, labels, centroids, iteration):
    pca_model = PCA(n_components=2)
    data_2d = pca_model.fit_transform(data)
    centroids_2d = pca_model.transform(centroids.T)
    plt.clf()
    plt.title(f'Iteration {iteration}')
    plt.scatter(x = data_2d[:,0], y =data_2d[:,1],c = labels)
    plt.scatter(x = centroids_2d[:,0], y = centroids_2d[:, 1], c = "black")
    plt.show()
    plt.pause(0.001)


#the actual algorithm
max_iterations = 100
k = int(input("How many clusters do you want? "))

centroids = random_centroids(data,k) #starting with random centroids
old_centroids = pd.DataFrame() #intializing a data frame to store our previous iteration to check later
iteration = 1

plt.ion()


while iteration < max_iterations and not centroids.equals(old_centroids):
    old_centroids = centroids

    labels = assign_labels(data,centroids)
    centroids = update_centroids(data,labels)
    plot_clusters(data,labels,centroids,iteration)
    iteration += 1

plt.ioff()
plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
import matplotlib.colors as mcolors

parser = argparse.ArgumentParser(description='Gaussian mixture and matplotlib sample script')
parser.add_argument(
	'-d', '--data',
	type = str,
	dest = 'data_file',
    required = True, 
    help = 'data file'
)

parser.add_argument(
    '-c', '--component',
    type = int,
    dest = 'num_component',
    required = False, 
    default = 10,
    help = 'The number of Gaussian components'
)
args = parser.parse_args()
data_file = args.data_file
num_component = args.num_component


#csv file 
data = pd.read_csv(data_file, index_col=0, delimiter=',', usecols=range(0, 3))
data = data.to_numpy()


#fit a Gaussian mixture model
max_iter = 5000
gmm = GaussianMixture(n_components=num_component, max_iter=max_iter, init_params='kmeans', random_state=2)
gmm.fit(data)

#predict the cluster assignments
prediction = gmm.predict(data)
print(prediction)


# Calculate the distance from each data point to each cluster center
cluster_centers = gmm.means_
distances = np.zeros((len(data), num_component))
for i in range(num_component):
    mean = cluster_centers[i,:]
    distances[:, i] = np.linalg.norm(data - mean, axis=1)

# Find the closest cluster center for each data point
closest_cluster = np.argmin(distances, axis=1)

# Find the 2nd closest cluster center for each data point
distances_copy = distances.copy()  # Make a copy of distances
for i in range(len(distances_copy)):
    closest_ind = closest_cluster[i]  # Indices of nearest clusters
    distances_copy[i, closest_ind] = np.inf  # Set distance of nearest cluster to infinity
second_closest_cluster = np.argmin(distances_copy, axis=1)  # Calculate index of 2nd nearest cluster


# Output the closest data point for each cluster center
print("Closest data points for each cluster center:")
for i in range(num_component):
    cluster_center = cluster_centers[i]
    closest_index = np.argmin(distances[:, i])
    closest_point = data[closest_index]
    second_closest_index = np.argmin(distances_copy[:, i])
    second_closest_point = data[second_closest_index]
    print("Cluster center:", cluster_center, "Closest data point:", closest_point, "Second closest data point:", second_closest_point)


# Pick up most and 2nd nearest data from cluster centers
cluster_center_indices = []
second_cluster_center_indices = []
for i in range(num_component):
    cluster_center = gmm.means_[i]
    distances = np.linalg.norm(data - cluster_center, axis=1)
    closest_point_index = np.argmin(distances)
    cluster_center_indices.append(closest_point_index)
    distances[closest_point_index] = np.inf  # To remove nearest data
    second_closest_point_index = np.argmin(distances)  # Find the 2nd closest cluster center indices
    second_cluster_center_indices.append(second_closest_point_index)



# Nearest data from cluster centers
closest_points = data[cluster_center_indices]

# 2nd nearest data from cluster centers
second_closest_points = data[second_cluster_center_indices]

# Get the index from the original pandas dataframe
index = pd.read_csv(data_file, index_col=0, delimiter=',').index[cluster_center_indices]
second_index = pd.read_csv(data_file, index_col=0, delimiter=',').index[second_cluster_center_indices]

# Output the closest data point for each cluster center, with index
closest_points_df = pd.DataFrame(closest_points, index=index)
closest_points_df.to_csv("nearest_data1.csv")

# Output the closest data point for each cluster center, with index
second_closest_points_df = pd.DataFrame(second_closest_points, index=second_index)
second_closest_points_df.to_csv("nearest_data2.csv")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pylab as plt
import sklearn

# Set seed for reproducibility
np.random.seed(55)

# Description of the code
print("This code performs K-means clustering on a given dataset and determines the optimal number of clusters using the silhouette score.")

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    '-d', '--data',
    type=str,
    dest='data_file',
    required=True,
    help='data file'
)
parser.add_argument(
    '-c', '--cluster',
    type=int,
    dest='num_cluster',
    required=False,
    default=10,
    help='The number of clusters'
)
args = parser.parse_args()
data_file = args.data_file
num_cluster = args.num_cluster

#2023/12/20 show version
print("sklearn_version=" + sklearn.__version__)
print("numpy_version=" + np.__version__)
print("pandas_version=" + pd.__version__)
print("matplot_version=" + plt.__version__)
print(f"Data file: {data_file}")

# Data loading
df_data = pd.read_csv(data_file, index_col=0, delimiter=',')

# Empty list to store silhouette scores
silhouette_scores = []

# Range of cluster numbers
for i in range(3, num_cluster):
    sil_scores = []
    for j in range(10):
        max_iter = 5000
        km = KMeans(n_clusters=i, max_iter=max_iter, random_state=j)
        clusters_sklearn = km.fit_predict(df_data)
        silhouette = silhouette_score(df_data, km.labels_)
        sil_scores.append(silhouette)
    sil_mean = np.mean(sil_scores)
    silhouette_scores.append(sil_mean)
    print(f"Number of clusters: {i}, Average silhouette score: {sil_mean:.4f}")

# Plot the silhouette scores
plt.plot(range(3, num_cluster), silhouette_scores)
plt.xlabel('Number of clusters')
plt.ylabel('Average silhouette score')
plt.show()

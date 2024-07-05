# Clustering and Finding Data Closest to Cluster Center Protocol for "Predicting Pathogenicity of Klebsiella pneumoniae through Holistic Gene Content Analysis"

## Overview

This protocol outlines the steps to cluster data and identify the data points closest to the cluster centers for predicting the pathogenicity of *Klebsiella pneumoniae*. The process involves preparing input data, trimming and plotting data, estimating the number of clusters, and finding the nearest data to cluster centers.

## Steps

### 1. Prepare Input Data
Use `binary_seq.py` from the [OSNAp repository](https://github.com/suzukimasahiro/OSNAp) with the `--mapping` option.

### 2. Data Trimming and Drawing PCA Plot
Use `data_trim_plot.py` to trim the data and draw a PCA plot.

For this example, `binary_seq.csv` was pretrimmed to remove unnecessary columns.

```sh
data_trim_plot.py -d binary_seq.csv -o binary_seq_trimmed --max 0.9 --min 0.1
```

If using the output file from `binary_seq.py` with the `--mapping` option, use the `-m` option:

```sh
data_trim_plot.py -m binary_seq_mapping.txt -o binary_seq_trimmed --max 0.9 --min 0.1
```

### 3. Estimate the Number of Clusters
Use `silhouette_average.py` to estimate the optimal number of clusters.

```sh
silhouette_average.py -d binary_seq_trimmed_PCA.csv -c 50 > silhouette_PCA.txt
```

### 4. Find Data Nearest to Cluster Centers
Use `nearest_data.py` to find the data points nearest to the cluster centers.

```sh
nearest_data.py -d binary_seq_trimmed_PCA.csv -c 18
```

### Drawing Figure 1
Use `fig1.py` to draw Figure 1.

```sh
fig1.py -d binary_seq_trimmed.csv -l fig1_cluster.csv
```

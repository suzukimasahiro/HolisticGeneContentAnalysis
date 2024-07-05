#!/usr/bin/python3
# -*- coding:utf-8 -*-

version = "Figure drawing only"

import argparse
import numpy as np
import pandas as pd
import csv
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import subprocess

parser = argparse.ArgumentParser(description='Plot 2D map for binary sequence')
parser.add_argument(
	'-d', '--data',
	type = str,
	dest = 'dataset',
    required = False
)
parser.add_argument(
    '-l', '--label',
    type = str,
    dest = 'label_file',
    required = False, 
    help = 'data label'
)
parser.add_argument(
	'--datalabel',
	dest = 'datalabel',
	action="store_true",
	default = False
)
parser.add_argument(
    '--max',
    type = float,
    dest = 'max_freq',
    help = 'maximum frequency of orf',
    default = 0.95
)
parser.add_argument(
    '--min',
    type = float,
    dest = 'min_freq',
    help = 'minimum frequency of orf',
    default = 0.05
)

args = parser.parse_args()

data_file = args.dataset
label_file = args.label_file
datalabel = args.datalabel
maxval = args.max_freq
minval = args.min_freq

##### Load files #####
if data_file:
	df_data = pd.read_csv(data_file, index_col=0)
else:
	print('--data (-d) or --binseqmap (-m) is required')
	sys.exit()
input_size = df_data.shape


##### Data trimming #####
f_lower = df_data.shape[0] * minval
f_upper = df_data.shape[0] * maxval
sr_sum = df_data.sum(axis=0, numeric_only=True)
drop_list = []
for index, value in sr_sum.items():
	if value < f_lower or value > f_upper:
		drop_list.append(index)
		#print(index, value)
df_data = df_data.drop(columns=drop_list)


dataindex_list = df_data.index.tolist()
print(df_data)

model = PCA(n_components=4)
model.fit(df_data)
data_pca = model.transform(df_data)
ratio1 = round(model.explained_variance_ratio_[0], 5)
ratio2 = round(model.explained_variance_ratio_[1], 5)
ratio3 = round(model.explained_variance_ratio_[2], 5)
ratio4 = round(model.explained_variance_ratio_[3], 5)
print(ratio1, ratio2, ratio3, ratio4)

color_map = ('tan','salmon','silver','gold','cyan','lime','plum','teal','turquoise','magenta','lightblue',
			'tomato','olive','aquamarine','wheat','khaki','pink','yellowgreen','black')
fig, ax = plt.subplots()

title = 'ml_dataplot.py'

#ax.set_title(title)
ax.set_xlabel('PC1 : ' + str(ratio1))
ax.set_ylabel('PC2 : ' + str(ratio2))

minor_g = (1,3,6,9,11,16)
gmellonella = (9, 11, 12, 13, 24, 26, 29, 39, 43, 51, 60, 
				61, 65, 78, 79, 83, 87, 93, 97, 105, 108, 
				110, 111, 114, 115, 117, 119, 120, 122, 
				123, 135, 142, 147, 160, 163, 164)
if label_file:
	df_label = pd.read_csv(label_file, index_col=0)
	print(df_label)
	labels = df_label['label'].values
	# Plot using colormap
	for i, row in enumerate(data_pca):
		symbol = 'o'
		marker_size = 25
		if labels[i] in minor_g:
			symbol = '^' 
			marker_size = 25
		ax.scatter(row[0], row[1], marker = symbol, color=color_map[labels[i]], s=marker_size)
		if i in gmellonella:
			symbol = 'x' 
			marker_size = 10
			ax.scatter(row[0], row[1], marker = symbol, color='black', s=marker_size)

else:
	for i, row in enumerate(data_pca):
		index = dataindex_list[i]
		ax.scatter(row[0], row[1], marker = '.', color='blue') 
		if datalabel:
			ax.text(row[0], row[1], index, fontsize=5) 

plt.show()


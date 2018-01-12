# SHefali Umrania
# 02-750 Fall 2017
# Final Project

import pandas as pd
import numpy as np
import random


# read file 
def load_data(filename):
	df = pd.read_csv(filename)

	size = df.shape[1] - 1
	x = df.ix[:, 0:size-1]
	y = df.ix[:, size]
	return x, y

# create a dictionary of labels
def load_labels(y):
	labels = {}
	ylist = list(set(y))
	sorted_list = sorted(ylist)
	for i in range(len(sorted_list)):
		labels[i] = sorted_list[i]
	return labels

# separate data into labeled and unlabeled pools
def pool_data(pool_size, xtrain, ytrain):

	xtrain_range = range(0, xtrain.shape[0])
	idx = random.sample(xtrain_range, pool_size)

	pool = xtrain.index.isin(idx)

	xtrain_label = np.array(xtrain[pool])
	xtrain_nolabel = np.array(xtrain[~pool])

	ytrain_label = np.array(ytrain[pool])
	ytrain_nolabel = np.array(ytrain[~pool])

	return xtrain_label, xtrain_nolabel, ytrain_label, ytrain_nolabel

# SHefali Umrania
# 02-750 Fall 2017
# Final Project

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import data_loader as dl
import heapq
import numpy as np
import random


def classify(xtrain_label, ytrain_label, xtrain_nolabel):

	classifier = OneVsRestClassifier(SVC(random_state=0))
	classifier.fit(xtrain_label, ytrain_label)

	idx = list(xrange(xtrain_label.shape[1]))
	dist = classifier.decision_function(xtrain_nolabel[:, idx])	
	return dist, classifier, idx

def find_query(dist, label_dict, bsize):

	qidx = []
	distances = []

	for i in range(len(dist)):
		row = np.array(dist[i])
		# choose maximum distance
		max_idx = np.argmax(row) 

		label = label_dict[max_idx]
		sorted_row = np.sort(row)[::-1]
		diff = np.abs(sorted_row[0] - sorted_row[1])
		heapq.heappush(distances, (diff, i, label))

	# min dist values
	min_dist = heapq.nsmallest(bsize, distances)
	for x in min_dist:
		qidx.append(x[1])

	return qidx

def active_update(xtrain_label, xtrain_nolabel, ytrain_label, ytrain_nolabel, qidx):

	# Update labeled and unlabeled xtrain with queried data
	xtrain_queried = xtrain_nolabel[qidx]
	xtrain_label = np.vstack((xtrain_label, xtrain_queried))
	xtrain_nolabel = np.delete(xtrain_nolabel, qidx, 0)
	
	# Update labeled and unlabeled ytrain with queried labels
	ytrain_queried = ytrain_nolabel[qidx] 
	ytrain_label = np.concatenate((ytrain_label, ytrain_queried))
	ytrain_nolabel = np.delete(ytrain_nolabel, qidx, None)

	return xtrain_label, xtrain_nolabel, ytrain_label, ytrain_nolabel

def random_update(xtrain_label, xtrain_nolabel, ytrain_label, ytrain_nolabel, bsize):

	rand_range = range(0, xtrain_nolabel.shape[0])
	rand_idx = random.sample(rand_range, bsize)

	# Update labeled and unlabeled xtrain with random data
	xtrain_random = xtrain_nolabel[rand_idx]
	xtrain_label = np.vstack((xtrain_label, xtrain_random))
	xtrain_nolabel = np.delete(xtrain_nolabel, rand_idx, 0)

	# Update labeled and unlabeled ytrain with random labels
	ytrain_random = ytrain_nolabel[rand_idx]
	ytrain_label = np.concatenate((ytrain_label, ytrain_random))
	ytrain_nolabel = np.delete(ytrain_nolabel, rand_idx, None)

	return xtrain_label, xtrain_nolabel, ytrain_label, ytrain_nolabel

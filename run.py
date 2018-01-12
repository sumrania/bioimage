# SHefali Umrania
# 02-750 Fall 2017
# Final Project

import data_loader as dl
import learner
from sklearn import metrics
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# params
pool_size = 100
bsize = 50
budget = 2500

# load data
xtrain, ytrain = dl.load_data('Data/DIFFICULT_TRAIN.csv')
xtest, ytest = dl.load_data('Data/DIFFICULT_TEST.csv')
xtest = np.array(xtest)

xtrain_label, xtrain_nolabel, ytrain_label, ytrain_nolabel = dl.pool_data(pool_size, xtrain, ytrain)
xtrain_label_rand, xtrain_nolabel_rand, ytrain_label_rand, ytrain_nolabel_rand = dl.pool_data(pool_size, xtrain, ytrain)

active_errors=[]
random_errors=[]
batches = []

min_error = 1
min_cost = 0
best_model = None

cost = pool_size
while cost < budget:

	# run active learner
	dist, classifier, idx = learner.classify(xtrain_label, ytrain_label, xtrain_nolabel) 
	label_dict = dl.load_labels(ytrain)
	idx_query = learner.find_query(dist, label_dict, bsize)
	xtrain_label, xtrain_nolabel, ytrain_label, ytrain_nolabel = learner.active_update(xtrain_label, xtrain_nolabel, ytrain_label, ytrain_nolabel, idx_query)

	# predict
	active_pred = classifier.predict(xtest[:,idx])
	active_error = 1.0 - metrics.accuracy_score(ytest, active_pred)
	active_errors.append(active_error)

	cost += bsize
	batches.append(cost)

	# save best model
	if active_error < min_error:
		min_cost = cost
		min_error = active_error
		best_model = deepcopy(classifier)

	# run random learner
	_, classifier_rand, idx_rand = learner.classify(xtrain_label_rand, ytrain_label_rand, xtrain_nolabel_rand) 
	xtrain_label_rand, xtrain_nolabel_rand, ytrain_label_rand, ytrain_nolabel_rand = learner.random_update(xtrain_label_rand, xtrain_nolabel_rand, ytrain_label_rand, ytrain_nolabel_rand, bsize)

	random_pred = classifier_rand.predict(xtest[:,idx_rand])
	random_error = 1.0 - metrics.accuracy_score(ytest, random_pred)
	random_errors.append(random_error)

	print "Batch:", cost, "  Active Error:", active_error, " Random Error:", random_error

# Plot
plt.plot(batches, active_errors, 'r-', label='Active Learner')
plt.plot(batches, random_errors, 'y-', label='Random Learner')
plt.title('Test Error v/s Budget')
plt.xlabel('Budget')
plt.ylabel('Error on test set')
plt.legend(loc=1)
plt.show()

# Blinded Prediction
df = pd.read_csv('Data/DIFFICULT_BLINDED.csv')
size = df.shape[1] - 1

xblind = np.array(df.ix[:, 1:size])

ypred = best_model.predict(xblind[:, idx])
instance_id = df.ix[:, 0]

save_df = pd.DataFrame({'instance_id':instance_id, 'prediction':ypred})
save_df.to_csv('blinded_pred.csv', header=False, index=False)






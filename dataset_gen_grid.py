#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:50:29 2017

@author: apple
"""

import numpy as np
import sys

#building Tree-Like MRF with inputsize number of variables
height = int(sys.argv[1])
width = int(sys.argv[2])
inputsize = height*width

num_of_train_samples = int(sys.argv[3])
num_of_valid_samples = int(sys.argv[4])
num_of_test_samples = int(sys.argv[5])
#for a very elemntry testing i considered graph shape like this:
#			*
#			|
#			*
#		   / \
#		  *   *
#
# Here i'm trying to make samples as data set relevant to random parameters
# for this tree

#adj = np.zeros([inputsize, inputsize])
#for r in range(0, height):
#    for c in range(0, width):
#        jj = r*width + c
#        if c > 0:
#            param = np.random.sample(1)
#            adj[jj-1][jj] = adj[jj][jj-1] = param
#        if r > 0:
#            param = np.random.sample(1)
#            adj[jj-width][jj] = adj[jj][jj-width] = param

with np.load('datasets/parameters.npz') as parameters:
    adj = parameters['parameter']
    
all_outcomes = np.ndarray(shape=(2**inputsize, inputsize), dtype=np.float32)
prob_of_outcomes = np.ndarray(shape=(2**inputsize), dtype=np.float32)
            
for i in range(2**inputsize):
    str_samp = ('{0:0' + str(height*width) + 'b}').format(i)
    asarr_samp = [int(d) for d in str_samp]
    all_outcomes[i][:] = asarr_samp
    sum_prod = 0
    for r in range(height):
        for c in range(width):
            jj = r*width + c
            if (c > 0):
                sum_prod += all_outcomes[i][jj]*all_outcomes[i][jj-1]*adj[jj][jj-1]
            if (r > 0):
                sum_prod += all_outcomes[i][jj]*all_outcomes[i][jj-width]*adj[jj][jj-width]
    p = np.exp(sum_prod)
    prob_of_outcomes[i] = p

sum_prob = sum(prob_of_outcomes)
prob_of_outcomes = np.divide(prob_of_outcomes, sum_prob)

cum_probs = []
s = 0
for x in prob_of_outcomes:
	s = s + x
	cum_probs.append(s)

train_data = np.ndarray(shape=(num_of_train_samples, inputsize), dtype=np.float32)
train_data_probs = np.ndarray(shape=(num_of_train_samples), dtype=np.float32)
for x in range(num_of_train_samples):
    p = np.random.uniform(0,1)
    i = np.searchsorted(cum_probs, p)
    train_data[x][:] = all_outcomes[i]
    train_data_probs[x] = prob_of_outcomes[i]


valid_data = np.ndarray(shape=(num_of_valid_samples, inputsize), dtype=np.float32)
valid_data_probs = np.ndarray(shape=(num_of_valid_samples), dtype=np.float32)
for x in range(num_of_valid_samples):
    p = np.random.uniform(0,1)
    i = np.searchsorted(cum_probs, p)
    valid_data[x][:] = all_outcomes[i]
    valid_data_probs[x] = prob_of_outcomes[i]

test_data = np.ndarray(shape=(num_of_test_samples, inputsize), dtype=np.float32)
test_data_probs = np.ndarray(shape=(num_of_test_samples), dtype=np.float32)
for x in range(num_of_test_samples):
    p = np.random.uniform(0,1)
    i = np.searchsorted(cum_probs, p)
    test_data[x][:] = all_outcomes[i]
    test_data_probs[x] = prob_of_outcomes[i]
#test_data = all_outcomes[prob_of_outcomes > 0][:]
#test_data_probs = prob_of_outcomes[prob_of_outcomes > 0]

file_name = 'datasets/grid_' + str(height) + 'x' + str(width) + '_' + str(num_of_train_samples) + str(num_of_valid_samples) + str(num_of_test_samples) + '.npz'
np.savez(file_name, 
         height=height,
         width=width,
         train_length=num_of_train_samples,
         train_data=train_data, 
         train_data_probs = train_data_probs, 
         valid_length=num_of_valid_samples, 
         valid_data=valid_data,
         valid_data_probs=valid_data_probs,
         test_length=num_of_test_samples, #test_data.shape[0]
         test_data=test_data,
         test_data_probs=test_data_probs,
         params=adj)

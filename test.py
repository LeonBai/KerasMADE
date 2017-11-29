#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 18:21:53 2017

@author: apple
"""
import numpy as np

height = 4
parameters = 0*np.random.sample(height-1)
inputsize = 4

all_outcomes = np.ndarray(shape=(2**inputsize, inputsize), dtype=np.float32)
prob_of_outcomes = np.ndarray(shape=(2**inputsize), dtype=np.float32)
            
for i in range(2**inputsize):
    str_samp = ('{0:0' + str(height) + 'b}').format(i)
    asarr_samp = [int(d) for d in str_samp]
    all_outcomes[i][:] = asarr_samp
    sum_prod = 0
    for j in range(height-1):
        sum_prod = sum_prod + all_outcomes[i][j]*parameters[j]*all_outcomes[i][j+1]
        
    p = np.exp(sum_prod)
    prob_of_outcomes[i] = p

sum_prob = np.sum(prob_of_outcomes)
prob_of_outcomes = np.divide(prob_of_outcomes, sum_prob)

cum_probs = []
s = 0
for x in prob_of_outcomes:
    s = s + x
    cum_probs.append(s)
        
num_of_train_samples = train_length = 10000
num_of_valid_samples = valid_length = 10000
num_of_test_samples = test_length = 16
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

#    test_data = np.ndarray(shape=(num_of_test_samples, inputsize), dtype=np.float32)
#    test_data_probs = np.ndarray(shape=(num_of_test_samples), dtype=np.float32)
#    for x in range(num_of_test_samples):
#        p = np.random.uniform(0,1)
#        i = np.searchsorted(cum_probs, p)
#        test_data[x][:] = all_outcomes[i]
#        test_data_probs[x] = prob_of_outcomes[i]
test_data = all_outcomes[prob_of_outcomes > 0][:]
test_data_probs = prob_of_outcomes[prob_of_outcomes > 0]


tprobs = np.zeros([height, 2])
tprobs[0][1] = np.sum(train_data, 0)[0]/train_length
tprobs[0][0] = 1-tprobs[0][1]

for i in range(1, height):
    tprobs[i][0] = np.sum(train_data[:,i]*(-1*train_data[:,i-1]+np.ones(train_length)))/(train_length - np.sum(train_data, 0)[i-1])
    tprobs[i][1] = np.sum(train_data[:,i]*train_data[:,i-1])/np.sum(train_data, 0)[i-1]
absolute_probs = np.ones([test_length, 1])   
for i in range(test_length):
    absolute_probs[i] = tprobs[0][int(test_data[i][0])]
    for j in range(1, height):
        if (test_data[i][j] == 1):
            absolute_probs[i] = absolute_probs[i]*tprobs[j][int(test_data[i][j-1])]
        else:
            absolute_probs[i] = absolute_probs[i]*(1-tprobs[j][int(test_data[i][j-1])])
            
abs_KL = -1*np.sum(np.multiply(test_data_probs, (np.log(absolute_probs) - np.log(test_data_probs))))
print('kl', abs_KL)
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 01:48:37 2017

@author: apple
"""
import numpy as np

height = 4
width = 4
inputsize = height*width

adj = np.zeros([inputsize, inputsize])
for r in range(0, height):
    for c in range(0, width):
        jj = r*width + c
        if c > 0:
            param = np.random.sample(1)
            adj[jj-1][jj] = adj[jj][jj-1] = param
        if r > 0:
            param = np.random.sample(1)
            adj[jj-width][jj] = adj[jj][jj-width] = param
            
np.savez('grid_parameters.npz',
         parameter=adj
         )

#params = 0*np.random.sample(height-1)
#np.savez('line_parameters.npz', parameter=params)
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from keras import layers
from keras.engine.topology import Layer
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
from keras import activations

class MaskedDenseLayer(Layer):

    def __init__(self, units, mask, **kwargs):
        #self.output_dim = output_dim
        super(MaskedDenseLayer, self).__init__(**kwargs)
        self.units = units
        #self.activation = activations.get(activation)
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MaskedDenseLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask):
        return K.dot(K.dot(x, self.kernel), mask)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

with np.load('datasets/simple_tree.npz') as dataset:
    inputsize = dataset['inputsize']
    train_length = dataset['train_length']
    train_data = dataset['train_data']
    valid_length = dataset['valid_length']
    valid_data = dataset['valid_data']
    test_length = dataset['test_length']
    test_data = dataset['test_data']
    params = dataset['params']
    
input_ds = Input(shape=(4,))
input_layer = MaskedLayer(4)#, activation='relu')
encoded1 = MaskedLayer(3)#, activation='relu')
decoded1 = MaskedLayer(3)#, activation='relu')
output_layer = MaskedLayer(4)#, activation='sigmoid')


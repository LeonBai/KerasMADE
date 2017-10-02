# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from keras import layers
from keras.engine.topology import Layer
from keras.callbacks import Callback
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
from keras import activations

state = np.random.randint(1,20)
all_masks = []

num_of_all_masks = 20
num_of_hlayer = 5
graph_size = 4
hlayer_size = 3

#masking lambdaCallback
class ReassignMask(Callback):
    def on_epoch_end(self, logs={}):
        global state 
        state = np.random.randint(1,20)

for i in range(0,20):
    #generating subsets as 3d matrix 
    subsets = np.random.randint(0,2,(num_of_hlayer,hlayer_size,graph_size))
    
    #generating masks as 3d matrix
    #masks = np.zeros([num_of_hlayer,hlayer_size,hlayer_size])
    masks = []
    
    #first layer mask
    mask = np.zeros([graph_size, hlayer_size])
    first_sets = np.eye(graph_size, dtype=int)
    for j in range(0, hlayer_size):
        for k in range(0, graph_size):
            if all( (subsets[0][j] - first_sets[k]) >=0 ):
                mask[k][j] = 1            
    masks.append(mask)
    
    #hidden layers mask
    for i in range(1, num_of_hlayer):
        mask = np.zeros([hlayer_size, hlayer_size])
        for j in range(0, hlayer_size):
            for k in range(0, hlayer_size):
                if all( (subsets[i][j] - subsets[i-1][k]) >= 0):
                    mask[k][j] = 1
        masks.append(mask)
    
    #last layer mask
    mask = np.zeros([hlayer_size, graph_size])
    last_sets = np.random.randint(0,2,(graph_size, graph_size))
    for j in range(0, graph_size):
        for k in range(0, hlayer_size):
            if all( (last_sets[j] - subsets[num_of_hlayer-1][k]) >=0 ):
                mask[k][j] = 1
    masks.append(mask)
    all_masks.append(masks)

class MaskedDenseLayer(Layer):
    def __init__(self, output_dim, layer_number, **kwargs):
        self.output_dim = output_dim
        super(MaskedDenseLayer, self).__init__(**kwargs)
        #self.units = units
        #self.activation = activations.get(activation)
        
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MaskedDenseLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        global all_masks
        return K.dot(K.dot(x, self.kernel), all_masks[state][layer_number])

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
input_layer = MaskedDenseLayer(4,0)#, activation='relu')
hlayer1 = MaskedDenseLayer(3,1)#, activation='relu')
hlayer2 = MaskedDenseLayer(3,2)#, activation='relu')
hlayer3 = MaskedDenseLayer(3,3)#, activation='relu')
hlayer4 = MaskedDenseLayer(3,4)#, activation='relu')
hlayer5 = MaskedDenseLayer(3,5)#, activation='relu')
output_layer = MaskedDenseLayer(4,6)#, activation='sigmoid')
            
autoencoder = Model(input_ds, output_layer)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
reassign_mask = ReassignMask()
autoencoder.fit(train_data, train_data,
                epochs=100,
                batch_size=20,
                shuffle=True,
                validation_data=(valid_data, valid_data),
                callbacks=[reassign_mask])

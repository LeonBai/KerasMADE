# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from keras.engine.topology import Layer
from keras.callbacks import Callback
from keras.models import Model
from keras import backend as K
from keras.layers import Input
import keras.activations as activations
from keras.layers.merge import Multiply

state = np.random.randint(0,20)

#masking lambdaCallback
class ReassignMask(Callback):
    def on_epoch_end(self, epoch, logs):
        global state 
        state = np.random.randint(0,20)

def generate_all_masks(num_of_all_masks, num_of_hlayer, hlayer_size, graph_size):
    all_masks = []
    for i in range(0,num_of_all_masks):
        #generating subsets as 3d matrix 
        subsets = np.random.randint(0, 2, (num_of_hlayer, hlayer_size, graph_size))
        
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
        
    all_masks = [[x*1.0 for x in y] for y in all_masks]
    
    return all_masks

class MaskedDenseLayer(Layer):
    def __init__(self, output_dim, layer_number, all_masks, activation, **kwargs):
        self.output_dim = output_dim
        super(MaskedDenseLayer, self).__init__(**kwargs)
        #self.units = units
        #self.activation = activations.get(activation)
        self._layer_number = layer_number
        self._all_masks = all_masks
        self._activation = activations.get(activation)
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MaskedDenseLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        print('x:', x)
        print('kernel:', self.kernel)
        print('mask:', self._all_masks[state][self._layer_number].shape)
        print('layer_number', self._layer_number)
        print('dot result:', K.dot(x, self.kernel))
        packed_mask = K.variable(value=self._all_masks[state][self._layer_number])
        masked = Multiply()([self.kernel, packed_mask])
        output = K.dot(x, masked)
        return self._activation(output)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    
def main():
    with np.load('datasets/simple_tree.npz') as dataset:
        inputsize = dataset['inputsize']
        train_length = dataset['train_length']
        train_data = dataset['train_data']
        valid_length = dataset['valid_length']
        valid_data = dataset['valid_data']
        test_length = dataset['test_length']
        test_data = dataset['test_data']
        params = dataset['params']
    
    num_of_all_masks = 20
    num_of_hlayer = 6
    hlayer_size = 3
    graph_size = 4
    
    all_masks = generate_all_masks(num_of_all_masks, num_of_hlayer, hlayer_size, graph_size)
    #all_masks = np.asarray(all_masks, dtype='float32')
    
    input_layer = Input(shape=(4,))
    hlayer1 = MaskedDenseLayer(3, 0, all_masks, 'relu')(input_layer)
    hlayer2 = MaskedDenseLayer(3, 1, all_masks, 'relu')(hlayer1)
    hlayer3 = MaskedDenseLayer(3, 2, all_masks, 'relu')(hlayer2)
    hlayer4 = MaskedDenseLayer(3, 3, all_masks, 'relu')(hlayer3)
    hlayer5 = MaskedDenseLayer(3, 4, all_masks, 'relu')(hlayer4)
    hlayer6 = MaskedDenseLayer(3, 5, all_masks, 'relu')(hlayer5)
    output_layer = MaskedDenseLayer(4,6, all_masks, 'sigmoid')(hlayer6)
    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    reassign_mask = ReassignMask()
    autoencoder.fit(train_data, train_data,
                epochs=100,
                batch_size=20,
                shuffle=True,
                validation_data=(valid_data, valid_data),
                callbacks=[reassign_mask],
                verbose=2)


if __name__=='__main__':
    main()
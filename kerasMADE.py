# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from keras.engine.topology import Layer
from keras.callbacks import Callback
from keras.models import Model
from keras.layers import Dense
from keras import backend as K
from keras.layers import Input
import keras.activations as activations
from keras.layers.merge import Multiply, multiply

state = np.random.randint(0,20)

#masking lambdaCallback
class ReassignMask(Callback):
    def on_epoch_end(self, epoch, logs):
        global state 
        state = np.random.randint(0,20)
        
class SaveWeights(Callback):
    def on_epoch_end(self, epoch, logs):
        for idx, layer in enumerate(model.layers):
            print ("layer", idx, "= ", layerlayer.get_wights)
            

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
    def __init__(self, output_dim, activation, **kwargs):
        self.output_dim = output_dim
        super(MaskedDenseLayer, self).__init__(**kwargs)
        #self.units = units
        #self.activation = activations.get(activation)
        #self._layer_number = layer_number
        self._activation = activations.get(activation)
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        #super(MaskedDenseLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, l):
        x = l[0]
        self._mask = l[1]
        print('x:', x)
        print('kernel:', self.kernel)
        #print('mask:', self._all_masks[state][self._layer_number].shape)
        print('dot result:', K.dot(x, self.kernel))
        #packed_mask = K.variable(value=self._mask)
        masked = Multiply()([self.kernel, K.reshape(mask,K.shape(kernel))])
        #masked = Multiply()([self.kernel, packed_mask])
        print('masked:', K.eval(masked))
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
    hlayer_size = 5
    graph_size = 4
    
    all_masks = generate_all_masks(num_of_all_masks, num_of_hlayer, hlayer_size, graph_size)
    
    mask_1 = Input( shape = (graph_size * hlayer_size,) )
    mask_2 = Input( shape = (graph_size * hlayer_size,) )
    mask_3 = Input( shape = (graph_size * hlayer_size,) )
    mask_4 = Input( shape = (graph_size * hlayer_size,) )
    mask_5 = Input( shape = (graph_size * hlayer_size,) )
    mask_6 = Input( shape = (graph_size * hlayer_size,) )
    mask_7 = Input( shape = (graph_size * hlayer_size,) )

    input_layer = Input(shape=(4,))
    
    hlayer1 = MaskedDenseLayer(hlayer_size, 'relu')(input_layer, mask_1)
    hlayer2 = MaskedDenseLayer(hlayer_size, 'relu')(hlayer1, mask_2)
    hlayer3 = MaskedDenseLayer(hlayer_size, 'relu')(hlayer2, mask_3)
    hlayer4 = MaskedDenseLayer(hlayer_size, 'relu')(hlayer3, mask_4)
    hlayer5 = MaskedDenseLayer(hlayer_size, 'relu')(hlayer4, mask_5)
    hlayer6 = MaskedDenseLayer(hlayer_size, 'relu')(hlayer5, mask_6)
    output_layer = MaskedDenseLayer(graph_size, 6, 'sigmoid')(hlayer6, mask_7)
    autoencoder = Model(input_layer, output_layer, mask_1, mask_2, mask_3,
                        mask_4, mask_5, mask_6, mask_7)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    #reassign_mask = ReassignMask()
    state = np.random.randint(0,20)
    
    for i in range(0, num_of_all_masks):
        autoencoder.fit(train_data, train_data, 
                        np.tile(all_masks[state][0].reshape([1,-1]),[x.shape[0],1]),
                        np.tile(all_masks[state][1].reshape([1,-1]),[x.shape[1],1]),
                        np.tile(all_masks[state][2].reshape([1,-1]),[x.shape[2],1]),
                        np.tile(all_masks[state][3].reshape([1,-1]),[x.shape[3],1]),
                        np.tile(all_masks[state][4].reshape([1,-1]),[x.shape[4],1]),
                        np.tile(all_masks[state][5].reshape([1,-1]),[x.shape[5],1]),
                        np.tile(all_masks[state][6].reshape([1,-1]),[x.shape[6],1]),
                        epochs=1,
                        batch_size=20,
                        shuffle=True,
                        validation_data=(valid_data, valid_data),
                        #callbacks=[reassign_mask],
                        verbose=1)
    
    #for idx, layer in enumerate(autoencoder.layers):
        #print ("layer", idx, "= ", layer.get_weights)
#    input_layer = Input(shape=(4,))
#    encoded = Dense(3, activation='relu')(input_layer)
#    encoded = Dense(3, activation='relu')(encoded)
#    encoded = Dense(3, activation='relu')(encoded)
#
#    decoded = Dense(3, activation='relu')(encoded)
#    decoded = Dense(3, activation='relu')(decoded)
#    decoded = Dense(4, activation='sigmoid')(decoded)
#    autoencoder = Model(input_layer, decoded)
#    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#    autoencoder.fit(train_data, train_data,
#                        epochs=100,
#                        batch_size=20,
#                        shuffle=True,
#                        validation_data=(valid_data, valid_data),
#                        verbose=2)
    
    a = autoencoder.trainable_weights
    b = autoencoder.predict(test_data)

if __name__=='__main__':
    main()
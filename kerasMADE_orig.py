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
from keras import optimizers

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
        #subsets = np.random.randint(0, 2, (num_of_hlayer, hlayer_size, graph_size))

        labels = np.zeros([num_of_hlayer, hlayer_size])
        min_label = 0
        for i in range(num_of_hlayer):
            labels[i][:] = np.random.randint(min_label, graph_size, (hlayer_size))
            min_label = np.amin(labels[i])
        #generating masks as 3d matrix
        #masks = np.zeros([num_of_hlayer,hlayer_size,hlayer_size])
        
        masks = []
        pi = np.random.permutation(graph_size)
        #first layer mask
        mask = np.zeros([graph_size, hlayer_size])
        for j in range(0, hlayer_size):
            for k in range(0, graph_size):
                if ((labels[0][j] >= pi[k]) and (pi[k] >= labels[0][j]-4)):
                    mask[k][j] = 1
        masks.append(mask)
        
        #hidden layers mask   
        for i in range(1, num_of_hlayer):
            mask = np.zeros([hlayer_size, hlayer_size])
            for j in range(0, hlayer_size):
                for k in range(0, hlayer_size):
                    if ((labels[i][j] >= labels[i-1][k]) and (labels[i][j] >= labels[i-1][k]-4)):
                        mask[k][j] = 1
            masks.append(mask)
        
        #last layer mask
        mask = np.zeros([hlayer_size, graph_size])
        #last_layer_label = np.random.randint(0, 4, graph_size)
        for j in range(0, graph_size):
            for k in range(0, hlayer_size):
                if (j > labels[-1][k]):
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
                                      shape=(input_shape[0][1], self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(MaskedDenseLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, l):
        self.x = l[0]
        self._mask = l[1][1]
        print('self._mask', self._mask)
        #print('x:', x)
        #print('dot result:', K.dot(x, self.kernel))
        kernel_shape = K.shape(self.kernel).eval(session=K.get_session())
        #mask_shape = K.shape(self._mask).eval(session=K.get_session())
        #print('khar')
        
        #tiled_kernel = K.tile(K.reshape(self.kernel, [1, kernel_shape[0], kernel_shape[1]]), [20, 1, 1])
        #print('kernel:', K.shape(self.kernel).eval(session=K.get_session()))
        masked = Multiply()([self.kernel, self._mask])
        #masked = Multiply()([self.kernel, packed_mask])
        print('masked:', masked)
        #print('x:', self.x)
        self._output = K.dot(self.x, masked)
        print('output:', self._output)
        return self._activation(self._output)

    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.output_dim)

    
def main():
     
    np.random.seed(4125)
    
    with np.load('datasets/grid_4x4_3000.npz') as dataset:
        height = dataset['height']
        width = dataset['width']
        #input_size = dataset['inputsize']
        train_length = dataset['train_length']
        train_data = dataset['train_data']
        train_data_probs = dataset['train_data_probs']
        valid_length = dataset['valid_length']
        valid_data = dataset['valid_data']
        valid_data_probs = dataset['valid_data_probs']
        test_length = dataset['test_length']
        test_data = dataset['test_data']
        test_data_probs = dataset['test_data_probs']
        params = dataset['params']
    
    num_of_exec = 10
    num_of_all_masks = 20
    num_of_hlayer = 6
    hlayer_size = 100
    graph_size = height.tolist()*width.tolist()
    fit_iter = 300
    
    KLs = []
    for ne in range(0, num_of_exec):   
        all_masks = generate_all_masks(num_of_all_masks, num_of_hlayer, hlayer_size, graph_size)
        
        input_layer = Input(shape=(graph_size,))
    
        mask_1 = Input(shape = (graph_size , hlayer_size))
        mask_2 = Input(shape = (hlayer_size , hlayer_size))
        mask_3 = Input(shape = (hlayer_size , hlayer_size))
        mask_4 = Input(shape = (hlayer_size , hlayer_size))
        mask_5 = Input(shape = (hlayer_size , hlayer_size))
        mask_6 = Input(shape = (hlayer_size , hlayer_size))
        mask_7 = Input(shape = (hlayer_size , graph_size))
    
        
        hlayer1 = MaskedDenseLayer(hlayer_size, 'relu')( [input_layer, mask_1] )
        hlayer2 = MaskedDenseLayer(hlayer_size, 'relu')( [hlayer1, mask_2] )
        hlayer3 = MaskedDenseLayer(hlayer_size, 'relu')( [hlayer2, mask_3] )
        hlayer4 = MaskedDenseLayer(hlayer_size, 'relu')( [hlayer3, mask_4] )
        hlayer5 = MaskedDenseLayer(hlayer_size, 'relu')( [hlayer4, mask_5] )
        hlayer6 = MaskedDenseLayer(hlayer_size, 'relu')( [hlayer5, mask_6] )
        output_layer = MaskedDenseLayer(graph_size, 'sigmoid')( [hlayer6, mask_7] )
        
        autoencoder = Model(inputs=[input_layer, mask_1, mask_2, mask_3,
                            mask_4, mask_5, mask_6, mask_7], outputs=[output_layer])
        
        AE_adam = optimizers.Adam(lr=0.0003, beta_1=0.1)
        autoencoder.compile(optimizer=AE_adam, loss='binary_crossentropy')
        #reassign_mask = ReassignMask()
        
        for i in range(0, fit_iter):
            state = np.random.randint(0,num_of_all_masks)
            autoencoder.fit(x=[train_data, 
                              np.tile(all_masks[state][0], [train_length, 1, 1]),
                              np.tile(all_masks[state][1], [train_length, 1, 1]),
                              np.tile(all_masks[state][2], [train_length, 1, 1]),
                              np.tile(all_masks[state][3], [train_length, 1, 1]),
                              np.tile(all_masks[state][4], [train_length, 1, 1]),
                              np.tile(all_masks[state][5], [train_length, 1, 1]),
                              np.tile(all_masks[state][6], [train_length, 1, 1])],
                            y=[train_data],
                            epochs=1,
                            batch_size=50,
                            shuffle=True,
                            #validation_data=(valid_data, valid_data),
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
        
        b = autoencoder.predict([test_data, 
                                np.tile(all_masks[state][0], [test_length, 1, 1]),
                                np.tile(all_masks[state][1], [test_length, 1, 1]),
                                np.tile(all_masks[state][2], [test_length, 1, 1]),
                                np.tile(all_masks[state][3], [test_length, 1, 1]),
                                np.tile(all_masks[state][4], [test_length, 1, 1]),
                                np.tile(all_masks[state][5], [test_length, 1, 1]),
                                np.tile(all_masks[state][6], [test_length, 1, 1])]
                                )
        made_probs = K.prod(b, 1).eval(session=K.get_session())
        print('made_probs', made_probs)
        print('test_probs', test_data_probs)
        #tmp = made_probs  train_data_probs
        KL = np.sum(np.multiply(made_probs, np.log(np.divide(made_probs, test_data_probs))))
        KLs.append(KL)
    
    mean = sum(KLs)/num_of_exec
    variance = 1.0/len(KLs) * np.sum(np.square([x - mean for x in KLs]))
      
    print('KLs:', KLs)
    print('mean:', mean)
    print('variance:', variance)

if __name__=='__main__':
    main()

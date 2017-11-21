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
import tensorflow as tf

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
                if (labels[0][j] >= pi[k]): #and (pi[k] >= labels[0][j]-4)):
                    mask[k][j] = 1
        masks.append(mask)
        
        #hidden layers mask   
        for i in range(1, num_of_hlayer):
            mask = np.zeros([hlayer_size, hlayer_size])
            for j in range(0, hlayer_size):
                for k in range(0, hlayer_size):
                    if (labels[i][j] >= labels[i-1][k]): #and (labels[i][j] >= labels[i-1][k]-4)):
                        mask[k][j] = 1
            masks.append(mask)
        
        #last layer mask
        mask = np.zeros([hlayer_size, graph_size])
        #last_layer_label = np.random.randint(0, 4, graph_size)
        for j in range(0, graph_size):
            for k in range(0, hlayer_size):
                if (j > labels[-1][k]): #and (j >= labels[-1][k]-4)):
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
        self._mask = l[1]
        bs = K.shape(self.x)[0]       
        ks = K.shape(self.kernel)
        masked_flat = Multiply()([K.tile(K.reshape(self.kernel,[1,ks[0]*ks[1]]),[bs,1]), K.reshape(self._mask,[bs,ks[0]*ks[1]])])        
        masked = K.reshape(masked_flat, [bs, ks[0], ks[1]])
        self._output = tf.matmul(K.reshape(self.x,[bs,1,ks[0]]), masked)
        return self._activation(K.reshape(self._output,[bs,self.output_dim]))
        
    '''
    def call(self, l):
        self.x = l[0]
        self._mask = l[1][1]
        
        self._mask_all = l[1]
        bs = K.shape(self.x)[0]       
        ks = K.shape(self.kernel)
        
        masked_flat = Multiply()([K.tile(K.reshape(self.kernel,[1,-1]),[bs,1]), K.reshape(self._mask,[bs,-1])])        
        masked_all = K.reshape(masked_flat, [bs, ks[0], ks[1]])
        output = K.dot(K.reshape(self.x,[bs,1,-1]), masked_all)
        a = self._activation(output)
        
        
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
    '''
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.output_dim)

    
def main():
     
    np.random.seed(4125)
    
    with np.load('datasets/grid_4x4_100.npz') as dataset:
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
    
    num_of_exec = 1
    num_of_all_masks = 10
    num_of_hlayer = 6
    hlayer_size = 100
    graph_size = height.tolist()*width.tolist()
    fit_iter = 1
    num_of_epochs = 10
    bach_s = 50
    
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
        reped_traindata = np.tile(train_data, [num_of_all_masks, 1])
        masks = [None]*(num_of_hlayer+1)
        for i in range(num_of_hlayer+1):
            for j in range(num_of_all_masks):
                tmp = np.tile(all_masks[j][i],[train_length,1,1])
                if j == 0:
                    masks[i] = tmp
                else:    
                    masks[i] = np.concatenate([masks[i], tmp], axis=0)
                
        for i in range(0, fit_iter):
            state = np.random.randint(0,num_of_all_masks)
            autoencoder.fit(x=[reped_traindata, 
                              masks[0], masks[1], masks[2],
                              masks[3], masks[4], masks[5], masks[6]],
                            y=[reped_traindata],
                            epochs=num_of_epochs,
                            batch_size=bach_s,
                            shuffle=True,
                            #validation_data=(valid_data, valid_data),
                            #callbacks=[reassign_mask],
                            verbose=1)
            
        #reped_testdata = np.tile(test_data, [num_of_all_masks, 1])
        
        all_avg_probs = np.array([])
        for i in range(test_length):
            made_probs = np.array([])
            for j in range(num_of_all_masks):
                b = autoencoder.predict([test_data[i].reshape(1, 16), 
                              all_masks[j][0].reshape(1, graph_size, hlayer_size),
                              all_masks[j][1].reshape(1, hlayer_size, hlayer_size), 
                              all_masks[j][2].reshape(1, hlayer_size, hlayer_size), 
                              all_masks[j][3].reshape(1, hlayer_size, hlayer_size), 
                              all_masks[j][4].reshape(1, hlayer_size, hlayer_size), 
                              all_masks[j][5].reshape(1, hlayer_size, hlayer_size), 
                              all_masks[j][6].reshape(1, hlayer_size, graph_size)]
                                )
                made_prob = np.prod(b, 1)
                made_probs = np.append(made_probs, made_prob)
            avg_prob = np.mean(made_probs)
            all_avg_probs = np.append(all_avg_probs, avg_prob)
            
        #print('made_probs', made_probs)
        #print('test_probs', test_data_probs)
        #tmp = made_probs  train_data_probs
        KL = np.sum(np.multiply(all_avg_probs, np.log(np.divide(all_avg_probs, test_data_probs))))
        KLs.append(KL)
    
    mean = sum(KLs)/num_of_exec
    variance = 1.0/len(KLs) * np.sum(np.square([x - mean for x in KLs]))
      
    print('KLs:', KLs)
    print('mean:', mean)
    print('variance:', variance)

if __name__=='__main__':
    main()

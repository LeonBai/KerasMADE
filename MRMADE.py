# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
import time
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

train_end_epochs = []

class MyEarlyStopping(Callback):
    def __init__(self, monitor='val_loss',
                 min_delta=0, patience=0, verbose=0, mode='auto'):
        super(MyEarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        global train_end_epochs
        train_end_epochs.append(self.stopped_epoch)
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
            
class MaskedDenseLayer(Layer):
    def __init__(self, output_dim, masks ,activation, **kwargs):
        self.output_dim = output_dim
        super(MaskedDenseLayer, self).__init__(**kwargs)
        self._mask = masks
        self._activation = activations.get(activation)
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[0][1], self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True,
                                      dtype="float32")
        super(MaskedDenseLayer, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, l):
        self.x = l[0]
        self._state = l[1]

        bs = K.shape(self.x)[0]
        ks = K.shape(self.kernel)

        tmp_mask = tf.gather(tf.constant(self._mask), K.reshape(self._state,[-1]))
        masked = tf.multiply(K.tile(K.reshape(self.kernel,[1,ks[0],ks[1]]),[bs,1,1]), tmp_mask)
        self._output = tf.matmul(K.reshape(self.x,[bs,1,ks[0]]), masked)
        return self._activation(K.reshape(self._output,[bs,self.output_dim]))
  
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.output_dim)

def dataset_gen_grid(height, width, trains, valids, tests, cum_probs, all_outcomes, prob_of_outcomes):
    height = height
    width = width
    inputsize = height*width
    
    num_of_train_samples = trains
    num_of_valid_samples = valids
    num_of_test_samples = tests
    
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
             test_data_probs=test_data_probs)
    return file_name
       
def generate_all_masks(height, width, num_of_all_masks, num_of_hlayer, hlayer_size, graph_size, algo):
    all_masks = []
    for i in range(0,num_of_all_masks):
        #generating subsets as 3d matrix 
        #subsets = np.random.randint(0, 2, (num_of_hlayer, hlayer_size, graph_size))
        labels = np.zeros([num_of_hlayer, hlayer_size], dtype=np.float32)
        min_label = 0
        for ii in range(num_of_hlayer):
            labels[ii][:] = np.random.randint(min_label, graph_size, (hlayer_size))
            min_label = np.amin(labels[ii])
        #generating masks as 3d matrix
        #masks = np.zeros([num_of_hlayer,hlayer_size,hlayer_size])
        
        masks = []
#        if (algo == 'orig'):
#            pi = np.random.permutation(graph_size)
#            #pi = np.array(range(graph_size))
#        else:
#            pi = np.array(range(graph_size))
        #first layer mask
        mask = np.zeros([graph_size, hlayer_size], dtype=np.float32)
        for j in range(0, hlayer_size):
            for k in range(0, graph_size):
                if (algo == 'orig'):
                    if (labels[0][j] >= k): #and (pi[k] >= labels[0][j]-width)):
                        mask[k][j] = 1.0
                else:
                    if ((labels[0][j] >= k) and (k - width <= labels[0][j])): #cant use permutation in our approach
                        mask[k][j] = 1.0
        masks.append(mask)
        
        #hidden layers mask   
        for i in range(1, num_of_hlayer):
            mask = np.zeros([hlayer_size, hlayer_size], dtype=np.float32)
            for j in range(0, hlayer_size):
                for k in range(0, hlayer_size):
                    if (algo == 'orig'):
                        if (labels[i][j] >= labels[i-1][k]): #and (labels[i][j] >= labels[i-1][k]-width)):
                            mask[k][j] = 1.0
                    else:
                        if ((labels[i][j] >= labels[i-1][k]) and (labels[i][j] - width <= labels[i-1][k] )):
                            mask[k][j] = 1.0
            masks.append(mask)
        
        #last layer mask
        mask = np.zeros([hlayer_size, graph_size], dtype=np.float32)
        #last_layer_label = np.random.randint(0, 4, graph_size)
        for j in range(0, graph_size):
            for k in range(0, hlayer_size):
                if (algo == 'orig'):
                    if (j > labels[-1][k]): #and (j >= labels[-1][k]-width)):
                        mask[k][j] = 1.0
                else:
                    if ((j > labels[-1][k]) and (j - width <= labels[-1][k])):
                        mask[k][j] = 1.0
        masks.append(mask)
        all_masks.append(masks)
        
    swapped_all_masks = []
    for i in range(num_of_hlayer+1):
        swapped_masks = []
        for j in range(num_of_all_masks):
            swapped_masks.append(all_masks[j][i])
        swapped_all_masks.append(swapped_masks)
        
    #all_masks = [[x*1.0 for x in y] for y in all_masks]
    
    return swapped_all_masks
    
def main():
    
    #parameter setup
    height = int(sys.argv[1])
    width = int(sys.argv[2])
    train_length = int(sys.argv[3])
    valid_length = int(sys.argv[4])
    test_length = int(sys.argv[5])
    algorithm = sys.argv[6]
    print ('algorithm', algorithm)  #original or minus-width for now
    
    np.random.seed(4125) 
    AE_adam = optimizers.Adam(lr=0.0003, beta_1=0.1)
    num_of_exec = 2
    num_of_all_masks = 10
    num_of_hlayer = 2
    hlayer_size = 24
    graph_size = height*width
    fit_iter = 1
    num_of_epochs = 2000   #max number of epoch if not reaches the ES condition
    batch_s = 50
    optimizer = AE_adam
    patience = 20
    inputsize = height*width
    
    with np.load('grid_parameters.npz') as parameters:
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
    
    sum_prob = np.sum(prob_of_outcomes)
    prob_of_outcomes = np.divide(prob_of_outcomes, sum_prob)
    
    cum_probs = []
    s = 0
    for x in prob_of_outcomes:
        s = s + x
        cum_probs.append(s)
        
    file_name = dataset_gen_grid(height, width, train_length, valid_length, test_length, 
                                 cum_probs, all_outcomes, prob_of_outcomes)
    with np.load(file_name) as dataset:
        print('Dataset:', file_name)
        train_data = dataset['train_data']
        train_data_probs = dataset['train_data_probs']
        valid_data = dataset['valid_data']
        valid_data_probs = dataset['valid_data_probs']
        test_data = dataset['test_data']
        test_data_probs = dataset['test_data_probs']
        
    NLLs = []
    KLs = []
    start_time = time.time()
    for ne in range(0, num_of_exec):                        
        all_masks = generate_all_masks(height, width, num_of_all_masks, num_of_hlayer, hlayer_size, graph_size, algorithm)
#        perm_matrix = np.zeros((test_length, graph_size))
#        for i in range(test_length):
#            for j in range(graph_size):
#                perm_matrix[i][j] = test_data[i][np.where(pi==j)[0][0]]
            #perm_matrix[i][j] = test_data[i][k]  :  pi[k] == j

        
        input_layer = Input(shape=(graph_size,))
        state = Input(shape=(1,), dtype = "int32")

        if (num_of_hlayer == 2): 
            mask_1 = Input(shape = (graph_size , hlayer_size))
            mask_2 = Input(shape = (hlayer_size , hlayer_size))
            mask_3 = Input(shape = (hlayer_size , graph_size))
        else:
            mask_1 = Input(shape = (graph_size , hlayer_size))
            mask_2 = Input(shape = (hlayer_size , hlayer_size))
            mask_3 = Input(shape = (hlayer_size , hlayer_size))
            mask_4 = Input(shape = (hlayer_size , hlayer_size))
            mask_5 = Input(shape = (hlayer_size , hlayer_size))
            mask_6 = Input(shape = (hlayer_size , hlayer_size))
            mask_7 = Input(shape = (hlayer_size , graph_size))
    
        if (num_of_hlayer == 2):
            hlayer1 = MaskedDenseLayer(hlayer_size, np.array(all_masks[0]), 'relu')( [input_layer, state] )
            hlayer2 = MaskedDenseLayer(hlayer_size, np.array(all_masks[1]), 'relu')( [hlayer1, state] )
            output_layer = MaskedDenseLayer(graph_size, np.array(all_masks[2]), 'sigmoid')( [hlayer2, state] )
        else:
            hlayer1 = MaskedDenseLayer(hlayer_size, np.array(all_masks[0]), 'relu')( [input_layer, state] )
            hlayer2 = MaskedDenseLayer(hlayer_size, np.array(all_masks[1]), 'relu')( [hlayer1, state] )
            hlayer3 = MaskedDenseLayer(hlayer_size, np.array(all_masks[2]), 'relu')( [hlayer2, state] )
            hlayer4 = MaskedDenseLayer(hlayer_size, np.array(all_masks[3]), 'relu')( [hlayer3, state] )
            hlayer5 = MaskedDenseLayer(hlayer_size, np.array(all_masks[4]), 'relu')( [hlayer4, state] )
            hlayer6 = MaskedDenseLayer(hlayer_size, np.array(all_masks[5]), 'relu')( [hlayer5, state] )
            output_layer = MaskedDenseLayer(graph_size, np.array(all_masks[6]), 'sigmoid')( [hlayer6, state] )
        if (num_of_hlayer == 6):
            autoencoder = Model(inputs=[input_layer, state], outputs=[output_layer])
        else:
            autoencoder = Model(inputs=[input_layer, state], outputs=[output_layer])
        
        autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
        #reassign_mask = ReassignMask()
        early_stop = MyEarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto')
        
        reped_state_train = np.arange(train_length*num_of_all_masks, dtype=np.int32)/train_length
        reped_state_valid = np.arange(valid_length*num_of_all_masks, dtype=np.int32)/valid_length
        reped_traindata = np.tile(train_data, [num_of_all_masks, 1])
        reped_validdata = np.tile(valid_data, [num_of_all_masks, 1])
                
        for i in range(0, fit_iter):
            if (num_of_hlayer == 6):
                autoencoder.fit(x=[reped_traindata, reped_state_train],
                                  y=[reped_traindata],
                                  epochs=num_of_epochs,
                                  batch_size=batch_s,
                                  shuffle=True,
                                  validation_data=([reped_validdata, reped_state_valid],
                                                    [reped_validdata]),
                                  callbacks=[early_stop],
                                  verbose=1)
            else:
                autoencoder.fit(x=[reped_traindata, 
                                  reped_state_train],
                                  y=[reped_traindata],
                                  epochs=num_of_epochs,
                                  batch_size=batch_s,
                                  shuffle=True,
                                  validation_data=([reped_validdata, reped_state_valid],
                                                    [reped_validdata]),
                                  callbacks=[early_stop],
                                  verbose=1)

        #reped_testdata = np.tile(test_data, [num_of_all_masks, 1])
        made_probs = np.zeros([num_of_all_masks, test_length])
        for j in range(num_of_all_masks):
            made_predict = autoencoder.predict([test_data, j * np.ones([test_length,1])])#.reshape(1, hlayer_size, graph_size)]
            
            corrected_probs = np.multiply(np.power(made_predict, test_data), 
                            np.power(np.ones(made_predict.shape) - made_predict, np.ones(test_data.shape) - test_data))
            made_prob = np.prod(corrected_probs, 1)
            made_probs[j][:] = made_prob
                  
        all_avg_probs = np.mean(made_probs, axis=0)
            
        KL = -1*np.sum(np.multiply(test_data_probs, (np.log(all_avg_probs) - np.log(test_data_probs))))
        NLL = -1*np.mean(np.log(all_avg_probs))
        NLLs.append(NLL)
        KLs.append(KL)
    
    mean_NLLs = sum(NLLs)/num_of_exec
    variance_NLLs = 1.0/len(NLLs) * np.sum(np.square([x - mean_NLLs for x in NLLs]))
    mean_KLs = sum(KLs)/num_of_exec
    variance_KLs = 1.0/len(KLs) * np.sum(np.square([x - mean_KLs for x in KLs]))
    
    total_time = time.time() - start_time
    global train_end_epochs
    print('End Epochs:', train_end_epochs)
    print('End Epochs Average', np.mean(train_end_epochs))
    print('NLLs:', NLLs)
    print('KLs:', KLs)
    print('Average KLs:', mean_KLs)
    print('Variance KLs:', variance_KLs)
    print('Average NLLs:', mean_NLLs)
    print('Variance NLLs:', variance_NLLs)
    print('Total Time:', total_time)

if __name__=='__main__':
    main()

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
        #masked_flat = Multiply()([K.tile(K.reshape(self.kernel,[1,ks[0]*ks[1]]),[bs,1]), K.reshape(self._mask,[bs,ks[0]*ks[1]])])        
        #masked = K.reshape(masked_flat, [bs, ks[0], ks[1]])
        masked = tf.multiply(K.tile(K.reshape(self.kernel,[1,ks[0],ks[1]]),[bs,1,1]), self._mask)
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

def dataset_gen_grid(height, width, trains, valids, tests, cum_probs, all_outcomes, prob_of_outcomes):
    height = height
    width = width
    inputsize = height
    
    num_of_train_samples = trains
    num_of_valid_samples = valids
    num_of_test_samples = tests
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
    
    test_data = np.ndarray(shape=(num_of_test_samples, inputsize), dtype=np.float32)
    test_data_probs = np.ndarray(shape=(num_of_test_samples), dtype=np.float32)
    for x in range(num_of_test_samples):
        p = np.random.uniform(0,1)
        i = np.searchsorted(cum_probs, p)
        test_data[x][:] = all_outcomes[i]
        test_data_probs[x] = prob_of_outcomes[i]
        #test_data = all_outcomes[prob_of_outcomes > 0][:]
        #test_data_probs = prob_of_outcomes[prob_of_outcomes > 0]

    file_name = 'datasets/line_' + str(height) + 'x' + str(width) + '_' + str(num_of_train_samples) + str(num_of_valid_samples) + str(num_of_test_samples) + '.npz'
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
                if (algo == 'orig'):
                    if (labels[0][j] >= pi[k]): #and (pi[k] >= labels[0][j]-width)):
                        mask[k][j] = 1
                else:
                    if ((labels[0][j] >= k) and (k >= labels[0][j]-1)): #cant use permutation in our approach
                        mask[k][j] = 1
        masks.append(mask)
        
        #hidden layers mask   
        for i in range(1, num_of_hlayer):
            mask = np.zeros([hlayer_size, hlayer_size])
            for j in range(0, hlayer_size):
                for k in range(0, hlayer_size):
                    if (algo == 'orig'):
                        if (labels[i][j] >= labels[i-1][k]): #and (labels[i][j] >= labels[i-1][k]-width)):
                            mask[k][j] = 1
                    else:
                        if ((labels[i][j] >= labels[i-1][k]) and (labels[i][j] >= labels[i-1][k]-1)):
                            mask[k][j] = 1
            masks.append(mask)
        
        #last layer mask
        mask = np.zeros([hlayer_size, graph_size])
        #last_layer_label = np.random.randint(0, 4, graph_size)
        for j in range(0, graph_size):
            for k in range(0, hlayer_size):
                if (algo == 'orig'):
                    if (j > labels[-1][k]): #and (j >= labels[-1][k]-width)):
                        mask[k][j] = 1
                else:
                    if ((j > labels[-1][k]) and (j >= labels[-1][k]-1)):
                        mask[k][j] = 1
        masks.append(mask)
        all_masks.append(masks)
        
    all_masks = [[x*1.0 for x in y] for y in all_masks]
    
    return all_masks
    
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
    num_of_exec = 10
    num_of_all_masks = 10
    num_of_hlayer = 2
    hlayer_size = 100
    graph_size = height
    fit_iter = 1
    num_of_epochs = 2000    #max number of epoch if not reaches the ES condition
    batch_s = 50
    optimizer = AE_adam
    patience = 20
    inputsize = height
    
    with np.load('line_parameters.npz') as parameters:
        params = parameters['parameter']
        
    all_outcomes = np.ndarray(shape=(2**inputsize, inputsize), dtype=np.float32)
    prob_of_outcomes = np.ndarray(shape=(2**inputsize), dtype=np.float32)
                
    for i in range(2**inputsize):
        str_samp = ('{0:0' + str(height) + 'b}').format(i)
        asarr_samp = [int(d) for d in str_samp]
        all_outcomes[i][:] = asarr_samp
        sum_prod = all_outcomes[i][0]*params[0]*all_outcomes[i][1] + \
        all_outcomes[i][1]*params[1]*all_outcomes[i][2] + \
        all_outcomes[i][2]*params[2]*all_outcomes[i][3]
#        for r in range(height):
#            for c in range(width):
#                jj = r*width + c
#                if (c > 0):
#                    sum_prod += all_outcomes[i][jj]*all_outcomes[i][jj-1]*adj[jj][jj-1]
#                if (r > 0):
#                    sum_prod += all_outcomes[i][jj]*all_outcomes[i][jj-width]*adj[jj][jj-width]
        p = np.exp(sum_prod)
        prob_of_outcomes[i] = p
    
    sum_prob = sum(prob_of_outcomes)
    prob_of_outcomes = np.divide(prob_of_outcomes, sum_prob)
    
    cum_probs = []
    s = 0
    for x in prob_of_outcomes:
        s = s + x
        cum_probs.append(s)
        
    NLLs = []
    KLs = []
    start_time = time.time()
    for ne in range(0, num_of_exec):   
    
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
                     
        all_masks = generate_all_masks(height, width, num_of_all_masks, num_of_hlayer, hlayer_size, graph_size, algorithm)
        
        input_layer = Input(shape=(graph_size,))
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
            hlayer1 = MaskedDenseLayer(hlayer_size, 'relu')( [input_layer, mask_1] )
            hlayer2 = MaskedDenseLayer(hlayer_size, 'relu')( [hlayer1, mask_2] )
            output_layer = MaskedDenseLayer(graph_size, 'sigmoid')( [hlayer2, mask_3] )
        else:
            hlayer1 = MaskedDenseLayer(hlayer_size, 'relu')( [input_layer, mask_1] )
            hlayer2 = MaskedDenseLayer(hlayer_size, 'relu')( [hlayer1, mask_2] )
            hlayer3 = MaskedDenseLayer(hlayer_size, 'relu')( [hlayer2, mask_3] )
            hlayer4 = MaskedDenseLayer(hlayer_size, 'relu')( [hlayer3, mask_4] )
            hlayer5 = MaskedDenseLayer(hlayer_size, 'relu')( [hlayer4, mask_5] )
            hlayer6 = MaskedDenseLayer(hlayer_size, 'relu')( [hlayer5, mask_6] )
            output_layer = MaskedDenseLayer(graph_size, 'sigmoid')( [hlayer6, mask_7] )
        if (num_of_hlayer == 6):
            autoencoder = Model(inputs=[input_layer, mask_1, mask_2, mask_3,
                            mask_4, mask_5, mask_6, mask_7], outputs=[output_layer])
        else:
            autoencoder = Model(inputs=[input_layer, mask_1, mask_2, mask_3], outputs=[output_layer])
        
        autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
        #reassign_mask = ReassignMask()
        early_stop = MyEarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto')
        
        reped_traindata = np.tile(train_data, [num_of_all_masks, 1])
        reped_validdata = np.tile(valid_data, [num_of_all_masks, 1])
        
        masks_train = [None]*(num_of_hlayer+1)
        for i in range(num_of_hlayer+1):
            for j in range(num_of_all_masks):
                tmp = np.tile(all_masks[j][i],[train_length,1,1])
                if j == 0:
                    masks_train[i] = tmp
                else:    
                    masks_train[i] = np.concatenate([masks_train[i], tmp], axis=0)
                    
        masks_valid = [None]*(num_of_hlayer+1)
        for i in range(num_of_hlayer+1):
            for j in range(num_of_all_masks):
                tmp = np.tile(all_masks[j][i],[valid_length,1,1])
                if j == 0:
                    masks_valid[i] = tmp
                else:    
                    masks_valid[i] = np.concatenate([masks_valid[i], tmp], axis=0)
                
        for i in range(0, fit_iter):
            state = np.random.randint(0,num_of_all_masks)
            if (num_of_hlayer == 6):
                autoencoder.fit(x=[reped_traindata, 
                                  masks_train[0], masks_train[1], masks_train[2],
                                  masks_train[3], masks_train[4], masks_train[5], masks_train[6]],
                                  y=[reped_traindata],
                                  epochs=num_of_epochs,
                                  batch_size=batch_s,
                                  shuffle=True,
                                  validation_data=([reped_validdata,
                                                    masks_valid[0], masks_valid[1], masks_valid[2],
                                                    masks_valid[3], masks_valid[4], masks_valid[5], masks_valid[6]],
                                                    [reped_validdata]),
                                  callbacks=[early_stop],
                                  verbose=1)
            else:
                autoencoder.fit(x=[reped_traindata, 
                                  masks_train[0], masks_train[1], masks_train[2]],
                                  y=[reped_traindata],
                                  epochs=num_of_epochs,
                                  batch_size=batch_s,
                                  shuffle=True,
                                  validation_data=([reped_validdata,
                                                    masks_valid[0], masks_valid[1], masks_valid[2]],
                                                    [reped_validdata]),
                                  callbacks=[early_stop],
                                  verbose=1)

        #reped_testdata = np.tile(test_data, [num_of_all_masks, 1])
        made_probs = np.zeros([num_of_all_masks, test_length])
        for j in range(num_of_all_masks):
            if (num_of_hlayer == 6):
                made_predict = autoencoder.predict([test_data, 
                                                    np.tile(all_masks[j][0], [test_length, 1, 1]),#.reshape(1, graph_size, hlayer_size),
                                                    np.tile(all_masks[j][1], [test_length, 1, 1]),#.reshape(1, hlayer_size, hlayer_size), 
                                                    np.tile(all_masks[j][2], [test_length, 1, 1]),#.reshape(1, hlayer_size, hlayer_size), 
                                                    np.tile(all_masks[j][3], [test_length, 1, 1]),#.reshape(1, hlayer_size, hlayer_size), 
                                                    np.tile(all_masks[j][4], [test_length, 1, 1]),#.reshape(1, hlayer_size, hlayer_size), 
                                                    np.tile(all_masks[j][5], [test_length, 1, 1]),#.reshape(1, hlayer_size, hlayer_size), 
                                                    np.tile(all_masks[j][6], [test_length, 1, 1])]#.reshape(1, hlayer_size, graph_size)]
                                                    )
            else:
                 made_predict = autoencoder.predict([test_data, 
                                                    np.tile(all_masks[j][0], [test_length, 1, 1]),#.reshape(1, graph_size, hlayer_size),
                                                    np.tile(all_masks[j][1], [test_length, 1, 1]),#.reshape(1, hlayer_size, hlayer_size), 
                                                    np.tile(all_masks[j][2], [test_length, 1, 1])]#.reshape(1, hlayer_size, hlayer_size), 
                                                    )

            made_prob = np.prod(made_predict, 1)
            made_probs[j][:] = made_prob
                  
        all_avg_probs = np.mean(made_probs, axis=0)
            
        #print('made_probs', made_probs)
        #print('test_probs', test_data_probs)
        #tmp = made_probs  train_data_probs
        KL = -1*np.mean(np.log(np.divide(all_avg_probs, test_data_probs)))
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

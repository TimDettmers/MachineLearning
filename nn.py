from theano import function, shared, Out, config, scan as s
from theano.sandbox.cuda.basic_ops import gpu_from_host as g
from theano.tensor.shared_randomstreams import RandomStreams
from numpy import asarray as arr
import theano.tensor as T
import numpy as np
import time
import math
import uuid
import matplotlib.pyplot as plt
import sys
import select
import scipy

from batch_creator import batch_creator
from neural_network import Neural_network
from neural_network import Cost_function
from neural_network_layer import Dropout
from neural_network_layer import Neural_network_layer
from neural_network_layer import Activation_function
from neural_network_layer import Weight_initialization_scheme
from Hyperparameters import Hyperparameters as H

'''Creates neural network and runs training on MNIST data
'''

b = batch_creator()
path_train = '/home/tim/development/train.csv'
#path_test = '/home/tim/development/test_X.csv'
result_dir = '/home/tim/development/results/'


batch_size = 150
set_sizes = [0.80,0.20]

data = b.create_batches([path_train], [0], set_sizes,batch_size, standardize = False)
X_test = np.float32(np.load('/home/tim/development/test_X.npy'))

data_X = data[0][0]
data_y = data[0][1]
data_t = data[0][2]
data_X_val = data[1][0]
data_y_val = data[1][1]
data_t_val = data[1][2]

input = Neural_network_layer(784,Activation_function.input_units, Dropout.drop_activation,
                              frequency = [0], dropout = 0.2, dropout_decay= [0.2], batch_size = batch_size)
hidden = Neural_network_layer(8000,Activation_function.sigmoid, Dropout.drop_activation,
                               frequency = [0], dropout = 0.5, dropout_decay=[0.2], batch_size = batch_size)

#hidden2 = Neural_network_layer(8000,Activation_function.sigmoid, Dropout.drop_activation,
#                              frequency = 100, dropout = 0.5, dropout_decay=0.2, batch_size = batch_size)

output = Neural_network_layer(10,Activation_function.softmax, Dropout.drop_none, 0,0, batch_size = batch_size, frequency = 0)

layers = [input, hidden,
          #hidden2,
          output]

H.layers = layers

H.set_learning_parameters(epochs = 100, 
                   learning_rate = 0.1, use_learning_rate_decay = False, learning_rate_decay = 0.01,
                    use_momentum = True, momentum = 0.5,transient_phase = 1500,
                    use_momentum_increase = True, momentum_increase = 0.001, use_bias = True)

H.set_standard_regularization_parameters(
                      use_early_stopping = True,
                      use_L2 = True, L2_penalty = 0.0001,
                      use_weight_decay = False, weight_decay = 0.05,weight_decay_decay = 0.01)


H.set_stopping_and_ensemble_parameter(use_ensemble = False, ensemble_count = 15,
                                    safe_weights_threshold = 0.011, epochs_force_reinitilization = 260)

H.set_initial_parameters()

nn = Neural_network(result_dir,data_X, data_y, data_t, data_X_val, data_y_val, data_t_val, X_test,
                    layers = layers, 
                    hyperparams = H,
                    weight_init_scheme=Weight_initialization_scheme.SparseInitialization15,
                    cost_function= Cost_function.cross_entropy, training_sequence = None)

nn.init_theano_network_functions()

activations = []

def print_tracker(batch):  
    '''Print activation of each neuron for a whole batch to an image.
    '''
    if len(activations) > 0 and batch == 0: 
        print activations[-1]
        print np.max(activations[-1])
        plt.imshow(activations[-1], interpolation='nearest')       
        scipy.misc.imsave(result_dir +'outfile.jpg', activations[-1])               
        
         
    if batch == 0:
        activations.append(np.int32(nn.activation_tracker_function(batch)[0]))
    else:       
        activations[-1] = np.add(activations[-1], np.int32(nn.activation_tracker_function(batch)[0]))
        
                
'''Use these two methods to test if averaging outputs with dropout is an improvement
   over using the mean net (multiplying outputs by 0.5).
'''
def multi_forward_pass(batch):
    '''Use training data to predict output by averaging prediction with active dropout.    
    '''
    np.set_printoptions(suppress=True)
    classifications = np.zeros((150,10))
    for i in range(15):
        classifications = np.add(classifications,(np.float64(arr(nn.feedforward_function(batch)))))    
        
    #print 'Mean ' + str(np.mean(classifications,axis=0))
    return nn.train_error_function_dropout(batch,np.float32(classifications/15.))[0]

def multi_forward_pass_epoch():
    '''use validation data predict output by averaging prediction with active dropout
    '''
    np.set_printoptions(suppress=True)
    classifications = np.zeros((4200,10))
    for i in range(50):
        classifications = np.add(classifications,(np.float64(arr(nn.feedforward_valid_drop_function()))))    
        
    #print 'Mean ' + str(np.mean(classifications,axis=0))
    return nn.cross_validation_function_dropout(np.float32(classifications/15.))[0]
       
    
#nn.hook_functions_crossvalid = multi_forward_pass
#nn.hook_functions_crossvalid_epoch = multi_forward_pass_epoch

nn.train()



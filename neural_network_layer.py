from theano.sandbox.cuda.basic_ops import gpu_from_host as g
from theano import function, shared
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as T
import numpy as np

class Dropout:
    drop_activation = 'drop_activation'
    drop_threshold = 'drop_threshold'
    drop_none = 'none'

class Weight_initialization_scheme:
    DimensionSquareRoot = 'DimensionSquareRoot'
    SparseInitialization15 = 'SparseInitialization15'

class Activation_function:
    sigmoid = T.nnet.sigmoid
    input_units =  staticmethod(lambda X: X)
    softmax = T.nnet.softmax

class Neural_network_layer:
    '''Represents the units within a layer and the units
       activations and dropout functions.
    '''
    
    def __init__(self, size, activation_function, dropout_type, dropout, dropout_decay, batch_size, frequency):
        
        
        self.drop_count = 0
        self.size = size  
        self.frequency = frequency
        self.dropout = dropout    
        self.dropout_init = dropout    
        self.dropout_decay = dropout_decay  
        self.dropout_type = dropout_type    
        self.rdm = RandomStreams(seed=1234)  
        self.batch_size = batch_size   
        self.sample_range = 100000       
        self.create_dropout_sample_functions()  
        self.activation_crossvalidation = activation_function 
        self.activation_function = self.set_dropout(dropout, activation_function)
        self.activation_derivative = lambda X: g(T.mul(X, (1.0 - X)))   
        self.activation_tracker = self.set_activation_tracker(activation_function)             
        
        pass
    
    
    def set_dropout(self, dropout, activation_function):
        action_with_drop = None
        if dropout > 0:
            action_with_drop = lambda X: T.mul(activation_function(X),self.dropout_function)            
            self.activation_cv_dropout = lambda X: T.mul(activation_function(X),self.dropout_function_cv)
        else:
            action_with_drop = activation_function
            self.activation_cv_dropout = activation_function
            
        return action_with_drop
     
    def set_activation_tracker(self, activation_function): 
        '''Sets a tracker function that logs the activations that exceed 0.75.
        '''
        if activation_function == Activation_function.sigmoid:
            activation_tracker = lambda X: T.gt(activation_function(X),0.75)
        else:
            activation_tracker = None
        return activation_tracker
    
    def create_dropout_sample_functions(self, reset = False):
        '''Creates functions of sample vectors which can be index with random
           integers to create a pseudo random sample for dropout. This greatly
           speeds up sampling as no new samples have to be created.
        '''
        if reset:
            self.dropout = self.dropout_init
            print 'Reset dropout to ' + str(self.dropout)
        
        self.dropout_function = None
        sample_function = None
        if self.dropout > 0:
            if self.dropout_type == Dropout.drop_activation:
                if reset:
                    self.bino_sample_vector.set_value(np.matrix(np.float32(
                                        np.random.binomial(1,1-self.dropout,(10000000,1)))),
                                        borrow=True) 
                else:
                    self.bino_sample_vector = shared(np.matrix(np.float32(
                                            np.random.binomial(1,1-self.dropout,(10000000,1)))),
                                            'float32', borrow=True) 
            
                sample_function = lambda rand: g(T.reshape(self.bino_sample_vector[rand:rand + (self.batch_size*self.size)],(self.batch_size,self.size)))
                sample_function_cv = lambda rand: g(T.reshape(self.bino_sample_vector[rand:rand + (4200*self.size)],(4200,self.size)))
                self.dropout_function = sample_function(self.rdm.random_integers(low=0, high=self.sample_range))  
                self.dropout_function_cv = sample_function_cv(self.rdm.random_integers(low=0, high=self.sample_range))  
             
                
    def handle_dropout_decay(self, epoch):
        '''Handles automatically the dropout decay by decreasing the dropout by
           the given amount after the given number of epochs.
        '''
        if self.dropout_function and self.frequency[self.drop_count] > 0 and epoch % self.frequency[self.drop_count] == 0  and epoch > 0:
            print 'Setting dropout from  '  + str(self.dropout)  + ' to ' + str(np.float32(self.dropout*(1-self.dropout_decay[self.drop_count])))   
            
            self.dropout = np.float32(self.dropout*(1-self.dropout_decay[self.drop_count]))       
            
            if self.dropout_type == Dropout.drop_activation:
                self.bino_sample_vector.set_value(np.matrix(np.float32( 
                                        np.random.binomial(1,1-self.dropout,(10000000,1)))),
                                        borrow=True) 
            self.drop_count += 1   
            if self.drop_count > len(self.dropout_decay)-1:
                self.drop_count -= 1
              
            
                 
                
                
                
                
                
                
         
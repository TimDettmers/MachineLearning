import numpy as np
import matplotlib.pyplot as plt
import sys
import select
import time
import theano.tensor as T

from theano.sandbox.cuda.basic_ops import gpu_from_host as g
from theano import function, shared, Out, config, scan as s
from numpy import asarray as arr

from neural_network_layer import Neural_network_layer
from neural_network_layer import Activation_function
from neural_network_layer import Weight_initialization_scheme
from neural_network_layer import Dropout

class Cost_function:
    cross_entropy = 'cross entropy'
 

class Neural_network:
    def __init__(self, result_dir,
                 data_X, data_y, data_t, valid_X, valid_y,valid_t, test_X,
                 layers, weight_init_scheme, cost_function, hyperparams, training_sequence):
        '''Class that takes layers and combines them into a neural network. Theano functions
            are created and the given data is then trained with the GPU on the constructed 
            neural network through the train method.
        '''
        
        self.cost_function = cost_function
        
        self.H = hyperparams
        if training_sequence:
            self.training_sequence = training_sequence
        else:
            self.training_sequence = self.default_training_sequence
        self.layers = layers
        self.rng = np.random.RandomState(1234)
        
        self.time1 = time.time() 
        
        self.ensemble = None
        self.ensemble_MNIST = None
        self.ensemble_softmax = None
        
        self.hook_functions = None
             
        self.weight_init_scheme = weight_init_scheme
        self.init_weight_and_data_variables(data_X, data_y, data_t, valid_X, valid_y, valid_t, test_X)
                        
        self.init_theano_variables()  

        self.train_history = np.float32(arr(range(self.H.L.epochs)))
        self.valid_history = np.float32(arr(range(self.H.L.epochs)))
        
        self.result_dir = result_dir
          
        self.hook_functions_batch = None
        self.hook_functions_crossvalid = None
        self.hook_functions_crossvalid_epoch = None
        pass

    def init_theano_network_functions(self):    
        self.create_feedforward_chains()
        self.create_cost_function()
        self.create_backprop_gradient_functions()
        self.create_weight_update_functions()
        self.create_nesterov_weight_update_functions()
        self.create_weight_update_with_momentum_functions()
        self.create_momentum_weight_update_functions()
        self.create_weight_decay_updates()
        self.create_validation_and_train_error_functions()        
        self.constaint_weight_function()
   
    def init_theano_variables(self):
        self.A = T.fmatrix("A")
        self.B = T.fmatrix("B")
        self.idx = T.iscalar("idx")
        self.alpha = T.fscalar("alpha")
        self.M = T.fscalar("M")
    
    def reinitilize_parameters(self, epoch):
        '''Reinitialization that is used to restart the neural network.
           Can be used to automatically create ensembles of neural networks.
        '''
        self.H.L.learning_rate =  self.H.initial_learning_rate
        
        self.H.R.weight_decay = self.H.initial_weight_decay           
        self.H.L.momentum = self.H.initial_momentum  
        self.H.S.saved_weights_error = 1
        self.H.S.safe_weights_threshold = self.H.initial_safe_weights_threshold
        self.H.S.last_save_epoch = epoch
        
        for layer in self.layers:
            layer.create_dropout_sample_functions(reset = True)
    
    def heardEnter(self):
        '''Returns true if enter was pressed.
        '''
        i,o,e = select.select([sys.stdin],[],[],0.0001)
        for s in i:
            if s == sys.stdin:
                input = sys.stdin.readline()
                return True
        return False    
    
    def reinitialize_weights(self):
        '''Reset weight to the starting conditions.        
        '''
        for i in range(len(self.layers) - 1):
            if self.weight_init_scheme == Weight_initialization_scheme.DimensionSquareRoot:
                self.weights[i].set_value(np.float32(
                self.rng.uniform(low=4 * -np.sqrt(6. / (self.layers[i].size + self.layers[i + 1].size)),
                                    high=4 * np.sqrt(6. / (self.layers[i].size + self.layers[i + 1].size)),
                                     size=(self.layers[i].size, self.layers[i + 1].size))), borrow=True)
                
                self.nesterov_weights[i].set_value(self.weights[i].get_value(), borrow=True)
                self.H.L.momentum_weights[i].set_value(np.float32(np.zeros((self.layers[i].size + self.H.L.use_bias, self.layers[i + 1].size))), borrow=True)
                
                self.biases[i].set_value(np.float32(np.tile(
                  self.rng.uniform(low=4 * -np.sqrt(6. / (self.layers[i].size + self.layers[i + 1].size)),
                                    high=4 * np.sqrt(6. / (self.layers[i].size + self.layers[i + 1].size)),
                                     size=(1, self.layers[i + 1].size)),(self.batch_size,1))), borrow=True)
            elif self.weight_init_scheme == Weight_initialization_scheme.SparseInitialization15:                
                weight = self.create_sparse_weight(self.layers[i].size, self.layers[i + 1].size, sparsity=15)                        
                self.weights[i].set_value(np.float32(weight), borrow = True)
                self.nesterov_weights[i].set_value(self.weights[i].get_value(), borrow=True)
                self.H.L.momentum_weights[i].set_value(np.float32(np.zeros((self.layers[i].size, self.layers[i + 1].size))), borrow=True)             
                self.biases[i].set_value(np.float32(np.zeros((self.batch_size,self.layers[i + 1].size))), borrow = True)
            else:
                raise Exception('Weight initalization scheme not implemented, yet!')        

    def initialize_weights(self):
        self.weights = []
        self.nesterov_weights = []
        self.H.L.momentum_weights = []
        self.biases = []
        for i in range(len(self.layers) - 1):
            if self.weight_init_scheme == Weight_initialization_scheme.DimensionSquareRoot:
                self.weights.append(shared(np.float32(
                  self.rng.uniform(low=4 * -np.sqrt(6. / (self.layers[i].size + self.layers[i + 1].size)),
                                    high=4 * np.sqrt(6. / (self.layers[i].size + self.layers[i + 1].size)),
                                     size=(self.layers[i].size, self.layers[i + 1].size))), 'float32', borrow=True))
                
                self.nesterov_weights.append(shared(self.weights[i].get_value(), 'float32', borrow=True))
                self.H.L.momentum_weights.append(shared(np.float32(np.zeros((self.layers[i].size, self.layers[i + 1].size))),'float32', borrow=True)) 
                self.biases.append(shared(np.float32(np.tile(
                  self.rng.uniform(low=4 * -np.sqrt(6. / (self.layers[i].size + self.layers[i + 1].size)),
                                    high=4 * np.sqrt(6. / (self.layers[i].size + self.layers[i + 1].size)),
                                     size=(1, self.layers[i + 1].size)),(self.batch_size,1))), 'float32', borrow=True)) 
            elif self.weight_init_scheme == Weight_initialization_scheme.SparseInitialization15:                
                weight = self.create_sparse_weight(self.layers[i].size, self.layers[i + 1].size, sparsity=15)                        
                self.weights.append(shared(np.float32(weight),'float32', borrow = True))
                self.nesterov_weights.append(shared(self.weights[i].get_value(), 'float32', borrow=True))
                self.H.L.momentum_weights.append(shared(np.float32(np.zeros((self.layers[i].size, self.layers[i + 1].size))),'float32', borrow=True))             
                self.biases.append(shared(np.float32(np.zeros((self.batch_size,self.layers[i + 1].size))),'float32', borrow = True))
            else:
                raise Exception('Weight initalization scheme not implemented, yet!')
            
    def create_sparse_weight(self, input_size, output_size, sparsity = 15):  
        '''Creates a sparse weight matrix where every output unit has only 15
           gaussian connections to the input units.  
           Sparsity drives more diverse features that have better generalization,
           also see Martens 2010:
           http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_Martens10.pdf
        '''      
        weight = np.zeros((input_size, output_size))
        for axon in range(output_size):            
            idxes = np.random.randint(0,input_size, (sparsity,))
            rdm_weights = np.random.randn(sparsity)
            for idx, rdm_weights in zip(idxes, rdm_weights):
                weight[idx,axon] = rdm_weights
       
                
        return weight
        
        
    def create_cost_function(self):
        cost_function = None
        
        if self.cost_function == Cost_function.cross_entropy:
            cost_function = -T.mean(T.sum(T.mul(self.t[self.idx], T.log(self.feedforward)),axis=0))
            
        
        self.cost_function = cost_function
        
    def create_feedforward_chains(self):
        #feedforward with dropout for train data
        self.a = []
        self.z = []
        for i in range(len(self.weights)):
            if i == 0: 
                #input units
                self.a.append(g(T.dot(self.layers[i].activation_function(self.X[self.idx]),self.weights[i])))                        
                self.z.append(g(self.layers[i+1].activation_function(T.add(self.a[i],self.biases[i]) if self.H.L.use_bias else self.a[i])))
            else:               
                self.a.append(g(T.dot(self.z[i-1],self.weights[i])))
                if len(self.layers) > i:
                    self.z.append(g(self.layers[i+1].activation_function(g(T.add(self.a[i],self.biases[i])) if self.H.L.use_bias else self.a[i])))     
                   
        self.feedforward = self.z[-1]
        
        self.feedforward_function = function(inputs=[self.idx],outputs=self.feedforward)
        #feedforward for validation data with dropout instead of mean net multiplication
        self.a_valid_drop = []
        self.z_valid_drop = []
        for i in range(len(self.weights)):
            if i == 0: 
                #input units
                self.a_valid_drop.append(g(T.dot(self.layers[i].activation_cv_dropout(self.X_val),self.weights[i])))                        
                self.z_valid_drop.append(g(self.layers[i+1].activation_cv_dropout(T.add(self.a_valid_drop[i],self.biases[i][0,:]) if self.H.L.use_bias else self.a_valid_drop[i])))
            else:               
                self.a_valid_drop.append(g(T.dot(self.z_valid_drop[i-1],self.weights[i])))
                if len(self.layers) > i:
                    self.z_valid_drop.append(g(self.layers[i+1].activation_cv_dropout(g(T.add(self.a_valid_drop[i],self.biases[i][0,:])) if self.H.L.use_bias else self.a_valid_drop[i])))     
                   
        self.feedforward_valid_drop = self.z_valid_drop[-1]
        
        self.feedforward_valid_drop_function = function(inputs=[],outputs=self.feedforward_valid_drop)
        
        #feedforward for validation data with mean net multiplication
        a_valid = []
        z_valid = []
        for i in range(len(self.weights)):
            if i == 0: 
                #input units
                a_valid.append(g(T.dot(self.X_val,self.weights[i])))               
                z_valid.append(g(T.mul(self.layers[i+1].activation_crossvalidation((g(T.add(self.biases[i][0,:],a_valid[i])) if self.H.L.use_bias else a_valid[i])),
                                       1-self.layers[i+1].dropout)))
            else:
                a_valid.append(g(T.dot(z_valid[i-1],self.weights[i])))
                if len(self.layers) > i:
                    z_valid.append(g(self.layers[i+1].activation_crossvalidation((g(T.add(self.biases[i][0,:],a_valid[i])) if self.H.L.use_bias else a_valid[i]))))
                    
        self.feedforward_valid = z_valid[-1]
        #feedforward for training data with mean net multiplication (for train error)
        a_train = []
        z_train = []
        for i in range(len(self.weights)):
            if i == 0: 
                #input units
                a_train.append(g(T.dot(self.X[self.idx],self.weights[i])))               
                z_train.append(g(T.mul(self.layers[i+1].activation_crossvalidation((g(T.add(self.biases[i],a_train[i])) if self.H.L.use_bias else a_train[i])),
                                       1-self.layers[i+1].dropout)))
            else:
                a_train.append(g(T.dot(z_train[i-1],self.weights[i])))
                if len(self.layers) > i:
                    z_train.append(g(self.layers[i+1].activation_crossvalidation((g(T.add(self.biases[i],a_train[i])) if self.H.L.use_bias else a_train[i]))))
                    
        self.feedforward_train = z_train[-1]
        #feedforward of test data with mean net multiplication
        a_predict = []
        z_predict = []
        for i in range(len(self.weights)):
            if i == 0: 
                #input units
                a_predict.append(g(T.dot(self.X_test,self.weights[i])))               
                z_predict.append(g(T.mul(self.layers[i+1].activation_crossvalidation((g(T.add(self.biases[i][0,:],a_predict[i])) if self.H.L.use_bias else a_predict[i])),
                                         1-self.layers[i+1].dropout)))
            else:
                a_predict.append(g(T.dot(z_predict[i-1],self.weights[i])))
                if len(self.layers) > i:
                    z_predict.append(g(self.layers[i+1].activation_crossvalidation(g(T.add(self.biases[i][0,:],a_predict[i])) if self.H.L.use_bias else a_predict[i])))
                    
        self.feedforward_predict = z_predict[-1]
      
  
        
    def create_validation_and_train_error_functions(self):    
        
        
        self.cross_validation_function = function([],outputs=[np.float32(1.0) - 
            T.mul(T.sum(T.eq(T.argmax(self.feedforward_valid,axis=1), self.y_val.T)),self.batch_size_divisor_cval)])
        

        
        self.c_entropy_cv = function([],outputs=[-T.mean(T.sum(T.mul(self.t_val,T.log(self.feedforward_valid)),axis=0))]) 
        
        self.train_error_function = function([self.idx],[1 - 
            T.mul(T.sum(T.eq(T.argmax(self.feedforward_train,axis=1), self.y[self.idx].T)),self.batch_size_divisor)])  
        
        self.train_error_function_dropout = function([self.idx, self.A],[1 - 
            T.mul(T.sum(T.eq(T.argmax(self.A,axis=1), self.y[self.idx].T)),self.batch_size_divisor)]) 
        
        self.cross_validation_function_dropout = function([self.A],outputs=[np.float32(1.0) - 
            T.mul(T.sum(T.eq(T.argmax(self.A,axis=1), self.y_val.T)),self.batch_size_divisor_cval)])
        
        self.c_entropy_train = function([self.idx],outputs=[-T.mean(T.sum(T.mul(self.t[self.idx],T.log(self.feedforward)),axis=0))])
        
        self.predict_function = function([],outputs=[T.argmax(self.feedforward_predict,axis=1)])   
        
        self.predict_softmax_function = function([],outputs=[self.feedforward_predict])    
        
        
        
    def create_backprop_gradient_functions(self):
        self.errors =[]
        self.error_gradients = []
        error_function = None
        error_gradient = None
        for i in range(len(self.weights)):
            if len(self.errors) == 0:
                #this is the last layer of the net: The error is X - t because of 
                #the combination of softmax and cross entropy cost function
                error_function = g(T.sub(self.feedforward,self.t[self.idx]))  
                self.errors.append(error_function)
                error_gradient = g(T.dot(self.z[-2].T,self.errors[i]))       
                error_gradient = self.apply_L2_penalties_error_gradients(error_gradient, -1)        
                self.error_gradients.append(error_gradient)
                
            elif (len(self.weights) - 1) == i:  
                #this involves the input X instead of z-values as it is the first weights that
                #need to be updated                   
                self.errors.append(g(T.mul(T.dot(self.errors[-1],self.weights[1].T),
                                         self.layers[1].activation_derivative(self.z[0])))) 
                
                error_gradient = g(T.dot(self.X[self.idx].T,self.errors[-1]))      
                #error_gradient = self.apply_L2_penalties_error_gradients(error_gradient, 0)  
                self.error_gradients.append(error_gradient)
            else:
                self.errors.append(g(T.mul(T.dot(self.errors[-1],self.weights[-i].T),
                                         self.layers[-(i+1)].activation_derivative(self.z[-(i+1)]))))
                
                error_gradient = g(T.dot(self.z[-(i+2)].T,self.errors[-1]))     
                #error_gradient = self.apply_L2_penalties_error_gradients(error_gradient, -(i+1))  
                self.error_gradients.append(error_gradient)     
           
    
    def apply_L2_penalties_error_gradients(self, error_gradient, weight_index):
        if self.H.R.use_L2:        
            error_gradient = g(T.add(error_gradient, T.mul(self.H.R.L2_penalty,self.weights[weight_index])))       
                               
        return error_gradient

                    
    def create_weight_update_functions(self):
        updates = []
        for i in range(len(self.error_gradients)):
            updates.append((self.weights[i], g(T.sub(self.weights[i],T.mul(T.mul(self.error_gradients[-(i+1)],self.alpha),self.batch_size_divisor)))))
            updates.append((self.biases[i],g(T.sub(self.biases[i],T.mul(T.mul(self.errors[-(i+1)], self.alpha),self.batch_size_divisor)))))
            
        self.update_weight_function = function(inputs=[self.idx,self.alpha],updates= updates) 
        
    def hook_end_of_batch(self, batch):
        '''Calls all functions of the array hook_functions_batch
           with the batch number argument.
        '''
        if self.hook_functions_batch != None:        
            for i in range(len(self.hook_functions_batch)):
                self.hook_functions_batch[i](batch)
                
    def hook_in_crossvalid(self, batch):
        '''Calls all functions of the array hook_functions_crossvalid
           with the batch number argument.
        '''
        error = 0
        if self.hook_functions_crossvalid != None:       
            error = self.hook_functions_crossvalid(batch) 
        return error     

    def hook_after_epoch_cross_valid(self):
        '''Calls all functions of the array hook_functions_crossvalid_epoch
           after each epoch.
        '''
        error = 0
        if self.hook_functions_crossvalid_epoch != None:       
            error = self.hook_functions_crossvalid_epoch() 
        return error    
        
        
    def create_nesterov_weight_update_functions(self):
        '''Creates functions for Nesterov accelerated gradient
           which is similar to momentum. The difference is that 
           the gradient is calculated with the current weights plus 
           the momentum vector; the result is used to update the weights
           and momentum matrices. This is generally better than normal momentum.
           Also see Sutskever, 2013:
           http://www.cs.toronto.edu/~hinton/absps/momentum.pdf           
        '''
        nesterov_updates = []
        
        for i in range(len(self.nesterov_weights)):
            nesterov_updates.append((self.weights[i], g(T.add(self.nesterov_weights[i],self.H.L.momentum_weights[i]))))
        
        for i in range(len(self.nesterov_weights)):
            nesterov_updates.append((self.nesterov_weights[i],  g(T.add(self.nesterov_weights[i],self.H.L.momentum_weights[i])))) 
            
        self.nesterov_update_function = function([],updates=nesterov_updates)
            
    def create_weight_update_with_momentum_functions(self):
        weight_updates_with_momentum = []
        for i in range(len(self.weights)):
            weight_updates_with_momentum.append((self.weights[i], g(T.add(self.weights[i],self.H.L.momentum_weights[i]))))          
            
        self.weight_updates_with_momentum_function = function(inputs=[],
                    updates=weight_updates_with_momentum)
            
    def create_momentum_weight_update_functions(self):
        momentum_updates = []
        for i in range(len(self.H.L.momentum_weights)):
            momentum_updates.append(
            (self.H.L.momentum_weights[i], 
             g(T.mul(self.batch_size_divisor,T.sub(T.mul(self.M,self.H.L.momentum_weights[i]),T.mul(self.alpha,self.error_gradients[-(i+1)]))))))
            
        self.H.L.momentum_update_function = function(inputs=[self.idx, self.M, self.alpha],                  
          updates=momentum_updates) 
        
    def constaint_weight_function(self):
        '''Calculates the max squared element of the weight vector
           to rescale it.
        '''
        max_values = []
        for w in self.weights:
            max_values.append(g(T.max(T.square(w))))
        self.weight_constaint_function = function([],outputs=max_values)
        
    def constaint_weights(self):
        '''Rescales the weights so that the maximum element of the matrix
           is 1.
        '''
        max_values = self.weight_constaint_function()
        for i in range(len(self.weights)):
            if arr(max_values[i]) > 1.0:
                print 'Constaining weights No ' + str(i) + ': Divide by ' + str(arr(max_values[i]))
                self.weights[i].set_value(np.float32(self.weights[i].get_value()/arr(max_values[i])))
                print 'New max value: ' + str(arr(self.weight_constaint_function()[i]))         
            
    
                                         
    def create_weight_decay_updates(self):
        '''Decays the weights exponentially.
        '''
        weight_updates = []
        for i in range(len(self.weights)):
            weight_updates.append((self.weights[i], g(T.mul(self.weights[i],self.alpha))))
              
        self.decay_weights = function(inputs=[self.alpha],updates=weight_updates)
   
    def init_weight_and_data_variables(self, data_X, data_y, data_t, valid_X, valid_y, valid_t, test_X):
        self.batch_size = data_X[0].shape[0]
        self.batches = data_X.shape[0]        
        
        self.batch_size_divisor = np.float32(1.0/np.float32(data_X.shape[1]))
        self.batch_size_divisor_cval = np.float32(1.0/np.float32(valid_X.shape[0]))    
        
        self.initialize_weights()
                   
        self.best_weights = []
        for i in range(len(self.weights)):
            self.best_weights.append(np.copy(arr(self.weights[i].get_value())))
        
        self.X = shared(data_X, 'float32', borrow=True)
        self.y = shared(data_y, 'float32', borrow=True)
        self.t = shared(data_t, 'float32', borrow=True)
                
        self.X_val = shared(valid_X, 'float32', borrow=True)
        self.y_val = shared(valid_y, 'float32', borrow=True)
        self.t_val = shared(valid_t, 'float32', borrow=True)  
        
        self.X_test = shared(np.float32(test_X), 'float32', borrow=True)
                    
        
    def plot_results(self, valid, train, epochs, filename):
        '''Plots train and crossvalidation error starting from 0.05.
        '''
        print valid.shape
        plt.axis([0,epochs,0,0.05])
        plt.title('Epochs: ' + str(epochs) + ', '  +'Hidden layer units: ')
        plt.plot(range(epochs),valid,color='blue')
        plt.plot(range(epochs),train,color='red')
        plt.tight_layout()
        plt.savefig(self.result_dir + filename +'.pdf')
     
    def plot_weights(self, weight, filename):
        '''Creates density plots of the weights
        '''
        hist, bins = np.histogram(weight,bins = 50)
        width = 0.7*(bins[1]-bins[0])
        center = (bins[:-1]+bins[1:])/2
        plt.bar(center, hist, align = 'center', width = width)
        plt.savefig(self.result_dir+ filename + '.pdf')
        
    def write_report(self, report_text, filename):
        '''Writes a file which lists all hyperparameter that were used in the training.
        '''
        with open(self.result_dir+ filename + '.txt', "w") as text_file:
            text_file.write(report_text)
            
    def print_and_predict_results(self, epoch):
        '''Writes report, creates plots, predicts test data and prints test error.
        '''
        print 'Printing results...'
        plt.hold(True)
        layer_sizes = ''    
        for i, layer in enumerate(self.layers):
            layer_sizes += '_' if i > 0 else '' 
            layer_sizes += str(layer.size) 
        filename = str(np.round(self.valid_history[epoch]*100,4))+'%_'+str(epoch)+'_'+ layer_sizes +'_'+str(self.batch_size)
        self.plot_results(self.valid_history[:epoch], self.train_history[:epoch], epoch, filename)
        plt.hold(False)
        for i in range(len(self.weights)):            
            self.plot_weights(arr(self.weights[i].get_value()),filename + '_W' + str(i+1))        
        print 'Writing report...'   
        self.write_report(self.H.get_reporttext()  + 'batch_size = ' + str(self.batch_size)  + '\n'  + 
                        'epochs = ' + str(epoch)  + '\n'  + 
                        'Execution time = ' + str(time.time()-self.time1) + ' seconds',
                        filename) 
     
        y_test = np.float32(np.load('/home/tim/development/test_y.npy'))
        X_test = np.float32(np.load('/home/tim/development/test_X.npy'))
        kaggle = np.float32(np.load('/home/tim/development/test_kaggle.npy'))
        self.X_test.set_value(X_test)
        prediction = arr(self.predict_function())
        print 'Test error: ' + str((1.0 - (np.sum(np.equal(prediction, y_test.T))/(prediction.shape[1]*1.0))))
        self.X_test.set_value(kaggle)
        prediction = np.int32(np.vstack([np.arange(1,1+kaggle.shape[0]), arr(self.predict_function())]).T)
        np.savetxt(self.result_dir + filename + '.csv',prediction, '%i',delimiter=',')

        
    def handle_epoch_decay_and_increases(self, epoch):
        '''Handles increases/decreases for momentum, learning rate, weight decay and dropout.
        '''
        if self.H.L.use_momentum and (epoch - self.H.S.last_save_epoch) > self.H.L.transient_phase:
            print 'Leaving transient phase... disabling momentum'
            self.H.L.use_momentum = False
        
        if self.H.L.use_momentum_increase:
            self.H.L.momentum = self.H.L.momentum + self.H.L.momentum_increase
            self.H.L.momentum = 0.999 if self.H.L.momentum > 0.999 else self.H.L.momentum        
        
        if self.H.R.use_weight_decay:            
            self.H.R.weight_decay = np.float32(self.H.R.weight_decay *(1 - self.H.R.weight_decay_decay))
            self.decay_weights(np.float32(1-self.H.R.weight_decay))
            
        if self.H.L.use_learning_rate_decay:
            self.H.L.learning_rate = np.float32(self.H.L.learning_rate*(1-self.H.L.learning_rate_decay))  
            
    
        for layer in self.layers:
            layer.handle_dropout_decay(epoch - self.H.S.last_save_epoch)
    
    def handle_early_stopping_and_ensembles(self, epoch):
        '''Saves the best weights for early stopping and also handles ensemble creation.
        '''
        if((self.H.S.safe_weights_threshold > self.H.S.saved_weights_error and self.H.R.use_early_stopping)
       or (self.H.S.last_save_epoch + self.H.S.epochs_force_reinitilization) < epoch and self.H.S.use_ensemble):            
            layer_sizes = ''    
            for i, layer in enumerate(self.layers):
                layer_sizes += '_' if i > 0 else '' 
                layer_sizes += str(layer.size) 
            
            filename = str(np.round(self.valid_history[epoch]*100,4))+'%_'+str(epoch)+'_'+layer_sizes+'_'+str(self.batch_size)
            if(not self.H.S.use_ensemble):
                self.H.S.safe_weights_threshold = self.H.S.saved_weights_error    
                print 'Saving best weights with error ' + str(self.H.S.saved_weights_error) + ' ...'
                for i in range(len(self.weights)):                    
                    np.save(self.result_dir + 'W' + str(i+1) + '.npy', self.best_weights[i])
               
                X_test = np.float32(np.load('/home/tim/development/test_X.npy'))
                kaggle = np.float32(np.load('/home/tim/development/test_kaggle.npy'))
                self.X_test.set_value(kaggle)
                prediction = np.int32(np.vstack([np.arange(1,1+arr(self.X_test.get_value()).shape[0]), arr(self.predict_function())]).T)
                np.savetxt(self.result_dir + 'prediction' + '.csv',prediction, '%i',delimiter=',')  
            else:
    
                if((self.H.S.last_save_epoch + self.H.S.epochs_force_reinitilization) < epoch and self.H.S.use_ensemble):
                    print "Forced reinitilization..."
                else:  
                    kaggle = np.float32(np.load('/home/tim/development/test_kaggle.npy'))
                    X_test = np.float32(np.load('/home/tim/development/test_X.npy'))
                    self.X_test.set_value(kaggle)
                    if self.ensemble == None:
                        print "Creating ensemble network..."
                        self.ensemble = np.matrix(arr(self.predict_function())).T
                        self.X_test.set_value(X_test)
                        self.ensemble_MNIST = np.matrix(arr(self.predict_function())).T                     
                        self.ensemble_softmax = arr(self.predict_softmax_function()[0])
                    else: 
                        print "Adding network " + str(self.ensemble.shape) + " to the ensemble"
                        self.ensemble = np.hstack([self.ensemble, np.matrix(arr(self.predict_function())).T])
                        X_test = np.float32(np.load('/home/tim/development/test_X.npy'))
                        self.X_test.set_value(X_test)
                        self.ensemble_MNIST = np.hstack([self.ensemble_MNIST, np.matrix(arr(self.predict_function())).T])
                        self.ensemble_softmax = np.vstack([self.ensemble_softmax, arr(self.predict_softmax_function()[0])] )
                                    
                    if(self.H.S.ensemble_count == self.ensemble.shape[1]):
                        print "Saving ensemble network..."
                        np.save(self.result_dir + 'ENSEMBLE_' + str(self.H.S.safe_weights_threshold) + '_' + str(self.H.S.ensemble_count) + '.npy',self.ensemble)                        
                        np.save(self.result_dir + 'ENSEMBLE_MNIST_' + str(self.H.S.safe_weights_threshold) + '_' + str(self.H.S.ensemble_count) + '.npy',self.ensemble_MNIST)
                        np.save(self.result_dir + 'ENSEMBLE_softmax_' + str(self.H.S.safe_weights_threshold) + '_' + str(self.H.S.ensemble_count) + '.npy',self.ensemble_softmax)
                
                self.reinitialize_weights()
                self.reinitilize_parameters(epoch)                 
            
            
    def handle_early_stopping(self, epoch):
        if(self.valid_history[epoch] < self.H.S.safe_weights_threshold):
            self.H.S.saved_weights_error = self.valid_history[epoch]
            for i in range(len(self.weights)):
                self.best_weights.append(np.copy(arr(self.weights[i].get_value())))
                
    def default_training_sequence(self, batch):
        if not self.H.L.use_momentum:                                  
            self.update_weight_function(batch,self.H.L.learning_rate)
        else:  
            self.weight_updates_with_momentum_function()   
            self.H.L.momentum_update_function(batch,self.H.L.momentum, self.H.L.learning_rate)
            self.nesterov_update_function()
            
    def train(self):
        '''Trains the neural network and prints the train and test error every 5 epochs.
        '''
        for epoch in range(self.H.L.epochs): 
            self.H.L.learning_rate
            for i in range(self.batches): 
                self.training_sequence(i)                    
                self.hook_end_of_batch(i)
                
            error = 0
            entropy_error = 0    
            error_custom = 0
            for i in range(self.batches):            
                error += arr(self.train_error_function(i))[0]
                error_custom += self.hook_in_crossvalid(i)/self.batches   
                #entropy_error += arr(self.c_entropy_train(i))[0]
                                        
            self.valid_history[epoch] = np.array(self.cross_validation_function())     
            self.train_history[epoch] = error/self.batches
            self.handle_early_stopping(epoch)    
                 
            if(epoch % 5 == 0):        
                print 'EPOCH: ' + str(epoch)                 
                print 'Cross validation error: ' + str(self.valid_history[epoch])
                print 'Train error: ' + str(self.train_history[epoch])
                if error_custom > 0:
                    print 'Custom error: ' + str(error_custom)
                cross_valid_custom_error = self.hook_after_epoch_cross_valid()
                if cross_valid_custom_error > 0:
                    print 'Custom cross valid error: ' + str(cross_valid_custom_error)
                #print 'Cross entropy validation error: ' + str(self.c_entropy_cv()[0])
                #print 'Cross entropy error: ' + str(entropy_error)                 
                        
            self.handle_epoch_decay_and_increases(epoch)
            
            
            #self.constaint_weights()
            
        
            if self.heardEnter():        
                self.print_and_predict_results(epoch) 
        
            self.handle_early_stopping_and_ensembles(epoch)
            
            if(self.ensemble != None):
                if(self.H.S.ensemble_count == self.ensemble.shape[1]):
                    break    
        
        self.print_and_predict_results(epoch)
            
import sys
import time
from batch_creator import batch_creator
sys.path.append('/home/tim/cudamat/')
import cudamat as cm
import gnumpy as gpu
import numpy as np
from util import Util


class RBM:
    '''Single layer restricted Boltzmann machine trained with 
       increasing contrast divergence.    
    '''
    
    def __init__(self):
        self.u = Util()
        gpu.board_id_to_use = 1
        print 'USE GPU' + str(gpu.board_id_to_use)
        gpu.expensive_check_probability = 0 

        b = batch_creator()
        path_train = '/home/tim/development/train.csv'
        path_test = '/home/tim/development/test_X.csv'
        
        batch_size = 100
        set_sizes = [1.00,0.00,0.00]
        
        data = b.create_batches([path_train, path_test], [0, -1], set_sizes,batch_size, standardize = False)
        self.data = gpu.garray(data[0][0]/255.)
        self.v_original = None
        
        
        #self.w = gpu.garray(self.u.create_sparse_weight(784, 800))
        self.w = gpu.garray(np.random.randn(784,800))*0.1        
        self.bias_h = gpu.zeros((1,800))
        self.bias_v = gpu.zeros((1,784))
        self.w_updt = gpu.zeros((784, 800))
        self.bias_h_updt = gpu.zeros((1,800))
        self.bias_v_updt = gpu.zeros((1,784))
        self.h = gpu.zeros((100,800))
        self.v = gpu.zeros((100,784))
        self.time_interval = 0


    def positive_phase(self):          
        self.h = gpu.logistic(gpu.dot(self.v,self.w)+self.bias_h)   
        self.w_updt += gpu.dot(self.v.T,self.h)        
        self.bias_h_updt += gpu.sum(self.h,axis=0)
        self.bias_v_updt += gpu.sum(self.v,axis=0)      
       
        
        
    def gibbs_updates(self): 
        self.h = (self.h > gpu.rand(100,800))    
        self.v = gpu.logistic(gpu.dot(self.h,self.w.T)+self.bias_v)
       
        
    def negative_phase(self): 
        self.h = gpu.logistic(gpu.dot(self.v,self.w)+self.bias_h)
        self.w_updt -= gpu.dot(self.v.T,self.h)
        self.bias_h_updt -= gpu.sum(self.h,axis=0)
        self.bias_v_updt -= gpu.sum(self.v,axis=0)


    def train(self):
        epochs = 50
        batches = self.data.shape[0]
        alpha = 0.1         
        self.time_interval = 0
        t1 = time.time()
        cd = 1
        cd3 = 10
        cd10 = 15   
   
     
        for epoch in xrange(epochs):    
            error = 0   
            for i in xrange(batches):
                self.w_updt = gpu.zeros((784, 800))
                self.bias_h_updt = gpu.zeros((1,800))
                self.bias_v_updt = gpu.zeros((1,784)) 
                          
                for j in range(cd):
                    self.v_original = gpu.garray(self.data[i])
                    self.v = self.v_original                  
                    self.positive_phase()
                    self.gibbs_updates()
                    self.negative_phase()                
            
                self.w += alpha*self.w_updt/100.
                self.bias_h += alpha*self.bias_h_updt/100.
                self.bias_v += alpha*self.bias_v_updt/100.
                t0 = time.time()
                error += gpu.mean((self.v-self.v_original)**2)
                self.time_interval += time.time() - t0
                
            print 'EPOCH: ' + str(epoch + 1)
            print 'Reconstruction error: ' + str(error/batches)
            
            if epoch == cd10:
                cd = 10
            elif epoch == cd3:
                cd = 3                   
          
        print 'Time interval: ' + str(self.time_interval)
        print 'Training time: ' + str(time.time() - t1)
        np.save('/home/tim/development/RBM_w1.npy',self.w.as_numpy_array())
        
        
    def train_cudamat(self):
        pass
        
        
        
rbm = RBM()
rbm.train()
        
        
        
        

import sys
import time
from batch_creator import batch_creator
sys.path.append('/home/tim/cudamat/')
import gnumpy as gpu
import numpy as np
from util import Util

'''Logistic regression to benchmark different momentum procedures.
'''

gpu.board_id_to_use = 0
print 'USE GPU' + str(gpu.board_id_to_use)
gpu.expensive_check_probability = 0 
rng = np.random.RandomState(1234)

rng = np.random.RandomState(1234)

u = Util()
b = batch_creator()
path_train = '/home/tim/development/train.csv'
result_dir = '/home/tim/development/results/'


batch_size = 128
set_sizes = [0.80,0.20]

data = b.create_batches([path_train], [0], set_sizes,batch_size, standardize = False)
X_test = np.float32(np.load('/home/tim/development/test_X.npy'))/255.


X = gpu.garray(data[0][0])/255.
t = gpu.garray(data[0][2])
y = gpu.garray(data[0][1])


X_val = gpu.garray(u.create_batches(data[1][0], 120))
y_val = u.create_batches(data[1][1].T,120)

w = gpu.garray(u.create_sparse_weight(784, 10))
m = gpu.zeros((784,10))
b = gpu.zeros((1,10))
mb = gpu.zeros((1,10))  

alpha = 0.1
momentum = 0.5
momentum_type = 1

for i in xrange(200):
    for i in xrange(X.shape[0]):        
        
        if momentum_type == 1:
            '''Use nesterov momentum to train the weights
            '''
            n = w + (m*momentum)
            nb = b + (mb*momentum)
            out = gpu.softmax(gpu.dot(X[i],n)+nb)
            gradb = gpu.dot(gpu.ones((1,batch_size)),out - t[i]) 
            grad = gpu.dot(X[i].T,out - t[i])
            
            m = m*momentum - (alpha*grad/128.)
            mb = mb*momentum - (alpha*gradb/128.)
            w += m
            b += mb
        elif momentum_type == 2:            
            '''Use classic momentum to train the weights
            '''
            out = gpu.softmax(gpu.dot(X[i],w)+b)
            gradb = gpu.dot(gpu.ones((1,batch_size)),out - t[i]) 
            grad = gpu.dot(X[i].T,out - t[i])
            
                       
            m = m*momentum - (alpha*grad/128.)
            mb = mb*momentum - (alpha*gradb/128.)
            
            w+=m
            b+=mb
            
        else:   
            '''Use no momentum to train the weights
            '''        
            out = gpu.softmax(gpu.dot(X[i],w)+b)
            gradb = gpu.dot(gpu.ones((1,batch_size)),out - t[i]) 
            grad = gpu.dot(X[i].T,out - t[i])          
            w -= (alpha*grad/128.)
            b -= (alpha*gradb/128.)
            
    momentum += 0.001  
    
    if momentum > 1: momentum = 0.99

    error = 0
    for i in xrange(X.shape[0]):
        error += 1.0 - (gpu.sum(np.argmax(gpu.dot(X[i],w)+b,axis=1) == y[i].as_numpy_array())/128.)
    print 'train: ' + str(error/(1.0*X.shape[0]))
    error = 0
    for i in range(100):
        error += 1.0 - (gpu.sum(np.argmax(gpu.dot(X_val[i],w)+b,axis=1) == y_val[i])/120.)
    print 'cross : ' + str(error/100.)
        
        
        
         
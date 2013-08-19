from theano import function, shared, Out, config, scan as s
from theano.sandbox.cuda.basic_ops import gpu_from_host as g
from theano.sandbox.cuda import  CudaNdarray as cu
from theano.tensor.shared_randomstreams import RandomStreams
from numpy import asarray as arr
import theano.tensor as T
import numpy as np
import time
import math
import h5py
import uuid
import matplotlib.pyplot as plt

from batch_creator import batch_creator

'''Restricted Boltzmann machine for the GPU written with theano
'''

b = batch_creator()
path_train = '/home/tim/development/train.csv'
path_test = '/home/tim/development/test_X.csv'
result_dir = '/home/tim/development/results/'

batch_size = 64
set_sizes = [1.00,0.00,0.00]

data = b.create_batches([path_train, path_test], [0, -1], set_sizes,batch_size, standardize = False)
data_X = data[0][0]/255.

# training parameters
epsilon = 0.0001

epochs = 50
batch_size = data_X.shape[1]
num_batches = data_X.shape[0]

dropout_input = 0.3

# model parameters
dim_visible = data_X.shape[2]
dim_hidden = 5000

rng = np.random.RandomState(1234)
rdm = RandomStreams(seed=1234)
sample_range = 10000000 - (dim_hidden*batch_size)
sample_range_dropout = 10000000 - (dim_visible*batch_size)

X = shared(data_X, 'float32', borrow=True)
v = shared(np.float32(np.zeros((batch_size,dim_visible))), 'float32', borrow=True)
# initialize weights
w_vh = shared(np.float32(rng.uniform(
                    low=4*-np.sqrt(6. / (dim_visible + dim_hidden)),
                    high=4*np.sqrt(6. / (dim_visible + dim_hidden)),
                    size=(dim_visible, dim_hidden))),'float32', borrow=True)

w_v = shared(np.float32(np.zeros((1,dim_visible))),'float32', borrow=True,  broadcastable = (True, False))
w_h = shared(np.float32(np.matrix(np.zeros((1,dim_hidden)))),'float32', borrow=True, broadcastable = (True, False))

# initialize weight updates
wu_vh = shared(np.float32(np.zeros((dim_visible,dim_hidden))),'float32', borrow=True)
wu_v = shared(np.float32(np.zeros((dim_visible,))),'float32', borrow=True)
wu_h = shared(np.float32(np.zeros((dim_hidden,))),'float32', borrow=True)

uniform_sample = shared(np.matrix(np.float32(np.random.rand(10000000,1))),'float32', borrow=True)
bino_input = shared(np.matrix(np.float32(np.random.binomial(1,1-dropout_input,(10000000,1)))),config.floatX, borrow=True)


t1 = T.fmatrix("t1")
a1 = T.fmatrix("a1")
e1 = T.fmatrix("e1")
idx = T.iscalar("idx")
bsize = T.fscalar("bsize")
alpha = T.fscalar("alpha")
cv_size = T.fscalar("cv_size")


drop_input = lambda rand: T.reshape(bino_input[rand:rand + (batch_size*dim_visible)],(batch_size,dim_visible))
input_drop = drop_input(rdm.random_integers(low=0, high=sample_range_dropout))

h = T.nnet.sigmoid(T.add(T.dot(v,w_vh),w_h))


u_w_plus = function([],updates=[(wu_vh, g(T.add(wu_vh,T.dot(v.T,h)))),
                            (wu_v,  g(T.add(T.sum(v[:],axis=0),wu_v))),
                            (wu_h, g(T.add(T.sum(h[:],axis=0),wu_h)))
                            ])

u_w_minus = function([],updates=[(wu_vh, g(T.sub(wu_vh,T.dot(v.T,h)))),
                            (wu_v,  g(T.sub(T.sum(v[:],axis=0),wu_v))),
                            (wu_h, g(T.sub(T.sum(h[:],axis=0),wu_h)))
                            ])

sample = lambda rdm: T.reshape(uniform_sample[rdm:rdm + (dim_hidden*batch_size)],(batch_size,dim_hidden))

gibbs = T.cast(T.gt(h, sample(rdm.random_integers(low=0, high=sample_range))),'float32')



update_v = function([],outputs=[g(T.nnet.sigmoid(T.add(T.dot(gibbs,w_vh.T),w_v)))])

update_w = function([alpha],updates=[(w_vh, g(T.add(w_vh,T.mul(alpha,wu_vh)))),
                                (w_v, g(T.add(w_v,T.mul(alpha,w_v)))),
                                (w_h, g(T.add(w_h,T.mul(alpha,w_h))))
                               ])

error = function([idx],[g(T.mean(T.square(T.sub(v,X[idx]))))])

drop_input = function([idx],T.mul(X[idx],input_drop))

   
time1 = time.time()

contrast_divergence_steps = 1
for epoch in range(epochs):
    for i in range(num_batches):   
        if(epoch > 15): 
            contrast_divergence_steps = 3
        elif (epoch > 20): 
            contrast_divergence_steps = 10
        else: 
            contrast_divergence_steps = 1                           
         
        for j in range(contrast_divergence_steps):
            v.set_value(np.float32(drop_input(i)),borrow=True)           
            u_w_plus()     
            v.set_value(np.float32(arr(update_v()[0])),borrow=True)
            u_w_minus() 
       
        update_w(np.float32(epsilon/np.float32(batch_size)))   
    print 'EPOCH: ' + str(epoch + 1)
    print 'Squared error: ' + str(arr(error(i)[0]))
    epsilon = epsilon * 0.90   
    
time2 = time.time()
print 'Function execution: '  + str((time2 - time1)) + ' seconds'
np.save('/home/tim/Downloads/RBM_weights_digits.npy', w_vh.get_value())

 


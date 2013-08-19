import sys
import time
from batch_creator import batch_creator
sys.path.append('/home/tim/cudamat/')
import cudamat as cm
import gnumpy as gpu
import numpy as np
from util import Util

u = Util()
gpu.board_id_to_use = 0
print 'USE GPU' + str(gpu.board_id_to_use)
gpu.expensive_check_probability = 0 
rng = np.random.RandomState(1234)

b = batch_creator()
path_train = '/home/tim/development/train.csv'
path_test = '/home/tim/development/test_X.csv'
result_dir = '/home/tim/development/results/'

batch_size = 100
set_sizes = [0.80,0.20,0.00]

data = b.create_batches([path_train, path_test], [0, -1], set_sizes,batch_size, standardize = False)
X_test = gpu.garray(np.load('/home/tim/development/test_X.npy'))
y_test = np.load('/home/tim/development/test_y.npy')
y_val = u.create_batches(data[1][1].T,120)
data_t_val = data[1][2]

X = gpu.garray(data[0][0]/255.)
y = gpu.garray(data[0][1])
t = gpu.garray(data[0][2])
X_val = gpu.garray(u.create_batches(data[1][0]/255., 120))
batches = X.shape[0]

alpha = 0.001
L2 = 0.0005
epochs = 1000
momentum = 0.5
n =  X.shape[1]*X.shape[0]*1.0

input = 784
hidden = 800
output = 10

w1 = gpu.garray(np.load('/home/tim/development/RBM_w1.npy'))
#w1 = gpu.garray(u.create_sparse_weight(784,hidden))
w2 = gpu.garray(u.create_sparse_weight(hidden,output))
b1 = gpu.zeros((1,hidden))
b2 = gpu.zeros((1,output))

m1 = gpu.zeros((input,hidden))
m2 = gpu.zeros((hidden,output))
mb1 = gpu.zeros((1,hidden))
mb2 = gpu.zeros((1,output))

d02 = gpu.garray(np.float32(np.random.binomial(1,0.8,(75, batch_size, input))))
d05 = gpu.garray(np.float32(np.random.binomial(1,0.5,(75, batch_size, hidden))))


t1 = time.time()   
time_softmax = 0
t0 = time.time()
cv = []
train = np.float32(np.array(range(epochs)))
cv = np.float32(np.array(range(epochs)))
for epoch in range(epochs):

    for i in xrange(batches):
        #nesterov accelerated gradient
        n1 = w1+(m1*momentum)#nesterov updates 2.2 sec
        n2 = w2+(m2*momentum)
        nb1 = b1+(mb1*momentum)
        nb2 = b2+(mb2*momentum)
  
        z0 = X[i]*d02[rng.randint(0,75)]
        z1 = (gpu.dot(z0,n1)+nb1).logistic()*d05[rng.randint(0,75)]#dropout and activations 7.1 sec 
        t0 = time.time()            
        feedforward = gpu.softmax(gpu.dot(z1,n2)+nb2)
        time_softmax += time.time() - t0      
        #softmax 0.48 sec
        #gradients
        e1 = (feedforward - t[i])
        grad2 = gpu.dot(z1.T,e1) 
        grad1 = gpu.dot(X[i].T,(gpu.dot(e1,n2.T)* z1*(1-z1)))#grads 6 sec
        gradb2 = gpu.dot(gpu.ones((1, batch_size)),e1)
        gradb1= gpu.dot(gpu.ones((1, batch_size)),(gpu.dot(e1,n2.T)* z1*(1-z1)))
        #momentum and weight updates
        m1 = (momentum*m1) - ((grad1 + n1*L2)*alpha/(batch_size*1.0))#momentum und weight updates 7.4 sec    
        m2 = (momentum*m2) - ((grad2 + n2*L2)*alpha/(batch_size*1.0)) 
        mb1 = (momentum*mb1) - ((gradb1 + nb1*L2)*alpha/(batch_size*1.0))
        mb2 = (momentum*mb2) - ((gradb2 + nb2*L2)*alpha/(batch_size*1.0))
      
        w1 = w1 + m1
        w2 = w2 + m2    
        b1 = b1 + mb1
        b2 = b2 + mb2

    momentum = momentum + 0.001
    
    if momentum > 0.95: momentum = 0.95    
        
    batch_error_cv = 0
    for i in range(100):
        batch_error_cv += 1.0 - (gpu.sum(np.argmax(gpu.dot(gpu.logistic(gpu.dot(X_val[i],w1))*0.5,w2),axis=1) == y_val[i])/120.)
  
    batch_error = 0   
    for i in xrange(batches):#train error 5.9 sec
        z1 = gpu.dot(X[i],w1).logistic()*0.5
        feedforward = gpu.dot(z1,w2)
        batch_error += 1. - (np.sum(np.equal(np.argmax(feedforward,axis=1),y.as_numpy_array()[i].T)/(batch_size*1.0)))
    '''    
    if gpu.max(w1)**2 > 9:
        print 'halving the weights of w1'
        w1 = w1/2.
        m1 = m1/2.
    
    if gpu.max(w2)**2 > 9:
        print 'halving the weights of w2'
        w2 = w2/2.
        m2 = m2/2.
    '''   
          
    train[epoch] = batch_error/batches
    cv[epoch] = batch_error_cv/100.
    
    if u.heardEnter():
        u.plot_results(cv[:epoch], train[:epoch], epoch, result_dir + str(np.round(cv[epoch],4)))
        u.plot_weights(w1.as_numpy_array(), result_dir + str(np.round(cv[epoch],4))+'_W1')
        u.plot_weights(w2.as_numpy_array(), result_dir + str(np.round(cv[epoch],4))+'_W2')
        print 'Test error: ' + str(1.0 - (gpu.sum(np.argmax(gpu.dot(gpu.logistic(gpu.dot(X_test,w1))*0.5,w2),axis=1) == y_test.T))/10000.)
    
    if epoch % 5 == 0:   
        print 'EPOCH: ' + str(epoch)
        print 'Cross validation: ' + str(batch_error_cv/100.)
        print 'Train error: ' + str(batch_error/batches)   
        
print 'run time: ' + str(time.time() - t1)
print 'time softmax: ' + str(time_softmax)

u.plot_results(cv[:epoch], train[:epoch], epoch, result_dir + str(np.round(cv[epoch],4)))
u.plot_weights(w1.as_numpy_array(), result_dir + str(np.round(cv[epoch],4))+'_W1')
u.plot_weights(w2.as_numpy_array(), result_dir + str(np.round(cv[epoch],4))+'_W2')
print 'Test error: ' + str(1.0 - (gpu.sum(np.argmax(gpu.dot(gpu.logistic(gpu.dot(X_test,w1))*0.5,w2),axis=1) == y_test.T))/10000.)









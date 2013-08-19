import os
import numpy as np


'''

'''
def vote_on_results(x, print_threshold):
    '''Combines result of ensembles by letting 
       them vote on the outcome. 
       Givens: print_threshold: cases below this 
                                threshold are printed
                                to the console
    
    '''
    result = np.zeros((x.shape[0],))
    count = np.zeros((10,))
    
    for i in range(x.shape[0]):
        count = np.zeros((10,))
        for j in range(10):
            count[j] = np.sum(x[i,:]==j)
            
        result[i] = np.argmax(count)
        if(np.max(count) < print_threshold):
            print x[i,:]
            print result[i]

is_MNIST_test = True

path_MNIST = '/home/tim/development/results/ENSEMBLE_MNIST_'
path_softmax = '/home/tim/development/results/ENSEMBLE_softmax_'

name1 = '0.01_11'
x1 = np.load('/home/tim/development/results/ENSEMBLE_'+ name1 + '.npy')

name2 = '0.01_15'
x2 = np.load('/home/tim/development/results/ENSEMBLE_'+ name2 + '.npy')
name3 = '0.0105_10'
x3 = np.load('/home/tim/development/results/ENSEMBLE_'+ name3 + '.npy')
name = '0.012_10_0.011_10_0.0105_20'

name1_MNIST = '0.01_15_rdm'
x1_MNIST = np.load(path_MNIST + name1_MNIST + '.npy')

name2_MNIST = '0.01_15'
x2_MNIST = np.load(path_MNIST + name2_MNIST + '.npy')

name3_MNIST = '0.0105_15'
x3_MNIST = np.load(path_MNIST + name3_MNIST + '.npy')

x = np.hstack([x1,x2])

#x= np.hstack([x1_MNIST, x2_MNIST, x3_MNIST])

vote_on_results(x, 13)
        
name1_softmax = '0.01_15'
x1_softmax = np.load(path_softmax + name1_softmax+ '.npy')

name2_softmax = '0.01_15_rdm'
x2_softmax = np.load(path_softmax + name2_softmax+ '.npy')

name3_softmax = '0.01_15'
x3_softmax = np.load(path_softmax + name3_softmax+ '.npy')

x = np.vstack([ x2_softmax])


#result = np.argmax(np.sum(x.reshape(15,10000,10),axis=0),axis=1)
result = np.argmax(x[30000:40000,:],axis=1)



#end_result = np.hstack([np.matrix(np.arange(1,x.shape[0]+1)).T, np.matrix(result).T])
if not is_MNIST_test:
    np.savetxt('/home/tim/development/results/end_result_ '+ name +'.csv',end_result, '%i', delimiter=',')
else:
    y_data = np.load('/home/tim/development/test_y.npy')    
    print 'Test error: ' + str((1.0 - (np.sum(np.equal(result, y_data.T))/10000.)))  
    print 'Test errors: ' +str(10000 - np.sum(np.equal(result, y_data.T)))  




import os
import numpy as np

class batch_creator:
    ''' Loads csv or numpy data and creates batches with balanced classes,
        i.e. X = [1, 1, 3, 3] -> batches = [[1,3],[1,3]]    
    '''
    

    def load_numpy_data(self, paths):
        np_paths = map(lambda i: i.replace('.csv', '.npy'),paths)
        return map(np.load, np_paths)
    
    def load_data(self, paths, load_csv, standardize):
        ''' Given 2 .csv paths loads .csv data with method load_csv - or if already created 
            .npy data with the same name. Creates .npy data when .csv data
            was read successfully.         
        '''
        if '.csv' in paths[0]:
            if os.path.isfile(
               os.path.dirname(paths[0]) + '/' + 
               os.path.basename(paths[0]).replace('.csv','.npy') ): 
                                           
                data_sets = self.load_numpy_data(paths)
            else:
                print 'Loading .csv files...'
                data_sets = map(load_csv, paths)
                
                if standardize:
                    data_sets = map(self.standardize_data, data_sets)                   
                  
                for path, data_set in zip(paths, data_sets):  
                    np.save(path.replace('.csv', '.npy'), data_set)
                print 'Saving .npy files...'
                
        return data_sets
        

    def check_integrity(self, paths, column_labels, set_sizes):
        if np.sum(set_sizes) > 1:
            raise Exception('The sum of the set size if bigger than 1.')
        if len(column_labels) > len(paths):
            raise Exception('There are more labels than data sets.')
        elif len(column_labels) < len(paths):
            raise Exception('Every data set needs to have a label column (-1 for no label column).')
        
    def get_label_from_set(self, data, label):
        if label != -1:
            return data[:,label]
        else:
            return None
    
    def remove_label_from_set(self, data, label):        
        if label == 0:
            return data[:,1:]
        elif label > 0:
            return data[:,:]
        else:
            return data
        
    def create_t_dataset(self, y):  
        '''Creates t matrix, i.e. a binary n times c matrix
           where n are the number of cases, and c the number of class labels.           
        '''      
        if y != None:
            Y = np.matrix(y)
            Y = Y.T if Y.shape[0] == 1 else Y
            
            no_labels = np.max(y)
            t = np.zeros((Y.shape[0],no_labels+1))
            for i in range(Y.shape[0]):
                t[i,Y[i,0]] = 1
                
            return t
        else:
            return None   
        
    def shuffle_set(self, data_set_X, data_set_y, data_set_t):
        '''Randomizes the data, label, and binary class matrix at the same time.
        '''
        n = data_set_X.shape[0]
        rdm_idx = np.arange(0,n)
        np.random.shuffle(rdm_idx)
        new_X = np.zeros((data_set_X.shape))
        new_y = np.zeros((data_set_y.shape))
        new_t = np.zeros((data_set_t.shape))
        for i in range(n):
            new_X[i,:] = data_set_X[rdm_idx[i],:]
            new_y[i] = data_set_y[rdm_idx[i]]
            new_t[i,:] = data_set_t[rdm_idx[i],:]
            
        return [np.float32(data_set_X), np.float32(data_set_y), np.float32(data_set_t)]
        
    def filter_for_full_train_label_set(self, Xyt):
        '''Filters for a full train label set that consists of data, labels,
           and binary class matrix
        '''
        if Xyt[0] != None and Xyt[1] != None and Xyt[2] != None:
            return True
        else:
            return False
        

    def standardize_data(self, set): 
        '''Standardizes data set:     x - mu
                                  x = ------
                                      sigma
        '''      
        n = set.shape[0]
        m = set[:,1:].shape[1]
        new_set = np.zeros((set.shape))
        new_set[:,0] = set[:,0]
        for i in range(1,m):
            print 'Standardizing variable ' + str(i+1)
            mu = np.mean(set[:,i])
            sigma = np.mean(set[:,i])
            for j in range(n): 
                new_set[j,i] = (set[j,i] - mu) / (sigma*1.0)
             
        return new_set
    
        
    def redistribute_cases(self, full_set):
        '''Balances the data set and saves a index vector
           for the data to the directory where the data came from.
           
           Example:
           X = [1, 1, 3, 3, 7, 4, 6, 4, 1] with batchsize = 3
           
           batches = [[1, 3, 4],
                      [1, 3, 6],
                      [1, 4, 7]]           
        '''
        y = full_set[1]
        no_labels = np.int32(np.max(y))+1
        n = full_set[0].shape[0]    
        
        
        
        proportions = np.zeros(no_labels)
        current_prop = np.zeros(no_labels)
        
        
        for i in range(no_labels):
            proportions[i] = np.sum(y== i) / (1.0*n)
         
        y_idx = np.int32(np.vstack([y, np.arange(n)]))        
        y_idx_new = np.int32(np.ones((n,2))*-1)
        
        def find_next_index(label): 
            #which are indices where one can find the next value of that label?
            idx_label = np.where(y_idx==label)[1]
            #find the first index which is not used from the index found above
            next_idx_label = np.where(np.in1d(idx_label,y_idx_new[:,1])==0)[0][0]                
            return idx_label[next_idx_label]         
                

        print 'Redistributing labels...'
        for row in range(n):
            
            if((row + 1) % 5000 == 0):
                print (row + 1)
            
            
            next_label = -1
            #next label which is underrepresented? -> minimize the ratio of actual proportion to target proportion
            next_label = np.argmin(current_prop,axis=0) if np.min(current_prop) == 0 else np.argmin(current_prop/proportions)
              
            #assign new values
            y_idx_new[row,0] = next_label
            y_idx_new[row,1] = find_next_index(next_label)
             
            #recalulate label proportions in the new set  
            for label in range(no_labels):                
                current_prop[label] = np.sum(y_idx_new[:,0]== label) / ((row +1.0)*1.0)
                
                
        print y_idx_new
        
        return y_idx_new
                    
        
    def calculate_proportions(self, set_sizes, n, batch_size):
        '''Search for batch sizes that that enables a split into 
           equally sized batches. The closed fitting value to
           the desired batch size is chosen.         
        '''
                
        start_lower = abs(n*set_sizes[0]*0.7)
        
        start_upper = abs(n*set_sizes[0]*1.3)
        target =  abs(n*set_sizes[0])        
        candidates = []
        for i in np.arange(start_lower,start_upper,1):
            if(i % batch_size == 0):
                candidates.append(i)
                
                
        idx = np.argmin(np.abs(np.array(candidates) - target))
        train_prop = (candidates[idx]/(target*1.0))* set_sizes[0]
        if len(set_sizes) == 2:
            valid_prop = (1-train_prop)
        else:
            valid_prop = ((1-train_prop)/(set_sizes[1]+ set_sizes[2]))*set_sizes[1]
            test_prop = ((1-train_prop)/(set_sizes[1]+ set_sizes[2]))*set_sizes[2]
       
        return [train_prop, valid_prop,0 if len(set_sizes) == 2 else test_prop]
         
    def redistribute_data_set(self, y_idx_new, full_set):
        '''Redistributes the data given an index vector.            
        '''
        n = full_set[0].shape[0]
        m = full_set[0].shape[1]
        
        
        y_new = y_idx_new[:,0]
        X_new = np.zeros((n,m))
        t_new = np.zeros((n, full_set[2].shape[1]))
        
        for i in range(n):
            X_new[i,:] = full_set[0][y_idx_new[i,1],:]
            t_new[i,:] = full_set[2][y_idx_new[i,1],:]
            
        return [X_new, y_new, t_new]
              
    def load_csv (self, path):
        return np.loadtxt(path, dtype='float32', delimiter=',', skiprows=1)       
        

    def create_batches(self, paths, column_labels, set_sizes, batch_size, standardize = True):
        '''Loads data and creates balanced batches of the given size.
           Givens: paths = [/ThisFolder/mytrainset.csv, /ThisFolder/mytestset.csv]
                   column_labels = [0,-1] column where the label is; -1 for no label, i.e. test set etc.
                   set_sizes = [0.8,0.1,0.1] split into train, validation and test set; must sum to one  
        
        '''
        
        self.check_integrity(paths, column_labels, set_sizes)   
        data_sets = self.load_data(paths, self.load_csv, standardize)        
                 
        y = map(self.get_label_from_set,  data_sets, column_labels)  
        X = map(self.remove_label_from_set, data_sets, column_labels)        
        t = map(self.create_t_dataset, y)         
        
        full_set = zip(X, y, t)                
        full_set = filter(self.filter_for_full_train_label_set, full_set)[0]
        
        full_set = self.shuffle_set(full_set[0], full_set[1], full_set[2])
        
        full_X = np.concatenate(X)
        
        if os.path.isfile(paths[0].replace('.csv','_redistributed_y.npy')):
            y_idx_new = np.load(paths[0].replace('.csv','_redistributed_y.npy'))            
        else:                  
            y_idx_new = self.redistribute_cases(full_set)
            np.save(paths[0].replace('.csv','_redistributed_y.npy'), y_idx_new)        
 
        full_set = self.redistribute_data_set(y_idx_new, full_set)
        
        
        set_sizes_new = self.calculate_proportions(set_sizes, full_set[0].shape[0], batch_size)        
        self.check_integrity(paths, column_labels, set_sizes_new)
        
        
        n = full_set[0].shape[0]
        size_train = np.int32(np.round(set_sizes_new[0]*n))
        size_valid = np.int32(np.round(set_sizes_new[1]*n))
        size_test = np.int32(np.round(set_sizes_new[2]*n)) 
        
        X_batches = full_set[0][:size_train,:]
        y_batches = full_set[1][:size_train]
        t_batches = full_set[2][:size_train,:]
        
        valid_X = np.float32(full_set[0][size_train:size_train+size_valid,:])
        valid_y = np.float32(full_set[1][size_train:size_train+size_valid])
        valid_t = np.float32(full_set[2][size_train:size_train+size_valid])
        
        test_X = np.float32(full_set[0][size_train+size_valid:size_train+size_valid+size_test,:])
        test_y = np.float32(full_set[1][size_train+size_valid:size_train+size_valid+size_test])
        
        X_batches = np.float32(np.array(np.split(X_batches, size_train/batch_size, axis=0)))
        y_batches = np.float32(np.array(np.split(y_batches, size_train/batch_size, axis=0)))
        t_batches = np.float32(np.array(np.split(t_batches, size_train/batch_size, axis=0)))
        
        return [[X_batches,y_batches,t_batches],
                [valid_X,valid_y, valid_t],
                [test_X,test_y]] 
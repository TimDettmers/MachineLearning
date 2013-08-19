
class Learning_params:
        epochs = 10000
        learning_rate = 0.1
        use_learning_rate_decay = True
        learning_rate_decay = 0.01        
        use_momentum = False
        momentum = 0.5
        transient_phase = 100        
        use_momentum_increase = True
        momentum_increase = 0.001
        use_bias = True
        transient_phase = 25   
        
class Regularization_params:        
    use_early_stopping = True
    use_L2 = False
    L2_penalty = 10
    use_weight_decay = True
    weight_decay = 0.05
    weight_decay_decay = 0.013
      
class Stopping_and_Ensemble_params:
    use_ensemble = False
    ensemble_count = 4,
    safe_weights_threshold = 0.013
    epochs_force_reinitilization = 250
    last_save_epoch = 0   
    best_error = 1
    saved_weights_error = 1

class Hyperparameters:
    L = Learning_params()
    R = Regularization_params()
    S = Stopping_and_Ensemble_params()
    layers = None
    
    initial_learning_rate = L.learning_rate 
    initial_weight_decay = R.weight_decay           
    initial_momentum = L.momentum      
    initial_safe_weights_threshold = S.safe_weights_threshold   
    
    @staticmethod
    def set_initial_parameters():
        Hyperparameters.initial_learning_rate = Hyperparameters.L.learning_rate 
        Hyperparameters.initial_weight_decay = Hyperparameters.R.weight_decay           
        Hyperparameters.initial_momentum = Hyperparameters.L.momentum      
        Hyperparameters.initial_safe_weights_threshold = Hyperparameters.S.safe_weights_threshold   
    
    @staticmethod
    def set_learning_parameters(epochs, learning_rate, use_learning_rate_decay,
                        learning_rate_decay, use_bias,
                        use_momentum, momentum, use_momentum_increase, momentum_increase,
                        transient_phase):

        Hyperparameters.L.learning_rate = learning_rate
        Hyperparameters.L.use_learning_rate_decay = use_learning_rate_decay
        Hyperparameters.L.learning_rate_decay = learning_rate_decay
        Hyperparameters.L.epochs = epochs
        Hyperparameters.L.use_bias = use_bias
        
        Hyperparameters.L.use_momentum = use_momentum
        Hyperparameters.L.momentum = momentum
        Hyperparameters.L.use_momentum_increase = use_momentum_increase
        Hyperparameters.L.momentum_increase = momentum_increase
        Hyperparameters.L.transient_phase = transient_phase 
    
    @staticmethod
    def set_standard_regularization_parameters(use_early_stopping,
                                            use_L2, L2_penalty,
                                            use_weight_decay, weight_decay, weight_decay_decay):
        
        Hyperparameters.R.use_early_stopping = use_early_stopping     
        Hyperparameters.R.use_L2 = use_L2     
        Hyperparameters.R.L2_penalty = L2_penalty 
                
        Hyperparameters.R.use_weight_decay = use_weight_decay
        Hyperparameters.R.weight_decay = weight_decay
        Hyperparameters.R.weight_decay_decay = weight_decay_decay
        
        
    @staticmethod        
    def set_stopping_and_ensemble_parameter(use_ensemble, ensemble_count, safe_weights_threshold, 
                                         epochs_force_reinitilization):
        Hyperparameters.S.use_ensemble = use_ensemble
        Hyperparameters.S.ensemble_count = ensemble_count
        Hyperparameters.S.safe_weights_threshold = safe_weights_threshold
        Hyperparameters.S.saved_weights_error = safe_weights_threshold
        Hyperparameters.S.epochs_force_reinitilization = epochs_force_reinitilization
        
    @staticmethod   
    def get_reporttext():
        text = 'layers = ' 
        for i, layer in enumerate(Hyperparameters.layers):
            text += ', ' if i > 0 else '' 
            text += str(layer.size) 
        text += '\n'
        text += 'learning_rate = ' + str(Hyperparameters.L.learning_rate)  + '\n\n'      
        text += 'use_bias = ' + str(Hyperparameters.L.use_bias)  + '\n'
        text += 'use_L2 = ' + str(Hyperparameters.R.use_L2)  + '\n'       
        text += 'use_weight_decay = ' + str(Hyperparameters.R.use_weight_decay)  + '\n'
        text += 'weight_decay_decay = ' + str(Hyperparameters.R.weight_decay_decay)  + '\n'
        text += 'use_learning_rate_decay = ' + str(Hyperparameters.L.use_learning_rate_decay)  + '\n'     
        text += 'L2_penalty = ' + str(Hyperparameters.R.L2_penalty)  + '\n'
        text += 'weight_decay = ' + str(Hyperparameters.R.weight_decay)  + '\n'
        text += 'learning_rate_decay = ' + str(Hyperparameters.L.learning_rate_decay)  + '\n\n'
        text += 'dropout_decays = ' 
        for i, layer in enumerate(Hyperparameters.layers):
            text += ', ' if i > 0 else '' 
            text += str(layer.dropout_decay) 
        text += '\n'
        text += 'dropout_decay_epoch_frequency = ' 
        for i, layer in enumerate(Hyperparameters.layers):
            text += ', ' if i > 0 else '' 
            text += str(layer.frequency)
        text += '\n' 
        text += 'dropout = ' 
        for i, layer in enumerate(Hyperparameters.layers):
            text += ', ' if i > 0 else '' 
            text += str(layer.dropout) 
        text += '\n'
        text += 'use_momentum = ' + str(Hyperparameters.L.use_momentum)  + '\n'
        text += 'momentum = ' + str(Hyperparameters.L.momentum)  + '\n'
        text += 'use_momentum_increase = ' + str(Hyperparameters.L.use_momentum_increase)  + '\n'
        text += 'momentum_increase = ' + str(Hyperparameters.L.momentum_increase)  + '\n'
        text += 'transient_phase = ' + str(Hyperparameters.L.transient_phase)  + '\n\n'
        text += 'best_error = ' + str(Hyperparameters.S.saved_weights_error)  + '\n'
        
        return text
    
    
    
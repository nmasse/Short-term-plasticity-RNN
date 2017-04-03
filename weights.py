import numpy as np
import matplotlib.pyplot as plt

class weight_initialization:
    
    # create weight and bias values that will be used to initialized the network
    def __init__(self, params):
        
        self.num_input = params['num_input_neurons']
        self.num_RNN = params['num_RNN_neurons']
        self.num_output = params['num_classes']
        self.EI_list = params['EI_list'] 
        self.weight_multiplier = params['weight_multiplier'] 
        self.connection_prob = params['connection_prob'] 
        
    def create_output_weights(self):
        
        w_out = self.weight_multiplier*np.random.gamma(shape=0.25, scale=1.0, size=[self.num_output, self.num_RNN])
        
        # will only project from excitatory RNN neurons to the output neurons
        # values in EI_list should be 1 for excitatory, -1 for inhibitory
        ind_inh = np.where(self.EI_list == -1)[0]
        w_out[:, ind_inh] = 0
        w_out = np.float32(w_out)
        mask = np.random.rand(self.num_output, self.num_RNN)
        mask = mask < self.connection_prob
        w_out *= mask
        b_out = np.zeros((self.num_output, 1), dtype = np.float32)
        
        return w_out, b_out
        
    def create_RNN_weights(self):
        
        w_rnn = self.weight_multiplier* np.random.gamma(shape=0.25, scale=1.0, size=[self.num_RNN, self.num_RNN])
        w_rnn = np.float32(w_rnn)
        mask = np.random.rand(self.num_RNN, self.num_RNN)
        mask = mask < self.connection_prob
        w_rnn *= mask
        
        # make sure neurons don't project onto themselves
        for i in range(self.num_RNN):
            w_rnn[i,i] = 0
        
        b_rnn = np.zeros((self.num_RNN, 1), dtype = np.float32)
        weight_mask = np.ones((self.num_RNN,self.num_RNN)) - np.identity(self.num_RNN)
        
        return w_rnn, b_rnn, weight_mask
        
    def create_input_to_RNN_weights(self):
        
        w_in = self.weight_multiplier*np.random.gamma(shape=0.25, scale=1.0, size=[self.num_RNN, self.num_input])
        w_in = np.float32(w_in)
        mask = np.random.rand(self.num_RNN, self.num_input)
        mask = mask < self.connection_prob
        w_in *= mask
        w_in = np.maximum(0, w_in)
        
        return w_in
        
    @staticmethod 
    def spectral_radius(A):
        
        #Compute the spectral radius of a matrix.
        return np.max(abs(np.linalg.eigvals(A)))

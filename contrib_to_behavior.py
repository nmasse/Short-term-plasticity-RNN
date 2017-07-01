import numpy as np
import pickle

class Analysis:
    
    def __init__(self, model_filename):
        
        x = pickle.load(open(model_filename, 'rb'))
        
        # reshape STP depression
        self.syn_x = np.stack(x['syn_x'],axis=2)
        self.syn_x = np.stack(self.syn_x,axis=1)
        if self.syn_x.shape[0] == 0:
            self.syn_x = None
        else:
            num_neurons, trial_length, num_blocks, trials_per_block = self.syn_x.shape
            self.syn_x = np.reshape(self.syn_x,(num_neurons,trial_length,num_blocks*trials_per_block))
        
        # reshape STP facilitation
        self.syn_u = np.stack(x['syn_u'],axis=2)
        self.syn_u = np.stack(self.syn_u,axis=1)
        if self.syn_u.shape[0] == 0:
            self.syn_u = None
        else:
            num_neurons, trial_length, num_blocks, trials_per_block = self.syn_u.shape
            self.syn_u = np.reshape(self.syn_u,(num_neurons,trial_length,num_blocks*trials_per_block))
        
        
        # reshape RNN outputs
        self.rnn_outputs = np.stack(x['hidden_state'],axis=2)
        self.rnn_outputs = np.stack(self.rnn_outputs,axis=1)
        num_neurons, trial_length, num_blocks, trials_per_block = self.rnn_outputs.shape
        self.rnn_outputs = np.reshape(self.rnn_outputs,(num_neurons,trial_length,num_blocks*trials_per_block))
        print('Mean RNN responses ', np.mean(self.rnn_outputs))
        
        # reshape desired outputs
        self.desired_outputs = x['desired_output']
        self.desired_outputs = np.transpose(self.desired_outputs,(0,1,2))
        
        # reshape train mask
        self.train_mask = x['train_mask']
        self.train_mask = np.transpose(self.train_mask,(0,1))

        
        # reshape RNN inputs
        self.rnn_inputs = x['rnn_input']
        #self.rnn_inputs = np.transpose(self.rnn_inputs,(2,0,1))
        self.rnn_inputs = np.transpose(self.rnn_inputs,(0,2,1))

        
        # reshape model outputs
        self.model_outputs = np.stack(x['model_outputs'],axis=2)
        self.model_outputs = np.stack(self.model_outputs,axis=1)
        num_classes = self.model_outputs.shape[0]
        self.model_outputs = np.reshape(self.model_outputs,(num_classes,trial_length,num_blocks*trials_per_block))
   
        
        # reshape trial_conds
        self.sample_dir = x['sample_dir']
        self.test_dir = x['test_dir']
        self.match = x['match']
        self.rule = x['rule']
        self.catch = x['catch']
        self.probe = x['probe']

        
        # other info
        #self.EI_list = x['params']['EI_list']
        self.num_rules = len(x['params']['possible_rules'])
        self.possible_rules = x['params']['possible_rules']
        self.num_motion_dirs = x['params']['num_motion_dirs']
        self.n_hidden = x['params']['n_hidden']
        self.U = x['params']['U']
        self.W_rnn = x['w_rnn']
        self.b_rnn = x['b_rnn']
        self.W_in = x['w_in']
        self.W_out = x['w_out']
        self.b_out = x['b_out']
        self.alpha_neuron = x['params']['alpha_neuron']
        self.alpha_std = x['params']['alpha_std']
        self.alpha_stf = x['params']['alpha_stf']
        self.noise_sd = x['params']['noise_sd']
        self.synapse_config = x['params']['synapse_config']
        self.EI_list = x['params']['EI_list']
        self.dt = x['params']['dt']
        #self.dt_sec = x['params']['dt_sec']
        self.dt_sec = x['params']['dt']/1000
        self.EI = x['params']['EI']
        self.test_onset = (x['params']['dead_time'] + x['params']['fix_time'] + x['params']['sample_time'] + x['params']['delay_time'])//self.dt
        
        """
        If E/I network is desired, then ensure XXXX
        """
        if self.EI:
            self.EI_matrix = np.diag(self.EI_list)
            self.W_rnn  = np.dot(np.maximum(0, self.W_rnn), self.EI_matrix)
            self.W_in  = np.maximum(0, self.W_in)
            self.W_out  = np.maximum(0, self.W_out)
            self.rnn_inputs = np.maximum(0, self.rnn_inputs)
            
            

    def simulate_network(self):
        
        
        num_neurons, trial_length, self.batch_train_size = self.rnn_outputs.shape
        self.hidden_init = self.rnn_outputs[:,self.test_onset-1,:]
        self.syn_x_init = self.syn_x[:,self.test_onset-1,:]
        self.syn_u_init = self.syn_u[:,self.test_onset-1,:]
        
        test_length = trial_length - self.test_onset
        
        #print(self.rnn_inputs.shape)
        self.input_data = np.split(self.rnn_inputs[self.test_onset:,:,:],test_length,axis=0)
        #print(self.input_data[0].shape)
        
        self.y = self.desired_outputs[self.test_onset:,:,:]

        
        """
        Calculating behavioral accuracy without shuffling
        """
        self.run_model()
        accuracy = self.get_perf()
        
        """
        Keep the synaptic values fixed, permute the neural activity
        """
        ind_shuffle = np.random.permutation(self.batch_train_size)

        self.hidden_init = self.hidden_init[:,ind_shuffle]
        self.run_model()
        accuracy_neural_shuffled = self.get_perf()
        
        """
        Keep the hidden values fixed, permute synaptic values
        """
        self.hidden_init = self.rnn_outputs[:,self.test_onset,:]
        self.syn_x_init = self.syn_x_init[:,ind_shuffle]
        self.syn_u_init = self.syn_u_init[:,ind_shuffle]
        self.run_model()
        accuracy_syn_shuffled = self.get_perf()
        
        return accuracy, accuracy_neural_shuffled, accuracy_syn_shuffled
    
    
    def run_model(self):
        
        """
        Run the reccurent network
        History of hidden state activity stored in self.hidden_state_hist
        """  
        self.rnn_cell_loop(self.input_data, self.hidden_init, syn_x = self.syn_x_init, syn_u = self.syn_u_init)
            
        """
        Network output 
        Only use excitatory projections from the RNN to the output layer
        """   
        print(self.hidden_state_hist[0].shape)
        print(len(self.hidden_state_hist))
        self.y_hat = [np.dot(self.W_out,h)+ self.b_out for h in self.hidden_state_hist]
        self.y_hat = np.stack(self.y_hat,axis=0)
        print('y_hat shape')
        print(self.y_hat.shape)    
            
    
    def rnn_cell_loop(self, x_unstacked, h, syn_x, syn_u):
            
        self.hidden_state_hist = []
        self.syn_x_hist = []
        self.syn_u_hist = []
        
        """
        Loop through the neural inputs to the RNN, indexed in time
        """
        for rnn_input in x_unstacked:
            h, syn_x, syn_u = self.rnn_cell(np.squeeze(rnn_input), h, syn_x, syn_u)
            self.hidden_state_hist.append(h)
            self.syn_x_hist.append(syn_x) 
            self.syn_u_hist.append(syn_u)  
     

    def rnn_cell(self, rnn_input, h, syn_x, syn_u):
               
        
        """
        Update the synaptic plasticity paramaters
        """ 
        
        #print(h.shape)
        #print(syn_x.shape)
        #print(syn_u.shape)
        #print(self.alpha_std.shape) 
        
        
        if self.synapse_config == 'std_stf':
            # implement both synaptic short term facilitation and depression
            syn_x += self.alpha_std*(1-syn_x) - self.dt_sec*syn_u*syn_x*h
            syn_u += self.alpha_stf*(self.U-syn_u) + self.dt_sec*self.U*(1-syn_u)*h
            syn_x = np.minimum(1, np.maximum(0, syn_x))
            syn_u = np.minimum(1, np.maximum(0, syn_u))
            h_post = syn_u*syn_x*h
            
        elif self.synapse_config == 'std':
            # implement synaptic short term derpression, but no facilitation
            # we assume that syn_u remains constant at 1
            syn_x += self.alpha_std*(1-syn_x) - self.dt_sec*syn_x*h
            syn_x = np.minimum(1, np.maximum(0, syn_x))
            h_post = syn_x*h

        elif self.synapse_config == 'stf':
            # implement synaptic short term facilitation, but no depression
            # we assume that syn_x remains constant at 1
            syn_u += self.alpha_stf*(self.U-syn_u) + self.dt_sec*self.U*(1-syn_u)*h
            syn_u = np.minimum(1, np.maximum(0, syn_u))
            h_post = syn_u*h
     
        else:
            # no synaptic plasticity
            h_post = h

        """
        Update the hidden state
        All needed rectification has already occured
        """ 
        h = np.maximum(0, h*(1-self.alpha_neuron) 
                       + self.alpha_neuron*(np.dot(self.W_in, rnn_input) 
                       + np.dot(self.W_rnn, h_post) + self.b_rnn)  
                       + np.random.normal(0, self.noise_sd,size=(self.n_hidden, self.batch_train_size))) 
            
        return h, syn_x, syn_u
    
    
    def get_perf(self):
    
        print(self.y_hat[0].shape)
        print(self.y.shape)
        
        y = np.argmax(self.y, axis = 2)
        y_hat = np.argmax(self.y_hat, axis = 1)
        
        

        return np.mean(np.float32(y == y_hat))
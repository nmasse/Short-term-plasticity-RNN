"""
2017/05/03 Nicolas Masse
"""

import tensorflow as tf
import numpy as np
import pickle
import stimulus
import time
from collections import namedtuple
from network import RNNCellSTP
from tensorflow.contrib.rnn import LSTMCell, BasicRNNCell
import sys
import os #added to ignore warning messages about CPU
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #added to ignore warning messages about CPU


class Model:

    def __init__(self, params, input_data, target_data, mask):  
        
        # added
        self.params = params

        """
        Load all variables from the params dictionary
        """
        for key,value in params.items():
            setattr(self, key, value) 
        
        """
        Load the input activity, the target data, and the training mask for this batch of trials
        """
        self.input_data = input_data
        self.target_data = target_data
        self.mask = mask
        
        """
        Load the intial hidden state activity, and the initial synaptic depression and facilitation terms
        to be used at the start of each trial
        """
        self.hidden_init = tf.constant(self.h_init)
        self.synapse_x_init = tf.constant(self.syn_x_init)
        self.synapse_u_init = tf.constant(self.syn_u_init)

        """
        Build the Tensorflow graph
        """
        self.run_model()
        
        """
        Train the model
        """
        self.optimize()
     

    def run_model(self):
        
        """
        Run the reccurent network
        History of hidden state activity stored in self.hidden_state_hist
        """  
    
        # original
        # self.rnn_cell_loop(self.input_data, self.hidden_init, syn_x = self.synapse_x_init, syn_u = self.synapse_u_init)
        
        # need to make tuples for the input_data, hidden_init, syn_x/u
        # double check synapse_config value

        if self.synapse_config is None:
            state = self.hidden_init
            hidden_state_size = sys.getsizeof(state)
        else:
            State_tuple = namedtuple('State', ['hidden', 'syn_x', 'syn_u'])
            state = State_tuple(self.hidden_init, self.synapse_x_init, self.synapse_u_init)
            hidden_state_size = sys.getsizeof(state)

        # print("hidden_state_size: ")
        # print(hidden_state_size)

        # cell = RNNCellSTP(self.params, hidden_state_size)
        cell = BasicRNNCell(hidden_state_size)
        self.hidden_state = tf.nn.dynamic_rnn(cell, self.input_data, initial_state=state, time_major=True)

        with tf.variable_scope('output'):
            W_out = tf.get_variable('W_out', initializer = np.float32(self.w_out0), trainable=True)
            b_out = tf.get_variable('b_out', initializer = np.float32(self.b_out0), trainable=True)
            
        """
        Network output 
        Only use excitatory projections from the RNN to the output layer
        """   
        self.y_hat = [tf.matmul(tf.nn.relu(W_out),h)+b_out for h in self.hidden_state_hist]

    
    def rnn_cell_loop(self, x_unstacked, h, syn_x, syn_u):
    
        """
        Initialize weights and biases
        """
        with tf.variable_scope('rnn_cell'):
            W_in = tf.get_variable('W_in', initializer = np.float32(self.w_in0), trainable=True)
            W_rnn = tf.get_variable('W_rnn', initializer = np.float32(self.w_rnn0), trainable=True)
            b_rnn = tf.get_variable('b_rnn', initializer = np.float32(self.b_rnn0), trainable=True)
            W_ei = tf.get_variable('EI', initializer = np.float32(self.EI_matrix), trainable=False)
               
        self.hidden_state_hist = []
        self.syn_x_hist = []
        self.syn_u_hist = []
        
        """
        Loop through the neural inputs to the RNN, indexed in time
        """
        for rnn_input in x_unstacked:
            h, syn_x, syn_u = self.rnn_cell(rnn_input, h, syn_x, syn_u)
            self.hidden_state_hist.append(h)
            self.syn_x_hist.append(syn_x) 
            self.syn_u_hist.append(syn_u)  
     

    def rnn_cell(self, rnn_input, h, syn_x, syn_u):
        
        """
        Main computation of the recurrent network
        """  
        with tf.variable_scope('rnn_cell', reuse=True):
            W_in = tf.get_variable('W_in')
            W_rnn = tf.get_variable('W_rnn')
            b_rnn = tf.get_variable('b_rnn')
            W_ei = tf.get_variable('EI')
            
        if self.EI:
            # ensure excitatory neurons only have postive outgoing weights, 
            # and inhibitory neurons have negative outgoing weights
            W_rnn_effective = tf.matmul(tf.nn.relu(W_rnn), W_ei)
        else:
            W_rnn_effective = W_rnn         
        
        # syn_x and syn_u will be capped at 1
        C = tf.constant(np.float32(1))
        
        """
        Update the synaptic plasticity paramaters
        """ 
        if self.synapse_config == 'std_stf':
            # implement both synaptic short term facilitation and depression
            syn_x += self.alpha_std*(1-syn_x) - self.dt_sec*syn_u*syn_x*h
            syn_u += self.alpha_stf*(self.U-syn_u) + self.dt_sec*self.U*(1-syn_u)*h
            syn_x = tf.minimum(C, tf.nn.relu(syn_x))
            syn_u = tf.minimum(C, tf.nn.relu(syn_u))
            h_post = syn_u*syn_x*h
            
        elif self.synapse_config == 'std':
            # implement synaptic short term derpression, but no facilitation
            # we assume that syn_u remains constant at 1
            syn_x += self.alpha_std*(1-syn_x) - self.dt_sec*syn_x*h
            syn_x = tf.minimum(C, tf.nn.relu(syn_x))
            syn_u = tf.minimum(C, tf.nn.relu(syn_u))
            h_post = syn_x*h

        elif self.synapse_config == 'stf':
            # implement synaptic short term facilitation, but no depression
            # we assume that syn_x remains constant at 1
            syn_u += self.alpha_stf*(self.U-syn_u) + self.dt_sec*self.U*(1-syn_u)*h
            syn_u = tf.minimum(C, tf.nn.relu(syn_u))
            h_post = syn_u*h
     
        else:
            # no synaptic plasticity
            h_post = h

     
        """
        Update the hidden state
        Only use excitatory projections from input layer to RNN
        All input and RNN activity will be non-negative
        """ 
        h = tf.nn.relu(h*(1-self.alpha_neuron) 
                       + self.alpha_neuron*(tf.matmul(tf.nn.relu(W_in), tf.nn.relu(rnn_input)) 
                       + tf.matmul(W_rnn_effective, h_post) + b_rnn)  
                       + tf.random_normal([self.n_hidden, self.batch_train_size], 0, self.noise_sd, dtype=tf.float32)) 
            
        return h, syn_x, syn_u
    
            
    def optimize(self):
        
        """
        Calculate the loss functions and optimize the weights
        """     
        perf_loss = [mask*tf.reduce_mean(tf.square(y_hat-desired_output),axis=0)
                     for (y_hat, desired_output, mask) in zip(self.y_hat, self.target_data, self.mask)]
        
        # L2 penalty term on hidden state activity to encourage low spike rate solutions
        spike_loss = [self.spike_cost*tf.reduce_mean(tf.square(h), axis=0) for (h, mask) 
                            in zip(self.hidden_state_hist, self.mask)]
      
        self.perf_loss = tf.reduce_mean(tf.stack(perf_loss, axis=0))
        self.spike_loss = tf.reduce_mean(tf.stack(spike_loss, axis=0))
        
        self.loss = self.perf_loss + self.spike_loss
        
        opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        grads_and_vars = opt.compute_gradients(self.loss)
        
        """
        Apply any applicable weights masks to the gradient and clip
        """ 
        capped_gvs = []
        for grad, var in grads_and_vars:
            if grad.shape == self.w_rec_mask.shape:
                print('Applying weight mask to w_rec')
                grad *= self.w_rec_mask
            elif grad.shape == self.w_out_mask.shape:
                print('Applying weight mask to w_out')
                grad *= self.w_out_mask
            capped_gvs.append((tf.clip_by_norm(grad, self.clip_max_grad_val), var))

        self.train_op = opt.apply_gradients(capped_gvs)

 
    
def main(params):
    

    """
    Create the stimulus class to generate trial paramaters and input activity
    """
    stim = stimulus.Stimulus(params)
   
    n_input, n_hidden, n_output = params['shape']  
    trial_length = params['trial_length']//params['dt'] 
    batch_size = params['batch_train_size']
    N = params['batch_train_size']*params['num_batches'] # trials per iteration, calculate gradients after batch_train_size
    
    """
    Define all placeholder
    """
    mask = tf.placeholder(tf.float32, shape=[trial_length, batch_size])
    x = tf.placeholder(tf.float32, shape=[n_input, trial_length, batch_size])  # input data
    y = tf.placeholder(tf.float32, shape=[n_output, trial_length, batch_size]) # target data
   
    with tf.Session() as sess:
        
        model = Model(params, x, y, mask)
        init = tf.global_variables_initializer()
        sess.run(init)
        t_start = time.time()
        
        saver = tf.train.Saver()
        # Restore variables from previous model if desired
        if params['load_previous_model']:
            saver.restore(sess, params['save_dir'] + params['ckpt_load_fn'])
            print('Model ' +  params['ckpt_load_fn'] + ' restored.')
        
        # keep track of the model performance across training
        model_performance = {'accuracy': [], 'loss': [], 'trial': [], 'time': []}
        
        for i in range(params['num_iterations']):
            
            var_delay = params['var_delay']
            end_training = False
            save_trial = False
            
            """
            Save the data if this is the last iteration, if performance threshold has been reached, and once every 500 trials.
            Before saving data, generate trials with a fixed delay period
            """
            if i>0 and (i == params['num_iterations']-1 or np.mean(accuracy) > params['stop_perf_th'] or (i+1)%25==0):
                var_delay = False
                save_trial = True
                model_results = create_save_dict()
                if np.mean(accuracy) > params['stop_perf_th']:
                    end_training = True
                
            # generate batch of N (batch_train_size X num_batches) trials
            trial_info = stim.generate_trial(N, var_delay=var_delay)
            
            # keep track of the model performance for this batch
            loss = []
            perf_loss = []
            spike_loss = []
            accuracy = []
            
            for j in range(params['num_batches']):
                   
                """
                Select batches of size batch_train_size 
                """
                ind = range(j*params['batch_train_size'],(j+1)*params['batch_train_size'])

                input_data = np.transpose(trial_info['neural_input'][:,:,ind],(1,2,0)).reshape((-1,batch_size,n_input))
                target_data = np.transpose(trial_info['desired_output'][:,:,ind],(1,2,0)).reshape((-1,n_output)) 
                train_mask = np.transpose(trial_info['train_mask'][:,ind],(0,1)).reshape((-1,1))

                # target_data = trial_info['desired_output'][:,:,ind]
                # input_data = trial_info['neural_input'][:,:,ind]
                # train_mask = trial_info['train_mask'][:,ind]

                """
                Run the model
                """
                _, loss_temp, perf_loss_temp, spike_loss_temp, y_hat, state_hist, syn_x_hist, syn_u_hist = sess.run([model.train_op, model.loss, model.perf_loss, model.spike_loss, model.y_hat, model.hidden_state_hist, model.syn_x_hist, model.syn_u_hist], {x: input_data, y: target_data, mask: train_mask})
          
                # append the data before saving
                if save_trial:
                    start_save_time = time.time()
                    model_results = append_hidden_data(model_results, y_hat, state_hist, syn_x_hist, syn_u_hist)
                
                # keep track of performance for each batch
                accuracy.append(get_perf(target_data, y_hat, train_mask))
                perf_loss.append(perf_loss_temp)
                spike_loss.append(spike_loss_temp)               
            
            iteration_time = time.time() - t_start 
            model_performance = append_model_performance(model_performance, accuracy, perf_loss, (i+1)*N, iteration_time)
                
            """
            Save the data and network model
            """
            if save_trial:
                model_results = append_fixed_data(model_results, trial_info, params)
                model_results['performance'] = model_performance
                with open(params['save_dir'] + params['save_fn'], 'wb') as f:
                    pickle.dump(model_results, f)
                    save_path = saver.save(sess,params['save_dir'] + params['ckpt_save_fn'])
                    print(params['save_fn'] + ' pickled!, file save time = ', time.time() - start_save_time)

               
                # output model performance to screen
                print('Trial {:7d}'.format((i+1)*N) + ' | Time {:0.2f} s'.format(iteration_time) + 
                  ' | Perf loss {:0.4f}'.format(np.mean(perf_loss)) + ' | Spike loss {:0.4f}'.format(np.mean(spike_loss)) + ' | Mean activity {:0.4f}'.format(np.mean(state_hist)) + ' | Accuracy {:0.4f}'.format(np.mean(accuracy)))
              
            if end_training:
                return None


def get_perf(y, y_hat, mask):
    
    """
    Calculate task accuracy by comparing the actual network output to the desired output
    only examine time points when test stimulus is on
    in another words, when y[0,:,:] is not 0
    """
    
    y_hat = np.stack(y_hat,axis=1)
    mask *= y[0,:,:]==0
    y = np.argmax(y, axis = 0)
    y_hat = np.argmax(y_hat, axis = 0)

    return np.sum(np.float32(y == y_hat)*np.squeeze(mask))/np.sum(mask)


def append_hidden_data(model_results, y_hat, state, syn_x, syn_u):  
    
    model_results['hidden_state'].append(state)
    model_results['syn_x'].append(syn_x)
    model_results['syn_u'].append(syn_u)
    model_results['model_outputs'].append(y_hat)
    
    return model_results


def append_model_performance(model_performance, accuracy, loss, trial_num, iteration_time):

    model_performance['accuracy'].append(accuracy)
    model_performance['loss'].append(loss)
    model_performance['trial'].append(trial_num)
    model_performance['time'].append(iteration_time)

    return model_performance

def append_fixed_data(model_results, trial_info, params):
    
    model_results['sample_dir'] = trial_info['sample']
    model_results['test_dir'] = trial_info['test']
    model_results['match'] = trial_info['match']
    model_results['catch'] = trial_info['catch']
    model_results['rule'] = trial_info['rule']
    model_results['probe'] = trial_info['probe']
    model_results['rnn_input'] = trial_info['neural_input']
    model_results['desired_output'] = trial_info['desired_output']
    model_results['train_mask'] = trial_info['train_mask']
    model_results['params'] = params
    
    # add extra trial paramaters for the ABBA task
    if 4 in params['possible_rules']:
        print('Adding ABB specific data')
        model_results['num_test_stim'] = trial_info['num_test_stim']
        model_results['repeat_test_stim'] = trial_info['repeat_test_stim']
        model_results['test_stim_code'] = trial_info['test_stim_code']

    with tf.variable_scope('rnn_cell', reuse=True):
        W_in = tf.get_variable('W_in')
        W_rnn = tf.get_variable('W_rnn')
        b_rnn = tf.get_variable('b_rnn')
        W_ei = tf.get_variable('EI')
            #W_rnn_effective = tf.matmul(tf.nn.relu(W_rnn), W_ei)
    with tf.variable_scope('output', reuse=True):
        W_out = tf.get_variable('W_out')
        b_out = tf.get_variable('b_out')
    
    model_results['w_in'] = W_in.eval()
    model_results['w_rnn'] = W_rnn.eval()
    model_results['w_out'] = W_out.eval()
    model_results['b_rnn'] = b_rnn.eval()
    model_results['b_out'] = b_out.eval()

    return model_results

def create_save_dict():

    model_results = {
        'syn_x' :          [],
        'syn_u' :          [],
        'syn_adapt' :      [],
        'hidden_state':    [],
        'desired_output':  [],
        'model_outputs':   [],
        'rnn_input':       [],
        'sample_dir':      [],
        'test_dir':        [],
        'match':           [],
        'rule':            [],
        'catch':           [],
        'probe':           [],
        'train_mask':      [],
        'U':               [],
        'w_in':            [],
        'w_rnn':           [],
        'w_out':           [],
        'b_rnn':           [],
        'b_out':           [],
        'num_test_stim':   [],
        'repeat_test_stim':[],
        'test_stim_code': []}
    
    return model_results

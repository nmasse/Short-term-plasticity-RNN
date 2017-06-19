"""
2017/05/03 Nicolas Masse
Edited: 2017/06/13 Gregory Grant
"""


"""
Default parameters for training.
"""

import numpy as np
import tensorflow as tf
import reader

class Parameters:

    def __init__(self):

        print('\nInitializing paramaters...')

        self.params = {
        'num_motion_tuned':      9, # usually 36
        'num_fix_tuned':         0,
        'num_rule_tuned':        0,
        'n_hidden':              20,
        'exc_inh_prop':          0.8, # the percentage of excitatory neurons, with inhibitory neurons being the rest
        'den_per_unit':          5, # adding dendrites
        'n_output':              1, # usually 3
        'dead_time':             400,
        'fix_time':              500,
        'sample_time':           500,
        'delay_time':            1000,
        'test_time':             500,
        'variable_delay_scale':  20,
        'variable_delay_max':    500,
        'possible_rules':        [0],
        'clip_max_grad_val':     0.25,
        'learning_rate':         5e-3,
        'membrane_time_constant':100,
        'num_motion_dirs':       8,
        'input_mean':            0,
        'input_sd':              0.1,
        'noise_sd':              0.5,
        'connection_prob':       0.25, # usually 1
        'dt':                    25,
        'catch_trial_pct':       0.2,
        'probe_trial_pct':       0,
        'probe_time':            25,
        'spike_cost':            5e-5,
        'wiring_cost':           5e-7,
        'match_test_prob':       0.3,
        'repeat_pct':            0.5,
        'max_num_tests':         4,
        'tau_fast':              200,
        'tau_slow':              1500,
        'U_stf':                 0.15,
        'U_std':                 0.45,
        'stop_perf_th':          1,
        'stop_error_th':         0.005,
        'batch_train_size':      128,
        'num_batches':           8,
        'num_iterations':        50,
        'synapse_config':        None,
        'stimulus_type':         'experimental',
        'load_previous_model':   False,
        'var_delay':             False,
        'debug_model':           False,
        'save_dir':              'D:/Masse/RNN STP/saved_models/',
        'profile_path' :         "./profiles/exp_events.txt"
        }

    def return_params(self):

        # re-calculate all dependencies and initial values and return paramaters
        self.create_dependencies()
        self.initialize_vals_weights_biases()

        return self.params

    def get_events(self):

        with open(self.params['profile_path']) as event_list:
            raw_content = event_list.read().split("\n")

        text = list(filter(None, raw_content))

        for line in range(len(text)):
            if text[line][0] == "0":
                content = text[line:]

        for line in range(len(content)):
            content[line] = content[line].split("\t")
            content[line][0] = int(content[line][0])

        return content

    def create_dependencies(self):

        """
        The ABBA task requires fairly specific trial params to properly function
        Here, if the ABBA task is used, we will use the suggested defaults
        """
        if 4 in self.params['possible_rules']:
            print('Using suggested ABBA trial params...')
            stim_duration = 240
            self.params['ABBA_delay']  = stim_duration
            self.params['test_time']  = stim_duration
            self.params['sample_time']  = stim_duration
            self.params['delay_time']  = 8*stim_duration
            self.params['dt']  = 20

        # Event list from stimulus profile
        self.params['events'] = self.get_events()
        # Length of trial based on stimulus profile
        self.params['trial_length'] = self.params['events'][-1][0]
        # Number of input neurons
        self.params['n_input'] = self.params['num_motion_tuned'] + self.params['num_fix_tuned'] + self.params['num_rule_tuned']
        # General network shape
        self.params['shape'] = (self.params['n_input'], self.params['n_hidden'],self.params['n_output'])

        # The time step in seconds
        self.params['dt_sec'] = np.float32(self.params['dt']/1000)
        # Number of time steps
        self.params['num_time_steps'] = self.params['trial_length']//self.params['dt']
        # The delay between test stimuli in the ABBA task (rule = 4)

        # If num_inh_units is set > 0, then neurons can be either excitatory or
        # inihibitory; is num_inh_units = 0, then the weights projecting from
        # a single neuron can be a mixture of excitatory or inhibiotry
        if self.params['exc_inh_prop'] < 1.:
            self.params['EI'] = True
            print('Using EI network.')
        else:
            self.params['EI'] = False
            print('Not using EI network.')

        self.params['num_exc_units'] = int(np.round(self.params['n_hidden']*(self.params['exc_inh_prop'])))
        self.params['num_inh_units'] = self.params['n_hidden'] - self.params['num_exc_units']

        self.params['EI_list'] = np.ones((self.params['n_hidden']), dtype=np.float32)
        self.params['EI_list'][-self.params['num_inh_units']:] = -1

        self.params['EI_matrix'] = np.diag(self.params['EI_list'])

        # Membrane time constant of RNN neurons
        self.params['alpha_neuron'] = self.params['dt']/self.params['membrane_time_constant']
        # The standard deviation of the gaussian noise added to each RNN neuron at each time step
        self.params['noise_sd'] = np.sqrt(2*self.params['alpha_neuron'])*self.params['noise_sd']

    def initialize(self, param, dims):
        # Takes a string such as 'w_in0' for param, generates a tensor of
        # randomly generated numbers, and checks it against connection probability.
        self.params[param] = np.float32(np.random.gamma(shape=0.25, scale=1.0, size=dims))
        self.params[param] *= (np.random.rand(*dims) < self.params['connection_prob'])

    def initialize_vals_weights_biases(self):

        self.params['h_init'] = 0.1*np.ones((self.params['n_hidden'], self.params['batch_train_size']), dtype=np.float32)

        input_to_hidden_dims = [self.params['n_hidden'],self.params['den_per_unit'],self.params['n_input']]
        rnn_to_rnn_dims = [self.params['n_hidden'],self.params['den_per_unit'],self.params['n_hidden']]

        # Initialize input weights
        self.initialize('w_in0', input_to_hidden_dims)

        # Initialize starting recurrent weights
        # If excitatory/inhibitory neurons desired, initializes with random matrix with zeroes on the diagonal
        # If not, initializes with a diagonal matrix
        if self.params['EI']:
            self.initialize('w_rnn0', rnn_to_rnn_dims)
            #for j in range(self.params['n_hidden']):
            #    self.params['w_rnn0'][j,j] = 0
            #ind_inh = np.where(self.params['EI_list']==-1)[0]
            #sum_inh = np.sum(self.params['w_rnn0'][:,ind_inh])
            #ind_exc = np.where(self.params['EI_list']==1)[0]
            #sum_exc = np.sum(self.params['w_rnn0'][:,ind_exc])

            eye = np.zeros([*rnn_to_rnn_dims], dtype=np.float32)
            for j in range(self.params['den_per_unit']):
                for i in range(self.params['n_hidden']):
                    eye[i][j][i] = 1

            self.params['w_rec_mask'] = np.ones((rnn_to_rnn_dims), dtype=np.float32) - eye
        else:
            self.params['w_rnn0'] = np.float32(0.975*np.identity(self.params['n_hidden']))
            self.params['w_rec_mask'] = np.ones((rnn_to_rnn_dims), dtype=np.float32)

        # Initialize starting recurrent biases
        # Note that the second dimension in the bias initialization term
        # can be either 1 or self.params['batch_train_size'].
        set_to_one = True
        bias_dim = (1 if set_to_one else self.params['batch_train_size'])
        self.params['b_rnn0'] = np.zeros((self.params['n_hidden'], bias_dim), dtype=np.float32)

        # Effective synaptic weights are stronger when no short-term synaptic
        # plasticity is used, so the strength of the recurrent weights
        # is reduced to compensate
        if self.params['synapse_config'] == None:
            self.params['w_rnn0'] /= 3

        # Initialize output weights and biases
        self.initialize('w_out0', [self.params['n_output'],self.params['n_hidden']])

        self.params['b_out0'] = np.zeros((self.params['n_output'],1), dtype=np.float32)
        self.params['w_out_mask'] = np.ones((self.params['n_output'],self.params['n_hidden']), dtype=np.float32)

        if self.params['EI']:
            ind_inh = np.where(self.params['EI_list']==-1)[0]
            self.params['w_out0'][:, ind_inh] = 0
            self.params['w_out_mask'][:, ind_inh] = 0

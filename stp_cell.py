"""
Catherine Chaihyun Lee 2018
RNN cell class using short-term synaptic plasticity
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, BasicRNNCell
from parameters import *

class STPCell(RNNCell):
    def __init__(self, activation=None, reuse=None):
        # set default values for RNN cell
        super(STPCell, self).__init__(_reuse=reuse)
        
        # network shape
        self.n_hidden = par['n_hidden']
        self.batch_train_size = par['batch_train_size']
        self.hidden_state_size = par['n_hidden']

        # initial weights, biases
        self.w_rnn0 = par['w_rnn0']
        self.w_in0 = par['w_in0']
        self.b_rnn0 = par['b_rnn0']
        # self.EI_matrix = par['EI_matrix']
        self.W_ei = par['EI']
        self.noise_rnn = par['noise_rnn']

        # time constants
        self.dt = par['dt']
        self.alpha_neuron = par['alpha_neuron']

        # synaptic plastcity params
        self.alpha_stf = par['alpha_stf']
        self.alpha_std = par['alpha_std']
        self.U = par['U']
        self.synapse_config = par['synapse_config']

        """
        Initialize weights and biases
        """
        with tf.variable_scope('rnn_cell'):
            self.W_in = tf.get_variable('W_in', initializer = par['w_in0'], trainable=True)
            self.W_rnn = tf.get_variable('W_rnn', initializer = par['w_rnn0'], trainable=True)
            self.b_rnn = tf.get_variable('b_rnn', initializer = par['b_rnn0'], trainable=True)

    @property
    def state_size(self):
        return par['n_hidden']

    @property
    def output_size(self):
        return par['n_hidden']

    def call(self, inputs, state, scope=None):
        """Leaky RNN: output = new_state = (1-alpha)*state + alpha*activation(W * input + U * state + B)."""   
        
        """
        We will update the state vector depending on whether the synapses are static (self._synapse_config == None)
        or dynamic (self._synapse_config == 'synapatic_adapt', 'stf', 'std' or 'std_stf')
        """ 
        
        """
        Define indices into the state vector
        State vector contains all the synaptic paramaters (if short-term plasticity is being applied),
        and the hiddent activity
        """            
        # inputs = [batch_size, max_time, ...] if time_major == False
        # state = [cell.output_size, batch_size]

        if par['synapse_config'] == 'stf': 
            # get data from state tuple
            hidden_state = state.hidden
            syn_u = state.syn_u_hist

            # implement synaptic short term facilitation, but no depression
            syn_u += par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*hidden_state
            syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
            state_post = syn_u*hidden_state
            
        elif par['synapse_config'] == 'std':
            # get data from state tuple
            hidden_state = state.hidden
            syn_x = state.syn_x

            # implement synaptic short term derpression, but no facilitation
            syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_x*hidden_state
            syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
            state_post = syn_x*hidden_state
            
        elif par['synapse_config'] == 'std_stf': 
            # get data from state tuple
            hidden_state = state.hidden
            syn_x = state.syn_x
            syn_u = state.syn_u

            # implement both synaptic short term facilitation and depression  
            syn_x += np.transpose(par['alpha_std'])*(1-syn_x) - par['dt_sec']*syn_u*syn_x*hidden_state
            syn_u += np.transpose(par['alpha_stf'])*(np.transpose(par['U'])-syn_u) + par['dt_sec']*np.transpose(par['U'])*(1-syn_u)*hidden_state 
            syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
            syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
            state_post = syn_u*syn_x*hidden_state     
        else:
            hidden_state = state
            state_post = state

            
        """
        Main calculation 
        If self.EI is True, then excitatory and inhibiotry neurons are desired, and will we ensure that recurrent enurons 
        are of only one type, and that W_in weights are non-negative 
        """
        # print("BEFORE EI")
        # if par['EI']:
        #     print(tf.nn.relu(self.W_rnn))
        #     print(self.W_ei)
        #     W_rnn_effective = tf.matmul(tf.nn.relu(self.W_rrn), par['EI_matrix'])
        # else:
        #     W_rnn_effective = self.W_rnn

        # print("AFTER EI")
        # print(tf.nn.relu(self.W_in))
        # print(tf.nn.relu(rnn_input))
        # new_state = tf.nn.relu(state*(1-par['alpha_neuron'])
        #                + par['alpha_neuron']*(tf.matmul(tf.nn.relu(self.W_in), tf.nn.relu(rnn_input))
        #                + tf.matmul(W_rnn_effective, h_post) + b_rnn)
        #                + tf.random_normal([par['n_hidden'], par['batch_train_size']], 0, par['noise_rnn'], dtype=tf.float32))


        if self.W_ei:
            # new_state = tf.nn.relu(hidden_state*(1-par['alpha_neuron'])
                       # + par['alpha_neuron']*(tf.matmul(tf.nn.relu(self.W_in), tf.nn.relu(tf.transpose(inputs)))
                       # + tf.matmul(tf.matmul(tf.nn.relu(self.W_rnn), par['EI_matrix']), state_post) + self.b_rnn)
                       # + tf.random_normal([par['n_hidden'], par['batch_train_size']], 0, par['noise_rnn'], dtype=tf.float32))
            new_state = tf.nn.relu((1-par['alpha_neuron'])*hidden_state 
                        + par['alpha_neuron']*(tf.matmul(tf.nn.relu(inputs), tf.nn.relu(tf.transpose(self.W_in))) 
                        + tf.matmul(state_post, tf.matmul(tf.nn.relu(self.W_rnn), par['EI_matrix'])) + tf.transpose(self.b_rnn)) 
                        + tf.random_normal(tf.shape(hidden_state), 0, self.noise_rnn, dtype=tf.float32))
        else:                                 
            new_state = tf.nn.relu((1-par['alpha_neuron'])*hidden_state 
                        + par['alpha_neuron']*(tf.matmul(self.W_in, inputs) 
                        + tf.matmul(self.W_rnn, state_post) + self.b_rnn) 
                        + tf.random_normal(tf.shape(hidden_state), 0, par['noise_rnn'], dtype=tf.float32))
            
        
        # load final output to state tuple
        if self.synapse_config == 'stf': 
            state = state._replace(hidden=new_state, syn_u=syn_u)
        elif self.synapse_config == 'std':
            state = state._replace(hidden=new_state, syn_x=syn_x)
        elif self.synapse_config == 'std_stf': 
            state = state._replace(hidden=new_state, syn_x=syn_x, syn_u=syn_u)
        else:
            state = new_state

        return state, state

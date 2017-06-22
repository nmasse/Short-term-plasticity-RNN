"""
The RNN cell with short-term synaptic plasticity
"""

from __future__ import division

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, BasicRNNCell
from tensorflow.python.ops import variable_scope as vs
#from tensorflow.python.platform import tf_logging as logging
#from tensorflow.python.ops import math_ops
#from tensorflow.python.util import nest


class RNNCellSTP(RNNCell):

	def __init__(self, params, hidden_state_size, activation=None, reuse=None):

		"""
		Load all variables from the params dictionary
		"""
		super(RNNCellSTP, self).__init__()
		self._num_units = hidden_state_size
		self._activation = activation
		self.hidden_state_size = hidden_state_size

		for key,value in params.items():
			setattr(self, key, value) 
		
	@property
	def state_size(self):
		return self.hidden_state_size

	@property
	def output_size(self):
		return self.hidden_state_size

	def call(self, inputs, state, scope=None):
		"""Leaky RNN: output = new_state = (1-alpha)*state + alpha*activation(W * input + U * state + B)."""   
		
		"""
		We will update the state vector depending on whether the synapses are static (self._synapse_config == None)
		or dynamic (self._synapse_config == 'synapatic_adapt', 'stf', 'std' or 'std_stf')
		""" 
		
		"""
		Define indices into the state vector
		State vector contains all the synapic paramaters (if short-term plasticity is being applied),
		and the hiddent activity
		"""

		if self.synapse_config == 'stf':
			hidden_state = state.hidden
			syn_u = state.syn_u
		elif self.synapse_config == 'std':
			hidden_state = state.hidden
			syn_x = state.syn_x
		elif self.synapse_config == 'std_stf': 
			hidden_state = state.hidden
			syn_x = state.syn_x
			syn_u = state.syn_u
		else:
			hidden_state = state


			
		if self.synapse_config == 'stf': 
			
			# implement synaptic short term facilitation, but no depression
			syn_u += self.alpha_stf*(self.U-syn_u) + self.dt*self.U*(1-syn_u)*hidden_state/1000 
			syn_u = tf.minimum(np.float32(1), tf.maximum(np.float32(0), syn_u))
			state_post = syn_u*hidden_state
			
		elif self.synapse_config == 'std':
			
			# implement synaptic short term derpression, but no facilitation
			syn_x += self.alpha_std*(1-syn_x) - self.dt*syn_x*hidden_state/1000 
			syn_x = tf.minimum(np.float32(1), tf.maximum(np.float32(0), syn_x))
			state_post = syn_x*hidden_state
			
		elif self.synapse_config == 'std_stf': 
			
			# implement both synaptic short term facilitation and depression  
			syn_u += self.alpha_stf*(self.U-syn_u) + self.dt*self.U*(1-syn_u)*hidden_state/1000 
			syn_x += self.alpha_std*(1-syn_x) - self.dt*syn_x*hidden_state/1000 
			syn_u = tf.minimum(np.float32(1), tf.maximum(np.float32(0), syn_u))
			syn_x = tf.minimum(np.float32(1), tf.maximum(np.float32(0), syn_x))
			state_post = syn_u*syn_x*hidden_state     
		else:
			state_post = state
			
		"""
		Initialize weights and biases
		"""
		# with tf.variable_scope(scope or 'rnn_cell'):
		with vs.variable_scope('rnn_cell'):
			W_in = vs.get_variable('W_in', initializer = self.w_in0, trainable=True)
			W_rnn = vs.get_variable('W_rnn', initializer = self.w_rnn0, trainable=True)
			b_rnn = vs.get_variable('b_rnn', initializer = self.b_rnn0, trainable=True)
			W_ei = vs.get_variable('EI', initializer = self.EI_matrix, trainable=False)
			
		"""
		Main calculation 
		If self.EI is True, then excitatory and inhibiotry neurons are desired, and will we ensure that recurrent enurons 
		are of only one type, and that W_in weights are non-negative 
		"""
		
		if self.EI:
		   
			#s1 = tf.matmul(tf.nn.relu(W_in), tf.nn.relu(inputs))
			#s2 = tf.matmul(tf.matmul(tf.nn.relu(W_rnn), W_ei), state_post)
			new_state = tf.nn.relu((1-self.alpha_neuron)*hidden_state + self.alpha_neuron*(tf.matmul(tf.nn.relu(inputs), tf.nn.relu(W_in)) + tf.matmul(state_post, tf.matmul(tf.nn.relu(W_rnn), W_ei)) + b_rnn) + tf.random_normal(tf.shape(hidden_state), 0, self.noise_sd, dtype=tf.float32))
		else:                                 
			new_state = tf.nn.relu((1-self.alpha_neuron)*hidden_state + self.alpha_neuron*(tf.matmul(W_in, inputs) + tf.matmul(W_rnn, state_post) + b_rnn) + tf.random_normal(tf.shape(hidden_state), 0, self.noise_sd, dtype=tf.float32))   
			
		
		if self.synapse_config == 'stf': 
			state.hidden = new_state
			state.syn_u = syn_u
		elif self.synapse_config == 'std':
			state.hidden = new_state
			state.syn_x = syn_x
		elif self.synapse_config == 'std_stf': 
			state.hidden = new_state
			state.syn_x = syn_x
			state.syn_u = syn_u
		else:
			state = new_state
 
		return state, state

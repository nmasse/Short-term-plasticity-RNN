import tensorflow as tf
import numpy as np
from model_saver import *
from parameters import *

#######################
### Dendrite Inputs ###
#######################

### Collects the weights and inputs for the input of each dendrite

# W_in      = [n_hidden_neurons x dendrites_per_neuron x n_input_neurons]
# W_rnn     = [n_hidden_neurons x dendrites_per_neuron x n_hidden_neurons]
# rnn_input = [n_input_neurons x batch_train_size]
# h_soma    = [n_hidden_neurons x batch_train_size]
# y         = [n_hidden_neurons x dendrites_per_neuron x batch_train_size]

def in_tensordot(W_in, W_rnn, rnn_input, h_soma):
    """
    Creation Date: 2017/6/30
    """

    # Matrix multiplies (generally) the inputs and weights, and sums them
    y = tf.tensordot(W_in, rnn_input, ([2],[0])) \
                            + tf.tensordot(W_rnn, h_soma, ([2],[0]))
    return y



##########################
### Dendrite Processes ###
##########################

### Processes each individual dendrite to simulate any internal processes

# x = [n_hidden_neurons x dendrites_per_neuron x batch_train_size]
# y = [n_hidden_neurons x dendrites_per_neuron x batch_train_size]

def pr_pass_through(x):
    """
    Creation Date: 2017/6/30
    """

    # Each dendrite does nothing but pass its input to its output
    y = x
    return x



#############################
### Dendrite Accumulators ###
#############################

### Accumulates the results of the dendrites per neuron and produces
### the input to the following hidden neuron

# x = [n_hidden_neurons x dendrites_per_neuron x batch_train_size]
# y = [n_hidden_neurons x batch_train_size]

def ac_simple_sum(x):
    """
    Creation Date: 2017/6/30
    """

    # Sums the dendrite outputs for each neuron
    y = tf.reduce_sum(x,1)/par['den_per_unit']
    return y



##############################
### Dendrite configuration ###
##############################

### Order of operations:
# Input functions (collects the weights and inputs for the input of each dendrite)
# Process functions (processes each individual dendrite to simulate any internal actions)
# Accumulator functions (accumulates the dendrite results per neuron)

def dendrite_function0001(W_in, W_rnn_effective, rnn_input, h_soma):
    """
    Creation Date: 2017/6/30
    Notes: The most basic of configurations.
    """

    h_den_in = in_tensordot(W_in, W_rnn_effective, rnn_input, h_soma)
    h_den_out = pr_pass_through(h_den_in)
    h_soma_in = ac_simple_sum(h_den_out)

    return h_soma_in

import tensorflow as tf
import numpy as np
from model_saver import *
from parameters import *


test = np.ones([5,2,5])
test = test * [1,2,3,4,5]

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
    y = tf.tensordot(W_in, rnn_input, ([2],[0])) + tf.tensordot(W_rnn, h_soma, ([2],[0]))

    return y


def in_dot_atten(W_in, W_rnn, rnn_input, h_soma):
    """
    Creation Date: 2017/7/3
    Notes: Attentuates the inputs by a constant.
    """
    y = tf.divide(in_tensordot(W_in, W_rnn, rnn_input, h_soma), 2)

    return y

##########################
### Dendrite Processes ###
##########################

### Processes each individual dendrite to simulate any internal processes

# x = [n_hidden_neurons x dendrites_per_neuron x batch_train_size]
# y = [n_hidden_neurons x dendrites_per_neuron x batch_train_size]

def pr_pass_through(x):
    """
    Each dendrite does nothing but pass its input to its output
    """
    y = x
    return x

def pr_bias(x):
    """
    Adds a negative bias to each dendrite
    """

    return x - 1

def pr_relu(x):
    """
    Performs a relu on each dendrite
    """

    return tf.nn.relu(x)

def pr_sigmoid(x):
    """
    Performs a sigmoid on each dendrite
    """

    return tf.nn.sigmoid(x)

def pr_retain(x, dend):
    """
    Performs a relu on each dendrite plus its previous decayed state
    """

    return tf.nn.sigmoid(x + par['alpha_dendrite']*dend)


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
    Notes: Sums the dendrite outputs for each neuron
    """

    y = tf.reduce_sum(x,1)/par['den_per_unit']
    return y


def ac_mult_all(x):
    """
    Creation Date: 2017/7/3
    Notes:  Multiplies the outputs of all of the dendrites together
    """

    y = tf.reduce_prod(x,1)/par['den_per_unit']
    return y


def ac_threshold(x):
    """
    Creation Date: 2017/7/3
    Notes: NONFUNCTIONAL
           Sums the dendrite outputs and checks if the sum is greater than 1
    """
    x0 = tf.zeros([par['n_hidden'], par['batch_train_size']])
    x1 = tf.ones([par['n_hidden'], par['batch_train_size']])
    x2 = tf.reduce_sum(x,1)
    y = tf.where(tf.greater_equal(x1, x2), x1, x0)
    return y

##############################
### Dendrite configuration ###
##############################

### Order of operations:
# Input functions (collects the weights and inputs for the input of each dendrite)
# Process functions (processes each individual dendrite to simulate any internal actions)
# Accumulator functions (accumulates the dendrite results per neuron)

def dendrite_function0001(W_in, W_rnn_effective, rnn_input, h_soma, dend):
    """
    Creation Date: 2017/6/30
    Notes: The most basic of configurations.
    """

    h_den_in = in_tensordot(W_in, W_rnn_effective, rnn_input, h_soma)
    h_den_out = pr_pass_through(h_den_in)
    h_soma_in = ac_simple_sum(h_den_out)

    return h_soma_in, h_den_out

def dendrite_function0002(W_in, W_rnn_effective, rnn_input, h_soma, dend):
    """
    Creation Date: 2017/6/30
    Notes: Biased and relu'ed.
    """

    h_den_in = in_tensordot(W_in, W_rnn_effective, rnn_input, h_soma)
    h_den_out = pr_bias(h_den_in)
    h_den_out = pr_relu(h_den_out)
    h_soma_in = ac_simple_sum(h_den_out)

    return h_soma_in, h_den_out

def dendrite_function0003(W_in, W_rnn_effective, rnn_input, h_soma, dend):
    """
    Creation Date: 2017/7/3
    Notes: Biased and relu'ed with contribution from previous state.
    """

    h_den_in = in_tensordot(W_in, W_rnn_effective, rnn_input, h_soma)

    h_den_out = pr_bias(h_den_in)
    h_den_out = pr_retain(h_den_out, dend)
    h_soma_in = ac_simple_sum(h_den_out)

    return h_soma_in, h_den_out

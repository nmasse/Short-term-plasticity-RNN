import tensorflow as tf
import numpy as np
from model_saver import *
from parameters import *

##############################
### Dendrite Direct Inputs ###
##############################

### Collects the weights and inputs for the input of each dendrite

# W_in      = [n_hidden_neurons x dendrites_per_neuron x n_input_neurons]
# rnn_input = [n_input_neurons x batch_train_size]
# y         = [n_hidden_neurons x dendrites_per_neuron x batch_train_size]

def in_tensordot(W_in, rnn_input):
    """
    Creation Date: 2017/6/30
    """

    # Matrix multiplies (generally) the inputs and weights, and sums them
    y = tf.tensordot(W_in, rnn_input, ([2],[0]))

    return y


def in_dot_atten(W_in, rnn_input):
    """
    Creation Date: 2017/7/3
    Notes: Attentuates the inputs by a constant.
    """
    y = tf.divide(in_tensordot(W_in, rnn_input), 2)

    return y


#################################
### Dendrite Recurrent Inputs ###
#################################

### Collects the weights and inputs for the recurrent input of each dendrites_init

# W_rnn     = [n_hidden_neurons x dendrites_per_neuron x n_hidden_neurons]
# h_soma    = [n_hidden_neurons x batch_train_size]
# y         = [n_hidden_neurons x dendrites_per_neuron x batch_train_size]

def EI_weights(W_rnn):
    """
    Produces a pair of weight matrices corresponding to excitatory and
    inhibitory weight selections
    """

    EI_exc = tf.constant(par['EI_matrix_d_exc'], name='W_ei_d_exc')
    EI_inh = tf.constant(par['EI_matrix_d_inh'], name='W_ei_d_inh')

    W_rnn_exc = tf.tensordot(tf.nn.relu(W_rnn), EI_exc, ([2],[0]))
    W_rnn_inh = tf.tensordot(tf.nn.relu(W_rnn), EI_inh, ([2],[0]))

    return W_rnn_exc, W_rnn_inh


def EI_activity(W_rnn, h_soma):
    """
    Takes excitatory and inhibitory weight matrices and applies them to the
    recurrent, somatic input to find excitatory and inhibitory activity
    """

    W_rnn_exc, W_rnn_inh = EI_weights(W_rnn)

    exc_activity = tf.tensordot(W_rnn_exc, h_soma, ([2],[0]))
    inh_activity = tf.tensordot(W_rnn_inh, h_soma, ([2],[0]))

    return exc_activity, inh_activity


def rin_basicEI(W_rnn, h_soma):
    """
    Creation Date: 2017/7/5
    Notes: Makes excitatory inputs positive and inhibitory inputs negative.
    """

    exc_activity, inh_activity = EI_activity(W_rnn, h_soma)
    # REMOVE y, takes TIME!
    y = (exc_activity - inh_activity)

    return y, exc_activity, inh_activity


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


def ac_two_groups(x):
    """
    Creation Date: 2017/7/7
    Notes: Meant to artificially incite dendrite grouping
    """
    group1, group2 = tf.split(x, 2, 1)

    return tf.reduce_sum(group1, 1) - tf.reduce_sum(group2, 1)


##############################
### Dendrite configuration ###
##############################

### Order of operations:
# Input functions (collects the weights and inputs for the input of each dendrite)
# Process functions (processes each individual dendrite to simulate any internal actions)
# Accumulator functions (accumulates the dendrite results per neuron)

def dendrite_function0001(W_in, W_td, W_rnn, stim_in, td_in, h_soma, dend):
    """
    Creation Date: 2017/6/30
    Notes: The most basic of configurations.
    """

    den_in = in_tensordot(W_in, stim_in) + in_tensordot(W_td, td_in)
    rnn_in, exc_activity, inh_activity = rin_basicEI(W_rnn, h_soma)

    h_den_in = den_in + rnn_in
    h_den_out = pr_pass_through(h_den_in)
    h_soma_in = ac_simple_sum(h_den_out)

    return h_soma_in, h_den_out, exc_activity, inh_activity

def dendrite_function0002(W_in, W_td, W_rnn, stim_in, td_in, h_soma, dend):
    """
    Creation Date: 2017/6/30
    Notes: Biased and relu'ed.
    """

    den_in = in_tensordot(W_in, stim_in) + in_tensordot(W_td, td_in)
    rnn_in, exc_activity, inh_activity = rin_basicEI(W_rnn, h_soma)

    h_den_in = den_in + rnn_in
    h_den_out = pr_bias(h_den_in)
    h_den_out = pr_relu(h_den_out)
    h_soma_in = ac_simple_sum(h_den_out)

    return h_soma_in, h_den_out, exc_activity, inh_activity

def dendrite_function0003(W_in, W_td, W_rnn, stim_in, td_in, h_soma, dend):
    """
    Creation Date: 2017/7/3
    Notes: Biased and relu'ed with contribution from previous state.
    """

    den_in = in_tensordot(W_in, stim_in) + in_tensordot(W_td, td_in)
    rnn_in, exc_activity, inh_activity = rin_basicEI(W_rnn, h_soma)

    h_den_in = den_in + rnn_in
    h_den_out = pr_bias(h_den_in)
    h_den_out = pr_retain(h_den_out, dend)
    h_soma_in = ac_simple_sum(h_den_out)

    return h_soma_in, h_den_out, exc_activity, inh_activity

def dendrite_function0004(W_in, W_td, W_rnn, stim_in, td_in, h_soma, dend):
    """
    Creation Date: 2017/7/5
    Notes: Inhibition will gate excitatory inputs. Using ReLu's
    """
    beta = tf.constant(np.float32(1))
    alpha = tf.constant(np.float32(0.5))

    den_in = in_tensordot(W_in, stim_in) + in_tensordot(W_td, td_in)
    _, exc_activity, inh_activity = rin_basicEI(W_rnn, h_soma)

    exc_activity += den_in

    h_den_out = (1-par['alpha_dendrite'])*dend + par['alpha_dendrite']*tf.nn.relu(exc_activity - beta)/(alpha+inh_activity)

    h_soma_in = ac_simple_sum(h_den_out)

    return h_soma_in, h_den_out, exc_activity, inh_activity

def dendrite_function0005(W_in, W_td, W_rnn, stim_in, td_in, h_soma, dend):
    """
    Creation Date: 2017/7/5
    Notes:Summing inputs, plus ReLu
    """

    den_in = in_tensordot(W_in, stim_in) + in_tensordot(W_td, td_in)
    _, exc_activity, inh_activity = rin_basicEI(W_rnn, h_soma)

    h_den_out = (1-par['alpha_dendrite'])*dend +  par['alpha_dendrite']*tf.nn.relu(exc_activity + den_in - inh_activity)

    h_soma_in = ac_simple_sum(h_den_out)

    return h_soma_in, h_den_out, exc_activity, inh_activity

def dendrite_function0006(W_in, W_td, W_rnn, stim_in, td_in, h_soma, dend):
    """
    Creation Date: 2017/7/5
    Notes:Summing inputs
    """

    den_in = in_tensordot(W_in, stim_in) + in_tensordot(W_td, td_in)
    _, exc_activity, inh_activity = rin_basicEI(W_rnn, h_soma)

    h_den_out = (1-par['alpha_dendrite'])*dend +  par['alpha_dendrite']*(exc_activity + den_in - inh_activity)

    h_soma_in = ac_simple_sum(h_den_out)

    return h_soma_in, h_den_out, exc_activity, inh_activity

def dendrite_function0007(W_in, W_td, W_rnn, stim_in, td_in, h_soma, dend):
    """
    Creation Date: 2017/7/7
    Notes: Dendrites are split into groups that are evaluated against
            each other, taking history into account  No bias.
    """

    den_in = in_tensordot(W_in, stim_in) + in_tensordot(W_td, td_in)
    rnn_in, exc_activity, inh_activity = rin_basicEI(W_rnn, h_soma)

    h_den_in = den_in + rnn_in
    h_den_out = pr_retain(h_den_in, dend)
    h_soma_in = ac_two_groups(h_den_out)

    return h_soma_in, h_den_out, exc_activity, inh_activity


def dendrite_function0008(W_in, W_td, W_rnn, stim_in, td_in, h_soma, dend):
    """
    Creation Date: 2017/7/11
    Notes: Inhibition gates inputs. EXC and INH are multiplied
    See "On Multiplicative Integration with Recurrent Neural Networks", Wu et al 2016
    """
    beta = tf.constant(np.float32(1))
    alpha = tf.constant(np.float32(1))

    den_in = in_tensordot(W_in, stim_in) + in_tensordot(W_td, td_in)
    _, exc_activity, inh_activity = rin_basicEI(W_rnn, h_soma)

    exc_activity += den_in

    h_den_out = (1-par['alpha_dendrite'])*dend + \
        par['alpha_dendrite']*tf.nn.relu(exc_activity - alpha)*tf.nn.relu(beta - inh_activity)

    h_soma_in = ac_simple_sum(h_den_out)

    return h_soma_in, h_den_out, exc_activity, inh_activity

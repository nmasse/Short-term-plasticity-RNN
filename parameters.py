"""
Overhauling the parameters setup
2017/06/21 Gregory Grant
"""

import numpy as np
import tensorflow as tf

"""
To have access to all parameters in other modules, put the following code
snippet at the top of the file.

import imp

def import_parameters():
    f = open('parameters.py')
    global par
    par = imp.load_source('data', '', f)
    f.close()

import_parameters()

"""

print("--> Loading parameters...")


##############################
### Independent parameters ###
##############################

# Setup parameters
stimulus_type           =   'dms'
profile_path            =   './profiles/motion.txt'
save_dir                =   './savedir/'
debug_model             =   False
load_previous_model     =   False

# Network configuration
synapse_config          =   None    # Full is 'std_stf'
exc_inh_prop            =   1.0     # Literature 0.8, for EI off 1
var_delay               =   False

# Network shape
num_motion_tuned        =   36
num_fix_tuned           =   0
num_rule_tuned          =   0
n_hidden                =   50
den_per_unit            =   1
n_output                =   3

# Timings and rates
dt                      =   25
learning_rate           =   5e-3
membrane_time_constant  =   100
connection_prob         =   1       # Usually 1

# Variance values
clip_max_grad_val       =   0.25
input_mean              =   0
input_sd                =   0.1   # from 0.1
noise_sd                =   0.5   # from 0.5
input_clip_max          =   10000   # keep this high unless limiting inputs

# Tuning function data
num_motion_dirs         =   8
tuning_height           =   1       # height scaling factor
kappa                   =   1       # concentration scaling factor for von Mises
catch_rate              =   0.2
match_rate              =   0.5     # tends a little higher than chosen rate
possible_rules          =   [0]

# Probe specs
probe_trial_pct         =   0
probe_time              =   25

# Cost parameters
spike_cost              =   5e-5
wiring_cost             =   5e-7

# Synaptic plasticity specs
tau_fast                =   200
tau_slow                =   1500
U_stf                   =   0.15
U_std                   =   0.45

# Performance thresholds
stop_perf_th            =   1
stop_error_th           =   1

# Training specs
batch_train_size        =   128
num_batches             =   8
num_iterations          =   1500
trials_between_outputs  =   5     # Ususally 500

# Pickle save paths
save_fn = 'DMS_stp_delay_' + str(0) + '_' + str(0) + '.pkl'
ckpt_save_fn = 'model_' + str(0) + '.ckpt'
ckpt_load_fn = 'model_' + str(0) + '.ckpt'


############################
### Dependent parameters ###
############################

# Number of input neurons
n_input = num_motion_tuned + num_fix_tuned + num_rule_tuned
# General network shape
shape = (n_input, n_hidden, n_output)
# The time step in seconds
dt_sec = dt/1000

# If num_inh_units is set > 0, then neurons can be either excitatory or
# inihibitory; is num_inh_units = 0, then the weights projecting from
# a single neuron can be a mixture of excitatory or inhibitory
if exc_inh_prop < 1.:
    EI = True
else:
    EI = False

num_exc_units = int(np.round(n_hidden*exc_inh_prop))
num_inh_units = n_hidden - num_exc_units

EI_list = np.ones(n_hidden, dtype=np.float32)
EI_list[-num_inh_units:] = -1.

EI_matrix = np.diag(EI_list)

# Membrane time constant of RNN neurons
alpha_neuron = dt/membrane_time_constant
# The standard deviation of the Gaussian noise added to each RNN neuron
# at each time step
noise_sd = np.sqrt(2*alpha_neuron)*noise_sd


def initialize(dims, connection_prob):
    n = np.float32(np.random.gamma(shape=0.25, scale=1.0, size=dims))
    n *= (np.random.rand(*dims) < connection_prob)
    return n


def get_profile(profile_path):
    """
    Gets profile information from the profile file
    """

    with open(profile_path) as neurons:
        raw_content = neurons.read().split("\n")

    text = list(filter(None, raw_content))

    for line in range(len(text)):
        text[line] = text[line].split("\t")

    name_of_stimulus = text[0][1]
    date_stimulus_created = text[1][1]
    author_of_stimulus_profile = text[2][1]

    return name_of_stimulus, date_stimulus_created, author_of_stimulus_profile


def get_events(profile_path):
    """
    Gets event information from the profile file
    """

    with open(profile_path) as event_list:
        raw_content = event_list.read().split("\n")

    text = list(filter(None, raw_content))

    for line in range(len(text)):
        if text[line][0] == "0":
            content = text[line:]

    for line in range(len(content)):
        content[line] = content[line].split("\t")
        content[line][0] = int(content[line][0])

    return content

# General event profile info
name_of_stimulus, date_stimulus_created, author_of_stimulus_profile = get_profile(profile_path)
# List of events that occur for the network
events = get_events(profile_path)
# Length of each trial in ms
trial_length = events[-1][0]
# Length of each trial in time steps
num_time_steps = trial_length//dt


# Number of rules - used in input tuning
num_rules = len(possible_rules)

####################################################################
### Setting up assorted intial weights, biases, and other values ###
####################################################################

h_init = 0.1*np.ones((n_hidden, batch_train_size), dtype=np.float32)

input_to_hidden_dims = [n_hidden, den_per_unit, n_input]
rnn_to_rnn_dims = [n_hidden, den_per_unit, n_hidden]

# Initialize input weights
w_in0 = initialize(input_to_hidden_dims, connection_prob)

# Initialize starting recurrent weights
# If excitatory/inhibitory neurons desired, initializes with random matrix with
#   zeroes on the diagonal
# If not, initializes with a diagonal matrix
if EI:
    w_rnn0 = initialize(rnn_to_rnn_dims, connection_prob)
    eye = np.zeros([*rnn_to_rnn_dims], dtype=np.float32)
    for j in range(den_per_unit):
        for i in range(n_hidden):
            eye[i,j,i] = 1
    w_rec_mask = np.ones((rnn_to_rnn_dims), dtype=np.float32) - eye
else:
    w_rnn0 = np.zeros([*rnn_to_rnn_dims], dtype=np.float32)
    for j in range(den_per_unit):
        for i in range(n_hidden):
            w_rnn0[i,j,i] = 0.975
    w_rec_mask = np.ones((rnn_to_rnn_dims), dtype=np.float32)

# Initialize starting recurrent biases
# Note that the second dimension in the bias initialization term can be either
# 1 or self.params['batch_train_size'].
set_to_one = True
bias_dim = (1 if set_to_one else batch_train_size)
b_rnn0 = np.zeros((n_hidden, bias_dim), dtype=np.float32)

# Effective synaptic weights are stronger when no short-term synaptic plasticity
# is used, so the strength of the recurrent weights is reduced to compensate
if synapse_config == None:
    w_rnn0 /= 3

# Initialize output weights and biases
w_out0 =initialize([n_output, n_hidden], connection_prob)

b_out0 = np.zeros((n_output, 1), dtype=np.float32)
w_out_mask = np.ones((n_output, n_hidden), dtype=np.float32)

if EI:
    ind_inh = np.where(EI_list == -1)[0]
    w_out0[:, ind_inh] = 0
    w_out_mask[:, ind_inh] = 0

######################################
### Setting up synaptic parameters ###
######################################

# 0 = static
# 1 = facilitating
# 2 = depressing

synapse_type = np.zeros(n_hidden, dtype=np.int8)

# only facilitating synapses
if synapse_config == 'stf':
    synapse_type = np.ones(n_hidden, dtype=np.int8)

# only depressing synapses
elif synapse_config == 'std':
    synapse_type = 2*np.ones(n_hidden, dtype=np.int8)

# even numbers facilitating, odd numbers depressing
elif synapse_config == 'std_stf':
    synapse_type = np.ones(n_hidden, dtype=np.int8)
    ind = range(1,n_hidden,2)
    synapse_type[ind] = 2

alpha_stf = np.ones((n_hidden, 1), dtype=np.float32)
alpha_std = np.ones((n_hidden, 1), dtype=np.float32)
U = np.ones((n_hidden, 1), dtype=np.float32)

# initial synaptic values
syn_x_init = np.zeros((n_hidden, batch_train_size), dtype=np.float32)
syn_u_init = np.zeros((n_hidden, batch_train_size), dtype=np.float32)

for i in range(n_hidden):
    if synapse_type[i] == 1:
        alpha_stf[i,0] = dt/tau_slow
        alpha_std[i,0] = dt/tau_fast
        U[i,0] = 0.15
        syn_x_init[i,:] = 1
        syn_u_init[i,:] = U[i,0]

    elif synapse_type[i] == 2:
        alpha_stf[i,0] = dt/tau_fast
        alpha_std[i,0] = dt/tau_slow
        U[i,0] = 0.45
        syn_x_init[i,:] = 1
        syn_u_init[i,:] = U[i,0]


print("--> Parameters successfully loaded.\n")

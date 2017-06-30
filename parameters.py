"""
Overhauling the parameters setup
2017/06/21 Gregory Grant
"""

import numpy as np
import tensorflow as tf

print("--> Loading parameters...")

##############################
### Independent parameters ###
##############################

global par

par = {
    # Setup parameters
    'stimulus_type'     : 'att',
    'profile_path'      : './profiles/attention.txt',
    'save_dir'          : './savedir/',
    'debug_model'       : False,
    'load_previous_model' : False,

    # Network configuration
    'synapse_config'    : None,      # Full is 'std_stf'
    'exc_inh_prop'      : 0.8,       # Literature 0.8, for EI off 1
    'use_dendrites'     : True,
    'var_delay'         : False,
    'catch_trials'      : False,     # Note that turning on var_delay implies catch_trials

    # Network shape
    'num_motion_tuned'  : 96,
    'num_fix_tuned'     : 0,
    'num_rule_tuned'    : 8,
    'n_hidden'          : 50,
    'den_per_unit'      : 1,
    'n_output'          : 3,

    # Timings and rates
    'dt'                : 25,
    'learning_rate'     : 5e-3,
    'membrane_time_constant'    : 100,
    'connection_prob'   : 1,         # Usually 1

    # Variance values
    'clip_max_grad_val' : 0.25,
    'input_mean'        : 0,
    'input_sd'          : 0.1,
    'noise_sd'          : 0.5,
    'input_clip_max'    : 10000,     # keep this high unless limiting inputs

    # Tuning function data
    'num_motion_dirs'   : 8,
    'tuning_height'     : 1,        # magnitutde scaling factor for von Mises
    'kappa'             : 1,        # concentration scaling factor for von Mises
    'catch_rate'        : 0.2,
    'match_rate'        : 0.5,      # tends a little higher than chosen rate
    'num_receptive_fields'  : 4,    # contributes to 'possible_rules'
    'num_categorizations'   : 2,    # contributes to 'possible_rules'
    'allowed_fields'        : [0,1,2,3],  # can hold 0 through num_fields - 1
    'allowed_categories'    : [0],  # Can be 0,1

    # Probe specs
    'probe_trial_pct'   : 0,
    'probe_time'        : 25,

    # Cost parameters
    'spike_cost'        : 5e-5,
    'wiring_cost'       : 5e-7,

    # Synaptic plasticity specs
    'tau_fast'          : 200,
    'tau_slow'          : 1500,
    'U_stf'             : 0.15,
    'U_std'             : 0.45,

    # Performance thresholds
    'stop_perf_th'      : 1,
    'stop_error_th'     : 1,

    # Training specs
    'batch_train_size'  : 128,
    'num_batches'       : 8,
    'num_iterations'    : 1200,
    'iterations_between_outputs'    : 5,        # Ususally 500

    # Pickle save paths
    'save_fn'           : 'model_data.json',
    'ckpt_save_fn'      : 'model_' + str(0) + '.ckpt',
    'ckpt_load_fn'      : 'model_' + str(0) + '.ckpt',
}

############################
### Dependent parameters ###
############################

def update_parameters(updates):
    """
    Takes a list of strings and values for updating parameters in the parameter dictionary
    Example: updates = [(key, val), (key, val)]
    """
    for (key, val) in updates:
        par[key] = val

    update_dependencies()

def update_dependencies():
    """
    Updates all parameter dependencies
    """

    # Number of input neurons
    par['n_input'] = par['num_motion_tuned'] + par['num_fix_tuned'] + par['num_rule_tuned']
    # General network shape
    par['shape'] = (par['n_input'], par['n_hidden'], par['n_output'])

    # Possible rules based on rule type values
    par['possible_rules'] = [par['num_receptive_fields'], par['num_categorizations']]
    # Number of rules - used in input tuning
    par['num_rules'] = len(par['possible_rules'])

    # Sets the number of accessible allowed fields, if equal, all allowed fields
    # are accessible, but no more than those.
    par['num_active_fields'] = len(par['allowed_fields'])
    # Checks to ensure valid receptor fields
    if len(par['allowed_fields']) < par['num_active_fields']:
        print("ERROR: More active fields than allowed receptor fields.")
        quit()
    elif par['num_active_fields'] <= 0:
        print("ERROR: Must have 1 or more active receptor fields.")
        quit()

    # If num_inh_units is set > 0, then neurons can be either excitatory or
    # inihibitory; is num_inh_units = 0, then the weights projecting from
    # a single neuron can be a mixture of excitatory or inhibitory
    if par['exc_inh_prop'] < 1.:
        par['EI'] = True
    else:
        par['EI']  = False

    par['num_exc_units'] = int(np.round(par['n_hidden']*par['exc_inh_prop']))
    par['num_inh_units'] = par['n_hidden'] - par['num_exc_units']

    par['EI_list'] = np.ones(par['n_hidden'], dtype=np.float32)
    par['EI_list'][-par['num_inh_units']:] = -1.

    par['EI_matrix'] = np.diag(par['EI_list'])

    # Membrane time constant of RNN neurons
    par['alpha_neuron'] = par['dt']/par['membrane_time_constant']
    # The standard deviation of the Gaussian noise added to each RNN neuron
    # at each time step
    par['noise_sd'] = np.sqrt(2*par['alpha_neuron'])*par['noise_sd']


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
    par['name_of_stimulus'], par['date_stimulus_created'], par['author_of_stimulus_profile'] = get_profile(par['profile_path'])
    # List of events that occur for the network
    par['events'] = get_events(par['profile_path'])
    # The time step in seconds
    par['dt_sec'] = par['dt']/1000
    # Length of each trial in ms
    par['trial_length'] = par['events'][-1][0]
    # Length of each trial in time steps
    par['num_time_steps'] = par['trial_length']//par['dt']


    ####################################################################
    ### Setting up assorted intial weights, biases, and other values ###
    ####################################################################

    def spectral_radius(A):
        """
        Compute the spectral radius of each dendritic dimension of a weight array,
        and normalize using square room of the sum of squares of those radii.
        """
        if A.ndim == 2:
            return np.max(abs(np.linalg.eigvals(A)))
        elif A.ndim == 3:
            # Assumes the second axis is the target (for dendritic setup)
            r = 0
            for n in range(np.shape(A)[1]):
                r = r + np.max(abs(np.linalg.eigvals(np.squeeze(A[:,n,:]))))

            return r / np.shape(A)[1]

    par['h_init'] = 0.1*np.ones((par['n_hidden'], par['batch_train_size']), dtype=np.float32)

    par['input_to_hidden_dims'] = [par['n_hidden'], par['den_per_unit'], par['n_input']]
    par['rnn_to_rnn_dims'] = [par['n_hidden'], par['den_per_unit'], par['n_hidden']]
    par['rnn_to_rnn_soma_dims'] = [par['n_hidden'], par['n_hidden']]

    # Initialize input weights
    par['w_in0'] = initialize(par['input_to_hidden_dims'], par['connection_prob'])

    # Initialize starting recurrent weights
    # If excitatory/inhibitory neurons desired, initializes with random matrix with
    #   zeroes on the diagonal
    # If not, initializes with a diagonal matrix
    if par['EI']:
        par['w_rnn0'] = initialize(par['rnn_to_rnn_dims'], par['connection_prob'])
        par['w_rnn_soma0'] = initialize(par['rnn_to_rnn_soma_dims'], par['connection_prob'])

        par['eye'] = np.zeros([*par['rnn_to_rnn_dims']], dtype=np.float32)
        for i in range(par['n_hidden']):
            par['w_rnn_soma0'][i,i] = 0
            par['eye'][i,:,i] = 1
            par['w_rnn0'][i,:,i] = 0

        par['w_rec_mask'] = np.ones((par['rnn_to_rnn_dims']), dtype=np.float32) - par['eye']
    else:
        par['w_rnn0'] = np.zeros([*par['rnn_to_rnn_dims']], dtype=np.float32)
        par['w_rnn_soma0'] = np.zeros([*par['rnn_to_rnn_soma_dims']], dtype=np.float32)

        for i in range(par['n_hidden']):
            par['w_rnn_soma0'][i,i] = 1
            par['w_rnn0'][i,:,i] = 1

        par['w_rec_mask'] = np.ones((par['rnn_to_rnn_dims']), dtype=np.float32)

    # Initialize starting recurrent biases
    # Note that the second dimension in the bias initialization term can be either
    # 1 or self.params['batch_train_size'].
    set_to_one = True
    par['bias_dim'] = (1 if set_to_one else par['batch_train_size'])
    par['b_rnn0'] = np.zeros((par['n_hidden'], par['bias_dim']), dtype=np.float32)

    # Effective synaptic weights are stronger when no short-term synaptic plasticity
    # is used, so the strength of the recurrent weights is reduced to compensate
    if par['synapse_config'] == None:
        par['w_rnn0'] = par['w_rnn0']/(2*spectral_radius(par['w_rnn0']))
        par['w_rnn_soma0'] = par['w_rnn_soma0']/(2*spectral_radius(par['w_rnn_soma0']))

    # Initialize output weights and biases
    par['w_out0'] =initialize([par['n_output'], par['n_hidden']], par['connection_prob'])

    par['b_out0'] = np.zeros((par['n_output'], 1), dtype=np.float32)
    par['w_out_mask'] = np.ones((par['n_output'], par['n_hidden']), dtype=np.float32)

    if par['EI']:
        par['ind_inh'] = np.where(par['EI_list'] == -1)[0]
        par['w_out0'][:, par['ind_inh']] = 0
        par['w_out_mask'][:, par['ind_inh']] = 0

    ######################################
    ### Setting up synaptic parameters ###
    ######################################

    # 0 = static
    # 1 = facilitating
    # 2 = depressing

    par['synapse_type'] = np.zeros(par['n_hidden'], dtype=np.int8)

    # only facilitating synapses
    if par['synapse_config'] == 'stf':
        par['synapse_type'] = np.ones(par['n_hidden'], dtype=np.int8)

    # only depressing synapses
    elif par['synapse_config'] == 'std':
        par['synapse_type'] = 2*np.ones(par['n_hidden'], dtype=np.int8)

    # even numbers facilitating, odd numbers depressing
    elif par['synapse_config'] == 'std_stf':
        par['synapse_type'] = np.ones(par['n_hidden'], dtype=np.int8)
        par['ind'] = range(1,par['n_hidden'],2)
        par['synapse_type'][par['ind']] = 2

    par['alpha_stf'] = np.ones((par['n_hidden'], 1), dtype=np.float32)
    par['alpha_std'] = np.ones((par['n_hidden'], 1), dtype=np.float32)
    par['U'] = np.ones((par['n_hidden'], 1), dtype=np.float32)

    # initial synaptic values
    par['syn_x_init'] = np.zeros((par['n_hidden'], par['batch_train_size']), dtype=np.float32)
    par['syn_u_init'] = np.zeros((par['n_hidden'], par['batch_train_size']), dtype=np.float32)

    for i in range(par['n_hidden']):
        if par['synapse_type'][i] == 1:
            par['alpha_stf'][i,0] = par['dt']/par['tau_slow']
            par['alpha_std'][i,0] = par['dt']/par['tau_fast']
            par['U'][i,0] = 0.15
            par['syn_x_init'][i,:] = 1
            par['syn_u_init'][i,:] = par['U'][i,0]

        elif par['synapse_type'][i] == 2:
            par['alpha_stf'][i,0] = par['dt']/par['tau_fast']
            par['alpha_std'][i,0] = par['dt']/par['tau_slow']
            par['U'][i,0] = 0.45
            par['syn_x_init'][i,:] = 1
            par['syn_u_init'][i,:] = par['U'][i,0]

update_dependencies()
print("--> Parameters successfully loaded.\n")

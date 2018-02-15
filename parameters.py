import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from itertools import product

print("--> Loading parameters...")

global par, analysis_par

"""
Independent parameters
"""
par = {
    # Setup parameters
    'save_dir'              : './savedir/CL4/',
    'save_analysis'         : False,
    'debug_model'           : False,
    'load_previous_model'   : False,
    'analyze_model'         : False,
    'stabilization'         : 'pathint',
    'no_gpu'                : False,

    # Network configuration
    'synapse_config'        : None, # Full is 'std_stf'
    'exc_inh_prop'          : 0.8,       # Literature 0.8, for EI off 1
    'var_delay'             : False,

    # Network shape
    'num_motion_tuned'      : 36*2,
    'num_fix_tuned'         : 20,
    'num_rule_tuned'        : 0,
    'n_hidden'              : 200,
    'n_dendrites'           : 1,

    # Euclidean shape
    'num_sublayers'         : 1,
    'neuron_dx'             : 1.0,
    'neuron_dy'             : 1.0,
    'neuron_dz'             : 10.0,

    # Timings and rates
    'dt'                    : 20,
    'learning_rate'         : 1e-3,
    'membrane_time_constant': 100,
    'connection_prob'       : 1.0,         # Usually 1

    # Variance values
    'clip_max_grad_val'     : 0.5,
    'input_mean'            : 0.0,
    'noise_in_sd'           : 0.1,
    'noise_rnn_sd'          : 0.5,

    # Tuning function data
    'num_motion_dirs'       : 8,
    'tuning_height'         : 4.0,        # magnitude scaling factor for von Mises
    'kappa'                 : 2.0,        # concentration scaling factor for von Mises

    # Cost parameters
    'spike_cost'            : 1e-3,
    'wiring_cost'           : 0, #1e-6,

    # Synaptic plasticity specs
    'tau_fast'              : 100,
    'tau_slow'              : 1500,
    'U_stf'                 : 0.15,
    'U_std'                 : 0.45,

    # Training specs
    'batch_train_size'      : 128,
    'num_iterations'        : 20,
    'iters_between_outputs' : 1,

    # Task specs
    'trial_type'            : 'limDMS',      # allowable types: DMS, DMRS45, DMRS90, DMRS180, DMC
    'num_tasks'             : 19,               # DMS+DMRS, ABBA, ABCA, dualDMS, multistim, twelvestim, limDMS
    'multistim_trial_length': 4000,
    'limDMS_trial_length'   : 1000,
    'rotation_match'        : 0,  # angular difference between matching sample and test
    'dead_time'             : 200,
    'fix_time'              : 200,
    'sample_time'           : 400,
    'delay_time'            : 200,
    'test_time'             : 400,
    'variable_delay_max'    : 400,
    'mask_duration'         : 100,  # duration of traing mask after test onset
    'catch_trial_pct'       : 0.0,
    'num_receptive_fields'  : 1,
    'num_rules'             : 1, # this will be two for the DMS+DMRS task
    'decoding_test_mode'    : False,

    # Save paths
    'save_fn'               : 'model_results.pkl',
    'ckpt_save_fn'          : 'model.ckpt',
    'ckpt_load_fn'          : 'model.ckpt',

    # Analysis
    'svm_normalize'         : True,
    'decoding_reps'         : 100,
    'simulation_reps'       : 100,
    'decode_test'           : False,
    'decode_rule'           : False,
    'decode_sample_vs_test' : False,
    'suppress_analysis'     : True,
    'analyze_tuning'        : True,

    # Omega parameters
    'omega_c'               : 0.0,
    'omega_xi'              : 0.1,
    'last_layer_mult'       : 2,
    'scale_factor'          : 1,

    # Projection of top-down activity
    'neuron_gate_pct'       : 0.0,
    'dendrite_gate_pct'     : 0.0,
    'dynamic_topdown'       : False,
    'num_tasks'             : 12,
    'td_cost'               : 0.0,

    # Fisher information parameters
    'EWC_fisher_calc_batch' : 8, # batch size when calculating EWC
    'EWC_fisher_num_batches': 256, # number of batches size when calculating EWC
}

"""
Parameters to be used before running analysis
"""
analysis_par = {
    'analyze_model'         : True,
    'load_previous_model'   : True,
    'num_iterations'        : 1,
    'batch_train_size'      : 1024,
    'var_delay'             : False,
    'learning_rate'         : 0,
    'catch_trial_pct'       : 0.0,
}

"""
Parameters to be used after running analysis
"""
revert_analysis_par = {
    'analyze_model'         : True,
    'load_previous_model'   : False,
    'decoding_test_mode'    : False
}


"""
Dependent parameters
"""

def update_parameters(updates):
    """
    Takes a list of strings and values for updating parameters in the parameter dictionary
    Example: updates = [(key, val), (key, val)]
    """
    #print('Updating parameters...')
    for key, val in updates.items():
        par[key] = val
        print(key, val)

    update_trial_params()
    update_dependencies()

def update_trial_params():

    """
    Update all the trial parameters given trial_type
    """

    par['num_rules'] = 1
    par['num_rule_tuned'] = 0

    if par['trial_type'] == 'DMS' or 'DMC' in par['trial_type']:
        par['rotation_match'] = 0

    elif par['trial_type'] == 'DMRS45':
        par['rotation_match'] = 45

    elif par['trial_type'] == 'DMRS45ccw':
        par['rotation_match'] = -45

    elif par['trial_type'] == 'DMRS90':
        par['rotation_match'] = 90

    elif par['trial_type'] == 'DMRS90ccw':
        par['rotation_match'] = -90

    elif par['trial_type'] == 'DMRS135':
        par['rotation_match'] = 135

    elif par['trial_type'] == 'DMRS135ccw':
        par['rotation_match'] = -135

    elif par['trial_type'] == 'DMRS180':
        par['rotation_match'] = 180

    elif 'Color_OneIntCat' in par['trial_type']:
        par['num_rules'] = 2
        par['rule_onset_time'] = par['dead_time']+par['fix_time']
        par['rule_offset_time'] = par['dead_time']+par['fix_time']+par['sample_time'] + par['delay_time'] + par['test_time']

    elif 'Color_DelayedCat' in par['trial_type']:
        par['num_rules'] = 2
        par['rule_onset_time'] = par['dead_time']+par['fix_time']+par['sample_time'] + par['delay_time']
        par['rule_offset_time'] = par['dead_time']+par['fix_time']+par['sample_time'] + par['delay_time'] + par['test_time']

    elif par['trial_type'] == 'dualDMS':
        par['catch_trial_pct'] = 0
        par['num_receptive_fields'] = 2
        par['num_rules'] = 2
        par['probe_trial_pct'] = 0
        par['probe_time'] = 10
        par['num_rule_tuned'] = 12
        par['sample_time'] = 500
        par['test_time'] = 500
        par['delay_time'] = 1000
        par['analyze_rule'] = True

    elif par['trial_type'] == 'ABBA' or par['trial_type'] == 'ABCA':
        par['catch_trial_pct'] = 0
        par['match_test_prob'] = 0.5
        par['max_num_tests'] = 3
        par['sample_time'] = 400
        par['delay_time'] = 2400
        #par['spike_cost'] = 1e-2
        par['ABBA_delay'] = par['delay_time']//par['max_num_tests']//2
        par['repeat_pct'] = 0
        par['analyze_test'] = True
        if par['trial_type'] == 'ABBA':
            par['repeat_pct'] = 0.5

    elif par['trial_type'] == 'DMS+DMRS' or par['trial_type'] == 'DMS+DMRS_early_cue':

        par['num_rules'] = 2
        par['num_rule_tuned'] = 12
        if par['trial_type'] == 'DMS+DMRS':
            par['rotation_match'] = [0, 90]
            par['rule_onset_time'] = par['dead_time']+par['fix_time']+par['sample_time'] + 500
            par['rule_offset_time'] = par['dead_time']+par['fix_time']+par['sample_time'] + 750
        else:
            par['rotation_match'] = [0, 45]
            par['rule_onset_time'] = par['dead_time']
            par['rule_offset_time'] = par['dead_time']+par['fix_time']+par['sample_time']

    elif par['trial_type'] == 'DMS+DMC':
        par['num_rules'] = 2
        par['num_rule_tuned'] = 12
        par['rotation_match'] = [0, 0]
        par['rule_onset_time'] = par['dead_time']+par['fix_time']+par['sample_time'] + 500
        par['rule_offset_time'] = par['dead_time']+par['fix_time']+par['sample_time'] + par['delay_time'] + par['test_time']

    elif par['trial_type'] in ['multistim', 'twelvestim', 'limDMS']:
        #print('Multistim params update placeholder.')
        pass

    else:
        print(par['trial_type'], 'not a recognized trial type')
        quit()


    # use this for all networks
    #par['num_rule_tuned'] = 12
    #par['num_fix_tuned'] = 12


def update_dependencies():
    """
    Updates all parameter dependencies
    """

    par['n_output'] = par['num_motion_dirs'] + 1

    # Number of input neurons
    par['n_input'] = par['num_motion_tuned'] + par['num_fix_tuned']
    # General network shape
    par['shape'] = (par['n_input'], par['n_hidden'], par['n_output'])

    # Create TD
    par['neuron_topdown'] = []
    par['dendrite_topdown'] = []
    for _ in range(par['num_tasks']):
        par['neuron_topdown'].append(np.float32(np.random.choice([0,1], par['n_hidden'], p= [par['neuron_gate_pct'], 1-par['neuron_gate_pct']])))
        par['dendrite_topdown'].append(np.float32(np.random.choice([0,1], [par['n_hidden'], par['n_dendrites']], p= [par['dendrite_gate_pct'], 1-par['dendrite_gate_pct']])))


    # If num_inh_units is set > 0, then neurons can be either excitatory or
    # inihibitory; is num_inh_units = 0, then the weights projecting from
    # a single neuron can be a mixture of excitatory or inhibitory
    if par['exc_inh_prop'] < 1:
        par['EI'] = True
    else:
        par['EI']  = False

    par['num_exc_units'] = int(np.round(par['n_hidden']*par['exc_inh_prop']))
    par['num_inh_units'] = par['n_hidden'] - par['num_exc_units']
    par['EI_list'] = np.ones(par['n_hidden'], dtype=np.float32)
    par['EI_list'][::par['n_hidden']//par['num_inh_units']] = -1.

    par['EI_matrix'] = np.diag(par['EI_list'])

    # Membrane time constant of RNN neurons
    par['alpha_neuron'] = np.float32(par['dt'])/par['membrane_time_constant']
    # The standard deviation of the Gaussian noise added to each RNN neuron
    # at each time step
    par['noise_rnn'] = np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd']
    par['noise_in'] = np.sqrt(2/par['alpha_neuron'])*par['noise_in_sd'] # since term will be multiplied by par['alpha_neuron']


    #print('noise ',par['noise_rnn'], par['noise_in'])

    # General event profile info
    #par['name_of_stimulus'], par['date_stimulus_created'], par['author_of_stimulus_profile'] = get_profile(par['profile_path'])
    # List of events that occur for the network
    #par['events'] = get_events(par['profile_path'])
    # The time step in seconds
    par['dt_sec'] = par['dt']/1000
    # Length of each trial in ms
    if par['trial_type'] == 'dualDMS':
        par['trial_length'] = par['dead_time']+par['fix_time']+par['sample_time']+2*par['delay_time']+2*par['test_time']
    elif par['trial_type'] in ['multistim', 'twelvestim']:
        par['trial_length'] = par['multistim_trial_length']
    elif par['trial_type'] in ['limDMS']:
        par['trial_length'] = par['limDMS_trial_length']
    else:
        par['trial_length'] = par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time']+par['test_time']
    # Length of each trial in time steps
    par['num_time_steps'] = par['trial_length']//par['dt']


    ####################################################################
    ### Setting up assorted intial weights, biases, and other values ###
    ####################################################################

    par['h_init'] = 0.1*np.ones((par['n_hidden'], par['batch_train_size']), dtype=np.float32)

    par['input_to_hidden_dims'] = [par['n_hidden'], par['n_dendrites'], par['n_input']]
    par['hidden_to_hidden_dims'] = [par['n_hidden'], par['n_dendrites'], par['n_hidden']]
    par['hidden_to_output_dims'] = [par['n_output'], par['n_hidden']]

    # Initialize input weights
    par['w_in0'] = initialize(par['input_to_hidden_dims'], par['connection_prob'])
    par['w_in_mask'] = np.ones(par['input_to_hidden_dims'], dtype = np.float32)

    # Initialize starting recurrent weights
    # If excitatory/inhibitory neurons desired, initializes with random matrix with
    #   zeroes on the diagonal
    # If not, initializes with a diagonal matrix
    if par['EI']:
        par['w_rnn0'] = initialize(par['hidden_to_hidden_dims'], par['connection_prob'])

        for i in range(par['n_hidden']):
            par['w_rnn0'][i,:,i] = 0
        par['w_rnn_mask'] = np.ones((par['hidden_to_hidden_dims']), dtype=np.float32) - np.eye(par['n_hidden'])[:,np.newaxis,:]
        #par['w_rnn0'][:,:,par['num_exc_units']:] *= par['exc_inh_prop']/(1-par['exc_inh_prop'])
    else:
        par['w_rnn0'] = np.concatenate([np.float32(0.5*np.eye(par['n_hidden']))[:,np.newaxis,:]]*par['n_dendrites'], axis=1)
        par['w_rnn_mask'] = np.ones((par['hidden_to_hidden_dims']), dtype=np.float32)

    par['b_rnn0'] = np.zeros((par['n_hidden'], 1), dtype=np.float32)



    # Effective synaptic weights are stronger when no short-term synaptic plasticity
    # is used, so the strength of the recurrent weights is reduced to compensate
    if par['synapse_config'] == None:
        par['w_rnn0'] = par['w_rnn0']/(spectral_radius(par['w_rnn0']))


    # Initialize output weights and biases
    par['w_out0'] = initialize(par['hidden_to_output_dims'], par['connection_prob'])
    par['b_out0'] = np.zeros((par['n_output'], 1), dtype=np.float32)
    par['w_out_mask'] = np.ones(par['hidden_to_output_dims'], dtype=np.float32)

    if par['EI']:
        par['ind_inh'] = np.where(par['EI_list'] == -1)[0]
        par['w_out0'][:, par['ind_inh']] = 0
        par['w_out_mask'][:, par['ind_inh']] = 0

    # Defining sublayers for the hidden layer
    n_per_sub = par['n_hidden']//par['num_sublayers']
    sublayers = []
    sublayer_sizes = []
    for i in range(par['num_sublayers']):
        if i == par['num_sublayers'] - 1:
            app = par['n_hidden']%par['num_sublayers']
        else:
            app = 0
        sublayers.append(range(i*n_per_sub,(i+1)*n_per_sub+app))
        sublayer_sizes.append(n_per_sub+app)

    # Determine physical sublayer positions
    input_pos = np.zeros([par['n_input'], 3])
    hidden_pos = np.zeros([par['n_hidden'], 3])
    output_pos = np.zeros([par['n_output'], 3])

    # Build layer geometry
    input_pos[:,0:2] = square_locs(par['n_input'], par['neuron_dx'], par['neuron_dy']).T
    input_pos[:,2] = 0

    for i, (s, l) in enumerate(zip(sublayers, sublayer_sizes)):
        hidden_pos[s,0:2] = square_locs(l, par['neuron_dx'], par['neuron_dy']).T
        hidden_pos[s,2] = (i+1)*par['neuron_dz']

    output_pos[:,0:2] = square_locs(par['n_output'], par['neuron_dx'], par['neuron_dy']).T
    output_pos[:,2] = np.max(hidden_pos[:,2]) + par['neuron_dz']

    # Apply physical positions to relative positional matrix
    par['w_in_pos'] = np.zeros(par['input_to_hidden_dims'])
    for i,j in product(range(par['n_input']), range(par['n_hidden'])):
        par['w_in_pos'][j,:,i] = np.sqrt(np.sum(np.square(input_pos[i,:] - hidden_pos[j,:])))

    par['w_rnn_pos'] = np.zeros(par['hidden_to_hidden_dims'])
    for i,j in product(range(par['n_hidden']), range(par['n_hidden'])):
        par['w_rnn_pos'][j,:,i] = np.sqrt(np.sum(np.square(hidden_pos[i,:] - hidden_pos[j,:])))

    par['w_out_pos'] = np.zeros(par['hidden_to_output_dims'])
    for i,j in product(range(par['n_hidden']), range(par['n_output'])):
        par['w_out_pos'][j,i] = np.sqrt(np.sum(np.square(hidden_pos[i,:] - output_pos[j,:])))

    # Specify connections to sublayers
    for i in range(1, par['num_sublayers']):
        par['w_in0'][sublayers[i],:,:] = 0
        par['w_in_mask'][sublayers[i],:,:] = 0

    # Only allow connections between adjacent sublayers
    for i,j in product(range(par['num_sublayers']), range(par['num_sublayers'])):
        if np.abs(i-j) > 1:
            for k,m in product(sublayers[i], sublayers[j]):
                par['w_rnn0'][k,:,m] = 0
                par['w_rnn_mask'][k,:,m] = 0

    # Specify connections from sublayers
    for i in range(par['num_sublayers'] - 1):
        par['w_out0'][:, sublayers[i]] = 0
        par['w_out_mask'][:, sublayers[i]] = 0

    # KLUDGE!
    # Specifying that fixation tuned neurons can only connect to first 8 RNN units
    par['fix_connections_in'] = np.ones_like(par['w_in0'])
    par['fix_connections_in'][:,:,:par['num_motion_tuned']] = 0
    par['fix_connections_rnn'] = np.ones_like(par['w_rnn0'])
    par['fix_connections_rnn'][:,:,8:] = 0
    par['fix_connections_out'] = np.ones_like(par['w_out0'])
    par['fix_connections_out'][:,8:] = 0
    for i in range(par['num_motion_tuned'], par['num_motion_tuned']+par['num_fix_tuned']):
        for j in range(8, par['n_hidden']):
            par['w_in0'][j,:,i] = 0
            par['w_in_mask'][j,:,i] = 0
            par['fix_connections_in'][j,:,i] = 0

    """
    Setting up synaptic parameters
    0 = static
    1 = facilitating
    2 = depressing
    """
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


def initialize(dims, connection_prob):
    w = np.random.gamma(shape=0.25, scale=1.0, size=dims)
    #w = np.random.uniform(low=0, high=0.5, size=dims)
    w *= (np.random.rand(*dims) < connection_prob)
    return np.float32(w)


def spectral_radius(A):
    if A.ndim == 3:
        return np.max(abs(np.linalg.eigvals(np.sum(A, axis=1))))
    else:
        return np.max(abs(np.linalg.eigvals(np.sum(A))))


def square_locs(num_locs, d1, d2):

    locs_per_side = np.int32(np.sqrt(num_locs))
    while locs_per_side**2 < num_locs:
        locs_per_side += 1

    x_set = np.repeat(d1*np.arange(locs_per_side)[:,np.newaxis], locs_per_side, axis=1).flatten()
    y_set = np.repeat(d2*np.arange(locs_per_side)[np.newaxis,:], locs_per_side, axis=0).flatten()
    locs  = np.stack([x_set, y_set])[:,:num_locs]

    locs[0,:] -= np.max(locs[0,:])/2
    locs[1,:] -= np.max(locs[1,:])/2

    return locs


update_trial_params()
update_dependencies()

print("--> Parameters successfully loaded.\n")

import numpy as np
import tensorflow as tf
import os

print("--> Loading parameters...")

"""
Independent parameters
"""

par = {
    # Setup parameters
    'save_dir'              : './savedir_sweep/',
    'debug_model'           : False,
    'load_previous_model'   : False,
    'analyze_model'         : True,

    # Network configuration
    'synapse_config'        : 'std_stf', # Full is 'std_stf'
    'exc_inh_prop'          : 0.8,       # Literature 0.8, for EI off 1
    'var_delay'             : False,

    # Network shape
    'num_motion_tuned'      : 36,
    'num_fix_tuned'         : 0,
    'num_rule_tuned'        : 12,
    'n_hidden'              : 100,
    'n_output'              : 3,

    # Timings and rates
    'dt'                    : 10,
    'learning_rate'         : 2e-2,
    'membrane_time_constant': 100,
    'connection_prob'       : 1,         # Usually 1


    # Variance values
    'clip_max_grad_val'     : 1,
    'input_mean'            : 0.0,
    'noise_in_sd'           : 0.1,
    'noise_rnn_sd'          : 0.5,

    # Tuning function data
    'num_motion_dirs'       : 8,
    'tuning_height'         : 4,        # magnitutde scaling factor for von Mises
    'kappa'                 : 2,        # concentration scaling factor for von Mises

    # Cost parameters
    'spike_cost'            : 2e-2,
    'weight_cost'           : 0.,

    # Synaptic plasticity specs
    'tau_fast'              : 200,
    'tau_slow'              : 1500,
    'U_stf'                 : 0.15,
    'U_std'                 : 0.45,

    # Training specs
    'batch_train_size'      : 1024,
    'num_iterations'        : 2000,
    'iters_between_outputs' : 100,

    # Task specs
    'trial_type'            : 'DMS', # allowable types: DMS, DMRS45, DMRS90, DMRS180, DMC, DMS+DMRS, ABBA, ABCA, dualDMS
    'rotation_match'        : 0,  # angular difference between matching sample and test
    'dead_time'             : 250,
    'fix_time'              : 500,
    'sample_time'           : 500,
    'delay_time'            : 1000,
    'test_time'             : 500,
    'variable_delay_max'    : 300,
    'mask_duration'         : 50,  # duration of traing mask after test onset
    'catch_trial_pct'       : 0.0,
    'num_receptive_fields'  : 1,
    'num_rules'             : 1, # this will be two for the DMS+DMRS task

    # Save paths
    'save_fn'               : 'model_results.pkl',

    # Analysis
    'svm_normalize'         : True,
    'decoding_reps'         : 100,
    'simulation_reps'       : 100,
    'decode_test'           : False,
    'decode_rule'           : False,
    'decode_sample_vs_test' : False,
    'suppress_analysis'     : False,
    'analyze_tuning'        : True,
    'decode_stability'      : False,

}


"""
Dependent parameters
"""

def update_parameters(updates):
    """
    Takes a list of strings and values for updating parameters in the parameter dictionary
    Example: updates = [(key, val), (key, val)]
    """
    print('Updating parameters...')
    for key, val in updates.items():
        par[key] = val
        #print('Updating ', key)

    update_trial_params()
    update_dependencies()

def update_trial_params():

    """
    Update all the trial parameters given trial_type
    """

    par['num_rules'] = 1
    par['num_rule_tuned'] = 0
    par['ABBA_delay' ] = 0
    par['rule_onset_time'] = [par['dead_time']]
    par['rule_offset_time'] = [par['dead_time']]

    if par['trial_type'] == 'DMS' or par['trial_type'] == 'DMC':
        par['rotation_match'] = 0

    elif par['trial_type'] == 'DMRS45':
        par['rotation_match'] = 45

    elif par['trial_type'] == 'DMRS90':
        par['rotation_match'] = 90

    elif par['trial_type'] == 'DMRS90ccw':
        par['rotation_match'] = -90

    elif  par['trial_type'] == 'DMRS180':
        par['rotation_match'] = 180

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
        par['num_motion_tuned'] = 36
        par['noise_in_sd']  = 0.1
        par['noise_rnn_sd'] = 0.5
        par['num_iterations'] = 5000

        par['dualDMS_single_test'] = False
        par['rule_onset_time'] = []
        par['rule_offset_time'] = []
        par['rule_onset_time'].append(par['dead_time'] + par['fix_time'] + par['sample_time'] + par['delay_time']/2)
        par['rule_offset_time'].append(par['dead_time'] + par['fix_time'] + par['sample_time'] + par['delay_time'] + par['test_time'])
        par['rule_onset_time'].append(par['dead_time'] + par['fix_time'] + par['sample_time'] + 3*par['delay_time']/2 + par['test_time'])
        par['rule_offset_time'].append(par['dead_time'] + par['fix_time'] + par['sample_time'] + 2*par['delay_time'] + 2*par['test_time'])


    elif par['trial_type'] == 'ABBA' or par['trial_type'] == 'ABCA':
        par['catch_trial_pct'] = 0
        par['match_test_prob'] = 0.5
        par['max_num_tests'] = 3
        par['sample_time'] = 400
        par['delay_time'] = 2400
        par['ABBA_delay'] = par['delay_time']//par['max_num_tests']//2
        par['repeat_pct'] = 0
        par['analyze_test'] = False
        if par['trial_type'] == 'ABBA':
            par['repeat_pct'] = 0.5

    elif par['trial_type'] == 'DMS+DMRS' or par['trial_type'] == 'DMS+DMRS_early_cue':

        par['num_rules'] = 2
        par['num_rule_tuned'] = 12
        if par['trial_type'] == 'DMS+DMRS':
            par['rotation_match'] = [0, 90]
            par['rule_onset_time'] = [par['dead_time']+par['fix_time']+par['sample_time'] + 500]
            par['rule_offset_time'] = [par['dead_time']+par['fix_time']+par['sample_time'] + 750]
        else:
            par['rotation_match'] = [0, 45]
            par['rule_onset_time'] = [par['dead_time']]
            par['rule_offset_time'] = [par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time']]

    elif par['trial_type'] == 'DMS+DMC':
        par['num_rules'] = 2
        par['num_rule_tuned'] = 12
        par['rotation_match'] = [0, 0]
        #par['rule_onset_time'] = par['dead_time']+par['fix_time']+par['sample_time'] + 500
        #par['rule_offset_time'] = par['dead_time']+par['fix_time']+par['sample_time'] + par['delay_time'] + par['test_time']
        par['rule_onset_time'] = [par['dead_time']]
        par['rule_offset_time'] = [par['dead_time']+par['fix_time']+par['sample_time'] + par['delay_time'] + par['test_time']]

    elif par['trial_type'] == 'DMS+DMRS+DMC':
        par['num_rules'] = 3
        par['num_rule_tuned'] = 18
        par['rotation_match'] = [0, 90, 0]
        par['rule_onset_time'] = [par['dead_time']]
        par['rule_offset_time'] = [par['dead_time']+par['fix_time']+par['sample_time'] + par['delay_time'] + par['test_time']]

    elif par['trial_type'] == 'location_DMS':
        par['num_receptive_fields'] = 3
        par['rotation_match'] = 0
        par['num_motion_tuned'] = 54

    elif par['trial_type'] == 'distractor':
        # this task will not use the create_tuning_functions in stimulus.py
        # instead, it will used a simplified neural input
        par['n_output'] = par['num_motion_dirs'] + 1
        par['sample_time'] = 300
        par['delay_time'] = 2300
        par['test_time'] = 500
        par['num_fix_tuned'] = 4
        par['simulation_reps'] = 0
        par['analyze_tuning'] = False
        par['num_receptive_fields'] = 1



    else:
        print(par['trial_type'], ' not a recognized trial type')
        quit()


def update_dependencies():
    """
    Updates all parameter dependencies
    """

    # Number of input neurons
    par['n_input'] = par['num_motion_tuned'] + par['num_fix_tuned'] + par['num_rule_tuned']
    # General network shape
    par['shape'] = (par['n_input'], par['n_hidden'], par['n_output'])
    # The time step in seconds
    par['dt_sec'] = par['dt']/1000
    # Length of each trial in ms
    if par['trial_type'] == 'dualDMS' and not par['dualDMS_single_test']:
        par['trial_length'] = par['dead_time']+par['fix_time']+par['sample_time']+2*par['delay_time']+2*par['test_time']
    else:
        par['trial_length'] = par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time']+par['test_time']
    # Length of each trial in time steps
    par['num_time_steps'] = par['trial_length']//par['dt']


    par['dead_time_rng'] = range(par['dead_time']//par['dt'])
    par['sample_time_rng'] = range((par['dead_time']+par['fix_time'])//par['dt'], (par['dead_time']+par['fix_time']+par['sample_time'])//par['dt'])
    par['rule_time_rng'] = [range(int(par['rule_onset_time'][n]/par['dt']), int(par['rule_offset_time'][n]/par['dt'])) for n in range(len(par['rule_onset_time']))]


    # Possible rules based on rule type values
    #par['possible_rules'] = [par['num_receptive_fields'], par['num_categorizations']]

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
    par['EI_list'][-par['num_inh_units']:] = -1.

    par['drop_mask'] = np.ones((par['n_hidden'],par['n_hidden']), dtype=np.float32)
    ind_inh = np.where(par['EI_list']==-1)[0]
    par['drop_mask'][:, ind_inh] = 0.
    par['drop_mask'][ind_inh, :] = 0.

    par['EI_matrix'] = np.diag(par['EI_list'])

    # Membrane time constant of RNN neurons
    par['alpha_neuron'] = np.float32(par['dt'])/par['membrane_time_constant']
    # The standard deviation of the Gaussian noise added to each RNN neuron
    # at each time step
    par['noise_rnn'] = np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd']
    par['noise_in'] = np.sqrt(2/par['alpha_neuron'])*par['noise_in_sd'] # since term will be multiplied by par['alpha_neuron']





    ####################################################################
    ### Setting up assorted intial weights, biases, and other values ###
    ####################################################################

    par['h_init'] = 0.1*np.ones((par['n_hidden'], par['batch_train_size']), dtype=np.float32)

    par['input_to_hidden_dims'] = [par['n_hidden'], par['n_input']]
    par['hidden_to_hidden_dims'] = [par['n_hidden'], par['n_hidden']]


    # Initialize input weights
    par['w_in0'] = initialize([par['n_hidden'], par['n_input']], par['connection_prob'])

    # Initialize starting recurrent weights
    # If excitatory/inhibitory neurons desired, initializes with random matrix with
    #   zeroes on the diagonal
    # If not, initializes with a diagonal matrix
    if par['EI']:
        par['w_rnn0'] = initialize([par['n_hidden'], par['n_hidden']], par['connection_prob'])

        for i in range(par['n_hidden']):
            par['w_rnn0'][i,i] = 0
        par['w_rnn_mask'] = np.ones((par['n_hidden'], par['n_hidden']), dtype=np.float32) - np.eye(par['n_hidden'])
    else:
        par['w_rnn0'] = 0.54*np.eye(par['n_hidden'])
        par['w_rnn_mask'] = np.ones((par['n_hidden'], par['n_hidden']), dtype=np.float32)

    par['b_rnn0'] = np.zeros((par['n_hidden'], 1), dtype=np.float32)

    # Effective synaptic weights are stronger when no short-term synaptic plasticity
    # is used, so the strength of the recurrent weights is reduced to compensate

    if par['synapse_config'] == None:
        par['w_rnn0'] = par['w_rnn0']/(spectral_radius(par['w_rnn0']))


    # Initialize output weights and biases
    par['w_out0'] =initialize([par['n_output'], par['n_hidden']], par['connection_prob'])
    par['b_out0'] = np.zeros((par['n_output'], 1), dtype=np.float32)
    par['w_out_mask'] = np.ones((par['n_output'], par['n_hidden']), dtype=np.float32)

    if par['EI']:
        par['ind_inh'] = np.where(par['EI_list'] == -1)[0]
        par['w_out0'][:, par['ind_inh']] = 0
        par['w_out_mask'][:, par['ind_inh']] = 0

    par['w_in_mask'] = np.ones_like(par['w_in0'])
    """
    par['w_out0'][:, 1:par['n_hidden']:2] = 0
    par['w_out_mask'][:, 1:par['n_hidden']:2] = 0
    par['w_in0'][1:par['n_hidden']:2, :] = 0
    par['w_in_mask'][1:par['n_hidden']:2, :] = 0
    """


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
    #w = np.random.uniform(0,0.25, size=dims)
    w *= (np.random.rand(*dims) < connection_prob)
    return np.float32(w)


def spectral_radius(A):

    return np.max(abs(np.linalg.eigvals(A)))

update_trial_params()
update_dependencies()

print("--> Parameters successfully loaded.\n")

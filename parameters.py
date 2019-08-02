import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

print("--> Loading parameters...")

"""
Independent parameters
"""

par = {
    # Setup parameters
    'save_dir'              : './savedir/',
    'save_fn'               : 'model_results.pkl',

    # Network configuration
    'synapse_config'        : 'None', # full is half facilitating, half depressing. See line 295 for all options
    'exc_inh_prop'          : 0.8,    # excitatory/inhibitory ratio, set to 1 so that units are neither exc or inh
    'balance_EI'            : True,
    'connection_prob'       : 1.,

    # Network shape
    'num_motion_tuned'      : 24,
    'num_fix_tuned'         : 0,
    'num_rule_tuned'        : 0,
    'n_hidden'              : 100,
    'n_output'              : 3,

    # Timings and rates
    'dt'                    : 10,
    'learning_rate'         : 2e-2,
    'membrane_time_constant': 100,

    # Input and noise
    'input_mean'            : 0.0,
    'noise_in_sd'           : 0.1,
    'noise_rnn_sd'          : 0.5,

    # Tuning function data
    'num_motion_dirs'       : 8,
    'tuning_height'         : 4,        # magnitutde scaling factor for von Mises
    'kappa'                 : 2,        # concentration scaling factor for von Mises

    # Loss parameters
    'spike_regularization'  : 'L2', # 'L1' or 'L2'
    'spike_cost'            : 2e-2,
    'weight_cost'           : 0.,
    'clip_max_grad_val'     : 0.1,

    # Synaptic plasticity specs
    'tau_fast'              : 200,
    'tau_slow'              : 1500,
    'U_stf'                 : 0.15,
    'U_std'                 : 0.45,

    # Training specs
    'batch_size'            : 1024,
    'num_iterations'        : 2000,
    'iters_between_outputs' : 100,

    # Task specs
    'trial_type'            : 'DMS', # allowable types: DMS, DMRS45, DMRS90, DMRS180, DMC, DMS+DMRS, ABBA, ABCA, dualDMS
    'rotation_match'        : 0,  # angular difference between matching sample and test
    'dead_time'             : 0,
    'fix_time'              : 500,
    'sample_time'           : 500,
    'delay_time'            : 1000,
    'test_time'             : 500,
    'variable_delay_max'    : 300,
    'mask_duration'         : 50,  # duration of traing mask after test onset
    'catch_trial_pct'       : 0.0,
    'num_receptive_fields'  : 1,
    'num_rules'             : 1, # this will be two for the DMS+DMRS task
    'test_cost_multiplier'  : 1.,
    'rule_cue_multiplier'   : 1.,
    'var_delay'             : False,
}



def update_parameters(updates):
    """ Takes a list of strings and values for updating parameters in the parameter dictionary
        Example: updates = [(key, val), (key, val)] """

    print('Updating parameters...')
    for key, val in updates.items():
        par[key] = val
        #print('Updating ', key)

    update_trial_params()
    update_dependencies()

def update_trial_params():
    """ Update all the trial parameters given trial_type """

    par['num_rules'] = 1
    par['num_receptive_fields'] = 1
    #par['num_rule_tuned'] = 0
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
        par['num_rule_tuned'] = 6
        par['sample_time'] = 500
        par['test_time'] = 500
        par['delay_time'] = 1000
        par['analyze_rule'] = True
        par['num_motion_tuned'] = 24*2
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
        par['ABBA_delay'] = 400
        par['delay_time'] = 6*par['ABBA_delay']
        par['repeat_pct'] = 0
        par['analyze_test'] = False
        if par['trial_type'] == 'ABBA':
            par['repeat_pct'] = 0.5

    elif 'DMS+DMRS' in par['trial_type']:

        par['num_rules'] = 2
        par['num_rule_tuned'] = 6
        if par['trial_type'] == 'DMS+DMRS':
            par['rotation_match'] = [0, 90]
            par['rule_onset_time'] = [par['dead_time']+par['fix_time']+par['sample_time'] + 500]
            par['rule_offset_time'] = [par['dead_time']+par['fix_time']+par['sample_time'] + 750]
        elif par['trial_type'] == 'DMS+DMRS_full_cue':
            par['rotation_match'] = [0, 90]
            par['rule_onset_time'] = [par['dead_time']]
            par['rule_offset_time'] = [par['dead_time']+par['fix_time']+par['sample_time']\
                +par['delay_time']+par['test_time']]
        else:
            par['rotation_match'] = [0, 90]
            par['rule_onset_time'] = [par['dead_time']]
            par['rule_offset_time'] = [par['dead_time']+par['fix_time']]

    elif par['trial_type'] == 'DMS+DMC':
        par['num_rules'] = 2
        par['num_rule_tuned'] = 12
        par['rotation_match'] = [0, 0]
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
        par['num_motion_tuned'] = 24*3

    else:
        print(par['trial_type'], ' not a recognized trial type')
        quit()

    if par['trial_type'] == 'dualDMS':
        par['trial_length'] = par['dead_time']+par['fix_time']+par['sample_time']+2*par['delay_time']+2*par['test_time']
    else:
        par['trial_length'] = par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time']+par['test_time']
    # Length of each trial in time steps
    par['num_time_steps'] = par['trial_length']//par['dt']


def update_dependencies():
    """ Updates all parameter dependencies """

    # Number of input neurons
    par['n_input'] = par['num_motion_tuned'] + par['num_fix_tuned'] + par['num_rule_tuned']
    # General network shape
    par['shape'] = (par['n_input'], par['n_hidden'], par['n_output'])
    # The time step in seconds
    par['dt_sec'] = par['dt']/1000
    # Length of each trial in ms

    par['dead_time_rng'] = range(par['dead_time']//par['dt'])
    par['sample_time_rng'] = range((par['dead_time']+par['fix_time'])//par['dt'], (par['dead_time']+par['fix_time']+par['sample_time'])//par['dt'])
    par['rule_time_rng'] = [range(int(par['rule_onset_time'][n]/par['dt']), int(par['rule_offset_time'][n]/par['dt'])) for n in range(len(par['rule_onset_time']))]

    # If exc_inh_prop is < 1, then neurons can be either excitatory or
    # inihibitory; if exc_inh_prop = 1, then the weights projecting from
    # a single neuron can be a mixture of excitatory or inhibitory
    if par['exc_inh_prop'] < 1:
        par['EI'] = True
    else:
        par['EI']  = False

    par['num_exc_units'] = int(np.round(par['n_hidden']*par['exc_inh_prop']))
    par['num_inh_units'] = par['n_hidden'] - par['num_exc_units']

    par['EI_list'] = np.ones(par['n_hidden'], dtype=np.float32)
    par['EI_list'][-par['num_inh_units']:] = -1.

    par['ind_inh'] = np.where(par['EI_list']==-1)[0]

    par['EI_matrix'] = np.diag(par['EI_list'])

    # Membrane time constant of RNN neurons
    par['alpha_neuron'] = np.float32(par['dt'])/par['membrane_time_constant']
    # The standard deviation of the Gaussian noise added to each RNN neuron
    # at each time step
    par['noise_rnn'] = np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd']
    par['noise_in'] = np.sqrt(2/par['alpha_neuron'])*par['noise_in_sd'] # since term will be multiplied by par['alpha_neuron']

    # initial neural activity
    par['h0'] = 0.1*np.ones((1, par['n_hidden']), dtype=np.float32)
    #par['h0'] = 0.1*np.ones((par['batch_size'], par['n_hidden']), dtype=np.float32)

    # initial input weights
    par['w_in0'] = initialize([par['n_input'], par['n_hidden']], par['connection_prob']/par['num_receptive_fields'], shape=0.2, scale=1.)

    # Initialize starting recurrent weights
    # If excitatory/inhibitory neurons desired, initializes with random matrix with
    #   zeroes on the diagonal
    # If not, initializes with a diagonal matrix
    if par['EI']:
        par['w_rnn0'] = initialize([par['n_hidden'], par['n_hidden']], par['connection_prob'])
        if par['balance_EI']:
            # increase the weights to and from inh units to balce excitation and inhibition
            par['w_rnn0'][:, par['ind_inh']] = initialize([par['n_hidden'], par['num_inh_units']], par['connection_prob'], shape=0.2, scale=1.)
            par['w_rnn0'][par['ind_inh'], :] = initialize([par['num_inh_units'], par['n_hidden']], par['connection_prob'], shape=0.2, scale=1.)

    else:
        par['w_rnn0'] = 0.54*np.eye(par['n_hidden'])


    # initial recurrent biases
    par['b_rnn0'] = np.zeros((1, par['n_hidden']), dtype=np.float32)

    # Effective synaptic weights are stronger when no short-term synaptic plasticity
    # is used, so the strength of the recurrent weights is reduced to compensate
    if par['synapse_config'] is None:
        par['w_rnn0'] = par['w_rnn0']/3.

    # initial output weights and biases
    par['w_out0'] = initialize([par['n_hidden'], par['n_output']], par['connection_prob'])
    par['b_out0'] = np.zeros((1, par['n_output']), dtype=np.float32)

    # for EI networks, masks will prevent self-connections, and inh to output connections
    par['w_rnn_mask'] = np.ones_like(par['w_rnn0'])
    par['w_out_mask'] = np.ones_like(par['w_out0'])
    par['w_in_mask'] = np.ones_like(par['w_in0'])
    if par['EI']:
        par['w_rnn_mask'] = np.ones((par['n_hidden'], par['n_hidden']), dtype=np.float32) - np.eye(par['n_hidden'])
        par['w_out_mask'][par['ind_inh'], :] = 0

    par['w_rnn0'] *= par['w_rnn_mask']
    par['w_out0'] *= par['w_out_mask']

    # for the location_DMS task, inputs from the 3 receptive fields project onto non-overlapping
    # units in the RNN. This tries to replicates what likely happesn in areas MST, which are retinotopic
    if par['trial_type'] == 'location_DMS':
        par['w_in_mask'] *= 0
        target_ind = [range(0, par['n_hidden'],3), range(1, par['n_hidden'],3), range(2, par['n_hidden'],3)]
        for n in range(par['n_input']):
            u = int(n//(par['n_input']/3))
            par['w_in_mask'][n, target_ind[u]] = 1
        par['w_in0'] = par['w_in0']*par['w_in_mask']

    synaptic_configurations = {
        'full'              : ['facilitating' if i%2==0 else 'depressing' for i in range(par['n_hidden'])],
        'fac'               : ['facilitating' for i in range(par['n_hidden'])],
        'dep'               : ['depressing' for i in range(par['n_hidden'])],
        'exc_fac'           : ['facilitating' if par['EI_list'][i]==1 else 'static' for i in range(par['n_hidden'])],
        'exc_dep'           : ['depressing' if par['EI_list'][i]==1 else 'static' for i in range(par['n_hidden'])],
        'inh_fac'           : ['facilitating' if par['EI_list'][i]==-1 else 'static' for i in range(par['n_hidden'])],
        'inh_dep'           : ['depressing' if par['EI_list'][i]==-1 else 'static' for i in range(par['n_hidden'])],
        'exc_dep_inh_fac'   : ['depressing' if par['EI_list'][i]==1 else 'facilitating' for i in range(par['n_hidden'])]
    }

    # initialize synaptic values
    par['alpha_stf'] = np.ones((1, par['n_hidden']), dtype=np.float32)
    par['alpha_std'] = np.ones((1, par['n_hidden']), dtype=np.float32)
    par['U'] = np.ones((1, par['n_hidden']), dtype=np.float32)
    par['syn_x_init'] = np.ones((par['batch_size'], par['n_hidden']), dtype=np.float32)
    par['syn_u_init'] = 0.3 * np.ones((par['batch_size'], par['n_hidden']), dtype=np.float32)
    par['dynamic_synapse'] = np.zeros((1, par['n_hidden']), dtype=np.float32)

    for i in range(par['n_hidden']):
        if not par['synapse_config'] in synaptic_configurations.keys():
            par['dynamic_synapse'][0,i] = 0
        elif synaptic_configurations[par['synapse_config']][i] == 'facilitating':
            par['alpha_stf'][0,i] = par['dt']/par['tau_slow']
            par['alpha_std'][0,i] = par['dt']/par['tau_fast']
            par['U'][0,i] = 0.15
            par['syn_u_init'][:, i] = par['U'][0,i]
            par['dynamic_synapse'][0,i] = 1

        elif synaptic_configurations[par['synapse_config']][i] == 'depressing':
            par['alpha_stf'][0,i] = par['dt']/par['tau_fast']
            par['alpha_std'][0,i] = par['dt']/par['tau_slow']
            par['U'][0,i] = 0.45
            par['syn_u_init'][:, i] = par['U'][0,i]
            par['dynamic_synapse'][0,i] = 1



def initialize(dims, connection_prob, shape=0.1, scale=1.0 ):
    w = np.random.gamma(shape, scale, size=dims)
    w *= (np.random.rand(*dims) < connection_prob)

    return np.float32(w)


update_trial_params()
update_dependencies()

print("--> Parameters successfully loaded.\n")

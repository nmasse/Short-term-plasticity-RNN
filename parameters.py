### Parameters for RNN research
### Authors: Nicolas Masse, Gregory Grant, Catherine Lee, Varun Iyer
### Date:    3 August, 2017

import numpy as np
import tensorflow as tf
import itertools

print("\n--> Loading parameters...")

##############################
### Independent parameters ###
##############################

global par

par = {
    # Setup parameters
    'stimulus_type'         : 'att',    # multitask, att, mnist
    'save_dir'              : './savedir/',
    'debug_model'           : False,
    'load_previous_model'   : False,
    'processor_affinity'    : [0,1],   # Default is [], for no preference

    # Network configuration
    'synapse_config'    : None,      # Full is 'std_stf'
    'exc_inh_prop'      : 0.6,       # Literature 0.8, for EI off 1
    'var_delay'         : False,
    'use_dendrites'     : True,
    'use_stim_soma'     : True,
    'df_num'            : '0008',    # Designates which dendrite function to use

    # hidden layer shape
    'n_hidden'          : 100,
    'den_per_unit'      : 8,

    # Timings and rates
    'dt'                        : 25,
    'learning_rate'             : 5e-3,
    'membrane_time_constant'    : 100,
    'dendrite_time_constant'    : 300,
    'connection_prob_in'        : 0.75,
    'connection_prob_rnn'       : 0.75,
    'connection_prob_out'       : 0.75,
    'mask_connectivity'         : 1.0,

    # Variance values
    'clip_max_grad_val' : 0.25,
    'input_mean'        : 0,
    'input_sd'          : 0.1/10,
    'internal_sd'       : 0.25,
    'xi'                : 0.001,     # Value used in Ganguli paper is 1e-3

    # Tuning function data
    'tuning_height'     : 1,        # magnitutde scaling factor for von Mises
    'kappa'             : 1,        # concentration scaling factor for von Mises
    'catch_rate'        : 0.2,      # catch rate when using variable delay
    'match_rate'        : 0.5,      # number of matching tests in certain tasks

    # Probe specs
    'probe_trial_pct'   : 0,
    'probe_time'        : 25,

    # Cost parameters/function
    'spike_cost'        : 5e-3,
    'dend_cost'         : 0,
    'wiring_cost'       : 5e-7,
    'motif_cost'        : 0,
    'omega_cost'        : 0,
    'loss_function'     : 'cross_entropy',    # cross_entropy or MSE

    # Synaptic plasticity specs
    'tau_fast'          : 200,
    'tau_slow'          : 1500,
    'U_stf'             : 0.15,
    'U_std'             : 0.45,

    # Performance thresholds
    'stop_perf_th'      : 1,
    'stop_error_th'     : 1,

    # Training specs
    'batch_train_size'  : 50,
    'num_train_batches' : 100,
    'num_test_batches'  : 20,
    'num_iterations'    : 10,
    'iterations_between_outputs'    : 1,        # Ususally 500
    'switch_rule_iteration'         : 4,

    # Save paths and other info
    'save_notes'        : '',
    'save_fn'           : 'model_data.json',
    'use_checkpoints'   : False,
    'ckpt_save_fn'      : 'model_' + str(0) + '.ckpt',
    'ckpt_load_fn'      : 'model_' + str(0) + '.ckpt',

    # Analysis
    'time_pts'          : [850, 1200, 1850, 2000],
    'num_category_rule' : 1,
    'roc_vars'          : None,
    'anova_vars'        : None, #['state_hist', 'dend_hist', 'dend_exc_hist', 'dend_inh_hist'],
    'tuning_vars'       : None, #['state_hist', 'dend_hist', 'dend_exc_hist', 'dend_inh_hist'],
    'modul_vars'        : True,

    # Meta weights
    'num_mw'            : 10,
    'use_metaweights'   : False,
    'alpha_mw'          : 1,
    'cascade_strength'  : 0.01,

    # Disinhibition circuit
    'use_connectivity'  : True,
    'use_disinhibition' : False
}

##############################
### Task parameter profile ###
##############################

def set_task_profile():
    """
    Depending on the stimulus type, sets the network
    to the appropriate configuration
    """

    if par['stimulus_type'] == 'mnist':
        par['profile_path'] = ['./profiles/mnist.txt']
        par['rules_map'] = None

        par['num_RFs']               = 1
        par['allowed_fields']        = [0]

        par['num_rules']             = 3
        par['allowed_rules']         = [0]          # 0 is regular, 1 is horizontal flilp, 2 is vertical flip

        par['permutation_id']        = 0

        par['num_stim_tuned']        = 784 * par['num_RFs']
        par['num_fix_tuned']         = 0
        par['num_rule_tuned']        = 0 * par['num_rules']
        par['num_spatial_cue_tuned'] = 0 * par['num_RFs']
        par['n_output']              = 11

        par['num_samples']           = 60000    # Number of available samples
        par['num_unique_samples']    = 10

    elif par['stimulus_type'] == 'att':
        par['profile_path'] = ['./profiles/attention.txt']
        par['rules_map'] = None

        par['num_RFs']               = 1            # contributes to 'possible_rules'
        par['allowed_fields']        = [0]     # can hold 0 through num_fields - 1

        par['num_rules']             = 2             # the number of possible judgements
        par['allowed_rules']         = [0,1]           # Can be 0 OR 1 OR 0, 1

        par['permutation_id']        = 0

        par['num_stim_tuned']        = 36 * par['num_RFs']
        par['num_fix_tuned']         = 0
        par['num_rule_tuned']        = 12 * par['num_rules']
        par['num_spatial_cue_tuned'] = 0 * par['num_RFs']
        par['n_output']              = 3

        par['num_samples']           = 12     # Number of motion directions
        par['num_unique_samples']    = 12

    elif par['stimulus_type'] == 'multitask':
        par['profile_path'] = ['./profiles/attention_multitask.txt', './profiles/motion_multitask.txt']
        par['rules_map'] = [0] * 2 + [1] * 5             # Maps rules to profiles

        par['num_RFs']               = 1          # contributes to 'possible_rules'
        par['allowed_fields']        = [0]     # can hold 0 through num_fields - 1

        par['num_rules']             = 7             # Possible tasks and rules in those tasks
        par['allowed_rules']         = [2,3]  # Can be 0 OR 1 OR 0, 1, etc.

        par['permutation_id']        = 0

        par['num_stim_tuned']        = 36 * par['num_RFs']
        par['num_fix_tuned']         = 0
        par['num_rule_tuned']        = 0 * par['num_rules']
        par['num_spatial_cue_tuned'] = 0 * par['num_RFs']
        par['n_output']              = 3

        par['num_samples']           = 12     # Number of motion directions
        par['num_unique_samples']    = 12

    else:
        print("ERROR: Bad stimulus type.")
        quit()

############################
### Dependent parameters ###
############################

def initialize(dims, connection_prob):
    n = np.float32(np.random.gamma(shape=0.25, scale=1.0, size=dims))
    n *= (np.random.rand(*dims) < connection_prob)
    return n


def get_events(paths):
    """
    Gets event information from the profile file
    """

    content = [0] * len(paths)
    for i in range(len(paths)):
        with open(paths[i]) as event_list:
            raw_content = event_list.read().split("\n")

        text = list(filter(None, raw_content))

        for line in range(len(text)):
            if text[line][0] == "0":
                content[i] = text[line:]

        for line in range(len(content[i])):
            content[i][line] = content[i][line].split("\t")
            content[i][line][0] = int(content[i][line][0])

    return content


def update_parameters(updates):
    """
    Takes a list of strings and values for updating parameters in the parameter dictionary
    Example: updates = [(key, val), (key, val)]
    """
    for (key, val) in updates.items():
        par[key] = val

    update_dependencies()

def define_neuron_type():

    # neuron_type = 0 (stimulus), 1 (top-down), 2 (EXC), 3 (PV), 4 (VIP),
    # 5 (SOM), 6 (output)
    par['num_exc_units'] = int(par['exc_inh_prop']*par['n_hidden'])
    par['num_inh_units'] = int(par['n_hidden'] - par['num_exc_units'])
    if par['num_inh_units']>0:
        par['EI'] = True
    else:
        par['EI'] = False

    # Number of input neurons
    par['n_input'] = par['num_stim_tuned'] + par['num_fix_tuned'] + par['num_rule_tuned'] + par['num_spatial_cue_tuned']

    n = par['num_inh_units']//4
    neuron_type = np.zeros((par['n_input'] + par['n_hidden'] + par['n_output']), dtype = np.uint8)
    neuron_type[par['num_stim_tuned']:par['n_input']] = 1 # top-down
    neuron_type[par['n_input']:par['n_input']+par['num_exc_units']] = 2 # EXC
    neuron_type[par['n_input']+par['num_exc_units']:par['n_input']+par['num_exc_units']+2*n] = 3 # PV
    neuron_type[par['n_input']+par['num_exc_units']+2*n:par['n_input']+par['num_exc_units']+3*n] = 4 # VIP
    neuron_type[par['n_input']+par['num_exc_units']+3*n:] = 5 # SOM
    neuron_type[-par['n_output']:] = 6 # output

    print(neuron_type, len(neuron_type))

    return neuron_type

def create_weights_masks_biases():

    # Define neccessary dimensions
    par['input_to_hidden_dend_dims'] = [par['n_hidden'], par['den_per_unit'], par['num_stim_tuned']]
    par['input_to_hidden_soma_dims'] = [par['n_hidden'], par['num_stim_tuned']]

    par['td_to_hidden_dend_dims']     = [par['n_hidden'], par['den_per_unit'], par['n_input'] - par['num_stim_tuned']]
    par['td_to_hidden_soma_dims']     = [par['n_hidden'], par['n_input'] - par['num_stim_tuned']]

    par['hidden_to_hidden_dend_dims'] = [par['n_hidden'], par['den_per_unit'], par['n_hidden']]
    par['hidden_to_hidden_soma_dims'] = [par['n_hidden'], par['n_hidden']]

    par['hidden_to_output_dims'] = [par['n_output'], par['n_hidden']]

    # Mask values
    par['w_rnn_dend_mask'] = np.zeros((par['hidden_to_hidden_dend_dims']), dtype=np.float32)
    par['w_rnn_soma_mask'] = np.zeros((par['hidden_to_hidden_soma_dims']), dtype=np.float32)

    par['w_stim_dend_mask'] = np.zeros((par['input_to_hidden_dend_dims']), dtype=np.float32)
    par['w_stim_soma_mask'] = np.zeros((par['input_to_hidden_soma_dims']), dtype=np.float32)

    par['w_td_dend_mask'] = np.zeros((par['td_to_hidden_dend_dims']), dtype=np.float32)
    par['w_td_soma_mask'] = np.zeros((par['td_to_hidden_soma_dims']), dtype=np.float32)

    par['w_out_mask'] = np.zeros((par['hidden_to_output_dims']), dtype=np.float32)

    # Intial weights
    par['w_rnn_dend0'] = np.zeros((par['hidden_to_hidden_dend_dims']), dtype=np.float32)
    par['w_rnn_soma0'] = np.zeros((par['hidden_to_hidden_soma_dims']), dtype=np.float32)

    par['w_stim_dend0'] = np.zeros((par['input_to_hidden_dend_dims']), dtype=np.float32)
    par['w_stim_soma0'] = np.zeros((par['input_to_hidden_soma_dims']), dtype=np.float32)

    par['w_td_dend0'] = np.zeros((par['td_to_hidden_dend_dims']), dtype=np.float32)
    par['w_td_soma0'] = np.zeros((par['td_to_hidden_soma_dims']), dtype=np.float32)

    par['w_out0'] = np.zeros((par['hidden_to_output_dims']), dtype=np.float32)

    # Initial biases
    par['b_rnn0'] = np.zeros((par['n_hidden'],1), dtype=np.float32)
    par['b_out0'] = np.zeros((par['n_output'],1), dtype=np.float32)
    par['b_rnn_mask'] = np.zeros((par['n_hidden'],1), dtype=np.float32)
    par['b_out_mask'] = np.zeros((par['n_output'],1), dtype=np.float32)

def define_connectivity():

    # define the connection probability between different neuron types
    # dim 0=0 refers to connections to soma, dim 0=1 refers to connections to dendrite
    par['connection_prob'] =  np.zeros((2,7,7))

    # define how to inialize weights
    # 0 = use gamma distribution, make weight variable
    # 1 = set weight to 1; make weight fixed
    par['connection_type'] =  np.zeros((2,7,7))

    # define how to inialize biases
    # 0 = intialize to 0; make bias variable
    # 1 = set bias to 1; make bias fixed
    par['baseline_type'] =  np.zeros((7))

    if par['use_connectivity']:

        par['connection_prob'][0, 1, 4] = 0.05 # td to VIP
        par['connection_prob'][0, 4, 5] = 0.15 # VIP to SOM
        par['connection_prob'][0, 2:4, 2:4] = 0 # EXC,PV to EXC, PV
        par['connection_prob'][0, 2, 6] = 0.25 # EXC to output
        par['connection_prob'][1, 0, 2:4] = 0.25 # stim to dendrites of EXC, PV
        par['connection_prob'][1, 5, 2:4] = 1 # SOM to dendrites of EXC, PV

        par['connection_type'][0, 2:4, 2:4] = 0 # EXC,PV to EXC, PV
        par['connection_type'][1, 0, 2:4] = 0 # stim to dendrites of EXC, PV

        par['connection_type'][0, 1, 4] = 1
        par['connection_type'][0, 4, 5] = 1
        par['connection_type'][1, 5, 2:4] = 1


        par['baseline_type'][5] = 1
    else:
        # stim to hidden
        par['connection_prob'][0,0,2:6] = 1
        par['connection_prob'][1,0,2:6] = 1
        # top-down to hidden
        par['connection_prob'][0,1,2:6] = 1
        par['connection_prob'][1,1,2:6] = 1
        # hidden to hidden
        par['connection_prob'][0,2:6,2:6] = 1
        par['connection_prob'][1,2:6,2:6] = 1
        # EXC to output
        par['connection_prob'][0,2,6] = 1

def fill_masks_weights_biases(neuron_type):

    for source in range(par['n_input'] + par['n_hidden']):
        source_type = neuron_type[source]

        if source_type >=2 and source_type<6:
            s = source - par['n_input']
            if par['baseline_type'][source_type] == 1:
                par['b_rnn0'][s] = 1
                par['b_rnn_mask'][s] = 0
            else:
                par['b_rnn0'][s] = 0
                par['b_rnn_mask'][s] = 1

        elif source_type == 6:
            s = source - par['n_input'] - par['n_hidden']
            if par['baseline_type'][source_type] == 1:
                par['b_out0'][s] = 1
                par['b_out_mask'][s] = 0
            else:
                par['b_out_mask'][s] = 1

        for target in range(par['n_input'] + par['n_hidden'] + par['n_output']):

            target_type = neuron_type[target]

            if source_type == 0 and target_type>=2 and target_type<6:
                s = np.array(source)
                t = target - par['n_input']
                weight_name = 'w_stim'
            elif source_type == 1 and target_type>=2 and target_type<6:
                s = source - par['num_stim_tuned']
                t = target - par['n_input']
                weight_name = 'w_td'
            elif source_type>=2 and source_type<6 and target_type>=2 and target_type<6:
                s = source - par['n_input']
                t = target - par['n_input']
                weight_name = 'w_rnn'
            elif source_type == 2 and target_type==6:
                s = source - par['n_input']
                t = target - par['n_input'] - par['n_hidden']
                weight_name = 'w_out'
            else:
                continue

            if weight_name == 'w_out':
                if par['connection_type'][0,source_type,target_type] == 0:
                    par[weight_name + '_mask'][t, s] = 1
                if np.random.rand() < par['connection_prob'][0,source_type,target_type]:
                    if par['connection_type'][0,source_type,target_type] == 1:
                        par[weight_name + '0'][t, s] = 1
                    else:
                        par[weight_name + '0'][t, s] = np.random.gamma(0.25, 1)

            elif weight_name == 'w_td':
                t1 = t - par['num_exc_units']-par['num_inh_units']//2
                if ((s<12 and t1<5) or (s>=12 and t1>=5)) and (source_type==1 and target_type==4):
                    par[weight_name + '_soma0'][t, s] = 1



            else:
                if par['connection_type'][0,source_type,target_type] == 0 and par['connection_prob'][0,source_type,target_type]>0:
                    par[weight_name + '_soma_mask'][t, s] = 1
                if par['connection_type'][1,source_type,target_type] == 0 and par['connection_prob'][1,source_type,target_type]>0:
                    par[weight_name + '_dend_mask'][t, :, s] = 1

                if np.random.rand() < par['connection_prob'][0,source_type,target_type]:
                    if par['connection_type'][0,source_type,target_type] == 1:
                        par[weight_name + '_soma0'][t, s] = 1
                    else:
                        par[weight_name + '_soma0'][t, s] = np.random.gamma(0.25, 1)/10

                ind = np.where(np.random.rand(par['den_per_unit']) < par['connection_prob'][1,source_type,target_type])[0]
                if not ind == [] and par['connection_type'][1,source_type,target_type] == 1:
                    par[weight_name + '_dend0'][t, ind, s] = 1
                elif not ind == [] and par['connection_type'][1,source_type,target_type] == 0:
                    par[weight_name + '_dend0'][t, ind, s] = np.random.gamma(0.25, 1, size = len(ind))/2


            if weight_name == 'w_stimXXX':
                print('w_sim sum ', ' ', s, ' ', t, ' ',np.sum(par[weight_name + '_dend0']))
    if par['EI']:
        for n in range(par['n_hidden']):
            par['w_rnn_soma0'][n,n] = 0
            par['w_rnn_dend0'][n,:,n] = 0
            par['w_rnn_soma_mask'][n,n] = 0
            par['w_rnn_dend_mask'][n,:,n] = 0


def generate_masks():
    """
    # for input neurons, stimulus tuned will have ID=0, rule/fix tuned will have ID =1
    input_type = np.zeros((par['n_input']), dtype = np.uint8)
    input_type[par['num_stim_tuned']:] = 1

    # for hidden nuerons, EXC will have ID=2, PV will have ID=3, VIP will have ID=4, SOM will have ID=5
    # 50% of INH neurons will be PV, 25% for VIP, SOM
    n = par['num_inh_units']//4
    hidden_type = 2*np.ones((par['n_hidden']), dtype = np.uint8)
    hidden_type[par['num_exc_units']:par['num_exc_units']+2*n] = 3
    hidden_type[par['num_exc_units']+2*n:par['num_exc_units']+3*n] = 4
    hidden_type[par['num_exc_units']+3*n::] = 5

    if par['use_connectivity']:
        # dim 0=0 refers to connections to soma, dim 0=1 refers to connections to dendrite
        connection_rate = np.zeros((2,6,6))
        connection_rate[0, 1, 4] = 0.4 # td to VIP
        connection_rate[0, 4, 5] = 0.9 # VIP to SOM
        connection_rate[0, 2:4, 2:4] = 1 # EXC,PV to EXC, PV
        connection_rate[1, 0, 2:4] = 1 # stim to dendrites of EXC, PV
        connection_rate[1, 5, 2:4] = 0.8 # SOM to dendrites of EXC, PV

        connection_type = np.zeros((2,6,6)) # 0 if intialize to 1, 1 if initialize with gamma
        connection_type[0, 2:4, 2:4] = 1 # EXC,PV to EXC, PV
        connection_type[1, 0, 2:4] = 1 # stim to dendrites of EXC, PV

        baseline = np.zeros((7))
        baseline[5] = 1
    else:
        connection_rate = np.ones((2,6,6))
        connection_type = np.ones((2,6,6)) # 0 if intialize to 1, 1 if initialize with gamma
        baseline = np.zeros((7))

    """
    if par['use_connectivity']:
        connectivity = np.zeros((2,6,6))
        connectivity[0, 1, 4] = 0.4 # td to VIP
        connectivity[0, 4, 5] = 0.9 # VIP to SOM
        connectivity[0, 2:4, 2:4] = 1 # EXC,PV to EXC, PV
        connectivity[1, 0, 2:4] = 1 # stim to dendrites of EXC, PV
        connectivity[1, 5, 2:4] = 1 # SOM to dendrites of EXC, PV

    else:
        connectivity = np.ones((2,6,6))

    # to soma
    # connectivity[0, 0, 2:4] = 1 # stim tuned will project to EXC,PV
    # connectivity[0, 2, 2:4] = 1 # EXC will project to EXC,PV
    # connectivity[0, 3, 2:4] = 1 # PV will project to EXC,PV
    # connectivity[0, 4, 5] = 1 # VIP will project to SOM

    # to dendrites
    # connectivity[1, 0, 2:4] = 1 # stim tuned will project to EXC,PV
    # connectivity[1, 1, 2:5] = 1 # rule tuned will project to EXC,PV,VIP
    # connectivity[1, 2, 2:4] = 1 # EXC will project to EXC,PV
    # connectivity[1, 4, 5] = 1 # VIP will project to SOM
    # connectivity[1, 5, 2:4] = 1 # SOM will project to EXC,PV
    """

    # input to hidden
    for source in range(par['num_stim_tuned']):
        for target in range(par['n_hidden']):
            par['w_stim_dend_mask'][target,:,source] = np.random.rand() < connectivity[1,input_type[source],hidden_type[target]]
            par['w_stim_soma_mask'][target,source] = np.random.rand() < connectivity[0,input_type[source],hidden_type[target]]

    # td to hidden
    for source in range(par['n_input'] - par['num_stim_tuned']):
        for target in range(par['n_hidden']):
            par['w_td_dend_mask'][target,:,source] = np.random.rand() < connectivity[1,input_type[par['num_stim_tuned'] + source],hidden_type[target]]:
            par['w_td_soma_mask'][target,source] = np.random.rand() < connectivity[0,input_type[par['num_stim_tuned'] + source],hidden_type[target]]

    # hidden to hidden
    for source in range(par['n_hidden']):
        for target in range(par['n_hidden']):
            par['w_rnn_dend_mask'][target,:,source] = np.random.rand() < connectivity[1,hidden_type[source],hidden_type[target]]
            par['w_rnn_soma_mask'][target,source] = np.random.rand() < connectivity[0,hidden_type[source],hidden_type[target]]
    """
    pass

def reduce_connectivity():
    # Clamp masks randomly
    r_nh    = range(par['n_hidden'])
    r_dpu   = range(par['den_per_unit'])
    r_nst   = range(par['num_stim_tuned'])
    r_ninst = range(par['n_input']-par['num_stim_tuned'])

    for i, j, k in itertools.product(r_nh, r_dpu, r_nst):
        if np.random.rand() > par['mask_connectivity']:
            par['w_stim_dend_mask'][i,j,k] = 0
        if np.random.rand() > par['mask_connectivity'] and j == 0:
            par['w_stim_soma_mask'][i,k] = 0

    for i, j, k in itertools.product(r_nh, r_dpu, r_ninst):
        if np.random.rand() > par['mask_connectivity']:
            par['w_td_dend_mask'][i,j,k] = 0
        if np.random.rand() > par['mask_connectivity'] and j == 0:
            par['w_td_soma_mask'][i,k] = 0

    for i, j, k in itertools.product(r_nh, r_dpu, r_nh):
        if np.random.rand() > par['mask_connectivity']:
            par['w_rnn_dend_mask'][i,j,k] = 0
        if np.random.rand() > par['mask_connectivity'] and j == 0:
            par['w_rnn_soma_mask'][i,k] = 0


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


def set_template(trial_rules, trial_locations):

    # Set up dendrite inhibition template for df0009
    template = 1000*np.ones((par['n_hidden'], par['den_per_unit'], par['batch_train_size']), dtype=np.float32)
    for n in range(par['batch_train_size']):
        r = trial_rules[n,0]*par['num_RFs'] + trial_locations[n,0]
        template[:,r,n] = 0

    return template


def apply_prob(mat, y0, y1, x0, x1, p):
    dims = np.shape(mat)
    if len(dims) == 3:
        for i, d, j in itertools.product(range(x0, x1), range(0, par['den_per_unit']), range(y0,y1)):
            num = np.random.randint(2)
            if num > p:
                mat[i,d,j] = 0
    else:
        for i, j in itertools.product(range(x0, x1), range(y0,y1)):
            num = np.random.randint(2)
            if num > p:
                mat[i,j] = 0
    return mat


def get_dend():
    n_0, n_1, n_2, n_3 = 24, 80, 80, 1
    p_vip, p_som = 0.275, 0.35
    beta = 1.0
    scale = 1.2*(n_1/40)+0.5

    num_iters = 1000
    dend = np.zeros((num_iters, 2))

    for i in range(num_iters):
        W_vip = np.ones((n_2, n_1))
        W_som = np.ones((n_3, n_2))
        W_vip = prune_connections(W_vip, p_vip)
        W_som = prune_connections(W_som, p_som)


        W_td = np.zeros((n_1, n_0))
        W_td2 = np.zeros((n_2, n_0))
        for k in range(n_0//2):
            for m in range(n_1//2):
                W_td[m,k] = 1
                W_td[m+(n_1//2),k+(n_0//2)] = 1
        for k in range(n_0//2):
            for m in range(n_2//2):
                W_td2[m,k] = 1
                W_td2[m+(n_2//2),k+(n_0//2)] = 1
        W_td = prune_connections(W_td, 0.2)
        W_td2 = prune_connections(W_td2, 0.5)


        for j in range(2):
            if j == 0:
                td = np.zeros((n_0,1))
                td[:n_0//2,0] = 2
            else:
                td = np.zeros((n_0,1))
                td[n_0//2:,0] = 2

            VIP = np.matmul(W_td, td)
            vip = np.matmul(W_vip, VIP)
            td2 = np.matmul(W_td2, td)
            SOM = np.maximum(0,beta - vip/scale + td2)
            dend[i,j] = np.matmul(W_som, SOM)<1

    print('Dendrite analysis...')
    print(np.mean(dend[:,0]), np.mean(dend[:,1]), np.mean(dend[:,0]*dend[:,1]), np.mean(dend[:,0])*np.mean(dend[:,1]))

def prune_connections(x, p):

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if np.random.rand() > p:
                x[i,j] = 0
    return x

def update_dependencies():
    """
    Updates all parameter dependencies
    """

    # Projected number of trials to take place
    par['projected_num_trials'] = par['batch_train_size']*par['num_train_batches']*par['num_iterations']

    # Number of input neurons
    par['n_input'] = par['num_stim_tuned'] + par['num_fix_tuned'] + par['num_rule_tuned'] + par['num_spatial_cue_tuned']

    # General network shape
    par['shape'] = (par['n_input'], par['n_hidden'], par['n_output'])

    # Possible rules based on rule type values
    par['possible_rules'] = [par['num_RFs'], par['num_rules']]
    # Number of rules - used in input tuning
    #par['num_rules'] = len(par['possible_rules'])

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
    if par['exc_inh_prop'] < 1:
        par['EI'] = True
    else:
        par['EI']  = False

    par['num_exc_units'] = int(np.round(par['n_hidden']*par['exc_inh_prop']))
    par['num_inh_units'] = par['n_hidden'] - par['num_exc_units']

    par['EI_list'] = np.ones(par['n_hidden'], dtype=np.float32)
    par['EI_list'][-par['num_inh_units']:] = -1.

    par['EI_matrix'] = np.diag(par['EI_list'])

    print('EI_mat', par['EI_matrix'])

    par['EI_list_d_exc'] = np.ones(par['n_hidden'], dtype=np.float32)
    par['EI_list_d_inh'] = np.ones(par['n_hidden'], dtype=np.float32)

    par['EI_list_d_exc'][-par['num_inh_units']:] = 0.
    par['EI_list_d_inh'][:par['num_exc_units']] = 0.

    par['EI_matrix_d_exc'] = np.diag(par['EI_list_d_exc'])
    par['EI_matrix_d_inh'] = np.diag(par['EI_list_d_inh'])

    # The previous exc_inh_prop addresses soma-to-soma interactions, but this
    # EI matrix addresses soma-to-dendrite interactions.  It follows the same
    # setup process, however.

    # [par['n_hidden'], par['den_per_unit'], par['n_hidden']]
    par['num_exc_dends'] = int(np.round(par['den_per_unit']))

    # Membrane time constant of RNN neurons
    par['alpha_neuron'] = par['dt']/par['membrane_time_constant']
    # The standard deviation of the Gaussian noise added to each RNN neuron
    # at each time step
    par['noise_sd'] = np.sqrt(2*par['alpha_neuron'])*par['internal_sd']

    # Dendrite time constant for dendritic branches
    par['alpha_dendrite'] = par['dt']/par['dendrite_time_constant']

    # Build seeded permutation list
    template = np.arange(par['num_stim_tuned']/par['num_RFs'])
    p = [[template]]
    np.random.seed(0)
    for n in range(1, 100):
        p.append([np.random.permutation(template)])
    np.random.seed(None)
    par['permutations'] = np.squeeze(p)

    # List of events that occur for the network
    par['events'] = get_events(par['profile_path'])
    # The time step in seconds
    par['dt_sec'] = par['dt']/1000
    # Length of each trial in ms
    par['trial_length'] = par['events'][0][-1][0]
    # Length of each trial in time steps
    par['num_time_steps'] = par['trial_length']//par['dt']


    ####################################################################
    ### Setting up assorted intial weights, biases, and other values ###
    ####################################################################

    par['h_init'] = 0.1*np.ones((par['n_hidden'], par['batch_train_size']), dtype=np.float32)
    par['d_init'] = 0.1*np.ones((par['n_hidden'], par['den_per_unit'], par['batch_train_size']), dtype=np.float32)

    """
    par['input_to_hidden_dend_dims'] = [par['n_hidden'], par['den_per_unit'], par['num_stim_tuned']]
    par['input_to_hidden_soma_dims'] = [par['n_hidden'], par['num_stim_tuned']]

    par['td_to_hidden_dend_dims']     = [par['n_hidden'], par['den_per_unit'], par['n_input'] - par['num_stim_tuned']]
    par['td_to_hidden_soma_dims']     = [par['n_hidden'], par['n_input'] - par['num_stim_tuned']]

    par['hidden_to_hidden_dend_dims'] = [par['n_hidden'], par['den_per_unit'], par['n_hidden']]
    par['hidden_to_hidden_soma_dims'] = [par['n_hidden'], par['n_hidden']]

    # Generate random masks
    generate_masks()

    #if par['mask_connectivity'] < 1:
        #reduce_connectivity()

    # Initialize input weights
    par['w_stim_dend0'] = initialize(par['input_to_hidden_dend_dims'], par['connection_prob_in'])
    par['w_stim_soma0'] = initialize(par['input_to_hidden_soma_dims'], par['connection_prob_in'])

    par['w_td_dend0'] = initialize(par['td_to_hidden_dend_dims'], par['connection_prob_in'])
    par['w_td_soma0'] = initialize(par['td_to_hidden_soma_dims'], par['connection_prob_in'])

    par['w_stim_dend0'] *= par['w_stim_dend_mask']
    par['w_stim_soma0'] *= par['w_stim_soma_mask']

    par['w_td_dend0'] *= par['w_td_dend_mask']
    par['w_td_soma0'] *= par['w_td_soma_mask']

    # Initialize starting recurrent weights
    # If excitatory/inhibitory neurons desired, initializes with random matrix with
    #   zeroes on the diagonal
    # If not, initializes with a diagonal matrix
    if par['EI']:
        par['w_rnn_dend0'] = initialize(par['hidden_to_hidden_dend_dims'], par['connection_prob_rnn'])
        par['w_rnn_soma0'] = initialize(par['hidden_to_hidden_soma_dims'], par['connection_prob_rnn'])

        # Remove connection based on the connection probability
        if par['use_disinhibition']:
            n = par['num_inh_units']//4
            vip = par['num_exc_units']+2*n
            par['w_td_soma0'] = apply_prob(par['w_td_soma0'], (par['n_input']-par['num_stim_tuned'])//2, (par['n_input']-par['num_stim_tuned']), vip, vip+n, 0.9)
            par['w_rnn_soma0'] = apply_prob(par['w_rnn_soma0'], vip, vip+n, vip+n, vip+2*n, 0.7)
            par['w_rnn_dend0'] = apply_prob(par['w_rnn_dend0'], vip+n, vip+2*n, 0, par['n_hidden'], 0.8)


        #par['w_rnn_dend_mask'] = np.ones((par['hidden_to_hidden_dend_dims']), dtype=np.float32)
        #par['w_rnn_soma_mask'] = np.ones((par['hidden_to_hidden_soma_dims']), dtype=np.float32) - np.eye(par['n_hidden'])

        for i in range(par['n_hidden']):
            par['w_rnn_soma0'][i,i] = 0
            #par['w_rnn_dend_mask'][i,:,i] = 0
            par['w_rnn_dend0'][i,:,i] = 0

    else:
        par['w_rnn_dend0'] = np.zeros(par['hidden_to_hidden_dend_dims'], dtype=np.float32)
        par['w_rnn_soma0'] = np.eye(par['hidden_to_hidden_soma_dims'], dtype=np.float32)

        for i in range(par['n_hidden']):
            par['w_rnn_dend0'][i,:,i] = 1

        #par['w_rnn_dend_mask'] = np.ones((par['hidden_to_hidden_dend_dims']), dtype=np.float32)
        #par['w_rnn_soma_mask'] = np.ones((par['hidden_to_hidden_soma_dims']), dtype=np.float32)

    par['w_rnn_dend0'] *= par['w_rnn_dend_mask']
    par['w_rnn_soma0'] *= par['w_rnn_soma_mask']

    # Initialize starting recurrent biases
    # Note that the second dimension in the bias initialization term can be either
    # 1 or self.params['batch_train_size'].
    set_to_one = True
    par['bias_dim'] = (1 if set_to_one else par['batch_train_size'])
    par['b_rnn0'] = np.zeros((par['n_hidden'], par['bias_dim']), dtype=np.float32)

    # Effective synaptic weights are stronger when no short-term synaptic plasticity
    # is used, so the strength of the recurrent weights is reduced to compensate

    if par['synapse_config'] == None:
        par['w_rnn_dend0'] /= (2*spectral_radius(par['w_rnn_dend0']))
        par['w_rnn_soma0'] /= (2*spectral_radius(par['w_rnn_soma0']))



    # Initialize output weights and biases
    par['w_out0'] = initialize([par['n_output'], par['n_hidden']], par['connection_prob_out'])

    par['b_out0'] = np.zeros((par['n_output'], 1), dtype=np.float32)
    par['w_out_mask'] = np.ones((par['n_output'], par['n_hidden']), dtype=np.float32)

    if par['EI']:
        par['ind_inh'] = np.where(par['EI_list'] == -1)[0]
        par['w_out0'][:, par['ind_inh']] = 0
        par['w_out_mask'][:, par['ind_inh']] = 0

    """
    # Describe which weights will be used in this model
    par['working_weights'] = []
    if par['use_dendrites']:
        par['working_weights'].append('W_stim_dend')
        par['working_weights'].append('W_td_dend')
        par['working_weights'].append('W_rnn_dend')
    if par['use_stim_soma']:
        par['working_weights'].append('W_stim_soma')
        par['working_weights'].append('W_td_soma')
    par['working_weights'].append('W_rnn_soma')
    par['working_weights'].append('W_out')

    # Establish the names and shapes of the Tensorflow graph placeholders
    par['general_placeholder_info'] = [('x_stim',               [par['num_stim_tuned'], par['num_time_steps'], par['batch_train_size']]),
                                       ('x_td',                 [par['n_input'] - par['num_stim_tuned'], par['num_time_steps'], par['batch_train_size']]),
                                       ('y',                    [par['n_output'], par['num_time_steps'], par['batch_train_size']]),
                                       ('mask',                 [par['num_time_steps'], par['batch_train_size']]),
                                       ('learning_rate',        [])]

    par['other_placeholder_info']   = [('dendrite_template',    [par['n_hidden'], par['den_per_unit'], par['batch_train_size']]),
                                       ('omega',                [])]

    par['weight_placeholder_info']  = [('W_stim_dend',          par['input_to_hidden_dend_dims']),
                                       ('W_td_dend',            par['td_to_hidden_dend_dims']),
                                       ('W_rnn_dend',           par['hidden_to_hidden_dend_dims']),
                                       ('W_stim_soma',          par['input_to_hidden_soma_dims']),
                                       ('W_td_soma',            par['td_to_hidden_soma_dims']),
                                       ('W_rnn_soma',           par['hidden_to_hidden_soma_dims']),
                                       ('W_out',                [par['n_output'], par['n_hidden']])]

    # Describe the mapping between working indices and placeholder indices
    par['weight_index_feed'] = []
    for w, p in itertools.product(range(len(par['working_weights'])), range(len(par['weight_placeholder_info']))):
        if par['working_weights'][w] == par['weight_placeholder_info'][p][0]:
            par['weight_index_feed'].append(p)

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


def set_disinhibitory_path():
    td = par['n_input'] - par['num_stim_tuned']
    n = par['num_inh_units']//4
    beta = 1.0
    # np.set_printoptions(threshold=np.nan)

    # set up td weight matrices
    # 1) td to VIP
    # 2) td to SOM
    par['w_td_soma0'][par['num_exc_units']+2*n:,:] = 0
    par['w_td_soma0'][(par['num_exc_units']+2*n):(par['num_exc_units']+int(2.5*n)),:(td//2)] = 1
    par['w_td_soma0'][(par['num_exc_units']+int(2.5*n)):(par['num_exc_units']+3*n),(td//2):] = 1
    par['w_td_soma0'][(par['num_exc_units']+3*n):(par['num_exc_units']+int(3.5*n)),:(td//2)] = 1
    par['w_td_soma0'][(par['num_exc_units']+int(3.5*n)):,(td//2):] = 1

    # prune connections for td to VIP & td to SOM based on their connection probabilities
    par['w_td_soma0'][(par['num_exc_units']+2*n):(par['num_exc_units']+3*n),:] = prune_connections(par['w_td_soma0'][(par['num_exc_units']+2*n):(par['num_exc_units']+3*n),:], 0.2)
    par['w_td_soma0'][(par['num_exc_units']+3*n):,:] = prune_connections(par['w_td_soma0'][(par['num_exc_units']+3*n):,:], 0.5)
    
    # connection from VIP to SOM
    par['w_rnn_soma0'][(par['num_exc_units']+3*n):,(par['num_exc_units']+2*n):(par['num_exc_units']+3*n)] = prune_connections(np.ones([n, n]), 0.275)
    
    # connection from SOM to EXC (?)
    par['w_rnn_dend0'][:par['num_exc_units'],:,(par['num_exc_units']+3*n):] = prune_connections(np.ones([par['num_exc_units'], par['den_per_unit'], n]), 1)

    # print(par['w_td_soma0'])
    # print(par['w_td_dend0'])

get_dend()
set_task_profile()
neuron_type = define_neuron_type()
create_weights_masks_biases()
define_connectivity()
fill_masks_weights_biases(neuron_type)
update_dependencies()
set_disinhibitory_path()
print("--> Parameters successfully loaded.\n")

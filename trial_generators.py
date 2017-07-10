"""
2017/07/10 Gregory Grant
"""

import numpy as np
import copy
from parameters import *


############################
### Adjustment functions ###
############################

def get_mnist():
    from mnist import MNIST
    mndata = MNIST('./resources/mnist/data/original')
    images, labels = mndata.load_training()

    return images, labels


def stimulus_permutation(m):
    m = np.atleast_2d(m)
    # Randomly permutes the inputs based on a set of seeded permutations
    output = np.zeros(np.shape(m))

    for n in range(np.shape(m)[0]):
        p_index = par['permutations'][par['permutation_id']][n]
        output[n] = m[int(p_index)]

    return output


#################################
### Trial generator functions ###
#################################

def trial_batch(N, stim_tuning, fix_tuning, rule_tuning, spatial_tuning, testing=False):

    # Pre-allocate inputs, outputs, rules, and samples
    fix_input       = np.zeros((N, par['n_input']), dtype=np.float32)
    sample_input    = np.zeros((N, par['n_input']), dtype=np.float32)

    fix_output      = np.zeros((N, par['n_output']), dtype=np.float32)
    fix_output[:,0] = 1
    sample_output   = np.zeros((N, par['n_output']), dtype=np.float32)

    location_index  = np.zeros((N, 1), dtype=np.float32)
    rule_index      = np.zeros((N, 1), dtype=np.float32)
    sample_index    = np.zeros((N, par['num_RFs']), dtype=np.float32)
    attended_sample_index = np.zeros((N, 1), dtype=np.float32)

    # Generate relevant indices (start = 0, end of spatial = n_input)
    eos = par['num_stim_tuned']             # End of sample neurons
    eof = eos + par['num_fix_tuned']            # End of fixation neurons
    eor = eof + par['num_rule_tuned']           # End of rule neurons

    neurons_per_field = par['num_stim_tuned']//par['num_RFs']
    if par['num_stim_tuned']/par['num_RFs'] != neurons_per_field:
        print("ERROR: Please use an multiple of RFs * neurons per RF.")
        quit()

    # Attention task helper:
    unit_circle = [0, par['num_samples']//4, par['num_samples']//2, \
                    3*par['num_samples']//4 , par['num_samples']]

    # MNIST task helper:
    if par['stimulus_type'] == 'mnist':
        from mnist import MNIST
        mndata = MNIST('./resources/mnist/data/original')
        images, labels = mndata.load_training()
        labels = np.array(labels)

    for n in range(N):
        # Generate fixation, spatial and rule cues
        fix         = 0
        location    = np.random.choice(par['allowed_fields'])
        rule        = np.random.choice(par['allowed_rules'])

        location_index[n, 0]    = location
        rule_index[n, 0]        = rule

        # Generate fixation period inputs from loop
        fix_input[n, eos:eof]   = fix_tuning[fix]           # Fixation neurons
        fix_input[n, eof:eor]   = rule_tuning[rule]         # Rule neurons
        fix_input[n, eor:]      = spatial_tuning[location]  # Spatial neurons

        # Decide on active fields and samples for those fields
        active_fields = np.sort(np.random.permutation(par['allowed_fields'])[0:par['num_active_fields']])
        sample_indices = np.random.randint(0, par['num_samples'], [par['num_RFs']])


        # Generate sample period inputs from the loop
        for f in active_fields:
            for i in range(f*neurons_per_field, (f+1)*neurons_per_field):
                sample_input[n, i] = stimulus_permutation(stim_tuning[sample_indices[f], i%neurons_per_field])

        # If MNIST task, match indices to translate to output
        if par['stimulus_type'] == 'mnist':
            for f in range(par['num_RFs']):
                sample_indices[f] = labels[sample_indices[f]]

        # Target desired stimulus
        target = sample_indices[location]
        sample_index[n,:] = sample_indices
        attended_sample_index[n] = target

        # Generate sample period outputs from the loop based on task-specific logic
        if par['stimulus_type'] == 'att':
            if rule == 0:
                if unit_circle[0] <= target < unit_circle[2]:
                    sample_output[n,1] = 1
                elif unit_circle[2] <= target < unit_circle[4]:
                    sample_output[n,2] = 1
                else:
                    print("Error in trial generator.")
                    quit()
            elif rule == 1:
                if unit_circle[1] <= target < unit_circle[3]:
                    sample_output[n, 1] = 1
                elif (unit_circle[3] <= target < unit_circle[4]) \
                or (unit_circle[0] <= target < unit_circle[1]):
                    sample_output[n, 2] = 1
                else:
                    print("Error in trial generator.")
                    quit()
        elif par['stimulus_type'] == 'mnist':
            sample_output[n, target] = 1
        else:
            print("ERROR: Bad stimulus type in trial generator.")

    inputs = {'fix' : fix_input,
              'stim': sample_input
             }

    outputs = {'fix' : fix_output,
               'stim': sample_output
              }

    trial_setup = {'inputs'                 : inputs,
                   'outputs'                : outputs,
                   'rule_index'             : rule_index,
                   'sample_index'           : sample_index,
                   'location_index'         : location_index,
                   'attended_sample_index'  : attended_sample_index
                  }

    return trial_setup

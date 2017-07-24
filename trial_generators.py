"""
2017/07/10 Gregory Grant
"""

import numpy as np
from parameters import *

###############################
### Task-specific functions ###
###############################

def stimulus_permutation(m):
    # Randomly permutes the inputs based on a set of seeded permutations
    output = np.zeros(np.shape(m))

    for n in range(np.shape(m)[0]):
        p_index = par['permutations'][par['permutation_id']][n]
        output[n] = m[int(p_index)]

    return output

def mnist_inversions(rule, sample_input_n, f, neurons_per_field):
    if rule == 1:
        sample_input_n[f*neurons_per_field:(f+1)*neurons_per_field] \
            = np.reshape(np.fliplr(np.reshape(sample_input_n[f*neurons_per_field:(f+1)*neurons_per_field], [28,28])), 784)
    elif rule == 2:
        sample_input_n[f*neurons_per_field:(f+1)*neurons_per_field] \
            = np.reshape(np.flipud(np.reshape(sample_input_n[f*neurons_per_field:(f+1)*neurons_per_field], [28,28])), 784)
    else:
        print("ERROR:  Bad MNIST rule.")
        quit()

    return sample_input_n


def handle_att(rule, target, sample_output_n):
    unit_circle = [0, par['num_samples']//4, par['num_samples']//2, \
                    3*par['num_samples']//4 , par['num_samples']]

    if rule == 0:
        if unit_circle[0] <= target < unit_circle[2]:
            sample_output_n[1] = 1
        elif unit_circle[2] <= target < unit_circle[4]:
            sample_output_n[2] = 1

    elif rule == 1:
        if unit_circle[1] <= target < unit_circle[3]:
            sample_output_n[1] = 1
        elif (unit_circle[3] <= target < unit_circle[4]) \
        or (unit_circle[0] <= target < unit_circle[1]):
            sample_output_n[2] = 1

    return sample_output_n


def handle_multitask_test(rule, location, target, f, active_fields, npf, stim_tuning, test_input_n):
    unit_circle = [0, par['num_samples']//4, par['num_samples']//2, \
                    3*par['num_samples']//4 , par['num_samples']]

    match = np.random.rand() < par['match_rate']

    for f in active_fields:
        if f == location:
            if rule in [2,3,4]:
                unit = (target + (rule-3)*unit_circle[1])%par['num_samples']
                if match:
                    index = unit
                else:
                    index = np.random.choice(np.setdiff1d(np.arange(par['num_samples']), unit))

            elif rule in [5, 6]:
                if rule == 5:
                    set1 = np.random.randint(unit_circle[0], unit_circle[2])
                    set2 = np.random.randint(unit_circle[2], unit_circle[4])
                else:
                    set1 = np.random.randint(unit_circle[1], unit_circle[3])
                    set2 = np.random.choice(np.concatenate([np.arange(unit_circle[3], unit_circle[4]), np.arange(unit_circle[0], unit_circle[1])]))

                if unit_circle[0 + (rule-5)] <= target < unit_circle[2 + (rule-5)]:
                    if match:
                        index = set1
                    else:
                        index = set2
                else:
                    if match:
                        index = set2
                    else:
                        index = set1

            test_input_n[f*npf:(f+1)*npf] = stimulus_permutation(stim_tuning[index])
        else:
            test_input_n[f*npf:(f+1)*npf] = stimulus_permutation(stim_tuning[np.random.randint(0, par['num_samples'])])

    return test_input_n, match


def handle_multitask(match, test_output_n):
    if match:
        test_output_n[1] = 1
    elif not match:
        test_output_n[2] = 1

    return test_output_n

#################################
### Trial generator functions ###
#################################

def trial_batch(N, stim_tuning, fix_tuning, rule_tuning, spatial_tuning, images, labels):

    # Pre-allocate inputs, outputs, rules, and samples
    fix_input       = np.zeros((N, par['n_input']), dtype=np.float32)
    sample_input    = np.zeros((N, par['n_input']), dtype=np.float32)
    test_input      = np.zeros((N, par['n_input']), dtype=np.float32)

    fix_output      = np.zeros((N, par['n_output']), dtype=np.float32)
    fix_output[:,0] = 1
    sample_output   = np.zeros((N, par['n_output']), dtype=np.float32)
    test_output     = np.zeros((N, par['n_output']), dtype=np.float32)

    location_index  = np.zeros((N, 1), dtype=np.uint8)
    rule_index      = np.zeros((N, 1), dtype=np.uint8)
    sample_index    = np.zeros((N, par['num_RFs']), dtype=np.uint8)
    attended_sample_index = np.zeros((N, 1), dtype=np.uint8)

    # Generate relevant indices (start = 0, end of spatial = n_input)
    eos = par['num_stim_tuned']                 # End of sample neurons
    eof = eos + par['num_fix_tuned']            # End of fixation neurons
    eor = eof + par['num_rule_tuned']           # End of rule neurons

    neurons_per_field = par['num_stim_tuned']//par['num_RFs']
    if par['num_stim_tuned']/par['num_RFs'] != neurons_per_field:
        print("ERROR: Please use an multiple of RFs * neurons per RF.")
        quit()

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

        # Generate sample period inputs from the loop, and flip matrices if the
        # rule dictates (currently MNIST only)
        for f in active_fields:
            sample_input[n, f*neurons_per_field:(f+1)*neurons_per_field] = \
                        stimulus_permutation(stim_tuning[sample_indices[f]])

            # Based on rules in the MNIST task, inverts the neural input in
            # each field vertically or horizontally.  Currently hard-coded to
            # a 784-long input array.
            if par['stimulus_type'] == 'mnist':
                if rule == 0:
                    pass
                else:
                    sample_input[n] = mnist_inversions(rule, sample_input[n], f, neurons_per_field)

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
            sample_output[n] = handle_att(rule, target, sample_output[n])
        elif par['stimulus_type'] == 'mnist':
            sample_output[n, target+1] = 1
        elif par['stimulus_type'] == 'multitask':
            if rule in [0, 1]:
                sample_output[n] = handle_att(rule, target, sample_output[n])
            elif rule in [2,3,4,5,6]:
                sample_output[n,0] = 1
        else:
            print("ERROR: Bad stimulus type in trial generator.")

        # Generate test input based on sample characteristics
        if par['stimulus_type'] == 'multitask':
            if rule in [0, 1]:
                pass
            elif rule in [2,3,4,5,6]:
                test_input[n], match = handle_multitask_test(rule, location, target, f, active_fields, neurons_per_field, stim_tuning, test_input[n])

        # Generate test period outputs from the loop based on task-specific logic
        if par['stimulus_type'] == 'multitask':
            if rule in [0, 1]:
                pass
            elif rule in [2,3,4,5,6]:
                test_output[n] = handle_multitask(match, test_output[n])

    # Assemble input and output dictionaries
    inputs = {'fix'   : fix_input,
              'sample': sample_input,
              'test'  : test_input
             }

    outputs = {'fix'   : fix_output,
               'sample': sample_output,
               'test'  : test_output
              }

    trial_setup = {'inputs'                 : inputs,
                   'outputs'                : outputs,
                   'rule_index'             : rule_index,
                   'sample_index'           : sample_index,
                   'location_index'         : location_index,
                   'attended_sample_index'  : attended_sample_index
                  }

    return trial_setup

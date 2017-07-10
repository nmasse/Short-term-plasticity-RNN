"""
2017/06/20 Gregory Grant
"""

import numpy as np
import copy
from parameters import *


#######################################
### Tuning and adjustment functions ###
#######################################

def tuning():
    """
    Generate tuning functions for the input neurons
    Neurons are selective for motion direction, fixation, or rules
    """

    def transform(d):
        # The tuning transformation, which transforms direction variances
        # into a distribution using a von Mises
        return par['tuning_height']*(np.exp(par['kappa']*d) \
                                     - np.exp(-par['kappa']))/np.exp(par['kappa'])

    # Essentially, creates matrices of the number of motion neurons
    # by the number of possible fields and states introduced to that neuron
    motion_tuning   = np.zeros((par['num_motion_tuned'], par['num_receptive_fields'], par['num_motion_dirs']))

    # Generates lists of preferred and possible motion directions
    pref_dirs   = np.float32(np.arange(0,2*np.pi,2*np.pi/(par['num_motion_tuned']//par['num_receptive_fields'])))
    stim_dirs   = np.float32(np.arange(0,2*np.pi,2*np.pi/par['num_motion_dirs']))

    for r in range(par['num_receptive_fields']):
        for n in range(r*par['num_motion_tuned']//par['num_receptive_fields'], (r+1)*par['num_motion_tuned']//par['num_receptive_fields']):
            for i in range(len(stim_dirs)):
                # Finds the difference between each motion diretion and preferred direction
                d = np.cos(stim_dirs[i] - pref_dirs[n%len(pref_dirs)])
                # Transforms the direction variances with a von Mises
                motion_tuning[n,r,i] = transform(d)

    # ------------------------------------------------------------------------ #

    # Dynamically generates a series of "rule matrices", then broadcasts
    # and concatenates them to create a rule_tuning array of size
    # [num_rule_tuned, max_num_cases_of_rule, num_rules]
    num_alloc = par['num_rule_tuned']//par['num_rules']
    rule_arrays = []
    largest = np.max(par['possible_rules'])
    for r in range(par['num_rules']):
        if num_alloc == par['possible_rules'][r]:
            rule_array = np.identity((num_alloc))
        else:
            # Simplify this to a num_alloc x num_alloc matrix, then plug in
            # the number of possible cases to generate the "rule matrix".
            # Assuming that should make this calculation MUCH easier, and much
            # more consistent.
            rule_array = np.zeros((num_alloc, par['possible_rules'][r]))
            #rule_array = np.zeros((num_alloc, num_alloc))
            for n in range(num_alloc):
                for c in range(par['possible_rules'][r]):
                    rule_array[0,0] = 1
                    if num_alloc >= par['possible_rules'][r]:
                        if (1/par['possible_rules'][r]) <= (c+1)/(n+1) < 1:
                            rule_array[n,c] = 1
                    if num_alloc <= par['possible_rules'][r]:
                        if (1/num_alloc) <= (n+1)/(c+1) < 1:
                            rule_array[n,c] = 1
        rule_arrays.append(rule_array)

    for i in range(len(rule_arrays)):
        #print(np.shape(rule_arrays[i]))
        if np.shape(rule_arrays[i])[1] == largest:
            #print("!")
            rule_arrays[i] = np.transpose(np.broadcast_to(rule_arrays[i], (par['num_rules'], np.shape(rule_arrays[i])[0], np.shape(rule_arrays[i])[1])))
        else:
            rule_arrays[i] = np.broadcast_to(rule_arrays[i], (largest, np.shape(rule_arrays[i])[0], np.shape(rule_arrays[i])[1]))
        #print(np.shape(rule_arrays[i]))

    #rule_tuning = np.transpose(np.concatenate([*rule_arrays], 1), (1,0,2))
    rule_tuning = np.transpose(np.concatenate([*rule_arrays], 1))
    #print(rule_tuning)

    fix_tuning = [0]

    return motion_tuning, fix_tuning, rule_tuning


def get_mnist():
    from mnist import MNIST
    mndata = MNIST('./resources/mnist/data/original')
    images, labels = mndata.load_training()

    return images, labels


def mnist_permutation(m):
    # Randomly permutes the inputs based on a set of seeds in parameters
    output = np.zeros(np.shape(m))

    for n in range(np.shape(m)[0]):
        p_index = par['permutations'][par['permutation_id']][n]
        output[n] = m[p_index]

    return output


#################################
### Trial generator functions ###
#################################

def attention(N):
    """
    Generates a set of random trials for attention tests based on the batch
    size, and returns the stimuli, tests, intended outputs, and the default
    inputs and outputs for the set.

    The output arrays are [batch_train_size x neurons] large, where [neurons]
    corresponds to the number of input or output neurons, depending on the array.
    """

    ### Tuning inputs
    stim_tuning, fix_tuning, rule_tuning = tuning()

    ### Pre-allocate inputs and outputs
    # TODO: do we need default_input, default_output?
    default_input   = np.zeros((N, par['n_input']), dtype=np.float32)
    rule_input      = np.zeros((N, par['n_input']), dtype=np.float32)
    sample_input    = np.zeros((N, par['n_input']), dtype=np.float32)

    default_output  = np.zeros((N, par['n_output']), dtype=np.float32)
    rule_output     = np.zeros((N, par['n_output']), dtype=np.float32)
    rule_output[:,0] = 1
    sample_output   = np.zeros((N, par['n_output']), dtype=np.float32)

    rule_index = par['num_motion_tuned']
    neurons_per_field = par['num_motion_tuned']//par['num_receptive_fields']

    unit_circle = [0, par['num_motion_dirs']//4, par['num_motion_dirs']//2, \
                    3*par['num_motion_dirs']//4 , par['num_motion_dirs']]

    location_rules = []
    category_rules = []
    #field_directions = []
    sample_index = np.zeros((N, par['num_receptive_fields']), dtype=np.uint8)
    target_directions = []

    for n in range(N):

        # Generate rule cues
        location_rule = np.random.choice(par['allowed_fields'])
        category_rule = np.random.choice(par['allowed_categories'])

        location_rules.append(location_rule)
        category_rules.append(category_rule)

        # Output rule_input[n] from loop
        rule_input[n, rule_index:rule_index+par['num_rule_tuned']] = rule_tuning[category_rule,:,location_rule]

        active_fields = np.sort(np.random.permutation(par['allowed_fields'])[0:par['num_active_fields']])
        sample_index[n,:] = np.random.randint(0, par['num_motion_dirs'], [par['num_receptive_fields']])

        # Output sample_input[n] from loop
        for f in active_fields:
            for i in range(f*neurons_per_field, (f+1)*neurons_per_field):
                sample_input[n,i] = stim_tuning[i, f, sample_index[n, f]]

        desired_dir = sample_index[n, location_rule]

        target_directions.append(desired_dir)

        # Output sample_output[n] from loop
        # Arbitrarily, up and left are judgement 1, and down and right are judgement 2
        # Up and down
        if category_rule == 0:
            if unit_circle[0] <= desired_dir < unit_circle[2]:
                sample_output[n][1] = 1
            elif unit_circle[2] <= desired_dir < unit_circle[4]:
                sample_output[n][2] = 1
        elif category_rule == 1:
            if unit_circle[1] <= desired_dir < unit_circle[3]:
                sample_output[n][1] = 1
            elif (unit_circle[3] <= desired_dir < unit_circle[4]) or (unit_circle[0] <= desired_dir < unit_circle[1]):
                sample_output[n][2] = 1

    inputs = {'none'    : default_input,
              'fix'     : rule_input,
              'stim'    : sample_input
              }

    outputs = {'none'    : default_output,
               'fix'     : rule_output,
               'stim'    : sample_output
               }

    trial_setup = {'default_input'  :   default_input,
                   'inputs'         :   inputs,
                   'default_output' :   default_output,
                   'outputs'        :   outputs,
                   'rule_index'     :   np.array([location_rules, category_rules]),
                   'sample_index'   :   sample_index,
                   'test_index'     :   []
                  }

    return trial_setup


def mnist(N):
    """
    Generates a set of random trials for timed MNIST tests based on the batch
    size, and returns the stimuli, tests, intended outputs, and the default
    intended output for the set

    The output arrays are [batch_train_size x neurons] large, where [neurons]
    corresponds to the number of input or output neurons, depending on the array.
    """

    ### Tuning inputs
    images, labels = get_mnist()

    ### Pre-allocate inputs and outputs
    # TODO: Do we need default_input, default_output?
    default_input   = np.zeros((N, par['n_input']), dtype=np.float32)
    #default_input = None
    fix_input       = np.zeros((N, par['n_input']), dtype=np.float32)
    sample_input    = np.zeros((N, par['n_input']), dtype=np.float32)

    default_output  = np.zeros((N, par['n_output']), dtype=np.float32)
    #default_output = None
    fix_output      = np.zeros((N, par['n_output']), dtype=np.float32)
    fix_output[:,0] = 1
    sample_output   =  np.zeros((N, par['n_output']), dtype=np.float32)

    presented_numbers = np.zeros((N, 1), dtype=np.float32)
    permutations      = np.zeros((N, 1), dtype=np.float32)

    for n in range(N):

        # Generate a number as stimulus
        num = np.random.randint(len(labels))
        label = labels[num]
        image = images[num]

        # Set permutation
        permutations[n,0] = par['permutation_id']
        #permutations.append(par['permutation_id'])
        image = mnist_permutation(image)/255

        # Output sample_input[n] from loop
        sample_input[n, :] = image

        # Output sample_output[n] from loop
        #presented_numbers.append(label)
        presented_numbers[n,0] = label
        sample_output[n, label] = 1

    inputs = {'none'    :   default_input,
              'fix'     :   fix_input,
              'stim'    :   sample_input
             }

    outputs = {'none'   :   default_output,
               'fix'    :   fix_output,
               'stim'   :   sample_output
              }

    trial_setup = {'default_input'  :   default_input,
                   'inputs'         :   inputs,
                   'default_output' :   default_output,
                   'outputs'        :   outputs,
                   'rule_index'     :   permutations,
                   'sample_index'   :   presented_numbers,
                   'test_index'     :   []
                  }

    return trial_setup


def direction_dms(N):
    """
    Generates a set of random trials for dms (delayed match to sample) tests
    based on the batch size, and returns the stimuli, tests, intended outputs,
    and the default intended output for the set.

    The output arrays are [batch_train_size x neurons] large, where [neurons]
    corresponds to the number of input or output neurons, depending on the array.
    """

    ### Tuning inputs
    stim_tuning, fix_tuning, rule_tuning = tuning()

    ### Default case
    default_input = np.zeros((N, par['n_input']), dtype=np.float32)
    default_output = np.array([[0. ,0. ,0.]] * N) # Standardize to zeros

    ### Fixation case
    fix_out = np.array([[1., 0., 0.]] * N)

    ### Sample case
    stim_tuning = np.transpose(stim_tuning)

    # Choose random samples to start test
    sample_setup = np.random.randint(0, par['num_motion_dirs'], size=N)

    # Map to the desired sample inputs
    sample = []
    for i in range(len(sample_setup)):
        sample.append(stim_tuning[sample_setup[i]])
    sample = np.array(sample)

    ### Test case

    # Choose test inputs:
    #   For each corresponding sample, decide if the test should match.
    #   If yes, keep the same direction number as the sample.  If no, generate
    #   a random direction and compare that.  If it is the same, try everything
    #   again.  If it is different, use that direction.
    output          = []
    match_out       = [0., 1., 0.]
    non_match_out   = [0., 0., 1.]

    test_setup = []
    for i in range(len(sample_setup)):
        applied = False
        while applied == False:
            if np.random.rand() < par['match_rate']:
                test_setup.append(sample_setup[i])
                output.append(match_out)
                applied = True
            else:
                y = np.random.randint(0, par['num_motion_dirs'])
                if y == sample_setup[i]:
                    pass
                else:
                    test_setup.append(y)
                    output.append(non_match_out)
                    applied = True

    # Map to the desired test inputs
    test = []
    for i in range(len(test_setup)):
        test.append(stim_tuning[test_setup[i]])

    ### End of cases

    # Putting together trial_setup
    inputs = {'none'        : default_input,
              'fix'         : default_input,
              'sample'      : sample,
              'test'        : test
             }

    outputs = {'none'       : default_output,
               'fix'        : fix_out,
               'sample'     : fix_out,
               'test'       : output
               }

    trial_setup = {'default_input'  :   default_input,
                   'inputs'         :   inputs,
                   'default_output' :   default_output,
                   'outputs'        :   outputs,
                   'rule_index'     :   [],
                   'sample_index'   :   [],
                   'test_index'     :   []
                  }

    return trial_setup

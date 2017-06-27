"""
2017/06/20 Gregory Grant
"""

import numpy as np
import imp
from parameters import *


#######################################
### Tuning and adjustment functions ###
#######################################

def motion_tuning():
    """
    Credit: Nicolas Masse
    Generate tuning functions for the input neurons
    Neurons are selective for motion direction, fixation, or rules
    """

    # Essentially, creates matrices of the number of requisite neurons
    # by the number of possible states introduced to that neuron
    motion_tuning   = np.zeros((par['num_motion_tuned'], par['num_motion_dirs']))
    fix_tuning      = np.zeros((par['num_fix_tuned'], 2))
    rule_tuning     = np.zeros((par['num_rule_tuned'], par['num_rules']))

    # Generates lists of preferred and possible motion directions
    pref_dirs   = np.float32(np.arange(0,2*np.pi,2*np.pi/par['num_motion_tuned']))
    motion_dirs = np.float32(np.arange(0,2*np.pi,2*np.pi/par['num_motion_dirs']))

    for n in range(par['num_motion_tuned']):
        for i in range(len(motion_dirs)):
            # Finds the difference between each motion diretion and preferred direction
            d = np.cos(motion_dirs[i] - pref_dirs[n])
            # Transforms the direction variances with a von Mises
            motion_tuning[n,i] = par['tuning_height']*(np.exp(par['kappa']*d) \
                                                     - np.exp(-par['kappa']))/np.exp(par['kappa'])

    for n in range(par['num_fix_tuned']):
        for i in range(2):
            if n%2 == i:
                fix_tuning[n,i] = scale

    for n in range(par['num_rule_tuned']):
        for i in range(par['num_rules']):
            if n%par['num_rules'] == i:
                rule_tuning[n,i] = scale

    return motion_tuning, fix_tuning, rule_tuning


def add_noise(m):
    """
    Add Gaussian noise to a matrix, and return only non-negative
    numbers within the matrix.  Used to provide noise to each time step.
    """

    gauss = np.random.normal(0, par['input_sd'], np.shape(m))
    m = np.clip(m + gauss, 0, par['input_clip_max'])

    return m


#################################
### Trial generator functions ###
#################################

def experimental(N):
    """
    Generates a set of random trials for the experimental tests
    based on the batch size, and returns the stimuli, tests,
    intended outputs, and the default intended output for the set.

    The output arrays are [batch_train_size] long.
    """
    default_input = [[0,0,0,0,0,0,0,0,0]] * N
    fixation = [[0,0,0,0,1,0,0,0,0]] * N
    stimuli =  [[0,1,0,0,0,1,0,0,0], [0,1,0,1,0,0,0,0,0], \
                [0,0,0,1,0,0,0,1,0], [0,0,0,0,0,1,0,1,0]]
    tests =     [[0,0,1,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], \
                [0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,0,1]]
    none =      [[0,0,0,0,0,0,0,0,0]] * N

    setup = np.random.randint(0,4,size=(2,N))

    stimulus = []
    test = []
    desired_output = np.transpose([np.float32(setup[0] == setup[1])])
    default_desired_output = np.transpose([[0.] * N])

    for i in range(N):
        stimulus.append(stimuli[setup[0,i]])
        test.append(tests[setup[1,i]])

    inputs = {'sample' : stimulus,
              'test' : test,
              'fix'  : fixation,
              'none' : none
              }
    outputs = {'sample' : default_desired_output,
               'test' : desired_output,
               'fix'  : default_desired_output,
               'none': default_desired_output}

    trial_setup = {'default_input' : default_input,
                   'inputs' : inputs,
                   'default_output' : default_desired_output,
                   'outputs' : outputs
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
    stim_tuning, fix_tuning, rule_tuning = motion_tuning()

    ### Default case
    default_input = np.zeros((N, par['n_input']), dtype=np.float32)
    default_output = np.array([[0. ,0. ,0.]] * N) # Standardize to zeros

    ### Fixation case
    fix_out = [[1., 0., 0.]] * N

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

    # Add noise to each input
    #default_input   = add_noise(default_input)
    #sample          = add_noise(sample)
    #test            = add_noise(test)

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

    trial_setup = {'default_input'  : default_input,
                   'inputs'         : inputs,
                   'default_output' : default_output,
                   'outputs'        : outputs
                   }

    return trial_setup

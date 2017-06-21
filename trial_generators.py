"""
2017/06/20 Gregory Grant
"""

import numpy as np
import imp

def import_parameters():
    print('Trial generator module:')
    f = open('parameters.py')
    global par
    par = imp.load_source('data', '', f)
    f.close()

import_parameters()

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
        stimulus.append(stimuli[setup[0][i]])
        test.append(tests[setup[1][i]])

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


def dms(N):
    """
    Generates a set of random trials for dms (delayed match to sample) tests
    based on the batch size, and returns the stimuli, tests, intended outputs,
    and the default intended output for the set.

    The output arrays are [batch_train_size x neurons] large, where [neurons]
    corresponds to the number of input or output neurons, depending on the array.
    """







    trial_setup = {'default_input' : default_input,
                   'inputs' : inputs,
                   'default_output' : default_desired_output,
                   'outputs' : outputs
                   }

    return trial_setup








def motion_tuning(num_pos, num_motion_tuned, num_fix_tuned, num_rule_tuned):



    return stim_tuning, fix_tuning, rule_tuning

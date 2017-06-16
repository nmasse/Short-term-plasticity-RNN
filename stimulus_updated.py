"""
2017/05/03 Nicolas Masse
2017/06/16 Gregory Grant
"""

import numpy as np
import matplotlib.pyplot as plt


class Stimulus:

    def __init__(self, params):

        # params is a dictionary containing all the trial parameters
        for key, value in params.items():
            setattr(self, key, value)

    def generate_trial(self, trials_per_batch=64):

        if self.stimulus_type == 'experimental':
            trial_setup = self.generate_experimental_trials()
        else:
            print("Invalid stimulus type.")
            quit()


        events = [[200, 'input', 'stim'], [400, 'mask', False], [400, 'input', 'test'], [1200, 'mask', True]]
        schedules = self.generate_schedule(events, trial_setup)


        """
        trial_info = {'desired_output'  :  np.zeros((self.n_output, trial_length, trials_per_batch),dtype=np.float32),
                      'train_mask'      :  np.ones((trial_length, trials_per_batch),dtype=np.float32),
                      'sample'          :  np.zeros((trials_per_batch,2),dtype=np.int8),
                      'test'            :  np.zeros((trials_per_batch,2,2),dtype=np.int8),
                      'test_mod'        :  np.zeros((trials_per_batch,2),dtype=np.int8),
                      'rule'            :  np.zeros((trials_per_batch,2),dtype=np.int8),
                      'match'           :  np.zeros((trials_per_batch,2),dtype=np.int8),
                      'catch'           :  np.zeros((trials_per_batch,2),dtype=np.int8),
                      'probe'           :  np.zeros((trials_per_batch,2),dtype=np.int8),
                      'neural_input'    :  np.random.normal(self.input_mean, self.input_sd, size=(self.n_input, trial_length, trials_per_batch))}
        """

        #print(trial_setup)
        quit()

        return trial_info

    def generate_experimental_trials(self):
        """
        Generates a set of random trials for the experimental tests
        based on the batch size, and returns the stimuli, tests,
        intended outputs, and the default intended output for the set.

        The output arrays are [batch_train_size] long.
        """
        default_input = [[0,0,0,0,0,0,0,0,0]] * self.batch_train_size
        stimuli =  [[0,1,0,0,0,1,0,0,0], [0,1,0,1,0,0,0,0,0], \
                    [0,0,0,1,0,0,0,1,0], [0,0,0,0,0,1,0,1,0]]
        tests =     [[0,0,1,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], \
                    [0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,0,1]]

        setup = np.random.randint(0,4,size=(2,self.batch_train_size))

        stimulus = []
        test = []
        desired_output = np.float32(setup[0] == setup[1])
        default_desired_output = [0.] * self.batch_train_size

        for i in range(self.batch_train_size):
            stimulus.append(stimuli[setup[0][i]])
            test.append(tests[setup[1][i]])

        inputs = {'stim' : stimulus,
                  'test' : test
                  }

        trial_setup = {'default_input' : default_input,
                       'inputs' : inputs,
                       'default_desired_output' : default_desired_output,
                       'desired_output' : desired_output
                       }

        return trial_setup

    def generate_schedule(self, events, trial_setup, mask_starts_on=True):
        """
        Takes in a sequence of events and trial setup information
        and returns three schedules:
            - Input schedule, or what the neural input sees at each time step
            - Mask schedule, or whether the mask is applied at each time step
            - Output schedule, or what the desired output is at each time step
        """

        num_events = len(events)
        finish_time = events[num_events-1][0]

        # Checks for a reasonable time step, and then sets it up if reasonable
        if self.dt >= finish_time:
            print("ERROR:  Time step is longer than entire trial time.")
            quit()
        else:
            steps = finish_time//self.dt

        # Converts times to time step values in the event list
        for i in range(num_events):
            events[i][0] = events[i][0]//self.dt

        # Sets up individual event lists, then picks out relevant
        # information for each
        input_events = []
        mask_events = []
        output_events = []

        for i in range(num_events):
            if events[i][1] == 'input':
                input_events.append(events[i])
            if events[i][1] == 'mask':
                mask_events.append(events[i])

        input_schedule = self.scheduler(steps, input_events, trial_setup['inputs'], trial_setup['default_input'])

    def scheduler(self, steps, events, inputs_batch, default_input):
        """
        Takes in a set of events and combines it with the batch_train_size
        to produce a neural input schedule of size [neurons, steps, batch_size]
        """
        print("\nInput neurons:", self.n_input)
        print("Steps:", steps)
        print("Batch size:", self.batch_train_size)
        print("Input events:", events)
        print("Default input:", np.shape(default_input), "\n")

        #schedule = np.zeros(self.n_input, steps, self.batch_train_size)
        value = default_input
        schedule = [np.zeros(steps)]

        # Starting with the 0th step, edits schedule
        step_val = 0
        for i in range(len(events)):
            schedule[step_val:events[i][0]] = [value]*(events[i][0]-step_val)
            step_val = events[i][0]
            value = inputs_batch[events[i][2]]

        # Edits the last portion of the mask
        schedule[step_val:steps] = [value]*(steps-step_val)

        # Transposes the schedule into the proper output form as described in
        # the docstring
        schedule = np.transpose(schedule, (2,0,1))
        print(np.shape(schedule))
        print(schedule)

        return schedule

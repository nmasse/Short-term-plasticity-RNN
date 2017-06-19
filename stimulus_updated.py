"""
2017/05/03 Nicolas Masse
2017/06/16 Gregory Grant
"""

import numpy as np
import matplotlib.pyplot as plt
import copy


class Stimulus:

    def __init__(self, params):

        # params is a dictionary containing all the trial parameters
        for key, value in params.items():
            setattr(self, key, value)


    def generate_trial(self, num):

        # Note that num will typically be self.batch_train_size * self.num_batches
        if self.stimulus_type == 'experimental':
            trial_setup = self.generate_experimental_trials(num)
        else:
            print("Invalid stimulus type.")
            quit()

        schedules = self.generate_schedule(copy.deepcopy(self.events), trial_setup, num)

        trial_info = {'desired_output' : schedules[1],
                        'train_mask' : np.asarray(schedules[2]),
                        'neural_input' : schedules[0]}

        return trial_info


    def generate_schedule(self, events, trial_setup, N, mask_starts_on=True):
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
        output_schedule = self.scheduler(steps, input_events, trial_setup['outputs'], trial_setup['default_output'])
        mask_schedule = self.scheduler(steps, mask_events, {'on':[0]*N,'off':[1]*N}, [0]*N if mask_starts_on else [1]*N, flag="mask")

        return input_schedule, output_schedule, mask_schedule


    def scheduler(self, steps, events, batch, default, flag=None):
        """
        Takes in a set of events and combines it with the batch_train_size
        to produce a neural input schedule of size [neurons, steps, batch_size]
        """

        value = default
        schedule = [np.zeros(steps)]

        # Starting with the 0th step, edits schedule
        step_val = 0
        for i in range(len(events)):
            schedule[step_val:events[i][0]] = [value]*(events[i][0]-step_val)
            step_val = events[i][0]
            value = batch[events[i][2]]

        # Edits the last portion of the mask
        schedule[step_val:steps] = [value]*(steps-step_val)

        if flag != "mask":
            # Transposes the schedule into the proper output orientation
            # Mask does not require this step -- it already is correctly shaped.
            schedule = np.transpose(schedule, (2,0,1))

        return schedule


    def generate_experimental_trials(self, N):
        """
        Generates a set of random trials for the experimental tests
        based on the batch size, and returns the stimuli, tests,
        intended outputs, and the default intended output for the set.

        The output arrays are [batch_train_size] long.
        """
        default_input = [[0,0,0,0,0,0,0,0,0]] * N
        stimuli =  [[0,1,0,0,0,1,0,0,0], [0,1,0,1,0,0,0,0,0], \
                    [0,0,0,1,0,0,0,1,0], [0,0,0,0,0,1,0,1,0]]
        tests =     [[0,0,1,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], \
                    [0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,0,1]]

        setup = np.random.randint(0,4,size=(2,N))

        stimulus = []
        test = []
        desired_output = np.transpose([np.float32(setup[0] == setup[1])])
        default_desired_output = np.transpose([[0.] * N])

        for i in range(N):
            stimulus.append(stimuli[setup[0][i]])
            test.append(tests[setup[1][i]])

        inputs = {'stim' : stimulus,
                  'test' : test
                  }
        outputs = {'stim' : default_desired_output,
                   'test' : desired_output}

        trial_setup = {'default_input' : default_input,
                       'inputs' : inputs,
                       'default_output' : default_desired_output,
                       'outputs' : outputs
                       }

        return trial_setup

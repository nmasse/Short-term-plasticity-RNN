"""
2017/06/16 Gregory Grant
"""

import numpy as np
import matplotlib.pyplot as plt
import trial_generators as gen
import copy
import imp

def import_parameters():
    print('Stimulus module:')
    f = open('parameters.py')
    global par
    par = imp.load_source('data', '', f)
    f.close()

import_parameters()


class Stimulus:

    def __init__(self):
        pass

    def generate_trial(self, num):

        # Note that num will typically be par.batch_train_size * par.num_batches
        if par.stimulus_type == 'exp':
            trial_setup = gen.experimental(num)
        elif par.stimulus_type == 'dms':
            trial_setup = gen.direction_dms(num)
        else:
            print("Invalid stimulus type.")
            quit()

        schedules = self.generate_schedule(copy.deepcopy(par.events), trial_setup, num)

        trial_info = {'desired_output' : schedules[1],
                        'train_mask' : np.asarray(schedules[2]),
                        'neural_input' : schedules[0]}

        return trial_info


    def generate_schedule(self, events, trial_setup, N, mask_starts_off=True):
        """
        Takes in a sequence of events and trial setup information
        and returns three schedules:
            - Input schedule, or what the neural input sees at each time step
            - Mask schedule, or whether the mask is applied at each time step
            - Output schedule, or what the desired output is at each time step
        """

        num_events = len(events)

        # Checks for a reasonable time step, and then sets it up if reasonable
        if par.dt >= par.trial_length:
            print("ERROR:  Time step is longer than entire trial time.")
            quit()
        else:
            steps = par.trial_length//par.dt

        # Converts times to time step values in the event list
        for i in range(num_events):
            events[i][0] = events[i][0]//par.dt

        # Sets up individual event lists, then picks out relevant
        # information for each.
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
        mask_schedule = self.scheduler(steps, mask_events, {'off':[0]*N,'on':[1]*N}, [0]*N if mask_starts_off else [1]*N, flag="mask")
        # Note that off means mask = 0 (block signal), on means mask = 1 (pass signal)

        if par.var_delay:
            input_schedule, output_schedule, mask_schedule = self.time_adjust(input_events, input_schedule, output_schedule, mask_schedule, N, steps)
        else:
            pass

        return input_schedule, output_schedule, mask_schedule


    def scheduler(self, steps, events, batch, default, flag=None):
        """
        steps = integer, events = list
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


    def time_adjust(self, input_events, input_schedule, output_schedule, mask_schedule, N, steps):
        """
        Uses np.random.exponential to add a variable delay to requisite parts
        of the input, output, and mask schedules.  Also implements catch trials,
        where there is no test and the output behaves correspondingly.
        """

        # Sets up a pair of timing arrays to be used in blocking out variable scopes
        timings_on = []
        timings_off = []

        for i in range(len(input_events)):
            if len(input_events[i]) > 3:
                timings_on.append(input_events[i][0])
                if len(input_events) > i + 1:
                    timings_off.append(input_events[i+1][0])

        if len(timings_on) != len(timings_off):
            timings_off.append(steps)

        mask_schedule = np.array(mask_schedule)

        # Finds the template mask lengths
        template = []
        mask_length = []
        c = 0
        for i in range(len(mask_schedule)):
            template.append(mask_schedule[i][0])
        for i in range(len(template)-1):
            if template[i] == 0 and template[i+1] == 0:
                c = c + 1
            elif template[i] == 0 and template[i+1] == 1:
                mask_length.append(c)
                c = 0
            elif template[i] == 1 and template[i+1] == 0:
                c = 1
            elif template[i] == 1 and template[i+1] == 1:
                c = 0

        # Removes dead time mask
        mask_length.pop(0)

        # Generate the timing variances and catches from the two timings arrays
        for s in range(len(timings_on)):
            steps_eff = (timings_off[s] - timings_on[s]) - (mask_length[s] + 1)
            var = np.int32(np.round(np.random.exponential(-(steps_eff)/np.log(par.catch_rate), N)))
            for n in range(N):
                for m in range(timings_on[s],timings_off[s]):
                    mask_schedule[m][n] = mask_schedule[m-1][n]
                # If there is not a catch:
                if var[n] <= steps_eff:
                    for d in range(var[n]):
                        for i in range(par.n_input):
                            input_schedule[i][timings_on[s]+d][n] = input_schedule[i][timings_on[s]-1][n]
                        for o in range(par.n_output):
                            output_schedule[o][timings_on[s]+d][n] = output_schedule[o][timings_on[s]-1][n]
                    for m in range(timings_on[s],timings_off[s]):
                        if m >= var[n]+timings_on[s]:
                            mask_schedule[m][n] = 0
                        if m >= var[n]+mask_length[s]+timings_on[s]:
                            mask_schedule[m][n] = 1
                # If there IS a catch:
                else:
                    for d in range(timings_off[s]-timings_on[s]):
                        for i in range(par.n_input):
                            input_schedule[i][timings_on[s]+d][n] = input_schedule[i][timings_on[s]-1][n]
                        for o in range(par.n_output):
                            output_schedule[o][timings_on[s]+d][n] = output_schedule[o][timings_on[s]-1][n]

        return input_schedule, output_schedule, mask_schedule

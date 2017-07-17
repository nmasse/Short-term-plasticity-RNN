"""
2017/06/16 Gregory Grant
"""

import numpy as np
import matplotlib.pyplot as plt
import trial_generators as gen
import psutil
import copy
from parameters import *


class Stimulus:

    def __init__(self):

        from mnist import MNIST
        mndata = MNIST('./resources/mnist/data/original')
        images, labels = mndata.load_training()
        self.mnist_images = images
        self.mnist_labels = np.array(labels)

        self.stim_tuning    = self.generate_stim_tuning()
        self.fix_tuning     = self.generate_fix_tuning()
        self.rule_tuning    = self.generate_rule_tuning()
        self.spatial_tuning = self.generate_spatial_tuning()

    def generate_trial(self, num):

        # Generate a set of trials for the desired task
        trial_setup = gen.trial_batch(num, self.stim_tuning, self.fix_tuning, \
                                           self.rule_tuning, self.spatial_tuning, \
                                           self.mnist_images, self.mnist_labels)

        # Propagate the trials through time
        schedules = self.generate_schedule(copy.deepcopy(par['events']), trial_setup, num)

        # Build the trial info dictionary for use in running the model
        trial_info = {'desired_output'          : schedules[1],
                      'train_mask'              : np.asarray(schedules[2]),
                      'neural_input'            : schedules[0],
                      'rule_index'              : trial_setup['rule_index'],
                      'location_index'          : trial_setup['location_index'],
                      'sample_index'            : trial_setup['sample_index'],
                      'attended_sample_index'   : trial_setup['attended_sample_index']
                     }

        return trial_info


    def generate_stim_tuning(self):
        """
        Based on the task type, return the possible neural input tunings
        """

        if par['stimulus_type'] == 'mnist':
            stim_tuning = np.array(self.mnist_images)

        elif par['stimulus_type'] == 'att':
            stim_tuning     = np.zeros([par['num_samples'], par['num_stim_tuned']//par['num_RFs']], dtype=np.float32)

            pref_dirs       = np.float32(np.arange(0, 2*np.pi, 2*np.pi/(par['num_stim_tuned']//par['num_RFs'])))
            stim_dirs       = np.float32(np.arange(0, 2*np.pi, 2*np.pi/par['num_samples']))
            for i in range(par['num_samples']):
                for n in range(par['num_stim_tuned']//par['num_RFs']):
                    d = np.cos(stim_dirs[i] - pref_dirs[n%len(pref_dirs)])
                    stim_tuning[i, n] = par['tuning_height']*(np.exp(par['kappa']*d) \
                                        - np.exp(-par['kappa']))/np.exp(par['kappa'])

        return stim_tuning


    def generate_fix_tuning(self):
        """
        Currently unused, but typically generates fixation tunings.
        """
        return np.array([0])


    def generate_rule_tuning(self):
        """
        Uses the number of rules and rule-tuned neurons to generate a neural
        input mapping for presenting rules to the network
        """

        rule_tuning = np.zeros([par['num_rules'], par['num_rule_tuned']])
        m = par['num_rule_tuned']//par['num_rules']
        if par['num_rule_tuned'] == 0:
            return np.array([0]*par['num_rules'])
        elif m == 0:
            print('ERROR: Use more rule neurons than rules.')
            quit()

        for r in range(par['num_rules']):
            if r == par['num_rules']-1:
                rule_tuning[r, r*m:] = 1
            else:
                rule_tuning[r, r*m:r*m+m] = 1
        return rule_tuning


    def generate_spatial_tuning(self):
        """
        Uses the number of RFs and spatially-tuned neurons to generate a neural
        input mapping for presenting location cues to the network
        """

        spatial_tuning = np.zeros([par['num_RFs'], par['num_spatial_cue_tuned']])
        m = par['num_spatial_cue_tuned']//par['num_RFs']
        if par['num_spatial_cue_tuned'] == 0:
            return np.array([0]*par['num_RFs'])
        elif m == 0:
            print('ERROR: Use more spatially tuned neurons than RFs.')
            quit()

        for r in range(par['num_RFs']):
            if r == par['num_RFs']-1:
                spatial_tuning[r, r*m:] = 1
            else:
                spatial_tuning[r, r*m:r*m+m] = 1

        return spatial_tuning


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
        if par['dt'] >= par['trial_length']:
            print("ERROR:  Time step is longer than entire trial time.")
            quit()
        else:
            steps = par['trial_length']//par['dt']

        # Converts times to time step values in the event list
        for i in range(num_events):
            events[i][0] = events[i][0]//par['dt']

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

        input_schedule = self.scheduler(steps, input_events, trial_setup['inputs'], np.zeros((N, par['n_input']), dtype=np.float32))
        output_schedule = self.scheduler(steps, input_events, trial_setup['outputs'], np.zeros((N, par['n_output']), dtype=np.float32))
        mask_schedule = self.scheduler(steps, mask_events, {'off':[0]*N,'on':[1]*N}, [0]*N if mask_starts_off else [1]*N, flag="mask")
        # Note that off means mask = 0 (block signal), on means mask = 1 (pass signal)

        if par['var_delay']:
            # Staggers the starting times of indicated events, and triggers catch trials for a proportion of those
            input_schedule, output_schedule, mask_schedule = self.time_adjust(input_events, input_schedule, output_schedule, mask_schedule, N, steps)
        elif par['catch_trials']:
            # Applies catch trials to a proportion of the indicated events
            input_schedule, output_schedule, mask_schedule = self.catch_trials(input_events, input_schedule, output_schedule, mask_schedule, N, steps)
        else:
            # Apply neither variable start times for tests nor catch trials
            pass

        # Add noise to each time step
        input_schedule = self.add_noise(input_schedule)

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

        # Edits the last portion of the schedule
        schedule[step_val:steps] = [value]*(steps-step_val)

        if flag != "mask":
            # Transposes the schedule into the proper output orientation
            # Mask does not require this step -- it already is correctly shaped.
            schedule = np.transpose(schedule, (2,0,1))
        else:
            schedule = np.array(schedule)

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
            template.append(mask_schedule[i,0])
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
            var = np.int32(np.round(np.random.exponential(-(steps_eff)/np.log(par['catch_rate']), N)))
            for n in range(N):
                for m in range(timings_on[s],timings_off[s]):
                    mask_schedule[m,n] = mask_schedule[m-1,n]
                # If there is not a catch:
                if var[n] <= steps_eff:
                    for d in range(var[n]):
                        for i in range(par['n_input']):
                            input_schedule[i,timings_on[s]+d,n] = input_schedule[i,timings_on[s]-1,n]
                        for o in range(par['n_output']):
                            output_schedule[o,timings_on[s]+d,n] = output_schedule[o,timings_on[s]-1,n]
                    for m in range(timings_on[s],timings_off[s]):
                        if m >= var[n]+timings_on[s]:
                            mask_schedule[m,n] = 0
                        if m >= var[n]+mask_length[s]+timings_on[s]:
                            mask_schedule[m,n] = 1
                # If there IS a catch:
                else:
                    for d in range(timings_off[s]-timings_on[s]):
                        for i in range(par['n_input']):
                            input_schedule[i,timings_on[s]+d,n] = input_schedule[i,timings_on[s]-1,n]
                        for o in range(par['n_output']):
                            output_schedule[o,timings_on[s]+d,n] = output_schedule[o,timings_on[s]-1,n]

        return input_schedule, output_schedule, mask_schedule


    def catch_trials(self, input_events, input_schedule, output_schedule, mask_schedule, N, steps):
        """
        Uses np.random.rand to match against a catch rate to block requisite parts
        of the input, output, and mask schedules.  These catch trials have no
        test, and the output behaves accordingly.
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

        for s in range(len(timings_on)):
            for n in range(N):
                # If there IS a catch
                if np.random.rand <= par['catch_rate']:
                    for d in range(timings_off[s]-timings_on[s]):
                        for i in range(par['n_input']):
                            input_schedule[i,timings_on[s]+d,n] = input_schedule[i,timings_on[s]-1,n]
                        for o in range(par['n_output']):
                            output_schedule[o,timings_on[s]+d,n] = output_schedule[o,timings_on[s]-1,n]
                        for m in range(timings_on[s],timings_off[s]):
                            mask_schedule[m,n] = mask_schedule[m-1,n]
                # If there is NOT a catch
                else:
                    # The input, output, and mask schedules stay the same
                    pass

        return input_schedule, output_schedule, mask_schedule


    def add_noise(self, m):
        """
        Add Gaussian noise to a matrix, and return only non-negative
        numbers within the matrix.
        """

        m = np.maximum(m + np.random.normal(0, par['input_sd'], np.shape(m)), 0)
        return m

import numpy as np
from parameters import *
import matplotlib.pyplot as plt

class Stimulus:

    def __init__(self):

        self.input_shape   = [par['n_input'], par['num_time_steps'], par['batch_train_size']]
        self.mask_shape     = [par['num_time_steps'], par['batch_train_size']]



    def generate_trial(self):

        self.trial_info = {
            'desired_output' : np.zeros(self.input_shape, dtype=np.float32),
            'train_mask'     : np.ones(self.mask_shape, dtype=np.float32)}

        return self.diagnostic()         # Returns the trial info and the task name


    def limited_DMS(self):

        mask_time     = int(par['num_time_steps']*0.1)
        fixation_time = int(par['num_time_steps']*0.2)
        stim_time     = int(par['num_time_steps']*0.3)

        t1 = mask_time
        t2 = t1 + fixation_time
        t3 = t2 + stim_time
        t4 = t3 + fixation_time

        response = np.zeros(self.input_shape)
        for b in range(par['batch_train_size']):
            d1 = np.random.choice(np.arange(par['num_motion_dirs']))
            if np.random.choice([True, False]):
                d2 = d1
            else:
                d2 = (d1 + par['num_motion_dirs']//2)%par['num_motion_dirs']

            # Think about this!
            response[-1, t1:t2, b] = 1
            response[d1, t2:t3, b] = 1
            response[d2, t4:, b] = 1

        mask = np.ones(self.mask_shape)
        mask[:t1, :] = 0
        mask[t3:t4]  = 0

        self.trial_info['desired_output'] = response
        self.trial_info['train_mask'] = mask

        return self.trial_info


    def diagnostic(self):

        mask_time     = int(par['num_time_steps']*0.1)
        fixation_time = int(par['num_time_steps']*0.2)
        stim_time     = int(par['num_time_steps']*0.3)

        t1 = mask_time
        t2 = t1 + fixation_time
        t3 = t2 + stim_time
        t4 = t3 + fixation_time

        response = np.zeros(self.input_shape)
        for b in range(par['batch_train_size']):
            for t in range(t2, t3):
                response[(t+b)%par['n_input'], t, :] = 1

            for t in range(t4, par['num_time_steps']):
                response[(t+b)%par['n_input'], t, :] = 1

        response[-1, t1:t2, b] = 1

        mask = np.ones(self.mask_shape)
        mask[:t1, :] = 0
        mask[t3:t4]  = 0

        self.trial_info['neural_input'] = response
        self.trial_info['train_mask'] = mask

        return self.trial_info

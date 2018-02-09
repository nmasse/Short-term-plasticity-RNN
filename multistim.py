import numpy as np
import matplotlib.pyplot as plt
from parameters import *

class MultiStimulus:

    def __init__(self):

        # Shape configuration
        self.input_shape    = [par['n_input'], par['num_time_steps'], par['batch_train_size']]
        self.output_shape   = [par['n_output'], par['num_time_steps'], par['batch_train_size']]
        self.stimulus_shape = [par['num_motion_tuned'], par['num_time_steps'], par['batch_train_size']]
        self.response_shape = [par['num_motion_dirs'], par['num_time_steps'], par['batch_train_size']]
        self.fixation_shape = [par['num_fix_tuned'], par['num_time_steps'], par['batch_train_size']]
        self.mask_shape     = [par['num_time_steps'], par['batch_train_size']]

        # Motion and stimulus configuration
        self.motion_dirs    = np.linspace(0,2*np.pi-2*np.pi/par['num_motion_dirs'],par['num_motion_dirs'])
        self.stimulus_dirs  = np.linspace(0,2*np.pi-2*np.pi/(par['num_motion_tuned']//2),(par['num_motion_tuned']//2))
        self.modality_size  = (par['num_motion_tuned'])//2

        # DM task stuff
        self.dm_c_set = np.array([-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08])
        self.dm_stim_lengths = np.array([400,800,1600])//par['dt']

        # DM Dly task stuff
        self.dm_dly_c_set = np.array([-0.32, -0.16, -0.08, 0.08, 0.16, 0.32])
        self.dm_dly_delay = np.array([200, 400, 800, 1600])//par['dt']

        # Matching task stuff
        self.match_delay = np.array([200, 400, 800, 1600])//par['dt']

        # Initialize task interface
        self.get_tasks()
        self.task_id = 0
        self.task_order = np.arange(len(self.task_types))


    def circ_tuning(self, theta):
        return par['tuning_height'] * np.exp(par['kappa'] \
             * np.cos(theta-self.stimulus_dirs[:,np.newaxis])) \
             / np.exp(par['kappa'])


    def get_tasks(self):
        if par['trial_type'] == 'multistim':
            self.task_types = [
                [self.task_go, 'go', 0],
                [self.task_go, 'rt_go', 0],
                [self.task_go, 'dly_go', 0],

                [self.task_go, 'go', np.pi],
                [self.task_go, 'rt_go', np.pi],
                [self.task_go, 'dly_go', np.pi],

                [self.task_dm, 'dm1'],
                [self.task_dm, 'dm2'],
                [self.task_dm, 'ctx_dm1'],
                [self.task_dm, 'ctx_dm2'],
                [self.task_dm, 'multsen_dm'],

                [self.task_dm_dly, 'dm1_dly'],
                [self.task_dm_dly, 'dm2_dly'],
                [self.task_dm_dly, 'ctx_dm1_dly'],
                [self.task_dm_dly, 'ctx_dm2_dly'],
                [self.task_dm_dly, 'multsen_dm_dly'],

                [self.task_matching, 'dms'],
                [self.task_matching, 'dmc'],
                [self.task_matching, 'dnms'],
                [self.task_matching, 'dnmc']
            ]
        elif par['trial_type'] == 'twelvestim':
            self.task_types = [
                [self.task_go, 'go', 0],
                [self.task_go, 'dly_go', 0],

                [self.task_dm, 'dm1'],
                [self.task_dm, 'dm2'],
                [self.task_dm, 'ctx_dm1'],
                [self.task_dm, 'ctx_dm2'],
                [self.task_dm, 'multsen_dm'],

                [self.task_dm_dly, 'dm1_dly'],
                [self.task_dm_dly, 'dm2_dly'],
                [self.task_dm_dly, 'ctx_dm1_dly'],
                [self.task_dm_dly, 'ctx_dm2_dly'],
                [self.task_dm_dly, 'multsen_dm_dly']
            ]
        else:
            raise Exception('Multistimulus task type \'{}\' not yet implemented.'.format(par['trial_type']))

        return self.task_types


    def generate_trial(self, current_task):

        self.trial_info = {
            'neural_input'   : np.random.normal(par['input_mean'], par['noise_in'], size=self.input_shape),
            'desired_output' : np.zeros(self.output_shape, dtype=np.float32),
            'train_mask'     : np.ones(self.mask_shape, dtype=np.float32)}

        self.trial_info['train_mask'][:par['dead_time']//par['dt'], :] = 0

        task = self.task_types[current_task]    # Selects a task from the list
        task[0](*task[1:])                      # Generates that task into trial_info
        return task[1], self.trial_info         # Returns the trial info and the task name


    def task_go(self, variant='go', offset=0):

        # Task parameters
        if variant == 'go':
            stim_onset = np.random.randint(500, 1500, par['batch_train_size'])//par['dt']
            stim_off = -1
            fixation_end = np.ones(par['batch_train_size'], dtype=np.int8)*1500//par['dt']
            resp_onset = fixation_end
        elif variant == 'rt_go':
            stim_onset = np.random.randint(500, 2500, par['batch_train_size'])//par['dt']
            stim_off = -1
            fixation_end = np.ones(par['batch_train_size'],dtype=np.int8)*par['num_time_steps']
            resp_onset = stim_onset
        elif variant == 'dly_go':
            stim_onset = 500//par['dt']*np.ones(par['batch_train_size'],dtype=np.int8)
            stim_off = 750//par['dt']
            fixation_end = stim_off + np.random.choice([200,400,800,1600], size=par['batch_train_size'])//par['dt']
            resp_onset = fixation_end
        else:
            raise Exception('Bad task variant.')

        # Need dead time
        self.trial_info['train_mask'][:par['mask_duration']//par['dt'], :] = 0

        fix_resp = np.tile(par['tuning_height'], (par['num_fix_tuned'],1))
        for b in range(par['batch_train_size']):

            # Input neurons index above par['num_motion_tuned'] encode fixation
            self.trial_info['neural_input'][par['num_motion_tuned']:,:fixation_end[b], b] += np.reshape(fix_resp,(-1,1))

            modality   = np.random.randint(2)
            neuron_ind = range(self.modality_size*modality, self.modality_size*(1+modality))
            stim_dir   = np.random.choice(self.motion_dirs)
            target_ind = int(np.round(par['num_motion_dirs']*(stim_dir+offset)/(2*np.pi))%par['num_motion_dirs'])

            self.trial_info['neural_input'][neuron_ind, stim_onset[b]:stim_off, b] += np.reshape(self.circ_tuning(stim_dir),(-1,1))
            self.trial_info['desired_output'][target_ind, resp_onset[b]:, b] = 1
            self.trial_info['desired_output'][-1,:resp_onset[b], b] = 1
            self.trial_info['train_mask'][resp_onset[b]:resp_onset[b]+par['mask_duration']//par['dt'], b] = 0


        return self.trial_info


    def task_dm(self, variant='dm1'):

        # Create trial stimuli
        stim_dir1 = np.random.choice(self.motion_dirs, [1, par['batch_train_size']])
        stim_dir2 = (stim_dir1 + np.pi/2 + np.random.choice(self.motion_dirs[::2], [1, par['batch_train_size']])/2)%(2*np.pi)
        stim1 = self.circ_tuning(stim_dir1)
        stim2 = self.circ_tuning(stim_dir2)

        # Determine the strengths of the stimuli in each modality
        c_mod1 = np.random.choice(self.dm_c_set, [1, par['batch_train_size']])
        c_mod2 = np.random.choice(self.dm_c_set, [1, par['batch_train_size']])
        mean_gamma = 0.8 + 0.4*np.random.rand(1, par['batch_train_size'])
        gamma_s1_m1 = mean_gamma + c_mod1
        gamma_s2_m1 = mean_gamma - c_mod1
        gamma_s1_m2 = mean_gamma + c_mod2
        gamma_s2_m2 = mean_gamma - c_mod2

        # Determine response directions and convert to output indices
        resp_dir_mod1 = np.where(gamma_s1_m1 > gamma_s2_m1, stim_dir1, stim_dir2)
        resp_dir_mod2 = np.where(gamma_s1_m2 > gamma_s2_m2, stim_dir1, stim_dir2)
        resp_dir_sum  = np.where(gamma_s1_m1 + gamma_s1_m2 > gamma_s2_m1 + gamma_s2_m2, stim_dir1, stim_dir2)

        resp_dir_mod1 = np.round(par['num_motion_dirs']*resp_dir_mod1/(2*np.pi))
        resp_dir_mod2 = np.round(par['num_motion_dirs']*resp_dir_mod2/(2*np.pi))
        resp_dir_sum  = np.round(par['num_motion_dirs']*resp_dir_sum/(2*np.pi))

        # Apply stimuli to modalities and build appropriate response
        if variant == 'dm1':
            modality1 = gamma_s1_m1*stim1 + gamma_s2_m1*stim2
            modality2 = np.zeros_like(stim1)
            resp_dirs = resp_dir_mod1
        elif variant == 'dm2':
            modality1 = np.zeros_like(stim1)
            modality2 = gamma_s1_m2*stim1 + gamma_s2_m2*stim2
            resp_dirs = resp_dir_mod2
        elif variant == 'ctx_dm1':
            modality1 = gamma_s1_m1*stim1 + gamma_s2_m1*stim2
            modality2 = gamma_s1_m2*stim1 + gamma_s2_m2*stim2
            resp_dirs = resp_dir_mod1
        elif variant == 'ctx_dm2':
            modality1 = gamma_s1_m1*stim1 + gamma_s2_m1*stim2
            modality2 = gamma_s1_m2*stim1 + gamma_s2_m2*stim2
            resp_dirs = resp_dir_mod2
        elif variant == 'multsen_dm':
            modality1 = gamma_s1_m1*stim1 + gamma_s2_m1*stim2
            modality2 = gamma_s1_m2*stim1 + gamma_s2_m2*stim2
            resp_dirs = resp_dir_sum
        else:
            raise Exception('Bad task variant.')

        resp = np.zeros([par['num_motion_dirs'], par['batch_train_size']])
        for b in range(par['batch_train_size']):
            resp[np.int8(resp_dirs[0,b]%8),b] = 1

        # Setting up arrays
        fixation = np.zeros(self.fixation_shape)
        response = np.zeros(self.response_shape)
        stimulus = np.zeros(self.stimulus_shape)
        mask     = np.ones(self.mask_shape)
        mask[:par['mask_duration']//par['dt'],:] = 0

        # Identify stimulus onset for each trial and build each trial from there
        stim_onset = 500//par['dt']
        stim_off   = stim_onset + np.random.choice(self.dm_stim_lengths, par['batch_train_size'])
        resp_time  = stim_off + 500//par['dt']
        for b in range(par['batch_train_size']):
            fixation[:,:resp_time[b],b] = 1
            stimulus[:,stim_onset:stim_off[b],b] = np.concatenate([modality1[:,b], modality2[:,b]], axis=0)[:,np.newaxis]
            response[:,resp_time[b]:,b] = resp[:,b,np.newaxis]
            mask[resp_time[b]:resp_time[b]+par['mask_duration']//par['dt'],b] = 0

        # Tweak the fixation array
        stim_fix = fixation
        resp_fix = fixation[0:1,:,:]

        # Merge activies and fixations into single vector
        stimulus = np.concatenate([stimulus, stim_fix], axis=0)
        response = np.concatenate([response, resp_fix], axis=0)

        self.trial_info['neural_input'] += stimulus
        self.trial_info['desired_output'] = response
        self.trial_info['train_mask'] = mask

        return self.trial_info


    def task_dm_dly(self, variant='dm1'):

        # Create trial stimuli
        stim_dir1 = 2*np.pi*np.random.rand(1, par['batch_train_size'])
        stim_dir2 = (stim_dir1 + np.pi/2 + np.pi*np.random.rand(1, par['batch_train_size']))%(2*np.pi)
        stim1 = self.circ_tuning(stim_dir1)
        stim2 = self.circ_tuning(stim_dir2)

        # Determine the strengths of the stimuli in each modality
        c_mod1 = np.random.choice(self.dm_dly_c_set, [1, par['batch_train_size']])
        c_mod2 = np.random.choice(self.dm_dly_c_set, [1, par['batch_train_size']])
        mean_gamma = 0.8 + 0.4*np.random.rand(1, par['batch_train_size'])
        gamma_s1_m1 = mean_gamma + c_mod1
        gamma_s2_m1 = mean_gamma - c_mod1
        gamma_s1_m2 = mean_gamma + c_mod2
        gamma_s2_m2 = mean_gamma - c_mod2

        # Determine the delay for each trial
        delay = np.random.choice(self.dm_dly_delay, [1, par['batch_train_size']])

        # Determine response directions and convert to output indices
        resp_dir_mod1 = np.where(gamma_s1_m1 > gamma_s2_m1, stim_dir1, stim_dir2)
        resp_dir_mod2 = np.where(gamma_s1_m2 > gamma_s2_m2, stim_dir1, stim_dir2)
        resp_dir_sum  = np.where(gamma_s1_m1 + gamma_s1_m2 > gamma_s2_m1 + gamma_s2_m2, stim_dir1, stim_dir2)

        resp_dir_mod1 = np.round(par['num_motion_dirs']*resp_dir_mod1/(2*np.pi))
        resp_dir_mod2 = np.round(par['num_motion_dirs']*resp_dir_mod2/(2*np.pi))
        resp_dir_sum  = np.round(par['num_motion_dirs']*resp_dir_sum/(2*np.pi))

        # Apply stimuli to modalities and build appropriate response
        if variant == 'dm1_dly':
            modality1_t1 = gamma_s1_m1*stim1
            modality2_t1 = np.zeros_like(stim1)
            modality1_t2 = gamma_s2_m1*stim2
            modality2_t2 = np.zeros_like(stim2)
            resp_dirs = resp_dir_mod1
        elif variant == 'dm2_dly':
            modality1_t1 = np.zeros_like(stim1)
            modality2_t1 = gamma_s1_m2*stim1
            modality1_t2 = np.zeros_like(stim2)
            modality2_t2 = gamma_s2_m2*stim2
            resp_dirs = resp_dir_mod2
        elif variant == 'ctx_dm1_dly':
            modality1_t1 = gamma_s1_m1*stim1
            modality2_t1 = gamma_s1_m2*stim1
            modality1_t2 = gamma_s2_m1*stim2
            modality2_t2 = gamma_s2_m2*stim2
            resp_dirs = resp_dir_mod1
        elif variant == 'ctx_dm2_dly':
            modality1_t1 = gamma_s1_m1*stim1
            modality2_t1 = gamma_s1_m2*stim1
            modality1_t2 = gamma_s2_m1*stim2
            modality2_t2 = gamma_s2_m2*stim2
            resp_dirs = resp_dir_mod2
        elif variant == 'multsen_dm_dly':
            modality1_t1 = gamma_s1_m1*stim1
            modality2_t1 = gamma_s1_m2*stim1
            modality1_t2 = gamma_s2_m1*stim2
            modality2_t2 = gamma_s2_m2*stim2
            resp_dirs = resp_dir_sum
        else:
            raise Exception('Bad task variant.')

        resp = np.zeros([par['num_motion_dirs'], par['batch_train_size']])
        for b in range(par['batch_train_size']):
            resp[np.int8(resp_dirs[0,b]%8),b] = 1

        # Setting up arrays
        fixation = np.zeros(self.fixation_shape)
        response = np.zeros(self.response_shape)
        stimulus = np.zeros(self.stimulus_shape)
        mask     = np.ones(self.mask_shape)
        mask[:par['mask_duration']//par['dt'],:] = 0

        # Identify stimulus onset for each trial and build each trial from there
        stim_on1   = 500//par['dt']
        stim_off1  = (500+300)//par['dt']
        stim_on2   = delay + stim_off1
        stim_off2  = stim_on2 + 300//par['dt']
        resp_time  = stim_off2 + 300//par['dt']
        for b in range(par['batch_train_size']):
            fixation[:,:resp_time[0,b],b] = 1
            stimulus[:,stim_on1:stim_off1,b] = np.concatenate([modality1_t1[:,b], modality2_t1[:,b]], axis=0)[:,np.newaxis]
            stimulus[:,stim_on2[0,b]:stim_off2[0,b],b] = np.concatenate([modality1_t2[:,b], modality2_t2[:,b]], axis=0)[:,np.newaxis]
            response[:,resp_time[0,b]:,b] = resp[:,b,np.newaxis]
            mask[resp_time[0,b]:resp_time[0,b]+par['mask_duration'],b] = 0

        # Tweak the fixation array
        stim_fix = fixation
        resp_fix = fixation[0:1,:,:]

        # Merge activies and fixations into single vectors
        stimulus = np.concatenate([stimulus, stim_fix], axis=0)
        response = np.concatenate([response, resp_fix], axis=0)

        self.trial_info['neural_input'] += stimulus
        self.trial_info['desired_output'] = response
        self.trial_info['train_mask'] = mask

        return self.trial_info


    def task_matching(self, variant='dms'):

        # Determine matches, and get stimuli
        if variant in ['dms', 'dnms']:
            stim1 = np.random.choice(self.motion_dirs, par['batch_train_size'])
            nonmatch = (stim1 + np.random.choice(self.motion_dirs[1:], par['batch_train_size']))%(2*np.pi)

            match = np.random.choice(np.array([True, False]), par['batch_train_size'])
            stim2 = np.where(match, stim1, nonmatch)

        elif variant in ['dmc', 'dnmc']:
            stim1 = np.random.choice(self.motion_dirs, par['batch_train_size'])
            stim2 = np.random.choice(self.motion_dirs, par['batch_train_size'])

            stim1_cat = np.logical_and(np.less(0, stim1), np.less(stim1, np.pi))
            stim2_cat = np.logical_and(np.less(0, stim2), np.less(stim2, np.pi))
            match = np.logical_not(np.logical_xor(stim1_cat, stim2_cat))
        else:
            raise Exception('Bad variant.')

        # Establishing stimuli
        stimulus1 = self.circ_tuning(stim1)
        stimulus2 = self.circ_tuning(stim2)

        # Convert to response
        stim1_int = np.round(par['num_motion_dirs']*stim1/(2*np.pi))
        stim2_int = np.round(par['num_motion_dirs']*stim2/(2*np.pi))

        if variant in ['dms', 'dmc']:
            resp = np.where(match, stim1_int, -1)
        elif variant in ['dnms', 'dnmc']:
            resp = np.where(match, -1, stim2_int)
        else:
            raise Exception('Bad variant.')

        # Setting up arrays
        modality_choice = np.random.choice(np.array([0,1], dtype=np.int8), [2, par['batch_train_size']])
        modalities = np.zeros([2, par['num_motion_tuned']//2, par['num_time_steps'], par['batch_train_size']])
        fixation = np.zeros(self.fixation_shape)
        response = np.zeros(self.response_shape)
        stimulus = np.zeros(self.stimulus_shape)
        mask     = np.ones(self.mask_shape)
        mask[:par['mask_duration']//par['dt'],:] = 0

        # Decide timings and build each trial
        stim1_on  = 300//par['dt']
        stim1_off = 600//par['dt']
        stim2_on  = stim1_off + np.random.choice(self.match_delay, par['batch_train_size'])
        stim2_off = stim2_on + 300//par['dt']
        resp_time = stim2_off
        resp_fix  = np.copy(fixation[0:1,:,:])
        for b in range(par['batch_train_size']):
            fixation[:,:resp_time[b],b] = 1
            modalities[modality_choice[0,b],:,stim1_on:stim1_off,b] = stimulus1[:,b,np.newaxis]
            modalities[modality_choice[1,b],:,stim2_on[b]:stim2_off[b],b] = stimulus2[:,b,np.newaxis]
            mask[resp_time[b]:resp_time[b]+par['mask_duration']//par['dt'],b] = 0
            if not resp[b] == -1:
                response[int(resp[b]),resp_time[b]:,b] = 1
            else:
                resp_fix[:,:,b] = 1

        # Tweak the fixation array
        stim_fix = fixation
        resp_fix = np.maximum(resp_fix, fixation[0:1,:,:])

        # Merge activies and fixations into single vectors)
        stimulus = np.concatenate([modalities[0], modalities[1], stim_fix], axis=0)
        response = np.concatenate([response, resp_fix], axis=0)

        self.trial_info['neural_input'] += stimulus
        self.trial_info['desired_output'] = response
        self.trial_info['train_mask'] = mask

        return self.trial_info

### EXAMPLE ###
"""
st = MultiStimulus()
for i in range(len(st.task_types)):
    print(i, st.task_types[i][1:])
    t, trial_info = st.generate_trial(i)

    s = trial_info['neural_input']
    r = trial_info['desired_output']
    m = trial_info['train_mask']

    print(t)
    print(s.shape)
    print(r.shape)
    print(m.shape)

    fig, axarr = plt.subplots(8, 3, sharex=True, sharey=True)
    for i in range(8):
        axarr[i,0].imshow(s[:,:,i])
        axarr[i,1].imshow(r[:,:,i])
        axarr[i,2].plot(m[:,i])

    plt.show()
quit()
"""

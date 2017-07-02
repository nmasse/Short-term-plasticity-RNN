import numpy as np
import matplotlib.pyplot as plt


class Stimulus:

    def __init__(self, params):

        # params is a dictionary containing all the trial parameters
        for key, value in params.items():
            setattr(self, key, value)

        self.num_rules = len(self.possible_rules)
        self.stimulus_type = 'motion'

        if self.stimulus_type == 'spatial':
            self.neural_tuning = self.create_spatial_tuning_function()
        elif self.stimulus_type == 'motion':
            if self.possible_rules[0] == 5:
                self.stim_tuning, self.fix_tuning, self.rule_tuning = self.create_tuning_functions_postle()
            else:
                self.motion_tuning, self.fix_tuning, self.rule_tuning = self.create_motion_tuning_function()


    def generate_trial(self, trials_per_batch=64, var_delay=True):


        """
        DMS: rule = 0
        DMrS (clockwise): rule = 1
        DMrS (counter-clockwise): rule = 2
        DMC: rule = 3
        ABBA: rule = 4
        Postle: rule = 5
        Can't mix ABBA with other rule types
        """

        if self.stimulus_type == 'spatial':
            # CURRENTLY NOT IS USE, NEEDS MAJOR WORK
            self.neural_tuning = self.create_spatial_tuning_function(trials_per_batch)
        elif self.stimulus_type == 'motion' and self.possible_rules[0] < 4:
            trial_info = self.generate_motion_working_memory_trial(trials_per_batch=trials_per_batch, var_delay=var_delay)
        elif self.stimulus_type == 'motion' and self.possible_rules[0] == 4:
            trial_info = self.generate_ABBA_trial(trials_per_batch=trials_per_batch)
        elif self.stimulus_type == 'motion' and self.possible_rules[0] == 5:
            trial_info = self.generate_postle_trial(trials_per_batch=trials_per_batch)

        return trial_info


    def generate_postle_trial(self, trials_per_batch):

        """
        Generate a trial based on "Reactivation of latent working memories with transcranial magnetic stimulation"

        Trial outline
        1. Dead period
        2. Fixation
        3. Two sample stimuli presented
        4. Delay (cue in middle, and possibly probe later)
        5. Test stimulus (to cued modality, match or non-match)
        6. Delay (cue in middle, and possibly probe later)
        7. Test stimulus

        INPUTS:
        1. sample_time (duration of sample stimlulus)
        2. test_time
        3. delay_time
        4. cue_time (duration of rule cue, always presented halfway during delay)
        5. probe_time (usually set to one time step, always presented 3/4 through delay


        """

        # end of trial epochs
        eodead = self.dead_time//self.dt
        eof = (self.dead_time+self.fix_time)//self.dt
        eos = (self.dead_time+self.fix_time+self.sample_time)//self.dt
        eod1 = (self.dead_time+self.fix_time+self.sample_time+self.delay_time)//self.dt
        eot1 = (self.dead_time+self.fix_time+self.sample_time+self.delay_time+self.test_time)//self.dt
        eod2 = (self.dead_time+self.fix_time+self.sample_time+2*self.delay_time+self.test_time)//self.dt
        trial_length = (self.dead_time+self.fix_time+self.sample_time+2*self.delay_time+2*self.test_time)//self.dt

        cue_time1 = (self.dead_time+self.fix_time+self.sample_time+self.delay_time//2)//self.dt
        cue_time2 = (self.dead_time+self.fix_time+self.sample_time+3*self.delay_time//2+self.test_time)//self.dt

        # probe_time1 will be right before the first test stimulus
        probe_time1 = (self.dead_time+self.fix_time+self.sample_time+9*self.delay_time//10)//self.dt
        # probe_time2 will be after first test stimulus, but before the second cue signal
        probe_time2 = (self.dead_time+self.fix_time+self.sample_time+14*self.delay_time//10+self.test_time)//self.dt

        # end of neuron indices
        est = self.num_motion_tuned
        ert = self.num_motion_tuned+self.num_rule_tuned
        eft = self.num_motion_tuned+self.num_rule_tuned+self.num_fix_tuned

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



        for t in range(trials_per_batch):

            # generate sample, match, rule and prob params
            for i in range(2):
                trial_info['sample'][t,i] = np.random.randint(self.num_motion_dirs)
                trial_info['match'][t,i] = np.random.randint(2)
                trial_info['rule'][t,i] = np.random.randint(2)
                trial_info['catch'][t,i] = np.random.rand() < self.catch_trial_pct
                if i == 1:
                    # only generate a pulse during 2nd delay epoch
                    trial_info['probe'][t,i] = np.random.rand() < self.probe_trial_pct


            # determine test stimulu based on sample and match status
            for i in range(2):

                # if trial is not a catch, the upcoming test modality (what the network should be attending to)
                # is given by the rule cue
                if not trial_info['catch'][t,i]:
                    trial_info['test_mod'][t,i] = trial_info['rule'][t,i]
                else:
                    trial_info['test_mod'][t,i] = (trial_info['rule'][t,i]+1)%2

                # cued test stimulus
                if trial_info['match'][t,i] == 1:
                    trial_info['test'][t,i,0] = trial_info['sample'][t,trial_info['test_mod'][t,i]]
                else:
                    sample = trial_info['sample'][t,trial_info['test_mod'][t,i]]
                    #bad_directions = [(i+sample+self.num_motion_dirs//2)%self.num_motion_dirs for i in range(1)]
                    #bad_directions.append(sample_dir)
                    bad_directions = [sample]
                    #possible_stim = np.setdiff1d(list(range(self.num_stim)), sample)
                    possible_stim = np.setdiff1d(list(range(self.num_motion_dirs)), bad_directions)
                    trial_info['test'][t,i,0] = possible_stim[np.random.randint(len(possible_stim))]

                # non-cued test stimulus
                trial_info['test'][t,i,1] = np.random.randint(self.num_motion_dirs)


            """
            Calculate input neural activity based on trial params
            """
            # SAMPLE stimuli
            trial_info['neural_input'][:est, eof:eos, t] += np.reshape(self.stim_tuning[:,0,trial_info['sample'][t,0]],(-1,1))
            trial_info['neural_input'][:est, eof:eos, t] += np.reshape(self.stim_tuning[:,1,trial_info['sample'][t,1]],(-1,1))

            # Cued TEST stimuli
            trial_info['neural_input'][:est, eod1:eot1, t] += np.reshape(self.stim_tuning[:,trial_info['test_mod'][t,0],trial_info['test'][t,0,0]],(-1,1))
            trial_info['neural_input'][:est, eod2:trial_length, t] += np.reshape(self.stim_tuning[:,trial_info['test_mod'][t,1],trial_info['test'][t,1,0]],(-1,1))

            # Non-cued TEST stimuli
            trial_info['neural_input'][:est, eod1:eot1, t] += np.reshape(self.stim_tuning[:,(1+trial_info['test_mod'][t,0])%2,trial_info['test'][t,0,1]],(-1,1))
            trial_info['neural_input'][:est, eod2:trial_length, t] += np.reshape(self.stim_tuning[:,(1+trial_info['test_mod'][t,1])%2,trial_info['test'][t,1,1]],(-1,1))


            # FIXATION
            trial_info['neural_input'][ert:eft,eodead:eod1,t] += np.reshape(self.fix_tuning[:,0],(-1,1)) #ON
            trial_info['neural_input'][ert:eft,eod1:eot1,t] += np.reshape(self.fix_tuning[:,1],(-1,1)) #OFF
            trial_info['neural_input'][ert:eft,eot1:eod2,t] += np.reshape(self.fix_tuning[:,0],(-1,1)) #ON
            trial_info['neural_input'][ert:eft,eod2:trial_length,t] += np.reshape(self.fix_tuning[:,1],(-1,1)) #OFF

            # RULE CUE
            trial_info['neural_input'][est:ert,cue_time1:eot1,t] += np.reshape(self.rule_tuning[:,trial_info['rule'][t,0]],(-1,1))
            trial_info['neural_input'][est:ert,cue_time2:trial_length,t] += np.reshape(self.rule_tuning[:,trial_info['rule'][t,1]],(-1,1))

            # PROBE
            # increase reponse of all stim tuned neurons by 10
            if trial_info['probe'][t,0]:
                trial_info['neural_input'][:est,probe_time1,t] += 10
            if trial_info['probe'][t,1]:
                trial_info['neural_input'][:est,probe_time2,t] += 10


            """
            Desired outputs
            """
            # FIXATION
            trial_info['desired_output'][0,:eod1, t] = 1
            trial_info['desired_output'][0,eot1:eod2, t] = 1
            # TEST 1
            if trial_info['match'][t,0] == 1:
                trial_info['desired_output'][2,eod1:eot1, t] = 1
            else:
                trial_info['desired_output'][1,eod1:eot1, t] = 1
            # TEST 2
            if trial_info['match'][t,1] == 1:
                trial_info['desired_output'][2,eod2:trial_length, t] = 1
            else:
                trial_info['desired_output'][1,eod2:trial_length, t] = 1

            # set to mask equal to zero during the dead time, and during the first times of test stimuli
            trial_info['train_mask'][:eodead, t] = 0
            trial_info['train_mask'][eod1, t] = 0
            trial_info['train_mask'][eod2, t] = 0

        #self.plot_neural_input(trial_info)
        #1/0

        return trial_info


    def generate_spatial_working_memory_trial(self, num_batches, trials_per_batch):

        """
        Generate a delayed match to sample spatial based task
        Goal is to determine whether the location of two stimuli, separated by a delay, are a match
        """

        num_classes = 2
        trial_length = self.fix_time + self.sample_time + self.delay_time + self.test_time
        sample_epoch = range(self.fix_time, self.fix_time + self.sample_time)
        test_epoch  = range(self.fix_time + self.sample_time + self.delay_time, trial_length)

        trial_info = {'desired_output' :  np.zeros((num_batches,num_classes, trial_length, trials_per_batch),dtype=np.float32),
                      'sample_location':  np.zeros((num_batches,trials_per_batch, 2),dtype=np.float32),
                      'test_location'  :  np.zeros((num_batches,trials_per_batch, 2),dtype=np.float32),
                      'match'          :  np.zeros((num_batches,trials_per_batch),dtype=np.int8),
                      'neural_input'   :  np.random.normal(self.input_mean, self.input_sd, size=(num_batches, self.num_neurons, trial_length, trials_per_batch))}


        for b in range(num_batches):
            for t in range(trials_per_batch):
                sample_x_loc = np.int_(np.floor(np.random.rand()*self.size_x))
                sample_y_loc = np.int_(np.floor(np.random.rand()*self.size_y))
                neural_multiplier = np.tile(1 + np.reshape(self.neural_tuning[:,0,sample_x_loc,sample_y_loc],(self.num_neurons,1)),(1,self.sample_time))
                trial_info['neural_input'][b, :, sample_epoch, t] *= neural_multiplier

                # generate test location
                if np.random.randint(2)==0:
                    # match trial
                    test_x_loc = sample_x_loc
                    test_y_loc = sample_y_loc
                    trial_info['match'][b,t] = 1
                    trial_info['desired_output'][b,0,test_epoch,t] = 1
                else:
                    d = 0
                    count = 0
                    trial_info['desired_output'][b,1,test_epoch,t] = 1
                    while d < self.non_match_separation:
                        test_x_loc = np.floor(np.random.rand()*self.size_x)
                        test_y_loc = np.floor(np.random.rand()*self.size_y)
                        d = np.sqrt((test_x_loc-sample_x_loc)**2 + (test_y_loc-sample_y_loc)**2)
                        count += 1
                        if count > 100:
                            print('Having trouble finding a test stimulus location. Consider decreasing non_match_separation')

                test_x_loc = np.int_(test_x_loc)
                test_y_loc = np.int_(test_y_loc)
                neural_multiplier = np.tile(1 + np.reshape(self.neural_tuning[:,0,test_x_loc,test_y_loc],(self.num_neurons,1)),(1,self.test_time))
                trial_info['neural_input'][:, test_epoch] *= neural_multiplier

                trial_info['sample_location'][b,t,0] = sample_x_loc
                trial_info['sample_location'][b,t,1] = sample_y_loc
                trial_info['test_location'][b,t,0] = test_x_loc
                trial_info['test_location'][b,t,1] = test_y_loc

        return trial_info



    def generate_spatial_cat_trial(self, num_batches, trials_per_batch):

        """
        Generate a spatial categorization task
        Goal is to determine whether the color and the location of a series of flashes fall
        inside color specific target zones
        """

        num_classes = 2
        trial_length = self.fix_time + self.num_flashes*self.cycle_time
        trial_info = {'desired_output' : np.zeros((num_batches,num_classes, trial_length, trials_per_batch),dtype=np.float32),
                      'flash_x_pos'    : np.random.randint(self.size_x, size=[num_batches, trials_per_batch, self.num_flashes]),
                      'flash_y_pos'    : np.random.randint(self.size_y, size=[num_batches, trials_per_batch, self.num_flashes]),
                      'flash_color'    : np.random.randint(self.num_colors, size=[num_batches, trials_per_batch, self.num_flashes]),
                      'flash_zone'     : np.zeros((num_batches, trials_per_batch, self.num_flashes), dtype=np.int8),
                      'neural_input'   : np.random.normal(self.input_mean, self.input_sd, size=(num_batches, self.num_neurons, trial_length, trials_per_batch))}

        # to start, assing all time points "non-target flash"
        trial_info['desired_output'][:,0,:,:] = 1

        for b in range(num_batches):
            for t in range(trials_per_batch):
                for f in range(self.num_flashes):
                    x_pos = trial_info['flash_x_pos'][b,t,f]
                    y_pos = trial_info['flash_y_pos'][b,t,f]
                    color = trial_info['flash_color'][b,t,f]
                    zone = self.get_category_zone(x_pos, y_pos)
                    trial_info['flash_zone'][b,t,f] = zone

                    # modify neural input according to flash stimulus
                    flash_time_ind = range(self.fix_time+f*self.cycle_time+1, self.fix_time+f*self.cycle_time+self.flash_time)
                    neural_multiplier = np.tile(1 + np.reshape(self.neural_tuning[:,color,x_pos,y_pos],(1,self.num_neurons)),(len(flash_time_ind),1))
                    #print(neural_multiplier.shape)
                    #print(trial_info['neural_input'][b, :, flash_time_ind, t].shape)
                    trial_info['neural_input'][b, :, flash_time_ind, t] *= neural_multiplier
                    cycle_time_ind = range(self.fix_time+f*self.cycle_time+1, self.fix_time+(f+1)*self.cycle_time)

                    # generate desired output based on whether the flash fell inside or outside a target zone
                    if zone == color:
                        # target flash
                        trial_info['desired_output'][b,0,cycle_time_ind,t] = 0
                        trial_info['desired_output'][b,1,cycle_time_ind,t] = 1

        #plt.imshow(trial_info['neural_input'][b, :, :, t])
        #plt.show()
        return trial_info



    @staticmethod
    def get_category_zone(x_pos, y_pos):

        """
        Used in association with generate_spatial_cat_trial to determine
        the spatial "zone" of each flash
        """

        if x_pos>=2 and x_pos<=4 and y_pos>=2 and y_pos<=4:
            return 1
        elif x_pos>=2 and x_pos<=4 and y_pos>=5 and y_pos<=7:
            return 2
        else:
            return -1


    def generate_motion_working_memory_trial(self, trials_per_batch=64, var_delay = False):

        """
        Generate a delayed matching task
        Goal is to determine whether the sample stimulus, possibly manipulated by a rule, is
        identicical to a test stimulus
        Sample and test stimuli are separated by a delay
        Rule = 0,1,2, or 3
        """

        # range of variable delay, in time steps
        var_delay_max = self.variable_delay_max//self.dt

        # rule signal can appear at the end of delay1_time
        trial_length = self.num_time_steps

        # end of trial epochs
        eodead = self.dead_time//self.dt
        eof = (self.dead_time+self.fix_time)//self.dt
        eos = (self.dead_time+self.fix_time+self.sample_time)//self.dt
        eod = (self.dead_time+self.fix_time+self.sample_time+self.delay_time)//self.dt
        probe_time = self.probe_time//self.dt

        # end of neuron indices
        emt = self.num_motion_tuned
        eft = self.num_fix_tuned+self.num_motion_tuned
        ert = self.num_fix_tuned+self.num_motion_tuned + self.num_rule_tuned
        self.num_input_neurons = ert

        trial_info = {'desired_output'  :  np.zeros((self.n_output, trial_length, trials_per_batch),dtype=np.float32),
                      'train_mask'      :  np.ones((trial_length, trials_per_batch),dtype=np.float32),
                      'sample'          :  np.zeros((trials_per_batch),dtype=np.float32),
                      'test'            :  np.zeros((trials_per_batch),dtype=np.float32),
                      'rule'            :  np.zeros((trials_per_batch),dtype=np.int8),
                      'match'           :  np.zeros((trials_per_batch),dtype=np.int8),
                      'catch'           :  np.zeros((trials_per_batch),dtype=np.int8),
                      'probe'           :  np.zeros((trials_per_batch),dtype=np.int8),
                      'neural_input'    :  np.random.normal(self.input_mean, self.input_sd, size=(self.n_input, trial_length, trials_per_batch))}


        # set to mask equal to zero during the dead time
        trial_info['train_mask'][:eodead, :] = 0

        # If the DMS and DMS rotate are being performed together,
        # or if I need to make the test more challenging, this will eliminate easry test directions
        # If so, reduce set of test stimuli so that a single strategy can't be used
        #limit_test_directions = self.possible_rules==[0,1] or self.possible_rules==[5]

        for t in range(trials_per_batch):

            """
            Generate trial paramaters
            """
            sample_dir = np.random.randint(self.num_motion_dirs)
            rule_ind = np.random.randint(len(self.possible_rules))
            rule = self.possible_rules[rule_ind]
            match = np.random.randint(2)
            catch = np.random.rand() < self.catch_trial_pct
            probe = np.random.rand() < self.probe_trial_pct

            """
            Determine the delay time for this trial
            The total trial length is kept constant, so a shorter delay implies a longer test stimulus
            """
            if var_delay:
                s = np.int_(np.random.exponential(scale=self.varialbe_delay_scale))
                if s <= var_delay_max:
                    eod_current = eod - var_delay_max + s
                else:
                    eod_current = eod
                    catch = 1
            else:
                eod_current = eod

            # set mask to zero during transition from delay to test
            trial_info['train_mask'][eod_current, t] = 0

            """
            Generate the sample and test stimuli based on the rule
            """
            # DMS
            if rule == 0:
                if match == 1: # match trial
                    test_dir = sample_dir
                else:
                    possible_dirs = np.setdiff1d(list(range(self.num_motion_dirs)), sample_dir)
                    test_dir = possible_dirs[np.random.randint(len(possible_dirs))]

            # DMrS (clockwise)
            elif rule == 1: # rotate sample +90 degs
                matching_dir = (sample_dir+self.num_motion_dirs//4)%self.num_motion_dirs
                if match == 1: # match trial
                    test_dir = matching_dir
                else:
                    possible_dirs = np.setdiff1d(list(range(self.num_motion_dirs)), matching_dir)
                    test_dir = possible_dirs[np.random.randint(len(possible_dirs))]

            # DMrS (counter-clockwise)
            elif rule == 2: # rotate sample -90 degs
                matching_dir = (sample_dir-self.num_motion_dirs//4)%self.num_motion_dirs
                if match == 1: # match trial
                    test_dir = matching_dir
                else:
                    possible_dirs = np.setdiff1d(list(range(self.num_motion_dirs)), matching_dir)
                    test_dir = possible_dirs[np.random.randint(len(possible_dirs))]

            # DMC
            elif rule == 3: # categorize between two equal size, contiguous zones
                sample_cat = np.floor(sample_dir/(self.num_motion_dirs/2))
                if match == 1: # match trial
                    # do not use sample_dir as a match test stimulus
                    dir0 = int(sample_cat*self.num_motion_dirs//2)
                    dir1 = int(self.num_motion_dirs//2 + sample_cat*self.num_motion_dirs//2)
                    possible_dirs = np.setdiff1d(list(range(dir0, dir1)), sample_dir)
                    test_dir = possible_dirs[np.random.randint(len(possible_dirs))]
                else:
                    test_dir = sample_cat*(self.num_motion_dirs//2) + np.random.randint(self.num_motion_dirs//2)
                    test_dir = np.int_((test_dir+self.num_motion_dirs//2)%self.num_motion_dirs)


            """
            Calculate neural input based on sample, tests, fixation, rule, and probe
            """
            # SAMPLE stimulus
            trial_info['neural_input'][:emt, eof:eos, t] += np.reshape(self.motion_tuning[:,sample_dir],(-1,1))

            # TEST stimulus
            if not catch:
                trial_info['neural_input'][:emt, eod_current:, t] += np.reshape(self.motion_tuning[:,test_dir],(-1,1))

            # PROBE: increase activity by 1 for all motion tuned neurons
            if probe:
                trial_info['neural_input'][probe_time, t, :emt] += 1

            # FIXATION cue
            if self.num_fix_tuned > 0:
                trial_info['neural_input'][emt:eft, eodead:eod_current, t] += np.reshape(self.fix_tuning[:,0],(-1,1))
                trial_info['neural_input'][emt:eft, eod_current:trial_length, t] += np.reshape(self.fix_tuning[:,1],(-1,1))

            # RULE CUE
            if len(self.possible_rules) > 1 and self.num_rule_tuned > 0:
                trial_info['neural_input'][eft:ert, self.rule_onset_time:self.rule_offset_time, t] += np.reshape(self.rule_tuning[:,rule_ind],(-1,1))

            """
            Determine the desired network output response
            """
            trial_info['desired_output'][0, eodead:eod_current, t] = 1
            if not catch:
                if match == 0:
                    trial_info['desired_output'][1, eod_current:, t] = 1
                else:
                    trial_info['desired_output'][2, eod_current:, t] = 1
            else:
                trial_info['desired_output'][0, eod_current:, t] = 1


            """
            Append trial info
            """
            trial_info['sample'][t] = sample_dir
            trial_info['test'][t] = test_dir
            trial_info['rule'][t] = rule
            trial_info['catch'][t] = catch
            trial_info['match'][t] = match


        # debugging: plot the neural input activity
        """
        plot_neural_input(trial_info)
        """

        return trial_info



    def generate_ABBA_trial(self, trials_per_batch=64):

        """
        Generate ABBA trials
        Sample stimulis is followed by up to max_num_tests test stimuli
        Goal is to to indicate when a test stimulus matches the sample
        Rule = 4
        """

        trial_length = self.num_time_steps
        ABBA_delay = self.ABBA_delay//self.dt

        # end of trial epochs
        eodead = self.dead_time//self.dt
        eof = (self.dead_time+self.fix_time)//self.dt
        eos = eof + ABBA_delay

        # end of neuron indices
        emt = self.num_motion_tuned
        eft = self.num_fix_tuned+self.num_motion_tuned
        ert = self.num_fix_tuned+self.num_motion_tuned + self.num_rule_tuned
        self.num_input_neurons = ert

        trial_info = {'desired_output'  :  np.zeros((self.n_output, trial_length, trials_per_batch),dtype=np.float32),
                      'train_mask'      :  np.ones((trial_length, trials_per_batch),dtype=np.float32),
                      'sample'          :  np.zeros((trials_per_batch),dtype=np.float32),
                      'test'            :  -1*np.ones((trials_per_batch,self.max_num_tests),dtype=np.float32),
                      'rule'            :  4*np.ones((trials_per_batch),dtype=np.int8),
                      'match'           :  np.zeros((trials_per_batch),dtype=np.int8),
                      'catch'           :  np.zeros((trials_per_batch),dtype=np.int8),
                      'probe'           :  np.zeros((trials_per_batch),dtype=np.int8),
                      'num_test_stim'   :  np.zeros((trials_per_batch),dtype=np.int8),
                      'repeat_test_stim':  np.zeros((trials_per_batch),dtype=np.int8),
                      'test_stim_code'  :  np.zeros((trials_per_batch),dtype=np.int32),
                      'neural_input'    :  np.random.normal(self.input_mean, self.input_sd, size=(self.n_input, trial_length, trials_per_batch))}


        # set to mask equal to zero during the dead time
        trial_info['train_mask'][:eodead, :] = 0

        # set fixation equal to 1 for all times; will then change
        trial_info['desired_output'][0, :, :] = 1

        for t in range(trials_per_batch):

            # generate trial params
            sample_dir = np.random.randint(self.num_motion_dirs)

            """
            Generate up to max_num_tests test stimuli
            Sequential test stimuli are identical with probability repeat_pct
            """
            stim_dirs = []
            test_stim_code = 0
            while len(stim_dirs) < self.max_num_tests:
                if np.random.rand() < self.match_test_prob:
                    stim_dirs.append(sample_dir)
                    test_stim_code += sample_dir*(10**len(stim_dirs)-1)
                    break
                else:
                    if len(stim_dirs) > 0  and np.random.rand() < self.repeat_pct:
                        #repeat last stimulus
                        stim_dirs.append(stim_dirs[-1])
                        trial_info['repeat_test_stim'][t] = 1
                        test_stim_code += stim_dirs[-1]*(10**len(stim_dirs)-1)
                    else:
                        possible_dirs = np.setdiff1d(list(range(self.num_motion_dirs)), [sample_dir])
                        distractor_dir = possible_dirs[np.random.randint(self.num_motion_dirs-1)]
                        stim_dirs.append(distractor_dir)
                        test_stim_code += distractor_dir*(10**len(stim_dirs)-1)

            trial_info['num_test_stim'][t] = len(stim_dirs)

            """
            Calculate input neural activity based on trial params
            """
            # SAMPLE stimuli
            trial_info['neural_input'][:emt, eof:eos, t] += np.reshape(self.motion_tuning[:,sample_dir],(-1,1))

            # TEST stimuli
            trial_info['catch'][t] = 1
            for i, stim_dir in enumerate(stim_dirs):
                trial_info['test'][t,i] = stim_dir
                test_rng = range(eos+(2*i+1)*ABBA_delay, eos+(2*i+2)*ABBA_delay)
                trial_info['neural_input'][:emt, test_rng, t] += np.reshape(self.motion_tuning[:,stim_dir],(-1,1))
                trial_info['train_mask'][eos+(2*i+1)*ABBA_delay, t] = 0
                trial_info['desired_output'][0, test_rng, t] = 0
                if stim_dir == sample_dir:
                    trial_info['desired_output'][2, test_rng, t] = 1
                    trial_info['match'][t] = 1
                    trial_info['catch'][t] = 0
                    trial_info['test_stim_code'][t] = test_stim_code
                    trial_info['train_mask'][eos+(2*i+2)*ABBA_delay:, t] = 0
                else:
                    trial_info['desired_output'][1, test_rng, t] = 1

            trial_info['sample'][t] = sample_dir

        # debugging: plot the neural input activity
        #self.plot_neural_input(trial_info)

        return trial_info



    def create_tuning_functions_postle(self):

        """
        Generate tuning functions for the Postle task
        """
        stim_tuning = np.zeros((self.num_motion_tuned, 2, self.num_motion_dirs))
        fix_tuning = np.zeros((self.num_fix_tuned, 2))
        rule_tuning = np.zeros((self.num_rule_tuned, 2))
        scale = 2

        # generate list of prefered directions
        # dividing neurons by 2 since two equal groups representing two modalities
        pref_dirs = np.float32(np.arange(0,360,360/(self.num_motion_tuned//2)))

        # generate list of possible stimulus directions
        stim_dirs = np.float32(np.arange(0,360,360/self.num_motion_dirs))

        for n in range(self.num_motion_tuned//2):
            for i in range(len(stim_dirs)):
                d = np.cos((stim_dirs[i] - pref_dirs[n])/180*np.pi)
                stim_tuning[n,0,i] = scale*(np.exp(scale*d)-np.exp(0))/np.exp(scale)
                stim_tuning[n,1,i] = 0
                stim_tuning[n+self.num_motion_tuned//2,0,i] = 0
                stim_tuning[n+self.num_motion_tuned//2,1,i] = scale*(np.exp(scale*d)-np.exp(0))/np.exp(scale)

        for n in range(self.num_fix_tuned):
            for i in range(2):
                if n%2 == i:
                    fix_tuning[n,i] = scale

        for n in range(self.num_rule_tuned):
            for i in range(2):
                if n%2 == i:
                    rule_tuning[n,i] = scale


        return stim_tuning, fix_tuning, rule_tuning


    def create_motion_tuning_function(self):

        """
        Generate tuning functions for the input neurons
        Here, neurons are either selective for motion direction, fixation, or the rule
        """
        motion_tuning = np.zeros((self.num_motion_tuned, self.num_motion_dirs))
        fix_tuning = np.zeros((self.num_fix_tuned, 2))
        rule_tuning = np.zeros((self.num_rule_tuned, self.num_rules))
        scale = 2

        # generate list of prefered motion directions
        #delta_pref_dir = int(360/self.num_motion_tuned)
        #pref_dirs = list(range(0,360,delta_pref_dir))
        pref_dirs = np.float32(np.arange(0,360,360/self.num_motion_tuned))

        #  generate list of possible motion directions
        #delta_motion_dir = int(360/self.num_motion_dirs)
        #motion_dirs = list(range(0,360,delta_motion_dir))
        motion_dirs = np.float32(np.arange(0,360,360/self.num_motion_dirs))

        for n in range(self.num_motion_tuned):
            for i in range(len(motion_dirs)):
                # cosine tuning
                #motion_tuning[n,i] = scale*np.cos((motion_dirs[i] - pref_dirs[n])/180*np.pi)
                # gaussian tuning
                d = np.cos((motion_dirs[i] - pref_dirs[n])/180*np.pi)
                motion_tuning[n,i] = scale*(np.exp(scale*d)-np.exp(-scale))/np.exp(scale)

        for n in range(self.num_fix_tuned):
            for i in range(2):
                if n%2 == i:
                    fix_tuning[n,i] = scale

        for n in range(self.num_rule_tuned):
            for i in range(self.num_rules):
                if n%self.num_rules == i:
                    rule_tuning[n,i] = scale


        return motion_tuning, fix_tuning, rule_tuning


    def create_spatial_tuning_function(self):

        """
        Generate tuning functions for the input neurons
        Here, neurons are either spatial or color selective, but not both
        """

        neural_tuning = np.zeros((self.num_neurons, self.num_colors, self.size_x, self.size_y))
        for n in range(self.num_spatial_tuned):
            pref_x = np.random.rand()*self.size_x
            pref_y = np.random.rand()*self.size_y
            for x in range(self.size_x):
                for y in range(self.size_y):
                    d = (x-pref_x)**2+(y-pref_y)**2
                    neural_tuning[n,:,x,y] = np.exp(-d/(2*self.spatial_tuning_sd**2))

        for n in range(self.num_color_tuned):
            for c in range(self.num_colors):
                if n%self.num_colors == c:
                    neural_tuning[n+self.num_spatial_tuned,c,:,:] = 0.5

        return neural_tuning


    def plot_neural_input(self, trial_info):

        print(trial_info['desired_output'][ :, 0, :].T)
        f = plt.figure(figsize=(8,4))
        ax = f.add_subplot(1, 1, 1)
        t = np.arange(0,400+500+2000,self.dt)
        t -= 900
        t0,t1,t2,t3 = np.where(t==-500), np.where(t==0),np.where(t==500),np.where(t==1500)
        #im = ax.imshow(trial_info['neural_input'][:,0,:].T, aspect='auto', interpolation='none')
        im = ax.imshow(trial_info['neural_input'][:,:,0], aspect='auto', interpolation='none')
        #plt.imshow(trial_info['desired_output'][:, :, 0], aspect='auto')
        ax.set_xticks([t0[0], t1[0], t2[0], t3[0]])
        ax.set_xticklabels([-500,0,500,1500])
        ax.set_yticks([0, 9, 18, 27])
        ax.set_yticklabels([0,90,180,270])
        f.colorbar(im,orientation='vertical')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('Motion direction')
        ax.set_xlabel('Time relative to sample onset (ms)')
        ax.set_title('Motion input')
        plt.show()
        plt.savefig('stimulus.pdf', format='pdf')
        print(trial_info['num_distract'][0])

        """
        f = plt.figure(figsize=(9,4))
        ax = f.add_subplot(1, 3, 1)
        ax.imshow(trial_info['sample_dir'],interpolation='none',aspect='auto')
        ax = f.add_subplot(1, 3, 2)
        ax.imshow(trial_info['test_dir'],interpolation='none',aspect='auto')
        ax = f.add_subplot(1, 3, 3)
        ax.imshow(trial_info['match'],interpolation='none',aspect='auto')
        plt.show()
        """

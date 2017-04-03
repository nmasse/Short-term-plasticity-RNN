import numpy as np
import matplotlib.pyplot as plt

class spatial_stimulus:
    
    """
    This class allows one to generate trials of different spatial based tasks
    Will produce the neural input, desired output, and all trial events
    """
    
    def __init__(self, trial_params):
        
        # trial_params is a dictionary containing all the trial parameters
        # should include size_x, size_y, num_colors, num_flashes, cycle_length, fix_time, sample_time, delay_time, test_time,
        # num_batches, trials_per_batch, non_match_separation
        for key, value in trial_params.items():
            setattr(self, key, value)
                
        self.num_neurons = self.num_color_tuned + self.num_spatial_tuned       
        self.neural_tuning = self.create_spatial_tuning_function()
        print('spatial_stimulus __init__ called')
        
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

    
class motion_stimulus:
    
    """
    This class allows one to generate trials of different spatial based tasks
    Will produce the neural input, desired output, and all trial events
    """
    
    def __init__(self, trial_params):
        
        # trial_params is a dictionary containing all the trial parameters
        # should include size_x, size_y, num_colors, num_flashes, cycle_length, fix_time, sample_time, delay_time, test_time,
        # num_batches, trials_per_batch, non_match_separation
        for key, value in trial_params.items():
            setattr(self, key, value)
                    
        self.motion_tuning, self.fix_tuning, self.rule_tuning = self.create_motion_tuning_function()
        print('motion_stimulus __init__ called')
        
    def generate_motion_working_memory_trial(self, num_batches, trials_per_batch):
        
        """
        Generate a delayed match to sample motion based task
        Goal is to determine whether the sample stimulus, possibly manipulated by a rule, is
        identicical to a test stimulus
        Sample and test stimuli are separated by a delay
        """
        
        num_classes = 3 # maintain fixation, match, non-match
        # rule signal can appear at the end of delay1_time
        trial_length = self.trial_length//self.delta_t
        
        # end of trial epochs
        eof = self.fix_time//self.delta_t
        eos = (self.fix_time+self.sample_time)//self.delta_t
        eod = (self.fix_time+self.sample_time+self.delay_time)//self.delta_t
        
        # end of neuron indices
        emt = self.num_motion_tuned
        eft = self.num_fix_tuned+self.num_motion_tuned
        ert = self.num_fix_tuned+self.num_motion_tuned + self.num_rule_tuned
        
        trial_info = {'desired_output'  :  np.zeros((num_batches,num_classes, trial_length, trials_per_batch),dtype=np.float32),
                      'sample_direction':  np.zeros((num_batches,trials_per_batch),dtype=np.float32),
                      'test_direction'  :  np.zeros((num_batches,trials_per_batch),dtype=np.float32),
                      'rule'            :  np.zeros((num_batches,trials_per_batch),dtype=np.int8),
                      'match'           :  np.zeros((num_batches,trials_per_batch),dtype=np.int8),
                      'catch'           :  np.zeros((num_batches,trials_per_batch),dtype=np.int8),
                      'neural_input'    :  np.random.normal(self.input_mean, self.input_sd, size=(num_batches, self.num_input_neurons, trial_length, trials_per_batch))}
        
        # maintain fixation during the fix epoch
        trial_info['desired_output'][:, 0, :eod, :] = 1
        
        for b in range(num_batches):
            for t in range(trials_per_batch):
                
                # generate trial params
                sample_dir = np.random.randint(self.num_motion_dirs)
                rule = np.random.randint(self.num_rules)
                match = np.random.randint(2)
                catch = np.random.rand() < self.catch_trial_pct
                if rule == 0: # don't rotate sample
                    if match == 1: # match trial
                        test_dir = sample_dir
                    else:
                        possible_dirs = np.setdiff1d(list(range(self.num_motion_dirs)), sample_dir)
                        test_dir = possible_dirs[np.random.randint(self.num_motion_dirs-1)]
                elif rule == 1: # rotate sample one positive index value
                    if match == 1: # match trial
                        test_dir = (sample_dir+1)%self.num_motion_dirs
                    else:
                        possible_dirs = np.setdiff1d(list(range(self.num_motion_dirs)), (sample_dir+1)%self.num_motion_dirs)
                        test_dir = possible_dirs[np.random.randint(self.num_motion_dirs-1)]                 
                elif rule == 2: # rotate sample one negative index value
                    if match == 1: # match trial
                        test_dir = (sample_dir-1)%self.num_motion_dirs
                    else:
                        possible_dirs = np.setdiff1d(list(range(self.num_motion_dirs)), (sample_dir-1)%self.num_motion_dirs)
                        test_dir = possible_dirs[np.random.randint(self.num_motion_dirs-1)]                         
                
                # modify neural input based on the trial params
                # sample motion direction
                trial_info['neural_input'][b, :emt, eof:eos, t] *= self.neural_mult(self.motion_tuning[:,sample_dir], eos-eof, self.num_motion_tuned)
                # fixation ON
                trial_info['neural_input'][b, emt:eft, :eod, t] *= self.neural_mult(self.fix_tuning[:,0], eod, self.num_fix_tuned)
                if not catch:
                    # test motion direction
                    trial_info['neural_input'][b, :emt, eod:, t] *= self.neural_mult(self.motion_tuning[:,test_dir], trial_length-eod, self.num_motion_tuned)
                    # fixation OFF
                    trial_info['neural_input'][b, emt:eft, eod:trial_length, t] *= self.neural_mult(self.fix_tuning[:,1], trial_length-eod, self.num_fix_tuned)
                
                # rule
                trial_info['neural_input'][b, eft:ert, self.rule_onset_time:, t] *= self.neural_mult(self.rule_tuning[:,0], trial_length-self.rule_onset_time, self.num_rule_tuned)
                
                if not catch:
                    if match == 0:
                        trial_info['desired_output'][b, 1, eod:, t] = 1
                    else:
                        trial_info['desired_output'][b, 2, eod:, t] = 1 
                else:
                    trial_info['desired_output'][b, 0, eod:, t] = 1 
                                                                
                trial_info['sample_direction'][b,t] = sample_dir
                trial_info['test_direction'][b,t] = test_dir
                trial_info['rule'][b,t] = rule
                trial_info['match'][b,t] = match
                trial_info['catch'][b,t] = catch
         
        # debugging: plot the neural input activity
        """
        f = plt.figure(figsize=(9,4))
        ax = f.add_subplot(1, 3, 1)
        ax.imshow(trial_info['sample_direction'],interpolation='none',aspect='auto')
        ax = f.add_subplot(1, 3, 2)
        ax.imshow(trial_info['test_direction'],interpolation='none',aspect='auto')
        ax = f.add_subplot(1, 3, 3)
        ax.imshow(trial_info['match'],interpolation='none',aspect='auto')
        plt.show()
        1/0  
        """
        return trial_info
                                                     
    @staticmethod                                                
    def neural_mult(tuning_function, trial_length, num_neurons):
                                                        
        return np.tile(1 + np.reshape(tuning_function,(num_neurons,1)),(1,trial_length))
                                                                  
    
    def create_motion_tuning_function(self):
        
        """
        Generate tuning functions for the input neurons
        Here, neurons are either selective for motion direction, fixation, or the rule
        """
        motion_tuning = np.zeros((self.num_motion_tuned, self.num_motion_dirs))
        fix_tuning = np.zeros((self.num_fix_tuned, 2))
        rule_tuning = np.zeros((self.num_rule_tuned, self.num_rules))
        scale = 0.5
        
        # generate list of prefered motion directions
        delta_pref_dir = int(360/self.num_motion_tuned)
        pref_dirs = list(range(0,360,delta_pref_dir))
        
        #  generate list of possible motion directions
        delta_motion_dir = int(360/self.num_motion_dirs)
        motion_dirs = list(range(0,360,delta_motion_dir))
        
        for n in range(self.num_motion_tuned): 
            for i in range(len(motion_dirs)):
                # cosine tuning
                motion_tuning[n,i] = scale*np.cos((motion_dirs[i] - pref_dirs[n])/180*np.pi)

        for n in range(self.num_fix_tuned):
            for i in range(2):
                if n%2 == i:
                    fix_tuning[n,i] = scale
                
        for n in range(self.num_rule_tuned):
            for i in range(self.num_rules):
                if n%self.num_rules == i:
                    rule_tuning[n,i] = scale
            
                    
        return motion_tuning, fix_tuning, rule_tuning


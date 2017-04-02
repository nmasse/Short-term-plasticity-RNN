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
        
    def generate_spatial_working_memory_trial(self):
        
        """
        Generate a delayed match to sample spatial based task
        Goal is to determine whether the location of two stimuli, separated by a delay, are a match
        """
        
        num_classes = 2
        trial_length = self.fix_time + self.sample_time + self.delay_time + self.test_time
        sample_epoch = range(self.fix_time, self.fix_time + self.sample_time)
        test_epoch  = range(self.fix_time + self.sample_time + self.delay_time, trial_length)
        
        trial_info = {'desired_output' :  np.zeros((self.num_batches,self.trials_per_batch,num_classes, trial_length),dtype=np.float32),
                      'sample_location':  np.zeros((self.num_batches,self.trials_per_batch, 2),dtype=np.float32),
                      'test_location'  :  np.zeros((self.num_batches,self.trials_per_batch, 2),dtype=np.float32),
                      'match'          :  np.zeros((self.num_batches,self.trials_per_batch),dtype=np.int8)}
        
        trial_info['neural_input'] = np.random.normal(self.input_mean, self.input_sd, size=(self.num_neurons,trial_length))
        
        for b in range(self.num_batches):
            for t in range(self.trials_per_batch):
                sample_x_loc = np.int_(np.floor(np.random.rand()*self.size_x))
                sample_y_loc = np.int_(np.floor(np.random.rand()*self.size_y))
                neural_multiplier = np.tile(1 + np.reshape(self.neural_tuning[:,0,sample_x_loc,sample_y_loc],(self.num_neurons,1)),(1,self.sample_time))
                trial_info['neural_input'][:, sample_epoch] *= neural_multiplier
                
                # generate test location
                if np.random.randint(2)==0:
                    # match trial
                    test_x_loc = sample_x_loc
                    test_y_loc = sample_y_loc
                    trial_info['match'][b,t] = 1
                    trial_info['desired_output'][b,t,0,test_epoch] = 1
                else:
                    d = 0
                    count = 0
                    trial_info['desired_output'][b,t,1,test_epoch] = 1
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
                
                
        
    def generate_spatial_cat_trial(self):
        
        """
        Generate a spatial categorization task
        Goal is to determine whether the color and the location of a series of flashes fall 
        inside color specific target zones
        """
        
        num_classes = 2
        trial_length = self.fix_time + self.num_flashes*self.cycle_time
        trial_info = {'desired_output' : np.zeros((self.num_batches,self.trials_per_batch,num_classes, trial_length),dtype=np.float32),
                      'flash_x_pos'    : np.random.randint(self.size_x, size=[self.num_batches, self.trials_per_batch, self.num_flashes]),
                      'flash_y_pos'    : np.random.randint(self.size_y, size=[self.num_batches, self.trials_per_batch, self.num_flashes]),
                      'flash_color'    : np.random.randint(self.num_colors, size=[self.num_batches, self.trials_per_batch, self.num_flashes]),
                      'flash_zone'     : np.zeros((self.num_batches, self.trials_per_batch, self.num_flashes), dtype=np.int8)}
        
        trial_info['neural_input'] = np.random.normal(self.input_mean, self.input_sd, size=(self.num_neurons,trial_length))
        
        for b in range(self.num_batches):
            for t in range(self.trials_per_batch):
                for f in range(self.num_flashes):
                    x_pos = trial_info['flash_x_pos'][b,t,f]
                    y_pos = trial_info['flash_y_pos'][b,t,f]
                    color = trial_info['flash_color'][b,t,f]
                    zone = self.get_category_zone(x_pos, y_pos)
                    trial_info['flash_zone'][b,t,f] = zone
                    
                    # modify neural input according to flash stimulus
                    flash_time_ind = range(self.fix_time+f*self.cycle_time+1, self.fix_time+f*self.cycle_time+self.flash_time)
                    neural_multiplier = np.tile(1 + np.reshape(self.neural_tuning[:,color,x_pos,y_pos],(self.num_neurons,1)),(1,len(flash_time_ind)))
                    trial_info['neural_input'][:, flash_time_ind] *= neural_multiplier
                    
                    cycle_time_ind = range(self.fix_time+f*self.cycle_time+1, self.fix_time+(f+1)*self.cycle_time)
                    
                    # generate desired output based on whether the flash fell inside or outside a target zone
                    if zone == color:
                        # target flash
                        trial_info['desired_output'][b,t,0,cycle_time_ind] = 0
                        trial_info['desired_output'][b,t,1,cycle_time_ind] = 1
                    else:
                        # non-target flash
                        trial_info['desired_output'][b,t,0,cycle_time_ind] = 1
                        trial_info['desired_output'][b,t,1,cycle_time_ind] = 0
              
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
                        




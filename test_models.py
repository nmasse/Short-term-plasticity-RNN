import tensorflow as tf
import numpy as np
import stimulus
import importlib
import pickle
import model
import paramaters 
importlib.reload(stimulus)
importlib.reload(paramaters)
importlib.reload(model)

p = paramaters.Paramaters()
p.params['synapse_config'] = 'std_stf'
p.params['num_iterations'] = 20000
p.params['spike_cost'] = 0.0002
#p.params['num_exc_units'] = 105
#p.params['num_inh_units'] = 20
p.params['num_exc_units'] = 105
p.params['num_inh_units'] = 20

def run_tests():
    
    
    for i in range(1,10):
             
        p.params['stimulus_type'] = 'MNIST'
        p.params['num_iterations'] = 2000
        p.params['probe_trial_pct'] = 0
        p.params['batch_train_size'] = 64
        p.params['num_batches'] = 1024//p.params['batch_train_size']
        p.params['spike_cost'] =  0.01
        p.params['wiring_cost'] = 0.02
        p.params['dt'] = 25
        p.params['input_sd'] = 0.01
        p.params['noise_sd'] = 0.25
        p.params['learning_rate'] = 0.005
        print(i, p.params['input_mean'], p.params['input_sd'], p.params['noise_sd'], p.params['spike_cost'])
        
        p.params['ckpt_save_fn'] = 'MNIST_wiring_' + str(i) + '.ckpt' 
        p.params['ckpt_load_fn'] = 'MNIST_wiring_' + str(i) + '.ckpt' 
        p.params['synapse_config'] = None
        p.params['possible_rules']=[-1]
        p.params['var_delay'] = False
        p.params['catch_trial_pct'] = 0
        p.params['save_fn'] = 'DMS_wiring_' + str(i) + '.pkl' 
        p.params['dead_time'] = 400
        p.params['connection_prob'] = 0.5
        p.params['load_previous_model'] = False
        params = p.return_params()
        tf.reset_default_graph()
        rnn = None
        model.main(params)


   
    # Go back to 16, best one yet
    
    for i in range(0,10):
            
        input_sd = [0.1]
        noise_sd = [0.5]   
        p.params['num_iterations'] = 10000
        p.params['probe_trial_pct'] = 0
        p.params['batch_train_size'] = 32
        p.params['spike_cost'] =  0.001
        p.params['wiring_cost'] = 0.02
        p.params['dt'] = 25
        p.params['input_sd'] = input_sd[0]
        p.params['noise_sd'] = noise_sd[0]
        p.params['learning_rate'] = 0.005
        print(i, p.params['input_mean'], p.params['input_sd'], p.params['noise_sd'], p.params['spike_cost'])
        
        p.params['ckpt_save_fn'] = 'postle_EI_' + str(i) + '.ckpt' 
        p.params['ckpt_load_fn'] = 'postle_EI_' + str(i) + '.ckpt' 
        p.params['num_rule_tuned'] = 0
        p.params['synapse_config'] = None
        p.params['possible_rules']=[5]
        p.params['var_delay'] = False
        p.params['catch_trial_pct'] = 0
        p.params['save_fn'] = 'postle_wiring_' + str(i) + '.pkl' 
        p.params['dead_time'] = 200
        p.params['fix_time'] = 400
        p.params['sample_time'] = 250
        p.params['delay_time'] = 400
        p.params['test_time'] = 400
        p.params['num_rule_tuned'] = 12
        p.params['cue_time'] = 0
        p.params['num_motion_dirs'] = 8
        p.params['load_previous_model'] = False
        params = p.return_params()
        tf.reset_default_graph()
        rnn = None
        model.main(params)

        
        """
        
        p.params['synapse_config'] = 'std_stf'
        p.params['possible_rules']=[1]
        p.params['var_delay'] = False
        p.params['catch_trial_pct'] = 0.2
        p.params['save_fn'] = 'DMrS_EI_std_stf_v' + str(i) + '.pkl' 
        params = p.return_params()
        tf.reset_default_graph()
        rnn = None
        model2.main(params)

        
        
        p.params['synapse_config'] = 'std_stf'
        p.params['possible_rules']=[3]
        p.params['var_delay'] = True
        p.params['catch_trial_pct'] = 0
        p.params['save_fn'] = 'DMC_EI_std_stf_vd_v' + str(i) + '.pkl' 
        params = p.return_params()
        tf.reset_default_graph()
        rnn = None
        model2.main(params)
        
        
        p.params['synapse_config'] = 'std_stf'
        p.params['possible_rules']=[0]
        p.params['var_delay'] = False
        p.params['catch_trial_pct'] = 0.2
        p.params['save_fn'] = 'DMS_EI_std_stf_v' + str(i) + '.pkl' 
        params = p.return_params()
        tf.reset_default_graph()
        rnn = None
        model2.main(params)
        """
        
           

 
    
    
    
    1/0
    
    params['possible_rules']=[3]
    tf.reset_default_graph()
    rnn = None
    params['save_fn'] = 'DMC' 
    params['synapse_config'] = None
    model2.main(params)
    
    

    params['possible_rules']=[0]
    tf.reset_default_graph()
    rnn = None
    params['save_fn'] = 'DMS_std_stf' 
    params['synapse_config'] = 'std_stf'
    model2.main(params)
    
    params['possible_rules']=[0]
    tf.reset_default_graph()
    rnn = None
    params['save_fn'] = 'DMS' 
    params['synapse_config'] = None
    model2.main(params)
    
    
    params['num_iterations'] = 2000
    
    params['possible_rules']=[1]
    tf.reset_default_graph()
    rnn = None
    params['save_fn'] = 'DMS_rotation_std_stf' 
    params['synapse_config'] = 'std_stf'
    model2.main(params)
    
    params['possible_rules']=[1]
    tf.reset_default_graph()
    rnn = None
    params['save_fn'] = 'DMS_rotation' 
    params['synapse_config'] = None
    model2.main(params)
    
    params['num_rule_tuned'] = 8
    
    params['num_input_units'] = params['num_motion_tuned'] + params['num_fix_tuned'] + params['num_rule_tuned']
    params['possible_rules']=[0,3]
    tf.reset_default_graph()
    rnn = None
    params['save_fn'] = 'DMS_DMC_std_stf' 
    params['synapse_config'] = 'std_stf'
    model2.main(params)
    
    params['possible_rules']=[0,3]
    tf.reset_default_graph()
    rnn = None
    params['save_fn'] = 'DMS_DMC' 
    params['synapse_config'] = None
    model2.main(params)
    
    
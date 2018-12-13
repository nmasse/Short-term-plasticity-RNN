import numpy as np
from parameters import *
import model
import sys
import os

#task_list = ['DMS+DMRS+DMC']
task_list = [ 'DMS']
task_list = [ 'DMRS45', 'DMRS180','ABBA']
task_list = [ 'DMS+DMRS','DMRS45', 'DMRS180']
#task_list = [ 'DMRS45', 'DMRS180']
#task_list = [ 'DMS','DMRS90']
#
#task_list = ['dualDMS']
task_list = ['DMS','DMRS90', 'DMRS45', 'DMRS180','DMS+DMRS','ABCA', 'ABBA','dualDMS','location_DMS']
#task_list = ['dualDMS','location_DMS']
#task_list = [ 'DMRS45', 'DMRS180','DMS+DMRS']
task_list = ['DMS+DMRS_early_cue']

def try_model(gpu_id):

    try:
        # Run model
        model.main(gpu_id)
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt')

# Second argument will select the GPU to use
# Don't enter a second argument if you want TensorFlow to select the GPU/CPU
try:
    gpu_id = sys.argv[1]
    print('Selecting GPU ', gpu_id)
except:
    gpu_id = None


spike_reg = 'L2'
if spike_reg == 'L1':
    spike_cost = [0., 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    weight_cost = [0., 1e-3, 1e-2, 1e-1, 1e0]
elif spike_reg == 'L2':
    spike_cost = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    w_rnn_weight_cost = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    w_in_weight_cost = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]

lr = 0.02
update_parameters({'simulation_reps':0,'analyze_tuning':False,'decoding_reps':5,'batch_train_size':1024, \
    'learning_rate':lr,'noise_rnn_sd': 0.5, 'noise_in_sd': 0.1,'num_iterations': 4000, \
    'savedir': './savedir_FINAL/','spike_regularization':spike_reg})

tcm = 2
bal_EI = 1
synaptic_config = 'full'


for n in [0]:
    for sc in [2,3,4]:
        for wrnn_wc in [3,4,5,6,7]:
            for win_wc in [3,4,5,6,7]:
                task = 'DMS+DMRS_early_cue'
                update_parameters({'w_rnn_weight_cost':w_rnn_weight_cost[wrnn_wc],'w_in_weight_cost':w_in_weight_cost[win_wc], 'spike_cost':spike_cost[sc], \
                    'test_cost_multiplier': np.float32(tcm), 'balance_EI': (bal_EI>0), 'synapse_config': synaptic_config, \
                    'tau_slow':1500, 'tau_fast': 200,'num_iterations': 1000})

                if synaptic_config == 'excitatory_facilitating':
                    save_fn = task + '_wc' + str(wc)+ '_sc' + str(sc) + '_tcm' + str(tcm) + '_balEI' + str(bal_EI) + '_EXC_FAC_lr1_v' + str(n) + '.pkl'
                elif synaptic_config == 'excitatory_half_facilitating':
                    save_fn = task + '_wc' + str(wc)+ '_sc' + str(sc) + '_tcm' + str(tcm) + '_balEI' + str(bal_EI) + '_EXC_H_FAC_v' + str(n) + '.pkl'
                elif synaptic_config == 'inhibitory_facilitating':
                    save_fn = task + '_wc' + str(wc)+ '_sc' + str(sc) + '_tcm' + str(tcm) + '_balEI' + str(bal_EI) + '_INH_FAC_v' + str(n) + '.pkl'
                elif synaptic_config == 'facilitating':
                    save_fn = task + '_wc' + str(wc)+ '_sc' + str(sc) + '_tcm' + str(tcm) + '_balEI' + str(bal_EI) + '_FAC_v' + str(n) + '.pkl'
                elif synaptic_config == 'excitatory_depressing':
                    save_fn = task + '_wc' + str(wc)+ '_sc' + str(sc) + '_tcm' + str(tcm) + '_balEI' + str(bal_EI) + '_EXC_DEP_v' + str(n) + '.pkl'
                elif synaptic_config == 'excitatory_depressing_inhibitory_facilitating':
                    save_fn = task + '_wc' + str(wc)+ '_sc' + str(sc) + '_tcm' + str(tcm) + '_balEI' + str(bal_EI) + '_EXC_DEP_INH_FAC_v' + str(n) + '.pkl'
                elif synaptic_config == 'inhibitory_depressing':
                    save_fn = task + '_wc' + str(wc)+ '_sc' + str(sc) + '_tcm' + str(tcm) + '_balEI' + str(bal_EI) + '_INH_DEP_v' + str(n) + '.pkl'
                elif synaptic_config == 'depressing':
                    save_fn = task + '_wc' + str(wc)+ '_sc' + str(sc) + '_tcm' + str(tcm) + '_balEI' + str(bal_EI) + '_DEP_v' + str(n) + '.pkl'
                elif synaptic_config == 'full':
                    #save_fn = task  +  '_N200' + '_wc' + str(wc)+ '_sc' + str(sc) + '_tcm' + str(tcm) + '_balEI' + str(bal_EI) + '_L2_lr' + str(int(lr*100)) + '_v' + str(n) + '.pkl'
                    #save_fn = task  + '_v2_wc' + str(wc) + '_sc' + str(sc) + '_tcm' + str(tcm) + '_balEI' + str(bal_EI) + '_L2_lr' + str(int(lr*100)) + '_v' + str(n) + '.pkl'
                    save_fn = task  + '_' + spike_reg  + '_rnn_wc' + str(wrnn_wc) + '_in_wc' + str(win_wc) + '_sc' + str(sc) + '_v' + str(n) + '.pkl'
                    if os.path.isfile(par['save_dir'] + save_fn):
                        pass
                        #break
                    print('save_fn ', save_fn)
                elif synaptic_config is None:
                    save_fn = task  + '_no_stp_wc' + str(wc)+ '_sc' + str(sc) + '_tcm' + str(tcm) + '_balEI' + str(bal_EI) + '_L2_lr2_v' + str(n) + '.pkl'
                update_parameters({'trial_type': task, 'save_fn': save_fn, 'simulation_reps':0, 'weight_multiplier': 1., \
                    'connection_prob': 1., 'membrane_time_constant': 100, 'n_hidden': 100, 'dead_time': 0})
                try_model(gpu_id)

1/0


task = 'ABBA'
update_parameters({'simulation_reps':0,'analyze_tuning':False,'dt':10,'spike_cost':1e-3,'decoding_reps':10,\
    'learning_rate':2e-2,'noise_rnn_sd': 0.5, 'noise_in_sd': 0.1,'savedir': './savedir_var_train/'})
for j in [0,1,2,3]:
    update_parameters({'num_iterations':2000})
    save_fn = task   + str(j) + '_long.pkl'
    updates = {'trial_type': task, 'save_fn': save_fn, 'simulation_reps':0}
    update_parameters(updates)
    try_model(gpu_id)

1/0
"""

task = 'distractor'
update_parameters({'num_iterations':4000,'simulation_reps':0,'analyze_tuning':False,'dt':10,'spike_cost':5e-3,'decoding_reps':3,\
    'learning_rate':1e-2,'noise_rnn_sd': 0.1, 'noise_in_sd': 0.05})
for j in range(1,2,2):
    save_fn = task   + str(j) + 'no_mask.pkl'
    updates = {'trial_type': task, 'save_fn': save_fn, 'simulation_reps':0}
    update_parameters(updates)
    try_model(gpu_id)

1/0




task = 'DMS+DMC'
update_parameters({'num_iterations':4000,'simulation_reps':100,'analyze_tuning':False,'dt':40,})
for j in range(0, 20, 2):
    save_fn = task  + '_seq4000_v' + str(j) + '.pkl'
    updates = {'trial_type': task, 'save_fn': save_fn, 'simulation_reps':100}
    update_parameters(updates)
    try_model(gpu_id)

1/0
"""
task_list = ['DMS']
for j in range(0,10,1):
    for i in range(4):
        for task in task_list:

            print('Training network on ', task,' task, network model number ', j, 'spike cost ', spike_cost[i])

            update_parameters({'sample_time':500, 'delay_time':1000,'analyze_test':False,'save_dir':'./savedir_spike_cost/'})

            save_fn = task + '_spike_cost' + str(i) + '_v' + str(j) + '.pkl'
            updates = {'trial_type': task, 'save_fn': save_fn, 'spike_cost': spike_cost[i], 'simulation_reps':0}
            update_parameters(updates)
            try_model(gpu_id)

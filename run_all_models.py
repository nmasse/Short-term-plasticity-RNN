import numpy as np
from parameters import *
import model
import sys

task_list = ['DMS', 'DMRS45', 'DMRS45ccw', 'DMRS90',  'DMRS90ccw','DMRS135', 'DMRS135ccw', 'DMRS180', \
    'DMC', 'DMC1', 'DMC2','DMC3', 'OneIntCat','OneIntCat1','OneIntCat2','OneIntCat3', 'Color_OneIntCat',\
    'Color_OneIntCat1','Color_OneIntCat2','Color_OneIntCat3','Color_DelayedCat','Color_DelayedCat1','Color_DelayedCat2',\
    'Color_DelayedCat3']
"""
task_list = ['DMS', 'DMRS45', 'DMRS45ccw', 'DMRS90',  'DMRS90ccw','DMRS135', 'DMRS135ccw', 'DMRS180', \
    'DMC', 'DMC1', 'DMC2','DMC3', 'OneIntCat','OneIntCat1','OneIntCat2','OneIntCat3']


task_list = ['DMC', 'DMC1', 'DMC2','DMC3', 'OneIntCat','OneIntCat1','OneIntCat2','OneIntCat3', 'Color_OneIntCat',\
    'Color_OneIntCat1','Color_OneIntCat2','Color_OneIntCat3','Color_DelayedCat','Color_DelayedCat1','Color_DelayedCat2',\
    'Color_DelayedCat3']
"""

#task_list = ['DelayedCat2','DelayedCat3']
#task_list = ['Color_DelayedCat2','Color_DelayedCat3']
#task_list = ['Color_OneIntCat1','Color_OneIntCat2']

task_list_order = [np.random.permutation(len(task_list)) for i in range(5)]
update_parameters({'num_iterations': 200, 'synapse_config': None, 'wiring_cost': 0})
update_parameters({'omega_c': 0.0})
update_parameters({'gate_pct': 0.75})
update_parameters({'stabilization': 'pathint'})
update_parameters({'n_hidden': 200})
update_parameters({'omega_xi': 0.01})
save_fn = 'test.pkl'
print('Proceeding to testing...')
gpu_id = str(sys.argv[1])

#update_parameters({'neuron_gate_pct': 0.0})
#model.train_and_analyze(save_fn, gpu_id)

model.train_and_analyze(save_fn, gpu_id)
quit()

#c_vals = [0.0, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.35, 0.5]
c_vals = [0.2]


#update_parameters({'synapse_config': None, 'delay_time': 400, 'spike_cost': 1e-7, 'noise_rnn_sd': 0.05})


for i in [0,1,2,3,4,5,6]:

    for j,c in enumerate(c_vals):
        current_task_list = [task_list[j] for j in task_list_order[i]]
        update_parameters({'omega_c': c, 'task_list': current_task_list})
        update_parameters({'gate_pct': 0.5})
        update_parameters({'omega_xi': 0.01})
        update_parameters({'n_hidden': 100})
        update_parameters({'stabilization': 'pathint'})
        save_fn = 'RNN_CL_SI_no_stp_h500_gating0_xi01_fixed_omega_' + str(1) + '_v' + str(i) + '.pkl'
        print(current_task_list)
        model.train_and_analyze(str(sys.argv[1]), save_fn)

quit()

for j,c in enumerate(c_vals):
    current_task_list = [task_list[j] for j in task_list_order[i]]
    update_parameters({'omega_c': c, 'task_list': current_task_list})
    update_parameters({'gate_pct': 0.0})
    update_parameters({'omega_xi': 0.01})
    update_parameters({'n_hidden': 500})
    update_parameters({'stabilization': 'pathint'})
    save_fn = 'RNN_CL_SI_h500_gating0_xi01_fixed_omega_' + str(j) + '_v' + str(i) + '.pkl'
    print(current_task_list)
    model.train_and_analyze(str(sys.argv[1]), save_fn)


for j,c in enumerate(c_vals):
    current_task_list = [task_list[j] for j in task_list_order[i]]
    update_parameters({'omega_c': c, 'task_list': current_task_list})
    update_parameters({'gate_pct': 0.75})
    update_parameters({'omega_xi': 0.001})
    update_parameters({'n_hidden': 500})
    update_parameters({'stabilization': 'pathint'})
    save_fn = 'RNN_CL_SI_h500_gating75_xi001_fixed_omega_' + str(j) + '_v' + str(i) + '.pkl'
    print(current_task_list)
    model.train_and_analyze(str(sys.argv[1]), save_fn)



for j,c in enumerate(c_vals):
    current_task_list = [task_list[j] for j in task_list_order[i]]
    update_parameters({'omega_c': c, 'task_list': current_task_list})
    update_parameters({'gate_pct': 0.0})
    update_parameters({'omega_xi': 0.001})
    update_parameters({'n_hidden': 500})
    update_parameters({'stabilization': 'pathint'})
    save_fn = 'RNN_CL_SI_h500_gating0_xi001_fixed_omega_' + str(j) + '_v' + str(i) + '.pkl'
    print(current_task_list)
    model.train_and_analyze(str(sys.argv[1]), save_fn)

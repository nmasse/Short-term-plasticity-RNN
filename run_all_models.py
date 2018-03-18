import numpy as np
from parameters import *
import model
import sys

task_list = ['DMC','ABBA','DMS','DMRS45']
cell_type = ['STP','LSTM','LSTM']
synapse_config = ['std_stf','','']
exc_inh_prop = [0.8, 1, 0.8]
spike_cost = [2e-2, 0, 2e-2]

for task in task_list:
    for i in range(3):
        for j in range(1):
            output = task+'_'+cell_type[i]+'_'+str(i)+'_result.txt'
            print('Training network on ', task,' task, network model number ', j)
            print('Cell type is ', cell_type[i], ' with iter number ', str(i))
            save_fn = task + str(j) + '.pkl'
            updates = {'trial_type':task, 'cell_type':cell_type[i], 'synapse_config':synapse_config[i], \
                        'exc_inh_prop':exc_inh_prop[i], 'spike_cost':spike_cost[i], 'file':output, 'save_fn': save_fn}
            update_parameters(updates)

            # Keep the try-except clauses to ensure proper GPU memory release
            if par['gpu']:
                try:
                    # GPU designated by first argument (must be integer 0-3)
                    try:
                        print('Selecting GPU ',  sys.argv[1])
                        assert(int(sys.argv[1]) in [0,1,2,3])
                    except AssertionError:
                        quit('Error: Select a valid GPU number.')

                    # Run model
                    model.train_and_analyze(sys.argv[1])
                except KeyboardInterrupt:
                    quit('Quit by KeyboardInterrupt')
            else:
                f = open(output,'a+')
                f.write('Task: '+task+'\nType: '+cell_type[i]+'\nsyn: '+synapse_config[i]\
                    +'\nEI: '+str(exc_inh_prop[i])+'\nSpike cost: '+str(spike_cost[i])+'\n')
                f.close()
                model.train_and_analyze(0)


# Command for observing python processes:
# ps -A | grep python

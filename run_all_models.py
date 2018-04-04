import numpy as np
from parameters import *
import model
import sys

#task_list = ['DMS+DMRS+DMC']
task_list = ['DMS']


for j in range(1, 80, 2):
    for task in task_list:
        print('Training network on ', task,' task, network model number ', j)

        save_fn = task + '_delay2500_' + str(j) + '.pkl'
        updates = {'trial_type': task, 'save_fn': save_fn, 'delay_time': 2500, 'num_iterations': 2000}
        """
        updates = {'trial_type': task, 'save_fn': save_fn, \
            'save_dir':'/media/masse/MySSDataStor1/Short-Term-Synaptic-Plasticity/savedir_motifs/', \
            'var_delay': True,'learning_rate':2e-2, 'decoding_test_mode':False,'n_hidden':100,'synapse_config':'std_stf','num_iterations':2000}
        """
        update_parameters(updates)


        # Keep the try-except clauses to ensure proper GPU memory release
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


# Command for observing python processes:
# ps -A | grep python

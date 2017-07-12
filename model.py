"""
2017/05/03 Nicolas Masse
Edited: 2017/06/13 Gregory Grant
"""

print("\nRunning model...\n")

import tensorflow as tf
import numpy as np

from paramters import *
from model_saver import *
import dendrite_functions as df
import stimulus
import analysis

import os
import time
import psutil

# Reset TensorFlow before running anythin
tf.reset_default_graph()

# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Allow for varied processor use (if on Windows)
if os.name == 'nt':
    p = psutil.Process(os.getpid())
    p.cpu_affinity(par['processor_affinity'])
    print('Running with PID', os.getpid(), "on processor(s)", \
            str(p.cpu_affinity()) + ".", "\n")


#################################
### Model setup and execution ###
#################################

class Model:

    def __init__(self, input_data, target_data, mask, learning_rate):

        # Load the input activity, the target data, and the training mask
        # for this batch of trials
        self.input_data = tf.unstack(input_data, axis=1)
        self.target_data = tf.unstack(target_data, axis=1)
        self.mask = tf.unstack(mask, axis=0)
        self.learning_rate = learning_rate

        # Load the initial hidden state activity to be used at
        # the start of each trial
        self.hidden_init = tf.constant(par['h_init'])
        self.dendrites_init = tf.constant(par['d_init'])

        # Load the initial synaptic depression and facilitation to be used at
        # the start of each trial
        self.synapse_x_init = tf.constant(par['syn_x_init'])
        self.synapse_u_init = tf.constant(par['syn_u_init'])

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()


    def run_model(self):
        """
        Run the recurrent network model using the set parameters
        """

        # The RNN Cell Loop contains the majority of the model calculations
        self.rnn_cell_loop(self.input_data, self.hidden_init, self.dendrites_init, self.synapse_x_init, self.synapse_u_init)

        # Describes the output variables and scope
        with tf.variable_scope('output'):
            W_out = tf.get_variable('W_out', initializer = np.float32(par['w_out0']), trainable=True)
            b_out = tf.get_variable('b_out', initializer = np.float32(par['b_out0']), trainable=True)

        # Setting the desired network output, considering only
        # excitatory RNN projections
        self.y_hat = [tf.matmul(tf.nn.relu(W_out),h)+b_out for h in self.hidden_state_hist]


    def rnn_cell_loop(self, x_unstacked, h, d, syn_x, syn_u):
        """
        Sets up the weights and baises for the hidden layer, then establishes
        the network computation
        """

        # Initialize weights and biases, with behavior changes based on
        # dendrite usage (with dendrites requies a rank higher of tensor)
        with tf.variable_scope('rnn_cell'):
            if par['use_dendrites']:
                W_rnn_dend = tf.get_variable('W_rnn_dend', initializer = np.float32(par['w_rnn_dend0']), trainable=True)
                W_in_dend = tf.get_variable('W_in_dend', initializer = np.float32(par['w_in_dend0']), trainable=True)

            W_in_soma = tf.get_variable('W_in_soma', initializer = np.float32(par['w_in_soma0']), trainable=True)
            W_rnn_soma = tf.get_variable('W_rnn_soma', initializer = np.float32(par['w_rnn_soma0']), trainable=True)
            b_rnn = tf.get_variable('b_rnn', initializer = np.float32(par['b_rnn0']), trainable=True)

        # Sets up the histories for the computation
        self.hidden_state_hist = []
        self.dendrites_hist = []
        self.dendrites_inputs_exc_hist = []
        self.dendrites_inputs_inh_hist = []
        self.syn_x_hist = []
        self.syn_u_hist = []

        # Loops through the neural inputs to the network, indexed in time
        for rnn_input in x_unstacked:
            h, d, syn_x, syn_u, e, i = self.rnn_cell(rnn_input, h, d, syn_x, syn_u)
            self.hidden_state_hist.append(h)
            self.dendrites_hist.append(d)
            self.dendrites_inputs_exc_hist.append(e)
            self.dendrites_inputs_inh_hist.append(i)
            self.syn_x_hist.append(syn_x)
            self.syn_u_hist.append(syn_u)


    def rnn_cell(self, rnn_input, h_soma, dend, syn_x, syn_u):
        """
        Run the main computation of the recurrent network
        """

        # Get the requisite variables for running the model.  Again, the
        # inclusion of dendrites changes the tensor shapes
        with tf.variable_scope('rnn_cell', reuse=True):
            if par['use_dendrites']:
                W_in_dend = tf.get_variable('W_in_dend')
                W_rnn_dend = tf.get_variable('W_rnn_dend')
                W_in_dend = tf.get_variable('W_in_dend')

            W_in_soma = tf.get_variable('W_in_soma')
            W_rnn_soma = tf.get_variable('W_rnn_soma')
            b_rnn = tf.get_variable('b_rnn')
            W_ei = tf.constant(par['EI_matrix'], name='W_ei')

        # If using an excitatory-inhibitory network, ensures that E neurons
        # have only positive outgoing weights, and that I neurons have only
        # negative outgoing weights
        if par['EI']:
            W_rnn_soma_effective = tf.matmul(tf.nn.relu(W_rnn_soma), W_ei)
        else:
            W_rnn_soma_effective = W_rnn_soma

        # Update the synaptic plasticity parameters
        if par['synapse_config'] == 'std_stf':
            # Implement both synaptic short term facilitation and depression
            syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_x*h_soma
            syn_u += par['alpha_std']*(par['U']-syn_u) \
                     + par['dt_sec']*par['U']*(1-syn_u)*h_soma
            syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
            syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
            h_post_syn = syn_u*syn_x*h_soma
        elif par['synapse_config'] == 'std':
            # Implement synaptic short term depression, but no facilitation
            # (assume that syn_u remains constant at 1)
            syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_x*h_soma
            syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
            syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
            h_post_syn = syn_x*h_soma
        elif par['synapse_config'] == 'stf':
            # Implement synaptic short term facilitation, but no depression
            # (assume that syn_x remains constant at 1)
            syn_u += par['alpha_stf']*(par['U']-syn_u) \
                     + par['dt_sec']*par['U']*(1-syn_u)*h_soma
            syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
            h_post_syn = syn_u*h_soma
        else:
            # Mo synaptic plasticity
            h_post_syn = h_soma

        """
        Update the hidden state by way of the dendrites
        Only use excitatory projections from input layer to RNN
        All input and RNN activity will be non-negative
        """

        # Apply a dendrite function to the inputs to the hidden layer,
        # if applicable.  The dendrite function in particular is specified
        # in the parameters file.
        if par['use_dendrites']:
            dendrite_function = getattr(df, 'dendrite_function' + par['df_num'])
            h_soma_in, dend_out, exc_activity, inh_activity = \
                dendrite_function(W_in_dend, W_rnn_dend, rnn_input, h_post_syn, dend)
        else:
            dend_out = dend
            h_soma_in = 0

        # Apply, in order: alpha decay, dendritic input, soma recurrence,
        # bias terms, and Gaussian randomness.  This generates the output of
        # the hidden layer.
        h_soma_out = tf.nn.relu(h_soma*(1-par['alpha_neuron']) \
                     + par['alpha_neuron']*(h_soma_in + tf.matmul(W_rnn_soma_effective, h_post_syn) \
                     + tf.matmul(tf.nn.relu(W_in_soma), tf.nn.relu(rnn_input)) + b_rnn) \
                     + tf.random_normal([par['n_hidden'], par['batch_train_size']], 0, par['noise_sd'], dtype=tf.float32))

        # Return the network information
        if par['use_dendrites']:
            return h_soma_out, dend_out, syn_x, syn_u, exc_activity, inh_activity
        else:
            return h_soma_out, dend_out, syn_x, syn_u, tf.constant(0), tf.constant(0)


    def optimize(self):
        """
        Calculate the loss functions for weight optimization, and apply weight
        masks where necessary.
        """

        # Calculate performance loss
        perf_loss = [mask*tf.reduce_mean(tf.square(y_hat-desired_output),axis=0) \
                     for (y_hat, desired_output, mask) in zip(self.y_hat, self.target_data, self.mask)]

        # L2 penalty term on hidden state activity to encourage low spike rate solutions
        spike_loss = [par['spike_cost']*tf.reduce_mean(tf.square(h), axis=0) \
                    for (h, mask) in zip(self.hidden_state_hist, self.mask)]

        # Aggregate loss values
        self.perf_loss = tf.reduce_mean(tf.stack(perf_loss, axis=0))
        self.spike_loss = tf.reduce_mean(tf.stack(spike_loss, axis=0))
        self.loss = self.perf_loss + self.spike_loss

        # Use TensorFlow's Adam optimizer, and then apply the results
        opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        grads_and_vars = opt.compute_gradients(self.loss)


        #Apply any applicable weights masks to the gradient and clip
        capped_gvs = []
        for grad, var in grads_and_vars:
            if var.name == "rnn_cell/W_rnn_dend:0" and par['use_dendrites']:
                grad *= par['w_rnn_dend_mask']
                print('Applied weight mask to w_rnn_dend.')
            elif var.name == "rnn_cell/W_rnn_soma:0":
                grad *= par['w_rnn_soma_mask']
                print('Applied weight mask to w_rnn_soma.')
            elif var.name == "output/W_out:0":
                grad *= par['w_out_mask']
                print('Applied weight mask to w_out.')

            if not str(type(grad)) == "<class 'NoneType'>":
                capped_gvs.append((tf.clip_by_norm(grad, par['clip_max_grad_val']), var))
        print("\n")
        self.train_op = opt.apply_gradients(capped_gvs)


def main():
    """
    Builds and runs a task with the specified model, based on the current
    parameter setup and task scenario.
    """

    print('Using dendrites:\t', par['use_dendrites'])
    print('Using EI network:\t', par['EI'])
    print('Synaptic configuration:\t', par['synapse_config'], "\n")

    # Create the stimulus class to generate trial paramaters and input activity
    stim = stimulus.Stimulus()

    # Define all placeholders
    mask = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_train_size']])
    x = tf.placeholder(tf.float32, shape=[par['n_input'], par['num_time_steps'], par['batch_train_size']])
    y = tf.placeholder(tf.float32, shape=[par['n_output'], par['num_time_steps'], par['batch_train_size']])
    learning_rate = tf.placeholder(tf.float32)

    # Create the TensorFlow session
    with tf.Session() as sess:
        # Create the model in TensorFlow and start running the session
        model = Model(x, y, mask, learning_rate)
        init = tf.global_variables_initializer()
        t_start = time.time()
        sess.run(init)

        # Restore variables from previous model if desired
        saver = tf.train.Saver()
        if par['load_previous_model']:
            saver.restore(sess, par['save_dir'] + par['ckpt_load_fn'])
            print('Model ' + par['ckpt_load_fn'] + ' restored.')

        # Generate an identifying timestamp and save directory for the model
        timestamp = "_D" + time.strftime("%y-%m-%d") \
                    + "_T" + time.strftime("%H-%M-%S")
        if par['use_dendrites']:
            dirpath = './savedir/model_' + par['stimulus_type'] + '_h' + \
                        str(par['n_hidden']) + '_df' + par['df_num']+ timestamp
        else:
            dirpath = './savedir/model_' + par['stimulus_type'] + '_h' + \
                        str(par['n_hidden']) + 'nd' + timestamp

        # Make new folder for parameters, results, and analysis
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        # Store a copy of the parameters setup in its default state
        json_save(par, dirpath + '/parameters.json')

        # Create summary file
        with open(dirpath + '/model_summary.txt', 'w') as f:
            f.write('Trial\tTime\tPerf loss\tSpike loss\tMean activity\tTest Accuracy\n')

        # Keep track of the model performance across training
        model_results = {'accuracy': [], 'loss': [], 'perf_loss': [], \
                         'spike_loss': [], 'mean_hidden': [], 'trial': [], 'time': []}

        # Loop through the desired number of iterations
        for i in range(par['num_iterations']):

            # Reset any altered task parameters back to their defaults, then
            # switch the allowed rules if the iteration number crosses a
            # specified threshold
            set_task_profile()
            switch_rule(i)

            """
            Training
            """
            for j in range(par['num_train_batches']):

                # generate batch of par['batch_train_size'] trials
                trial_info = stim.generate_trial(par['batch_train_size'])

                # train the model
                _ = sess.run(model.train_op, {x: trial_info['neural_input'], y: trial_info['desired_output'], \
                    mask: trial_info['train_mask'], learning_rate: par['learning_rate']})

                # show model progress
                progress = (j+1)/par['num_train_batches']
                bar = int(np.round(progress*20))
                print("Training Model:\t [{}] ({:>3}%)\r".format("#"*bar + " "*(20-bar), int(np.round(100*progress))), end='\r')
            print("\nTraining session {:} complete.\n".format(i))


            # Allows all fields and rules for testing purposes
            par['allowed_fields']       = np.arange(par['num_RFs'])
            par['allowed_rules']        = np.arange(par['num_rules'])
            par['num_active_fields']    = len(par['allowed_fields'])

            # keep track of the model performance for this batch
            test_data = initialize_test_data()

            """
            Testing
            """
            for j in range(par['num_test_batches']):

                # generate batch of testing trials
                trial_info = stim.generate_trial(par['batch_train_size'])

                # run the model
                _, test_data['loss'][j], test_data['perf_loss'][j], test_data['spike_loss'][j], y_hat, \
                state_hist_batch, dend_hist_batch, dend_exc_hist_batch, dend_inh_hist_batch \
                = sess.run([model.train_op, model.loss, model.perf_loss, model.spike_loss, \
                    model.y_hat, model.hidden_state_hist, model.dendrites_hist, \
                    model.dendrites_inputs_exc_hist, model.dendrites_inputs_inh_hist], \
                    {x: trial_info['neural_input'], y: trial_info['desired_output'], \
                    mask: trial_info['train_mask'], learning_rate: 0})

                # calculate model accuracy and the mean activity of the hidden neurons for analysis
                test_data = append_test_data(test_data, trial_info, state_hist_batch, dend_hist_batch, dend_exc_hist_batch, dend_inh_hist_batch, j)
                test_data['accuracy'][j] = get_perf(trial_info['desired_output'], y_hat, trial_info['train_mask'])
                test_data['mean_hidden'][j] = np.mean(state_hist_batch)

                # show model progress
                progress = (j+1)/par['num_test_batches']
                bar = int(np.round(progress*20))
                print("Testing Model:\t [{}] ({:>3}%)\r".format("#"*bar + " "*(20-bar), int(np.round(100*progress))), end='\r')
            print("\nTesting session {:} complete.\n".format(i))


            """
            Analyze the data and save the results
            """

            testing_conditions = {'stimulus_type': par['stimulus_type'], 'allowed_fields' : par['allowed_fields'], 'allowed_rules' : par['allowed_rules']}
            iteration_time = time.time() - t_start

            model_results = append_model_performance(model_results, test_data, (i+1)*N, iteration_time)
            #model_results['weights'] = extract_weights()

            analysis_val = analysis.get_analysis(test_data)

            model_results = append_analysis_vals(model_results, analysis_val)
            print_data(dirpath, model_results, analysis_val)

            json_save([testing_conditions, analysis_val], dirpath + '/iter{}_results.json'.format(i))
            json_save(model_results, dirpath + '/model_results.json')


    print('\nModel execution complete.\n')


def switch_rule(iteration):

    if (iteration+1)%par['switch_rule_iteration'] == 0:
        new_allowed_rule = (par['allowed_rules'][0]+1)%par['num_rules']
        par['allowed_rules'] = [new_allowed_rule]
        print('allowed_rules now equal to ', new_allowed_rule)


def get_perf(y, y_hat, mask):

    """
    Calculate task accuracy by comparing the actual network output to the desired output
    only examine time points when test stimulus is on
    in another words, when y[0,:,:] is not 0
    """

    y_hat = np.stack(y_hat,axis=1)
    mask *= y[0,:,:]==0
    y = np.argmax(y, axis = 0)
    y_hat = np.argmax(y_hat, axis = 0)

    return np.sum(np.float32(y == y_hat)*np.squeeze(mask))/np.sum(mask)


def print_data(dirpath, model_results, analysis):

    hud.update_data(model_results['trial'][-1], model_results['perf_loss'][-1], model_results['accuracy'][-1])

    with open(dirpath + '/model_summary.txt', 'a') as f:
        # In order, Trial | Time | Perf Loss | Spike Loss | Mean Activity | Accuracy
        f.write('{:7d}'.format(model_results['trial'][-1]) \
            + '\t{:0.2f}'.format(model_results['time'][-1]) \
            + '\t{:0.4f}'.format(model_results['perf_loss'][-1]) \
            + '\t{:0.4f}'.format(model_results['spike_loss'][-1]) \
            + '\t{:0.4f}'.format(model_results['mean_hidden'][-1]) \
            + '\t{:0.4f}'.format(model_results['accuracy'][-1]) \
            + '\n')

    # output model performance to screen
    print('Trial: {:13.0f} | Time: {:15.2f} s |'.format(model_results['trial'][-1], model_results['time'][-1]))
    print('Perf. Loss: {:8.4f} | Mean Activity: {:8.4f} | Accuracy: {:8.4f}'.format( \
        model_results['perf_loss'][-1], model_results['mean_hidden'][-1], model_results['accuracy'][-1]))

    if not analysis['anova'] == []:
        anova_print = [k[:-5] + ':{:8.3f} '.format(np.mean(v[:,:,:,0]<0.001)) for k,v in analysis['anova'].items() if k.count('pval')>0]
        anova_print = ' | '.join(anova_print)
        print('Anova P<0.001, ' + anova_print)
    if not analysis['roc'] == []:
        roc_print = [k[:-5] + ':{:8.3f} '.format(np.percentile(np.abs(v[:,:,:,0]), 98)) for k,v in analysis['roc'].items()]
        roc_print = ' | '.join(roc_print)
        print('98th prctile t-stat, ' + roc_print)


    """
    print('ROC Value (Neuron): \t\t ROC Value (Dendrites):')
    for i in range(len(analysis['roc']['neurons'][1])):
        print(analysis['roc']['neurons'][1][i].round(2), '\t\t\t', analysis['roc']['dendrites'][1][i].round(2))

    print('ROC Value (Dend_excitatory): \t ROC Value (Dend_inhibitory):')
    for i in range(len(analysis['roc']['neurons'][1])):
        print(analysis['roc']['dendrite_exc'][1][i].round(2), '\t\t\t', analysis['roc']['dendrite_inh'][1][i].round(2))
    """
    print("\n")


def append_model_performance(model_results, test_data, trial_num, iteration_time):

    model_results['accuracy'].append(np.mean(test_data['accuracy']))
    model_results['loss'].append(np.mean(test_data['loss']))
    model_results['spike_loss'].append(np.mean(test_data['spike_loss']))
    model_results['perf_loss'].append(np.mean(test_data['perf_loss']))
    model_results['mean_hidden'].append(np.mean(test_data['mean_hidden']))
    model_results['trial'].append(trial_num)
    model_results['time'].append(iteration_time)

    return model_results


def extract_weights():

    with tf.variable_scope('rnn_cell', reuse=True):
        if par['use_dendrites']:
            W_in_dend = tf.get_variable('W_in_dend')
            W_rnn_dend = tf.get_variable('W_rnn_dend')
        W_in_soma = tf.get_variable('W_in_soma')
        W_rnn_soma = tf.get_variable('W_rnn_soma')
        b_rnn = tf.get_variable('b_rnn')

    with tf.variable_scope('output', reuse=True):
        W_out = tf.get_variable('W_out')
        b_out = tf.get_variable('b_out')

    weights = {'w_in_soma': W_in_soma.eval(),
        'w_rnn_soma': W_rnn_soma.eval(),
        'w_out': W_out.eval(),
        'b_rnn': b_rnn.eval(),
        'b_out': b_out.eval()}

    if par['use_dendrites']:
        weights['w_in_dend'] = W_in_dend.eval()
        weights['w_rnn_dend'] = W_rnn_dend.eval()

    return weights


def append_test_data(test_data, trial_info, state_hist_batch, dend_hist_batch, dend_exc_hist_batch, dend_inh_hist_batch, batch_num):

    trial_ind = range(batch_num*par['batch_train_size'], (batch_num+1)*par['batch_train_size'])

    # add stimulus information
    test_data['sample_index'][trial_ind,:] = trial_info['sample_index']
    test_data['rule_index'][trial_ind] = trial_info['rule_index']
    test_data['location_index'][trial_ind] = trial_info['location_index']

    # add neuronal activity
    test_data['state_hist'][:,:,trial_ind] = state_hist_batch
    if par['use_dendrites']:
        test_data['dend_hist'][:,:,:,trial_ind] = dend_hist_batch
        test_data['dend_exc_hist'][:,:,:,trial_ind] = dend_exc_hist_batch
        test_data['dend_inh_hist'][:,:,:,trial_ind] = dend_inh_hist_batch

    return test_data


def initialize_test_data():

    N = par['batch_train_size']*par['num_test_batches']

    test_data = {
        'loss'          : np.zeros((par['num_test_batches']), dtype=np.float32),
        'perf_loss'     : np.zeros((par['num_test_batches']), dtype=np.float32),
        'spike_loss'    : np.zeros((par['num_test_batches']), dtype=np.float32),
        'mean_hidden'   : np.zeros((par['num_test_batches']), dtype=np.float32),
        'accuracy'      : np.zeros((par['num_test_batches']), dtype=np.float32),

        'sample_index'  : np.zeros((N, par['num_RFs']), dtype=np.uint8),
        'location_index': np.zeros((N, 1), dtype=np.uint8),
        'rule_index'    : np.zeros((N, 1), dtype=np.uint8),
        'state_hist'    : np.zeros((par['num_time_steps'], par['n_hidden'], N), dtype=np.float32)
    }

    if par['use_dendrites']:
        test_data['dend_hist'] = np.zeros((par['num_time_steps'], par['n_hidden'], par['den_per_unit'], N), dtype=np.float32)
        test_data['dend_exc_hist'] = np.zeros((par['num_time_steps'], par['n_hidden'], par['den_per_unit'], N), dtype=np.float32)
        test_data['dend_inh_hist'] = np.zeros((par['num_time_steps'], par['n_hidden'], par['den_per_unit'], N), dtype=np.float32)

    return test_data


def append_analysis_vals(model_results, analysis_val):

    for k in analysis_val.keys():
        if not analysis_val[k] == []:
            for k1,v in analysis_val[k].items():
                current_key = k + '_' + k1
                if not current_key in model_results.keys():
                    model_results[current_key] = [v]
                else:
                    model_results[current_key].append([v])

    return model_results

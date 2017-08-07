###############################################################################
### Project: Recurrent Neural Network Research with Tensorflow              ###
### Authors: Nicolas Masse, Gregory Grant, Catherine Lee, Varun Iyer        ###
### Date:    3 August, 2017                                                 ###
###############################################################################

import tensorflow as tf
import numpy as np

from parameters import *
from model_saver import *
import model_utils as mu
import dendrite_functions as df
import metaweight as mw
import stimulus
import analysis

import os
import time

#################################
### Model setup and execution ###
#################################

class Model:

    def __init__(self, input_data, td_data, target_data, mask, learning_rate, template, *external):

        # Load the input activity, the target data, and the training mask
        # for this batch of trials
        self.input_data     = tf.unstack(input_data, axis=1)
        self.td_data        = tf.unstack(td_data, axis=1)
        self.target_data    = tf.unstack(target_data, axis=1)
        self.mask           = tf.unstack(mask, axis=0)
        self.learning_rate  = learning_rate
        self.template       = template
        self.weights, self.omegas   = mu.split_list(external)
        self.split_indices, _       = mu.split_list(par['external_index_feed'])


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
        self.rnn_cell_loop(self.input_data, self.td_data, self.hidden_init, self.dendrites_init, self.synapse_x_init, self.synapse_u_init)

        # Describes the output variables and scope
        with tf.variable_scope('output'):
            W_out = tf.get_variable('W_out', initializer = np.float32(par['w_out0']), trainable=True)
            b_out = tf.get_variable('b_out', initializer = np.float32(par['b_out0']), trainable=True)

        # Setting the desired network output, considering only
        # excitatory RNN projections
        self.y_hat = [tf.matmul(tf.nn.relu(W_out),h)+b_out for h in self.hidden_state_hist]


    def rnn_cell_loop(self, x_unstacked, td_unstacked, h, d, syn_x, syn_u):
        """
        Sets up the weights and baises for the hidden layer, then establishes
        the network computation
        """

        # Initialize weights and biases, with behavior changes based on
        # dendrite usage (with dendrites requies a rank higher of tensor)
        with tf.variable_scope('rnn_cell'):
            if par['use_dendrites']:

                W_rnn_dend = tf.get_variable('W_rnn_dend', initializer = np.float32(par['w_rnn_dend0']), trainable=True)
                W_stim_dend = tf.get_variable('W_stim_dend', initializer = np.float32(par['w_stim_dend0']), trainable=True)
                W_td_dend = tf.get_variable('W_td_dend', initializer = np.float32(par['w_td_dend0']), trainable=True)

            if par['use_stim_soma']:
                W_stim_soma = tf.get_variable('W_stim_soma', initializer = np.float32(par['w_stim_soma0']), trainable=True)
                W_td_soma = tf.get_variable('W_td_soma', initializer = np.float32(par['w_td_soma0']), trainable=True)

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
        for n_in in range(len(x_unstacked)):
            h, d, syn_x, syn_u, e, i = self.rnn_cell(x_unstacked[n_in], td_unstacked[n_in], h, d, syn_x, syn_u, self.template)
            self.hidden_state_hist.append(h)
            self.dendrites_hist.append(d)
            self.dendrites_inputs_exc_hist.append(e)
            self.dendrites_inputs_inh_hist.append(i)
            self.syn_x_hist.append(syn_x)
            self.syn_u_hist.append(syn_u)


    def rnn_cell(self, stim_in, td_in, h_soma, dend, syn_x, syn_u, template):
        """
        Run the main computation of the recurrent network
        """

        # Get the requisite variables for running the model.  Again, the
        # inclusion of dendrites changes the tensor shapes
        with tf.variable_scope('rnn_cell', reuse=True):
            if par['use_dendrites']:
                W_stim_dend = tf.get_variable('W_stim_dend')
                W_td_dend   = tf.get_variable('W_td_dend')
                W_rnn_dend  = tf.get_variable('W_rnn_dend')

            if par['use_stim_soma']:
                W_stim_soma = tf.get_variable('W_stim_soma')
                W_td_soma   = tf.get_variable('W_td_soma')
            else:
                W_stim_soma = np.zeros([par['n_hidden'], par['num_stim_tuned']], dtype=np.float32)
                W_td_soma   = np.zeros([par['n_hidden'], par['n_input'] - par['num_stim_tuned']], dtype=np.float32)

            W_rnn_soma  = tf.get_variable('W_rnn_soma')
            b_rnn       = tf.get_variable('b_rnn')
            W_ei        = tf.constant(par['EI_matrix'], name='W_ei')

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
            if par['df_num'] == '0009':
                h_soma_in, dend_out, exc_activity, inh_activity = \
                    dendrite_function(tf.nn.relu(W_stim_dend), tf.nn.relu(W_td_dend), \
                        tf.nn.relu(W_rnn_dend), stim_in, td_in, h_post_syn, dend, template)
            else:
                h_soma_in, dend_out, exc_activity, inh_activity = \
                    dendrite_function(tf.nn.relu(W_stim_dend), tf.nn.relu(W_td_dend), \
                        tf.nn.relu(W_rnn_dend), stim_in, td_in, h_post_syn, dend)
        else:
            dend_out = dend
            h_soma_in = 0

        # Apply, in order: alpha decay, dendritic input, soma recurrence,
        # bias terms, and Gaussian randomness.  This generates the output of
        # the hidden layer.
        h_soma_out = tf.nn.relu(h_soma*(1-par['alpha_neuron']) \
                     + par['alpha_neuron']*(h_soma_in + tf.matmul(W_rnn_soma_effective, h_post_syn) \
                     + tf.matmul(tf.nn.relu(W_stim_soma), tf.nn.relu(stim_in)) \
                     + tf.matmul(tf.nn.relu(W_td_soma), tf.nn.relu(td_in)) + b_rnn) \
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
        if par['loss_function'] == 'MSE':
            perf_loss = [mask*tf.reduce_mean(tf.square(y_hat-desired_output),axis=0) \
                for (y_hat, desired_output, mask) in zip(self.y_hat, self.target_data, self.mask)]

        elif par['loss_function'] == 'cross_entropy':
            perf_loss = [mask*tf.nn.softmax_cross_entropy_with_logits(logits = y_hat, labels = desired_output, dim=0) \
                for (y_hat, desired_output, mask) in zip(self.y_hat, self.target_data, self.mask)]

        # L2 penalty term on hidden state activity to encourage low spike rate solutions
        spike_loss = [par['spike_cost']*tf.reduce_mean(tf.square(h), axis=0) \
                    for (h, mask) in zip(self.hidden_state_hist, self.mask)]

        # L1 penalty term on dendrite activity to encourage sparseness
        if par['use_dendrites']:
            dend_loss = [par['dend_cost']*tf.reduce_mean(tf.abs(d), axis=0) \
                        for (d, mask) in zip(self.dendrites_hist, self.mask)]
        else:
            dend_loss = tf.constant(0, dtype=tf.float32)

        # L1 penalty term to encourage motifs
        # work in progress!
        m = np.zeros((par['n_hidden'], par['n_hidden']), dtype = np.float32)
        m[:par['n_hidden']//5, :par['n_hidden']//5] = 1
        mask = tf.constant(m)
        with tf.variable_scope('rnn_cell', reuse=True):
            W_rnn_soma  = tf.get_variable('W_rnn_soma')
        self.motif_loss = par['motif_cost']*tf.reduce_sum(mask*(tf.abs(tf.nn.relu(W_rnn_soma) - tf.transpose(tf.nn.relu(W_rnn_soma)))))
        """
        n = ((800//par['dt']) - 1) - 400//par['dt']
        # u_0, u_1, v_0, v_1, cov = [tf.placeholder(tf.float32, shape = n)]*5

        # print(self.hidden_state_hist)

        # u_0, v_0 = zip(*[tf.nn.moments(h, axes=[0,1]) for h in self.hidden_state_hist[(400//par['dt']):(800//par['dt'])-1]])
        # u_1, v_1 = zip(*[tf.nn.moments(h, axes=[0,1]) for h in self.hidden_state_hist[(400//par['dt'])+1:(800//par['dt'])]])
        desired_corr = np.zeros((40, 40))
        for i,j in itertools.product(range(20), range(20)):
            if i!=j:
                desired_corr[i, j] = 0.25

        desired_corr = tf.constant(desired_corr, dtype=tf.float32)

        mse = []
        for h_0, h_1 in zip(self.hidden_state_hist[(400//par['dt']):(800//par['dt'])-1], self.hidden_state_hist[(400//par['dt'])+1:(800//par['dt'])]):
            u_0, v_0 = tf.nn.moments(h_0, axes=1)
            u_1, v_1 = tf.nn.moments(h_1, axes=1)
            cov = tf.matmul(h_0 - tf.tile(tf.reshape(u_0, (40,1)), (1,100)), tf.transpose(h_1 - tf.tile(tf.reshape(u_1, (40,1)), (1,100))))/(100*100)
            b = tf.matmul(tf.reshape(v_0, (40,1)), tf.reshape(v_1, (1,40)))
            mse.append(tf.pow((cov - (desired_corr * b)), 2))
        """

        # Calculate omega loss
        # So far, only works if two rules/tasks
        weight_tf_vars = []
        with tf.variable_scope('rnn_cell', reuse=True):
            for name in par['working_weights'][:-1]:
                weight_tf_vars.append(tf.get_variable(name))
        with tf.variable_scope('output', reuse=True):
            weight_tf_vars.append(tf.get_variable(par['working_weights'][-1]))

        weight_prev_vars = []
        for i in range(len(self.weights)):
            if i in self.split_indices:
                weight_prev_vars.append(self.weights[i])

        omega_vars = []
        for i in range(len(self.omegas)):
            if i in self.split_indices:
                omega_vars.append(self.omegas[i])

        self.omega_loss = 0.
        for w1, w2, omega in zip(weight_prev_vars, weight_tf_vars, omega_vars):
            self.omega_loss += par['omega_cost'] * tf.reduce_sum(tf.multiply(omega, tf.square(w1 - w2)))

        # Aggregate loss values
        self.perf_loss = tf.reduce_mean(tf.stack(perf_loss, axis=0))
        self.spike_loss = tf.reduce_mean(tf.stack(spike_loss, axis=0))
        self.dend_loss = tf.reduce_mean(tf.stack(dend_loss, axis=0))
        #mse = tf.reduce_mean(tf.stack(mse, axis=0))

        self.loss = self.perf_loss + self.spike_loss + self.dend_loss + self.omega_loss

        # Use TensorFlow's Adam optimizer, and then apply the results
        opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        grads_and_vars = opt.compute_gradients(self.loss)

        # Print out the active, trainable TensorFlow variables
        print('Active TensorFlow variables:')
        for grad, var in grads_and_vars:
            print('  ', var)

        #Apply any applicable weights masks to the gradient and clip
        print('\nWeight masks:')
        self.capped_gvs = []
        for grad, var in grads_and_vars:
            if var.name == "rnn_cell/W_rnn_dend:0" and par['use_dendrites']:
                grad *= par['w_rnn_dend_mask']
                print('  Applied weight mask to w_rnn_dend.')
            elif var.name == "rnn_cell/W_rnn_soma:0":
                grad *= par['w_rnn_soma_mask']
                print('  Applied weight mask to w_rnn_soma.')

            elif var.name == "rnn_cell/W_stim_soma:0":
                grad *= par['w_stim_soma_mask']
                print('  Applied weight mask to w_stim_soma.')
            elif var.name == "rnn_cell/W_stim_dend:0":
                grad *= par['w_stim_dend_mask']
                print('  Applied weight mask to w_stim_dend.')

            elif var.name == "rnn_cell/W_td_soma:0":
                grad *= par['w_td_soma_mask']
                print('  Applied weight mask to w_td_soma.')
            elif var.name == "rnn_cell/W_td_dend:0":
                grad *= par['w_td_dend_mask']
                print('  Applied weight mask to w_td_dend.')

            elif var.name == "output/W_out:0":
                grad *= par['w_out_mask']
                print('  Applied weight mask to w_out.')

            if not str(type(grad)) == "<class 'NoneType'>":
                self.capped_gvs.append((tf.clip_by_norm(grad, par['clip_max_grad_val']), var))
        print("\n")

        self.train_op = opt.apply_gradients(self.capped_gvs)


def main():
    """
    Builds and runs a task with the specified model, based on the current
    parameter setup and task scenario.
    """

    print("\nRunning model...\n")

    # Reset TensorFlow before running anythin
    tf.reset_default_graph()

    # Show a chosen selection of parameters in the console
    mu.print_startup_info()

    # Create the stimulus class to generate trial parameters and input activity
    stim = stimulus.Stimulus()

    # Create graph placeholders
    g = mu.create_placeholders(par['general_placeholder_info'])
    o = mu.create_placeholders(par['other_placeholder_info'])
    e = mu.create_placeholders(par['external_placeholder_info'], True)

    # Create the TensorFlow session
    with tf.Session() as sess:
        # Create the model in TensorFlow and start running the session
        model = Model(*g, *o, *e)
        init = tf.global_variables_initializer()
        t_start = time.time()
        sess.run(init)

        # Restore variables from previous model if desired
        saver = tf.train.Saver()
        if par['load_previous_model']:
            saver.restore(sess, par['save_dir'] + par['ckpt_load_fn'])
            print('Model ' + par['ckpt_load_fn'] + ' restored.')

        # Generate an identifying timestamp and save directory for the model
        timestamp, dirpath = mu.create_save_dir()

        # Keep track of the model performance across training
        model_results = mu.initialize_model_results()
        test_data = mu.initialize_test_data()

        # Assemble the list of metaweights metaweights to use
        weight_tf_vars = []
        with tf.variable_scope('rnn_cell', reuse=True):
            for name in par['working_weights'][:-1]:
                weight_tf_vars.append(tf.get_variable(name))
        with tf.variable_scope('output', reuse=True):
            weight_tf_vars.append(tf.get_variable(par['working_weights'][-1]))

        # Ensure that the correct task settings are in place
        set_task_profile()

        # Loop through the desired number of iterations
        for i in range(par['num_iterations']):

            # Print iteration header
            print('='*40 + '\n' + '=== Iteration {:>3}'.format(i) + ' '*20 + '===\n' + '='*40 + '\n')

            # Reset any altered task parameters back to their defaults, then switch
            # the allowed rules if the iteration number crosses a specified threshold
            set_rule(i)

            # Reset omega_k
            w_k = [0]*(len(par['external_index_feed'])//2)

            # Training loop
            for j in range(par['num_train_batches']):

                # Generate batch of par['batch_train_size'] trials
                trial_info = stim.generate_trial(par['batch_train_size'])
                trial_stim  = trial_info['neural_input'][:par['num_stim_tuned'], :, :]
                trial_td    = trial_info['neural_input'][par['num_stim_tuned']:, :, :]

                # Allow for special dendrite functions
                template = set_template(trial_info['rule_index'], trial_info['location_index'])

                # Build feed_dict
                feed_stream = [trial_stim, trial_td, trial_info['desired_output'], trial_info['train_mask'], par['learning_rate'], template]
                feed_places = [*g, *o]

                e_feed_stream = []
                e_feed_places = []
                if (i > 0):
                    e_feed_stream = [*previous_weights, *omegas]
                    for es in par['external_index_feed']:
                        e_feed_places.append(e[es])

                feed_dict = mu.zip_to_dict(feed_places + e_feed_places, feed_stream + e_feed_stream)

                # Train the model
                _, grads, *new_weights = sess.run([model.train_op, model.capped_gvs, *weight_tf_vars], feed_dict)

                # Calculate metaweight values if desired, then plug them back into the graph
                print("MW Before: ", np.sum(new_weights[0]))
                print("MW Before: ", np.sum(new_weights[1]))
                print("MW Before: ", np.sum(new_weights[2]))
                print("MW Before: ", np.sum(new_weights[3]))
                if par['use_metaweights']:
                    new_weights = sess.run(list(map((lambda u, v, n: u.assign(mw.adjust(v, n))), weight_tf_vars, new_weights, par['working_weights'])))
                    print("MW After: ", np.sum(new_weights[0]))
                    print("MW After: ", np.sum(new_weights[1]))
                    print("MW After: ", np.sum(new_weights[2]))
                    print("MW After: ", np.sum(new_weights[3]))

                # Update omega_k
                z = 0
                num_bs = 0
                for grad, var in grads:
                    if np.shape(grad)[1] != 1:
                        w_k[z] += par['learning_rate'] * np.square(grad)
                        z += 1
                    else:
                        num_bs += 1
                        if num_bs > 2:
                            print("ERROR: Check number of bias matrices or make some weight matrix not have size 1 on axis 1")
                            quit()
                print('-'*20)

                # Generate weight matrix storage on the first trial
                if i == 0 and j == 0:
                    previous_weights = []
                    for l in range(len(new_weights)):
                        previous_weights.append(np.zeros(np.shape(new_weights[l])))

                # Show model progress
                progress = (j+1)/par['num_train_batches']
                bar = int(np.round(progress*20))
                print("Training Model:\t [{}] ({:>3}%)\r".format("#"*bar + " "*(20-bar), int(np.round(100*progress))), end='\r')
            print("\nTraining session {:} complete.\n".format(i))

            # Allows all fields and rules for testing purposes
            par['allowed_fields']       = np.arange(par['num_RFs'])
            par['allowed_rules']        = np.arange(par['num_rules'])
            par['num_active_fields']    = len(par['allowed_fields'])

            # Testing loop
            for j in range(par['num_test_batches']):

                # Generate batch of testing trials
                trial_info = stim.generate_trial(par['batch_train_size'])
                trial_stim  = trial_info['neural_input'][:par['num_stim_tuned'],:,:]
                trial_td    = trial_info['neural_input'][par['num_stim_tuned']:,:,:]

                # Allow for special dendrite functions
                template = set_template(trial_info['rule_index'], trial_info['location_index'])

                # Build feed_dict
                feed_stream = [trial_stim, trial_td, trial_info['desired_output'], trial_info['train_mask'], par['learning_rate'], template]
                feed_places = [*g, *o]

                e_feed_stream = []
                e_feed_places = []
                if (i > 0):
                    e_feed_stream = [*previous_weights, *omegas]
                    for es in par['external_index_feed']:
                        e_feed_places.append(e[es])

                feed_dict = mu.zip_to_dict(feed_places + e_feed_places, feed_stream + e_feed_stream)

                # Run the model
                test_data['y'][j], state_hist_batch, dend_hist_batch, dend_exc_hist_batch, dend_inh_hist_batch,\
                test_data['loss'][j], test_data['perf_loss'], test_data['spike_loss'], test_data['dend_loss'], test_data['omega_loss']\
                = sess.run([model.y_hat, model.hidden_state_hist, model.dendrites_hist,\
                    model.dendrites_inputs_exc_hist, model.dendrites_inputs_inh_hist,\
                    model.loss, model.perf_loss, model.spike_loss, model.dend_loss,\
                    model.omega_loss], feed_dict)

                # Aggregate the test data for analysis
                test_data = mu.append_test_data(test_data, trial_info, state_hist_batch, dend_hist_batch, dend_exc_hist_batch, dend_inh_hist_batch, j)
                test_data['y_hat'][j] = trial_info['desired_output']
                test_data['train_mask'][j] = trial_info['train_mask']
                test_data['mean_hidden'][j] = np.mean(state_hist_batch)

                # Show model progress
                progress = (j+1)/par['num_test_batches']
                bar = int(np.round(progress*20))
                print("Testing Model:\t [{}] ({:>3}%)\r".format("#"*bar + " "*(20-bar), int(np.round(100*progress))), end='\r')
            print("\nTesting session {:} complete.\n".format(i))

            # Calculate this iteration's omega value and reset the previous weight values
            omegas, previous_weights = calculate_omega(w_k, new_weights, previous_weights)
            mw.set_g(omegas)

            # Analyze the data and save the results
            iteration_time = time.time() - t_start
            N = par['batch_train_size']*par['num_train_batches']
            model_results = mu.append_model_performance(model_results, test_data, (i+1)*N, iteration_time)
            model_results['weights'] = mu.extract_weights()

            analysis_val = analysis.get_analysis(test_data, model_results['weights'])
            model_results = mu.append_analysis_vals(model_results, analysis_val)

            mu.print_data(dirpath, model_results, analysis_val)

            testing_conditions = {'stimulus_type': par['stimulus_type'], 'allowed_fields' : par['allowed_fields'], 'allowed_rules' : par['allowed_rules']}
            json_save([testing_conditions, analysis_val], dirpath + '/iter{}_results.json'.format(i))
            json_save(model_results, dirpath + '/model_results.json')

            if par['use_checkpoints']:
                saver.save(sess, os.path.join(dirpath, par['ckpt_save_fn']), i)

    print('\nModel execution complete.\n')


def set_rule(iteration):
    par['allowed_rules'] = [(iteration//par['switch_rule_iteration'])%2]
    print('Allowed task rule(s):', par['allowed_rules'])


def calculate_omega(w_k, new_weights, previous_weights):
    omegas = []
    for w_k_i, a, b in zip(w_k, new_weights, previous_weights):
        w_d = np.square(a-b)
        omega_array = w_k_i/(w_d + par['xi'])
        omegas.append(omega_array)

    return omegas, new_weights

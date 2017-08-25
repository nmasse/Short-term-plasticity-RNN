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
import ctypes

#################################
### Model setup and execution ###
#################################

class Model:

    def __init__(self, input_data, td_data, target_data, mask, learning_rate, template, lesion, *external):

        # Load the input activity, the target data, and the training mask
        # for this batch of trials
        self.input_data     = tf.unstack(input_data, axis=1)
        self.td_data        = tf.unstack(td_data, axis=1)
        self.target_data    = tf.unstack(target_data, axis=1)
        self.mask           = tf.unstack(mask, axis=0)
        self.learning_rate  = learning_rate
        self.template       = template
        self.lesion         = lesion
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

        # Initialize all variables
        self.initialize_variables()
        self.delta_mw = tf.constant(0)

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()

        # Engage the metaweights
        self.delta_mw = tf.constant(0)
        if par['use_metaweights']:
            self.run_metaweights()


    def initialize_variables(self):
        with tf.variable_scope('parameters'):
            if par['use_stim_soma']:
                with tf.variable_scope('soma'):
                    tf.get_variable('W_stim_soma',      initializer=np.float32(par['w_stim_soma0']),    trainable=True)
                    tf.get_variable('W_td_soma',        initializer=np.float32(par['w_td_soma0']),      trainable=True)
                    tf.get_variable('W_rnn_soma',       initializer=np.float32(par['w_rnn_soma0']),     trainable=True)
            if par['use_dendrites']:
                with tf.variable_scope('dendrite'):
                    tf.get_variable('W_stim_dend',      initializer=np.float32(par['w_stim_dend0']),    trainable=True)
                    tf.get_variable('W_td_dend',        initializer=np.float32(par['w_td_dend0']),      trainable=True)
                    tf.get_variable('W_rnn_dend',       initializer=np.float32(par['w_rnn_dend0']),     trainable=True)
            with tf.variable_scope('standard'):
                tf.get_variable('W_out',                initializer=np.float32(par['w_out0']),          trainable=True)
                tf.get_variable('b_rnn',                initializer=np.float32(par['b_rnn0']),          trainable=True)
                tf.get_variable('b_out',                initializer=np.float32(par['b_out0']),          trainable=True)

        if par['use_metaweights']:
            with tf.variable_scope('meta'):
                if par['use_stim_soma']:
                    with tf.variable_scope('soma'):
                        tf.get_variable('W_stim_soma',  initializer=np.float32(par['U_stim_soma0']),    trainable=False)
                        tf.get_variable('W_td_soma',    initializer=np.float32(par['U_td_soma0']),      trainable=False)
                        tf.get_variable('W_rnn_soma',   initializer=np.float32(par['U_rnn_soma0']),     trainable=False)
                if par['use_dendrites']:
                    with tf.variable_scope('dendrite'):
                        tf.get_variable('W_stim_dend',  initializer=np.float32(par['U_stim_dend0']),    trainable=False)
                        tf.get_variable('W_td_dend',    initializer=np.float32(par['U_td_dend0']),      trainable=False)
                        tf.get_variable('W_rnn_dend',   initializer=np.float32(par['U_rnn_dend0']),     trainable=False)
                with tf.variable_scope('standard'):
                    tf.get_variable('W_out',            initializer=np.float32(par['U_out0']),          trainable=False)
                    tf.get_variable('b_rnn',            initializer=np.float32(par['U_brnn0']),         trainable=False)
                    tf.get_variable('b_out',            initializer=np.float32(par['U_bout0']),         trainable=False)

        if par['use_metaweights']:
            with tf.variable_scope('engine'):
                if par['use_stim_soma']:
                    with tf.variable_scope('soma'):
                        tf.get_variable('W_stim_soma',  initializer=np.float32(par['U_stim_soma0'][...,0]),    trainable=False)
                        tf.get_variable('W_td_soma',    initializer=np.float32(par['U_td_soma0'][...,0]),      trainable=False)
                        tf.get_variable('W_rnn_soma',   initializer=np.float32(par['U_rnn_soma0'][...,0]),     trainable=False)
                if par['use_dendrites']:
                    with tf.variable_scope('dendrite'):
                        tf.get_variable('W_stim_dend',  initializer=np.float32(par['U_stim_dend0'][...,0]),    trainable=False)
                        tf.get_variable('W_td_dend',    initializer=np.float32(par['U_td_dend0'][...,0]),      trainable=False)
                        tf.get_variable('W_rnn_dend',   initializer=np.float32(par['U_rnn_dend0'][...,0]),     trainable=False)
                with tf.variable_scope('standard'):
                    tf.get_variable('W_out',            initializer=np.float32(par['U_out0'][...,0]),          trainable=False)
                    tf.get_variable('b_rnn',            initializer=np.float32(par['U_brnn0'][...,0]),         trainable=False)
                    tf.get_variable('b_out',            initializer=np.float32(par['U_bout0'][...,0]),         trainable=False)


    def run_model(self):
        """
        Run the recurrent network model using the set parameters
        """

        # The RNN Cell Loop contains the majority of the model calculations
        self.rnn_cell_loop(self.input_data, self.td_data, self.hidden_init, self.dendrites_init, self.synapse_x_init, self.synapse_u_init)

        # Describes the output variables and scope
        with tf.variable_scope('parameters', reuse=True), tf.variable_scope('standard'):
            W_out = tf.get_variable('W_out')
            b_out = tf.get_variable('b_out')

        # Setting the desired network output, considering only
        # excitatory RNN projections
        self.y_hat = [tf.matmul(tf.nn.relu(W_out),h)+b_out for h in self.hidden_state_hist]


    def rnn_cell_loop(self, x_unstacked, td_unstacked, h, d, syn_x, syn_u):
        """
        Sets up the weights and baises for the hidden layer, then establishes
        the network computation
        """

        ######## Used to initialize variables here ##################
        """
        Basically initialized anything in RNN_Cell
        """

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

        # Get the requisite variables for running the model.  Note that
        # the inclusion of dendrites changes some of the tensor shapes
        with tf.variable_scope('parameters', reuse=True):
            if par['use_dendrites']:
                with tf.variable_scope('dendrite'):
                    W_rnn_dend  = tf.get_variable('W_rnn_dend')
                    W_stim_dend = tf.get_variable('W_stim_dend')

                    W_td_dend   = tf.get_variable('W_td_dend')
            if par['use_stim_soma']:
                with tf.variable_scope('soma'):
                    W_rnn_soma  = tf.get_variable('W_rnn_soma')
                    W_stim_soma = tf.get_variable('W_stim_soma')
                    W_td_soma   = tf.get_variable('W_td_soma')
            else:
                W_stim_soma     = np.zeros([par['n_hidden'], par['num_stim_tuned']], dtype=np.float32)
                W_td_soma       = np.zeros([par['n_hidden'], par['n_input'] - par['num_stim_tuned']], dtype=np.float32)

            with tf.variable_scope('standard'):
                b_rnn           = tf.get_variable('b_rnn')

            W_ei                = tf.constant(par['EI_matrix'], name='W_ei')

        # If using an excitatory-inhibitory network, ensures that E neurons
        # have only positive outgoing weights, and that I neurons have only
        # negative outgoing weights.  Dendritic EI is taken care of in the
        # dendrite functions.
        if par['EI']:
            W_rnn_soma_effective = tf.multiply(tf.matmul(tf.nn.relu(W_rnn_soma), W_ei), self.lesion)
        else:
            W_rnn_soma_effective = tf.multiply(W_rnn_soma, self.lesion)

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


    def run_metaweights(self):
        omega_vars, par_vars = mu.intersection_by_shape(mu.sort_tf_vars(self.omegas), mu.get_vars_in_scope('parameters'))
        par_vars, meta_vars  = mu.intersection_by_shape(par_vars, mu.get_vars_in_scope('meta'), flag='meta')
        par_vars, eng_vars   = mu.intersection_by_shape(par_vars, mu.get_vars_in_scope('engine'))
        par_vars = mu.filter_adams(par_vars)

        self.delta_mw = tf.constant(0., dtype=tf.float32)
        R = []
        for weight, U, omega, R in zip(par_vars, meta_vars, omega_vars, eng_vars):
            delta_weight, delta_U, delta_R = tf.py_func(mw.adjust, [weight, U, tf.ones_like(omega), par['C_multiplier']*omega+1, R], [tf.float32, tf.float32, tf.float32], name='MWAdjust')
            weight += delta_weight
            U += delta_U

            R += delta_R

            self.delta_mw += tf.reduce_mean(weight - tf.reduce_mean(U, -1))


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
        """
        m = np.zeros((par['n_hidden'], par['n_hidden']), dtype = np.float32)
        m[:par['n_hidden']//5, :par['n_hidden']//5] = 1
        mask = tf.constant(m)
        with tf.variable_scope('rnn_cell', reuse=True):
            W_rnn_soma  = tf.get_variable('W_rnn_soma')
        self.motif_loss = par['motif_cost']*tf.reduce_sum(mask*(tf.abs(tf.nn.relu(W_rnn_soma) - tf.transpose(tf.nn.relu(W_rnn_soma)))))
        """

        # Calculate omega loss
        weight_prev_vars = []
        for i in range(len(self.weights)):
            if i in self.split_indices:
                weight_prev_vars.append(self.weights[i])
        weight_prev_vars = mu.sort_tf_vars(weight_prev_vars)

        omega_vars = []
        for i in range(len(self.omegas)):
            if i in self.split_indices:
                omega_vars.append(self.omegas[i])
        omega_vars = mu.sort_tf_vars(omega_vars)

        # Checks the omega and prev_weight vars, then restricts the tf vars
        # according to the allowed shapes
        omega_vars, weight_prev_vars = mu.intersection_by_shape(omega_vars, weight_prev_vars)
        weight_tf_vars, omega_vars = mu.intersection_by_shape(mu.get_vars_in_scope('parameters'), omega_vars)

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
        self.grads_and_vars = mu.sort_tf_grads_and_vars(opt.compute_gradients(self.loss))

        # Print out the active, trainable TensorFlow variables
        print('Active TensorFlow variables:')
        for grad, var in self.grads_and_vars:
            print('  ', var.name.ljust(36), 'shape =', var.shape)

        #Apply any applicable weights masks to the gradient and clip
        print('\nWeight masks:')
        self.capped_gvs = []
        if par['use_metaweights'] and par['use_mw_engine']:
            eng = mu.get_vars_in_scope('engine')
        else:
            eng = [0]*len(self.grads_and_vars)
        for (grad, var), R in zip(self.grads_and_vars, eng):

            if par['use_metaweights'] and par['use_mw_engine']:
                grad /= (2/(1+tf.exp(5*R)))

            if var.name == "parameters/soma/W_rnn_soma:0":
                grad *= par['w_rnn_soma_mask']
                print('   Applied weight mask to w_rnn_soma.')
            elif var.name == "parameters/dendrite/W_rnn_dend:0" and par['use_dendrites']:
                grad *= par['w_rnn_dend_mask']
                print('   Applied weight mask to w_rnn_dend.')

            elif var.name == "parameters/soma/W_stim_soma:0":
                grad *= par['w_stim_soma_mask']
                print('   Applied weight mask to w_stim_soma.')
            elif var.name == "parameters/dendrite/W_stim_dend:0":
                grad *= par['w_stim_dend_mask']
                print('   Applied weight mask to w_stim_dend.')

            elif var.name == "parameters/soma/W_td_soma:0":
                grad *= par['w_td_soma_mask']
                print('   Applied weight mask to w_td_soma.')
            elif var.name == "parameters/dendrite/W_td_dend:0":
                grad *= par['w_td_dend_mask']
                print('   Applied weight mask to w_td_dend.')

            elif var.name == "output/W_out:0":
                grad *= par['w_out_mask']
                print('   Applied weight mask to w_out.')

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

    lesion_results = {
        'accuracy_test' : [],
        'accuracy_diff' : [],
        'imp_synapse'   : []
    }

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
        weight_tf_vars = [var for var in mu.get_vars_in_scope('parameters') \
        if any(val in var.name for val in par['working_weights']) and ('Adam' not in var.name)]
        weight_tf_vars = mu.sort_tf_vars(weight_tf_vars)

        # Ensure that the correct task settings are in place
        set_task_profile()

        if par['use_lesion']:
            accuracy_diff = np.zeros((par['num_iterations']//par['lesion_iter'], par['n_hidden'],par['n_hidden'],par['num_rules']), dtype=np.float32)
            accuracy_test = np.zeros((par['num_iterations']//par['lesion_iter'], par['n_hidden'],par['n_hidden'],par['num_rules']), dtype=np.float32)
            imp_synapse = np.zeros((par['num_iterations']//par['lesion_iter'], par['num_rules'], 2))

        # Loop through the desired number of iterations
        for i in range(par['num_iterations']):
            q = np.ones((par['n_hidden'],par['n_hidden']))
            # Print iteration header
            print('='*40 + '\n' + '=== Iteration {:>3}'.format(i) + ' '*20 + '===\n' + '='*40 + '\n')

            # Reset any altered task parameters back to their defaults, then switch
            # the allowed rules if the iteration number crosses a specified threshold
            set_rule(i)

            # Reset omega_k and previous performance loss
            w_k = [0]*(len(par['external_index_feed'])//2)
            previous_loss = np.float32(0)

            # Training loop
            for j in range(par['num_train_batches']):

                # Generate batch of par['batch_train_size'] trials
                trial_info  = stim.generate_trial(par['batch_train_size'])
                trial_stim  = trial_info['neural_input'][:par['num_stim_tuned'], :, :]
                trial_td    = trial_info['neural_input'][par['num_stim_tuned']:, :, :]

                # Allow for special dendrite functions
                template = set_template(trial_info['rule_index'], trial_info['location_index'])

                # Build feed_dict
                feed_stream = [trial_stim, trial_td, trial_info['desired_output'], trial_info['train_mask'], par['learning_rate'], template, q]
                feed_places = [*g, *o]

                e_feed_stream = []
                e_feed_places = []
                if (i > 0):
                    e_feed_stream = [*previous_weights, *omegas]
                    for es in par['external_index_feed']:
                        e_feed_places.append(e[es])

                feed_dict = mu.zip_to_dict(feed_places + e_feed_places, feed_stream + e_feed_stream)

                # Train the model
                _, delta_mw, perf_loss, grads_and_vars, *new_weights = \
                    sess.run([model.train_op, model.delta_mw, model.perf_loss, \
                    model.grads_and_vars, *weight_tf_vars], feed_dict)

                #Performance loss difference
                loss_diff = np.abs(perf_loss - previous_loss)
                previous_loss = perf_loss

                num_bs = 0
                z = 0
                if j == 0:
                    prev_grad = [0]*(len(par['external_index_feed'])//2) #prev_grad = [0]*(len(par['external_index_feed'])//2)
                    prev_var = [0]*(len(par['external_index_feed'])//2)  #prev_var = [0]*(len(par['external_index_feed'])//2)
                for grad, var in mu.sort_grads_and_vars(grads_and_vars):
                    if np.shape(var)[1] != 1:
                        if j == 0:
                            prev_var[z] = var
                            prev_grad[z] = grad
                            z += 1
                        if j > 0:
                            w_k[z] -= (var - prev_var[z])*prev_grad[z]
                            prev_var[z] = var
                            prev_grad[z] = grad
                            z += 1
                    else:
                        num_bs += 1
                        if num_bs > 2:
                            print("ERROR: Check number of bias matrices or make some weight matrix not have size 1 on axis 1")
                            quit()

                # Generate weight matrix storage on the first trial
                if i == 0 and j == 0:
                    previous_weights = []
                    for l in range(len(new_weights)):
                        previous_weights.append(np.zeros(np.shape(new_weights[l])))

                # Show model progress
                progress = (j+1)/par['num_train_batches']
                bar = int(np.round(progress*20))
                print("Training Model:\t [{}] ({:>3}%) --- Delta MW: {:.4f}\r".format("#"*bar + " "*(20-bar), int(np.round(100*progress)), delta_mw), end='\r')
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
                feed_stream = [trial_stim, trial_td, trial_info['desired_output'], trial_info['train_mask'], par['learning_rate'], template, q]
                feed_places = [*g, *o]

                e_feed_stream = []
                e_feed_places = []
                if (i > 0):
                    e_feed_stream = [*previous_weights, *omegas]
                    for es in par['external_index_feed']:
                        e_feed_places.append(e[es])

                feed_dict = mu.zip_to_dict(feed_places + e_feed_places, feed_stream + e_feed_stream)

                # Run the model
                if par['test_with_optimizer']:
                    test_data['y'][j], state_hist_batch, dend_hist_batch, dend_exc_hist_batch, dend_inh_hist_batch,\
                    test_data['loss'][j], test_data['perf_loss'], test_data['spike_loss'], test_data['dend_loss'], test_data['omega_loss']\
                    = sess.run([model.y_hat, model.hidden_state_hist, model.dendrites_hist,\
                        model.dendrites_inputs_exc_hist, model.dendrites_inputs_inh_hist,\
                        model.loss, model.perf_loss, model.spike_loss, model.dend_loss,\
                        model.omega_loss], feed_dict)
                else:
                    test_data['y'][j], state_hist_batch, dend_hist_batch, dend_exc_hist_batch, dend_inh_hist_batch,\
                    = sess.run([model.y_hat, model.hidden_state_hist, model.dendrites_hist,\
                                model.dendrites_inputs_exc_hist, model.dendrites_inputs_inh_hist], feed_dict)


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

            # Analyze the data and save the results
            iteration_time = time.time() - t_start
            N = par['batch_train_size']*par['num_train_batches']
            model_results = mu.append_model_performance(model_results, test_data, (i+1)*N, iteration_time)
            model_results['weights'] = mu.extract_weights()

            analysis_val = analysis.get_analysis(test_data, model_results['weights'])
            model_results = mu.append_analysis_vals(model_results, analysis_val)

            mu.print_data(dirpath, model_results, analysis_val)


            # LESION WEIGHTS
            if par['use_lesion'] and (i%par['lesion_iter']==(par['lesion_iter']-1)):
                for n1 in range(par['n_hidden']):
                    for n2 in range(par['n_hidden']):
                        # simulate network
                        stim = stimulus.Stimulus()
                        test_data = mu.initialize_test_data()

                        # lesion weights
                        q = np.ones((par['n_hidden'],par['n_hidden']))
                        q[n1,n2] = 0
                        for j in range(par['num_test_batches']):
                            trial_info = stim.generate_trial(par['batch_train_size'])
                            trial_stim  = trial_info['neural_input'][:par['num_stim_tuned'],:,:]
                            trial_td    = trial_info['neural_input'][par['num_stim_tuned']:,:,:]

                            # Allow for special dendrite functions
                            template = set_template(trial_info['rule_index'], trial_info['location_index'])

                            # Build feed_dict
                            feed_stream = [trial_stim, trial_td, trial_info['desired_output'], trial_info['train_mask'], 0, template, q]
                            feed_places = [*g, *o]

                            feed_dict = mu.zip_to_dict(feed_places, feed_stream)

                            # Run the model
                            test_data['y'][j], state_hist_batch, dend_hist_batch,\
                            = sess.run([model.y_hat, model.hidden_state_hist, model.dendrites_hist], feed_dict)
                            test_data['y_hat'][j] = trial_info['desired_output']
                            test_data['train_mask'][j] = trial_info['train_mask']

                            trial_ind = range(j*par['batch_train_size'], (j+1)*par['batch_train_size'])
                            test_data['rule_index'][trial_ind] = trial_info['rule_index']

                        _, accuracy_test[i//par['lesion_iter'], n1,n2] = analysis.get_perf(test_data)

                        # Show lesion progress
                        progress = (n1*par['n_hidden']+n2+1)/(par['n_hidden']*par['n_hidden'])
                        bar = int(np.round(progress*20))
                        print("Lesioning Model:\t [{}] ({:>3}%)\r".format("#"*bar + " "*(20-bar), int(np.round(100*progress))), end='\r')

                        accuracy_diff[i//par['lesion_iter'], n1, n2] = (analysis_val['rule_accuracy'][0] - accuracy_test[i//par['lesion_iter'], n1,n2][0], \
                                                                        analysis_val['rule_accuracy'][1] - accuracy_test[i//par['lesion_iter'], n1,n2][1])

                for rule in range(par['num_rules']):
                    temp = np.argmax(accuracy_diff[i//par['lesion_iter'],:,:,rule])
                    imp_synapse[i//par['lesion_iter'], rule] = (temp//par['n_hidden'], temp%par['n_hidden'])

            testing_conditions = {'stimulus_type': par['stimulus_type'], 'allowed_fields' : par['allowed_fields'], 'allowed_rules' : par['allowed_rules']}
            json_save([testing_conditions, analysis_val], dirpath + '/iter{}_results.json'.format(i))
            json_save(model_results, dirpath + '/model_results.json')

            if par['use_checkpoints']:
                saver.save(sess, os.path.join(dirpath, par['ckpt_save_fn']), i)

        if par['use_lesion']:
            lesion_results['accuracy_test'] = accuracy_test
            lesion_results['accuracy_diff'] = accuracy_diff
            lesion_results['imp_synapse'] = imp_synapse
            json_save(lesion_results, dirpath+'/lesion_weights.json')

    print('\nModel execution complete.\n')

    if par['notify_on_completion']:
        ctypes.windll.user32.MessageBoxW(0, 'Model execution complete.', 'NN Notification', 1)


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

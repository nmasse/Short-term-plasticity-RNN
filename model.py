"""
Nicolas Masse 2017
Contributions from Gregory Grant, Catherine Lee
"""

import tensorflow as tf
import numpy as np
import stimulus
import time
import analysis
from parameters import *
import os, sys

# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

print('Using EI Network:\t', par['EI'])
print('Synaptic configuration:\t', par['synapse_config'], "\n")

"""
Model setup and execution
"""

class Model:

    def __init__(self, input_data, target_data, mask):

        # Load the input activity, the target data, and the training mask for this batch of trials
        self.input_data = tf.unstack(input_data, axis=1)
        self.target_data = tf.unstack(target_data, axis=1)
        self.mask = tf.unstack(mask, axis=0)

        # Load the initial hidden state activity to be used at the start of each trial
        self.hidden_init = tf.constant(par['h_init'])

        # Load the initial synaptic depression and facilitation to be used at the start of each trial
        self.synapse_x_init = tf.constant(par['syn_x_init'])
        self.synapse_u_init = tf.constant(par['syn_u_init'])

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()


    def run_model(self):

        """
        Run the reccurent network
        History of hidden state activity stored in self.hidden_state_hist
        """
        self.rnn_cell_loop(self.input_data, self.hidden_init, self.synapse_x_init, self.synapse_u_init)

        with tf.variable_scope('output'):
            W_out = tf.get_variable('W_out', initializer = par['w_out0'], trainable=True)
            b_out = tf.get_variable('b_out', initializer = par['b_out0'], trainable=True)

        """
        Network output
        Only use excitatory projections from the RNN to the output layer
        """
        self.y_hat = [tf.matmul(tf.nn.relu(W_out),h)+b_out for h in self.hidden_state_hist]


    def rnn_cell_loop(self, x_unstacked, h, syn_x, syn_u):

        """
        Initialize weights and biases
        """
        with tf.variable_scope('rnn_cell'):
            W_in = tf.get_variable('W_in', initializer = par['w_in0'], trainable=True)
            W_rnn = tf.get_variable('W_rnn', initializer = par['w_rnn0'], trainable=True)
            b_rnn = tf.get_variable('b_rnn', initializer = par['b_rnn0'], trainable=True)
        self.W_ei = tf.constant(par['EI_matrix'])

        self.hidden_state_hist = []
        self.syn_x_hist = []
        self.syn_u_hist = []

        """
        Loop through the neural inputs to the RNN, indexed in time
        """
        for rnn_input in x_unstacked:
            h, syn_x, syn_u = self.rnn_cell(rnn_input, h, syn_x, syn_u)
            self.hidden_state_hist.append(h)
            self.syn_x_hist.append(syn_x)
            self.syn_u_hist.append(syn_u)


    def rnn_cell(self, rnn_input, h, syn_x, syn_u):

        """
        Main computation of the recurrent network
        """
        with tf.variable_scope('rnn_cell', reuse=True):
            W_in = tf.get_variable('W_in')
            W_rnn = tf.get_variable('W_rnn')
            b_rnn = tf.get_variable('b_rnn')

        if par['EI']:
            # ensure excitatory neurons only have postive outgoing weights,
            # and inhibitory neurons have negative outgoing weights
            W_rnn_effective = tf.matmul(tf.nn.relu(W_rnn), self.W_ei)
        else:
            W_rnn_effective = W_rnn_drop

        """
        Update the synaptic plasticity paramaters
        """
        if par['synapse_config'] == 'std_stf':
            # implement both synaptic short term facilitation and depression
            syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_u*syn_x*h
            syn_u += par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
            syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
            syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
            h_post = syn_u*syn_x*h

        elif par['synapse_config'] == 'std':
            # implement synaptic short term derpression, but no facilitation
            # we assume that syn_u remains constant at 1
            syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_x*h
            syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
            syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
            h_post = syn_x*h

        elif par['synapse_config'] == 'stf':
            # implement synaptic short term facilitation, but no depression
            # we assume that syn_x remains constant at 1
            syn_u += par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
            syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
            h_post = syn_u*h

        else:
            # no synaptic plasticity
            h_post = h

        """
        Update the hidden state
        Only use excitatory projections from input layer to RNN
        All input and RNN activity will be non-negative
        """
        h = tf.nn.relu(h*(1-par['alpha_neuron'])
                       + par['alpha_neuron']*(tf.matmul(tf.nn.relu(W_in), tf.nn.relu(rnn_input))
                       + tf.matmul(W_rnn_effective, h_post) + b_rnn)
                       + tf.random_normal([par['n_hidden'], par['batch_train_size']], 0, par['noise_rnn'], dtype=tf.float32))

        return h, syn_x, syn_u


    def optimize(self):

        """
        Calculate the loss functions and optimize the weights

        perf_loss = [mask*tf.reduce_mean(tf.square(y_hat-desired_output),axis=0)
                     for (y_hat, desired_output, mask) in zip(self.y_hat, self.target_data, self.mask)]
        """
        """
        cross_entropy
        """
        perf_loss = [mask*tf.nn.softmax_cross_entropy_with_logits(logits = y_hat, labels = desired_output, dim=0) \
                for (y_hat, desired_output, mask) in zip(self.y_hat, self.target_data, self.mask)]

        self.MSE = [mask*tf.reduce_mean(tf.square(y_hat-desired_output),axis=0)
                     for (y_hat, desired_output, mask) in zip(self.y_hat, self.target_data, self.mask)]


        # L2 penalty term on hidden state activity to encourage low spike rate solutions
        spike_loss = [par['spike_cost']*tf.reduce_mean(tf.square(h), axis=0) for h in self.hidden_state_hist]


        with tf.variable_scope('rnn_cell', reuse = True):
            W_in = tf.get_variable('W_in')
            W_rnn = tf.get_variable('W_rnn')
        with tf.variable_scope('output', reuse = True):
            W_out = tf.get_variable('W_out')
        self.wiring_loss = tf.reduce_sum(tf.nn.relu(W_in)) + tf.reduce_sum(tf.nn.relu(W_rnn)) + tf.reduce_sum(tf.nn.relu(W_out))
        self.wiring_loss *= par['wiring_cost']

        self.perf_loss = tf.reduce_mean(tf.stack(perf_loss, axis=0))
        self.spike_loss = tf.reduce_mean(tf.stack(spike_loss, axis=0))

        self.loss = self.perf_loss + self.spike_loss + self.wiring_loss

        opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        self.grads_and_vars = opt.compute_gradients(self.loss)

        """
        Apply any applicable weights masks to the gradient and clip
        """
        self.capped_gvs = []
        grads_vars_to_apply = []
        for grad, var in self.grads_and_vars:
            if var.name == "rnn_cell/W_rnn:0":
                grad *= par['w_rnn_mask']
                self.delta_w_rnn = grad
                print('Applied weight mask to w_rnn.')
            elif var.name == "output/W_out:0":
                grad *= par['w_out_mask']
                print('Applied weight mask to w_out.')
                grads_vars_to_apply.append((tf.clip_by_norm(grad, par['clip_max_grad_val']), var))
            elif var.name == "rnn_cell/W_in:0":
                self.delta_w_in = grad
                print('DELTA WIN', self.delta_w_in)
            elif var.name == "output/b_out:0":
                grads_vars_to_apply.append((tf.clip_by_norm(grad, par['clip_max_grad_val']), var))
            if not str(type(grad)) == "<class 'NoneType'>":
                self.capped_gvs.append((tf.clip_by_norm(grad, par['clip_max_grad_val']), var))


        #self.train_op = opt.apply_gradients(grads_vars_to_apply)
        self.train_op = opt.apply_gradients(self.capped_gvs)



class Local_learning_model():

    def __init__(self, neural_output, sum_exc_current, sum_inh_current, mse, delta_W_in, delta_W_rnn, *synaptic_currents):

        self.neural_output = neural_output
        self.sum_exc_current = sum_exc_current
        self.sum_inh_current = sum_inh_current
        print('mse original', mse)
        self.mse =  tf.unstack(mse, axis = 0)
        self.delta_W_in = delta_W_in
        self.delta_W_rnn = delta_W_rnn
        self.synaptic_currents = synaptic_currents

        self.transforms = [tf.constant(t) for t in par['target_transforms']]


        self.create_graph()

        self.optimize()

    def create_graph(self):

        # neural_activity = tf.placeholder(tf.float32, shape=[par['num_input_neurons']+par['n_hidden'], par['num_time_steps'], par['batch_train_size']])

        initial_tc = np.float32(0.9)
        initial_u = np.float32(0.1)
        initial_alpha = np.float32(1.)
        initial_bias = np.float32(0.)

        delta_weights = []

        for (i ,name) in enumerate(['input-exc', 'exc-exc', 'inh-exc', 'input-inh', 'exc-inh', 'inh-inh']):
            with tf.variable_scope(name):
                # tc and u are involved in dx/dt = -x/tc + x*u
                # both low pass and instantaneous are combined as alpha*x + bias

                # low pass of single synaptic input
                pre_low_tc = tf.get_variable('pre_low_tc', initializer = initial_tc, trainable=True)
                pre_low_u = tf.get_variable('pre_low_u', initializer = initial_u, trainable=True)
                pre_low_bias = tf.get_variable('pre_low_bias', initializer = initial_bias, trainable=True)
                pre_low_alpha = tf.get_variable('pre_low_alpha', initializer = initial_alpha, trainable=True)

                # instananeuos single synaptic input
                pre_bias = tf.get_variable('pre_bias', initializer = initial_bias, trainable=True)
                pre_alpha = tf.get_variable('pre_alpha', initializer = initial_alpha, trainable=True)

                # low pass of sum of EXC synaptic input
                pre_exc_low_tc = tf.get_variable('pre_exc_low_tc', initializer = initial_tc, trainable=True)
                pre_exc_low_u = tf.get_variable('pre_exc_low_u', initializer = initial_u, trainable=True)
                pre_exc_low_bias = tf.get_variable('pre_exc_low_bias', initializer = np.float32(0.), trainable=True)
                pre_exc_low_alpha = tf.get_variable('pre_exc_low_alpha', initializer = initial_alpha, trainable=True)

                # instananeuos of sum of EXC synaptic input
                pre_exc_bias = tf.get_variable('pre_exc_bias', initializer = initial_bias, trainable=True)
                pre_exc_alpha = tf.get_variable('pre_exc_alpha', initializer = initial_alpha, trainable=True)

                # low pass of sum of INH synaptic input
                pre_inh_low_tc = tf.get_variable('pre_inh_low_tc', initializer = initial_tc, trainable=True)
                pre_inh_low_u = tf.get_variable('pre_inh_low_u', initializer = initial_u, trainable=True)
                pre_inh_low_bias = tf.get_variable('pre_inh_low_bias', initializer = np.float32(0.), trainable=True)
                pre_inh_low_alpha = tf.get_variable('pre_inh_low_alpha', initializer = initial_alpha, trainable=True)

                # instananeuos of sum of INH synaptic input
                pre_inh_bias = tf.get_variable('pre_inh_bias', initializer = initial_bias, trainable=True)
                pre_inh_alpha = tf.get_variable('pre_inh_alpha', initializer = initial_alpha, trainable=True)

                # low pass of neuronal output
                post_low_tc = tf.get_variable('post_low_tc', initializer = initial_tc, trainable=True)
                post_low_u = tf.get_variable('post_low_u', initializer = initial_u, trainable=True)
                post_low_bias = tf.get_variable('post_low_bias', initializer = initial_bias, trainable=True)
                post_low_alpha = tf.get_variable('post_low_alpha', initializer = initial_alpha, trainable=True)

                # instananeuos neuronal output
                post_bias = tf.get_variable('post_bias', initializer = initial_bias, trainable=True)
                post_alpha = tf.get_variable('post_alpha', initializer = initial_alpha, trainable=True)

                # low pass of mse
                mse_low_tc = tf.get_variable('mse_low_tc', initializer = initial_tc, trainable=True)
                mse_low_u = tf.get_variable('mse_low_u', initializer = initial_u, trainable=True)
                mse_low_bias = tf.get_variable('mse_low_bias', initializer = initial_bias, trainable=True)
                mse_low_alpha = tf.get_variable('mse_low_alpha', initializer = initial_alpha, trainable=True)

                # instananeuos mse
                mse_bias = tf.get_variable('mse_bias', initializer = initial_bias, trainable=True)
                mse_alpha = tf.get_variable('mse_alpha', initializer = initial_alpha, trainable=True)

                # FINAL output
                #output_tc = tf.get_variable('output_tc', initializer = initial_tc, trainable=True)
                #output_u = tf.get_variable('output_u', initializer = initial_u, trainable=True)

                inp = tf.unstack(self.synaptic_currents[i], axis = 1)
                out = tf.unstack(self.neural_output, axis = 1)

                x_low = tf.constant(np.zeros(par['synaptic_current_sizes'][i], dtype = np.float32))
                y_low = tf.constant(np.zeros(par['synaptic_current_sizes'][i], dtype = np.float32))
                exc_low = tf.constant(np.zeros(par['synaptic_current_sizes'][i], dtype = np.float32))
                inh_low = tf.constant(np.zeros(par['synaptic_current_sizes'][i], dtype = np.float32))
                #pred_delta_weight = tf.constant(np.zeros(par['synaptic_current_sizes'][i], dtype = np.float32))
                mse_low = tf.constant(np.zeros(par['synaptic_current_sizes'][i], dtype = np.float32))
                pred_delta_weight = []

                # local learning model weights
                W0 = tf.get_variable('W0', initializer = tf.random_uniform([15, 20], -0.01, 0.01), trainable=True)
                W1 = tf.get_variable('W1', initializer = tf.random_uniform([15, 20], -0.01, 0.01), trainable=True)
                b = tf.get_variable('b', initializer = tf.random_uniform([15, 1, 1], -0.001, 0.001), trainable=True)
                W2 = tf.get_variable('W2', initializer = tf.random_uniform([1, 15], -0.01, 0.01), trainable=True)
                b2 = tf.get_variable('b2', initializer = tf.random_uniform([1, 1, 1], -0.001, 0.001), trainable=True)

                for (x, y_prev, y, mse) in zip(inp[1:], out[0:-1], out[1:], self.mse[1:]):

                    x_inst = pre_alpha*x + pre_bias
                    x_low = pre_low_alpha*(pre_low_tc*x_low + pre_low_u*x) + pre_low_bias

                    y = tf.matmul(self.transforms[i], y)
                    y_inst = post_alpha*y + post_bias
                    y_low = post_low_alpha*(post_low_tc*x_low + post_low_u*x) + post_low_bias

                    exc_pre = tf.matmul(self.transforms[i], y_prev)
                    inh_pre = tf.matmul(self.transforms[i], y_prev)

                    exc_inst = pre_exc_alpha*exc_pre + pre_exc_bias
                    inh_inst = pre_inh_alpha*inh_pre + pre_inh_bias

                    exc_low = pre_exc_low_alpha*(pre_exc_low_tc*exc_low + pre_exc_low_u*exc_pre) + pre_exc_low_bias
                    inh_low = pre_inh_low_alpha*(pre_inh_low_tc*inh_low + pre_inh_low_u*inh_pre) + pre_inh_low_bias

                    mse_tiled = tf.tile(tf.reshape(mse, (1,-1)), (par['synaptic_current_sizes'][i][0], 1))
                    mse_inst =  mse_alpha*mse_tiled + mse_bias
                    mse_low = mse_low_alpha*(mse_low_tc*mse_low + mse_low_u*mse_tiled) + mse_low_bias

                    V = []
                    for a in range(2):
                        act = tf.nn.relu if a == 0 else tf.nn.sigmoid
                        V.append(act(x_inst))
                        V.append(act(x_low))
                        V.append(act(y_inst))
                        V.append(act(y_low))
                        V.append(act(exc_inst))
                        V.append(act(inh_inst))
                        V.append(act(exc_low))
                        V.append(act(inh_low))
                        V.append(act(mse_inst))
                        V.append(act(mse_low))

                    V = tf.stack(V)
                    z0 = tf.tensordot(W0, V, axes = ([1], [0]))
                    z1 = tf.tensordot(W1, V, axes = ([1], [0]))
                    z = tf.nn.relu(z0*(1. + z1) + b)
                    #z = tf.nn.relu(z0 + b)
                    current_pred_d_w = tf.tensordot(W2, z, axes = ([1], [0])) + b2
                    #pred_delta_weight = output_tc*pred_delta_weight + output_u*current_pred_d_w
                    pred_delta_weight.append(current_pred_d_w)

                pred_delta_weight = tf.stack(pred_delta_weight, axis = 1)

                pred_delta_weight = tf.reduce_mean(tf.reduce_mean(pred_delta_weight, axis = 3), axis = 1)

                delta_weights.append(pred_delta_weight)

        self.pred_delta_w_in = tf.concat([tf.reshape(delta_weights[0], [par['num_exc_units'], par['n_input']]), \
            tf.reshape(delta_weights[1], [par['num_inh_units'], par['n_input']])], axis = 0)

        delta_w_rnn0 = tf.concat([tf.reshape(delta_weights[2], [par['num_exc_units'], par['num_exc_units']]), \
            tf.reshape(delta_weights[3], [par['num_inh_units'], par['num_exc_units']])], axis = 0)
        delta_w_rnn1 = tf.concat([tf.reshape(delta_weights[4], [par['num_exc_units'], par['num_inh_units']]), \
            tf.reshape(delta_weights[5], [par['num_inh_units'], par['num_inh_units']])], axis = 0)
        self.pred_delta_w_rnn = tf.concat([delta_w_rnn0, delta_w_rnn1], axis = 1)


    def optimize(self):

        """
        self.w_in_loss = tf.reduce_mean(self.delta_W_in*self.pred_delta_w_in/tf.norm(self.delta_W_in)/tf.norm(self.pred_delta_w_in))
        self.w_rnn_loss = tf.reduce_mean(self.delta_W_rnn*self.pred_delta_w_rnn/tf.norm(self.delta_W_rnn)/tf.norm(self.pred_delta_w_rnn))
        self.loss = -self.w_in_loss - self.w_rnn_loss
        """
        self.w_in_loss = tf.reduce_mean(tf.square(self.delta_W_in - self.pred_delta_w_in))
        self.w_rnn_loss = tf.reduce_mean(tf.square(self.delta_W_rnn - self.pred_delta_w_rnn))
        self.loss = self.w_in_loss + self.w_rnn_loss
        self.w_in_dot_prod = tf.reduce_sum(self.delta_W_in*self.pred_delta_w_in)/tf.norm(self.delta_W_in)/tf.norm(self.pred_delta_w_in)
        self.w_rnn_dot_prod = tf.reduce_sum(self.delta_W_rnn*self.pred_delta_w_rnn)/tf.norm(self.delta_W_rnn)/tf.norm(self.pred_delta_w_rnn)


        opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        print('Creating minimize opt...')
        self.train_op = opt.minimize(self.loss)


def train_and_analyze(gpu_id):

    tf.reset_default_graph()
    main(gpu_id)
    update_parameters(revert_analysis_par)


def main(gpu_id):

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    """
    Reset TensorFlow before running anything
    """
    tf.reset_default_graph()

    """
    Create the stimulus class to generate trial paramaters and input activity
    """
    stim = stimulus.Stimulus()

    n_input, n_hidden, n_output = par['shape']
    N = par['batch_train_size'] # trials per iteration, calculate gradients after batch_train_size

    """
    Define all model placeholders
    """
    mask = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_train_size']])
    x = tf.placeholder(tf.float32, shape=[n_input, par['num_time_steps'], par['batch_train_size']])  # input data
    y = tf.placeholder(tf.float32, shape=[n_output, par['num_time_steps'], par['batch_train_size']]) # target data

    """
    Define all Local_learning_model placeholders
    """
    neural_output = tf.placeholder(tf.float32, shape=[par['n_hidden'], par['num_time_steps'], par['batch_train_size']])
    sum_exc_current = tf.placeholder(tf.float32, shape=[par['n_hidden'], par['num_time_steps'], par['batch_train_size']])
    sum_inh_current = tf.placeholder(tf.float32, shape=[par['n_hidden'], par['num_time_steps'], par['batch_train_size']])
    mse = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_train_size']])
    delta_W_in = tf.placeholder(tf.float32, shape=[par['n_hidden'], par['n_input']])
    delta_W_rnn = tf.placeholder(tf.float32, shape=[par['n_hidden'], par['n_hidden']])
    synaptic_placeholders = create_synaptic_placeholder()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    # enter "config=tf.ConfigProto(log_device_placement=True)" inside Session to check whether CPU/GPU in use
    with tf.Session(config=config) as sess:

        with tf.device("/gpu:0"):
            model = Model(x, y, mask)
            learning_model = Local_learning_model(neural_output, sum_exc_current, sum_inh_current, \
                mse, delta_W_in, delta_W_rnn, *synaptic_placeholders)
            init = tf.global_variables_initializer()
        sess.run(init)
        t_start = time.time()

        saver = tf.train.Saver()
        # Restore variables from previous model if desired
        if par['load_previous_model']:
            saver.restore(sess, par['save_dir'] + par['ckpt_load_fn'])
            print('Model ' +  par['ckpt_load_fn'] + ' restored.')

        # keep track of the model performance across training
        model_performance = {'accuracy': [], 'loss': [], 'perf_loss': [], 'spike_loss': [], 'trial': [], 'time': []}
        w_in_dp = -1
        w_rn_dp = -1
        l2l_loss = -1

        for i in range(par['num_iterations']):

            # generate batch of batch_train_size
            trial_info = stim.generate_trial()

            """
            Run the model
            """
            _, loss, perf_loss, spike_loss, y_hat, state_hist, syn_x_hist, syn_u_hist, mse_hist, d_w_in, d_w_rnn = \
                sess.run([model.train_op, model.loss, model.perf_loss, model.spike_loss, model.y_hat, \
                model.hidden_state_hist, model.syn_x_hist, model.syn_u_hist, model.MSE, model.delta_w_in, model.delta_w_rnn], \
                {x: trial_info['neural_input'], y: trial_info['desired_output'], mask: trial_info['train_mask']})

            accuracy, _, _ = analysis.get_perf(trial_info['desired_output'], y_hat, trial_info['train_mask'])

            synaptic_current_dict, exc_current, inh_current = calculate_synaptic_inputs(trial_info['neural_input'], state_hist, synaptic_placeholders)
            state_hist = np.stack(state_hist, axis = 1)
            mse_hist = np.stack(mse_hist, axis = 0)

            if i > 50:
                for _ in range(25):
                    _, l2l_loss, d_W_in, d_W_rnn, w_in_dp, w_rn_dp = sess.run([learning_model.train_op, learning_model.loss, \
                        learning_model.pred_delta_w_in, learning_model.pred_delta_w_rnn, learning_model.w_in_dot_prod, learning_model.w_rnn_dot_prod], \
                        {neural_output: state_hist, sum_exc_current: exc_current, sum_inh_current: inh_current, \
                        mse: mse_hist, delta_W_in: d_w_in, delta_W_rnn: d_w_rnn,  **synaptic_current_dict})


            iteration_time = time.time() - t_start
            model_performance = append_model_performance(model_performance, accuracy, loss, perf_loss, \
                spike_loss, (i+1)*N, iteration_time)

            """
            Save the network model and output model performance to screen
            """
            if i%par['iters_between_outputs']==0 and i > 0:
                print_results(i, N, iteration_time, perf_loss, spike_loss, state_hist, accuracy, l2l_loss, w_in_dp, w_rn_dp)


        """
        Save model, analyze the network model and save the results
        """
        #save_path = saver.save(sess, par['save_dir'] + par['ckpt_save_fn'])
        if par['analyze_model']:
            weights = eval_weights()
            analysis.analyze_model(trial_info, y_hat, state_hist, syn_x_hist, syn_u_hist, model_performance, weights, \
                simulation = True, lesion = False, tuning = False, decoding = False, load_previous_file = False, save_raw_data = False)


            # Generate another batch of trials with decoding_test_mode = True (sample and test stimuli
            # are independently drawn), and then perform tuning and decoding analysis
            update = {'decoding_test_mode': True, 'learning_rate': 0}
            update_parameters(update)
            trial_info = stim.generate_trial()
            y_hat, state_hist, syn_x_hist, syn_u_hist = \
                sess.run([model.y_hat, model.hidden_state_hist, model.syn_x_hist, model.syn_u_hist], \
                {x: trial_info['neural_input'], y: trial_info['desired_output'], mask: trial_info['train_mask']})
            analysis.analyze_model(trial_info, y_hat, state_hist, syn_x_hist, syn_u_hist, model_performance, weights, \
                simulation = False, lesion = False, tuning = par['analyze_tuning'], decoding = True, load_previous_file = True, save_raw_data = False)

            if False and par['trial_type'] == 'dualDMS':
                # run an additional session with probe stimuli
                save_fn = 'probe_' + par['save_fn']
                update = {'probe_trial_pct': 1, 'save_fn': save_fn}
                update_parameters(update)
                trial_info = stim.generate_trial()
                y_hat, state_hist, syn_x_hist, syn_u_hist = \
                    sess.run([model.y_hat, model.hidden_state_hist, model.syn_x_hist, model.syn_u_hist], \
                    {x: trial_info['neural_input'], y: trial_info['desired_output'], mask: trial_info['train_mask']})
                analysis.analyze_model(trial_info, y_hat, state_hist, syn_x_hist, \
                    syn_u_hist, model_performance, weights, simulation = False, tuning = False, decoding = True, \
                    load_previous_file = False, save_raw_data = False)

def create_synaptic_placeholder():

    synaptic_placeholders = []
    synaptic_placeholders.append(tf.placeholder(tf.float32, shape=[par['num_exc_units']*par['n_input'], par['num_time_steps'], par['batch_train_size']]))
    synaptic_placeholders.append(tf.placeholder(tf.float32, shape=[par['num_inh_units']*par['n_input'], par['num_time_steps'], par['batch_train_size']]))
    synaptic_placeholders.append(tf.placeholder(tf.float32, shape=[par['num_exc_units']*par['num_exc_units'], par['num_time_steps'], par['batch_train_size']]))
    synaptic_placeholders.append(tf.placeholder(tf.float32, shape=[par['num_exc_units']*par['num_inh_units'], par['num_time_steps'], par['batch_train_size']]))
    synaptic_placeholders.append(tf.placeholder(tf.float32, shape=[par['num_exc_units']*par['num_inh_units'], par['num_time_steps'], par['batch_train_size']]))
    synaptic_placeholders.append(tf.placeholder(tf.float32, shape=[par['num_inh_units']*par['num_inh_units'], par['num_time_steps'], par['batch_train_size']]))

    return synaptic_placeholders


def calculate_synaptic_inputs(input_activity, hidden_activity, synaptic_placeholders):

    with tf.variable_scope('rnn_cell', reuse=True):
        W_in = tf.get_variable('W_in').eval()
        W_rnn = tf.get_variable('W_rnn').eval()

    W_in = np.maximum(0, W_in)
    W_rnn = np.dot(np.maximum(0, W_rnn), par['EI_matrix'])

    hidden_activity = np.stack(hidden_activity, axis=1)

    inp_current = np.tensordot(W_in, input_activity, axes = ([1], [0]))
    sum_exc_current = inp_current + np.tensordot(W_rnn[:, :par['num_exc_units']], hidden_activity[:par['num_exc_units'], :, :], axes = ([1], [0]))
    sum_inh_current = np.tensordot(W_rnn[:, par['num_exc_units']:], hidden_activity[par['num_exc_units']:, :, :], axes = ([1], [0]))

    # input to EXC
    Wx = np.reshape(W_in[:par['num_exc_units'], :], (-1,1,1))
    hx = np.tile(input_activity,(par['num_exc_units'],1,1))
    syn_inp_exc = Wx*hx

    # input to INH
    Wx = np.reshape(W_in[par['num_exc_units']:, :], (-1,1,1))
    hx = np.tile(input_activity,(par['num_inh_units'],1,1))
    syn_inp_inh = Wx*hx

    # EXC to EXC
    Wx = np.reshape(W_rnn[:par['num_exc_units'], :par['num_exc_units']], (-1,1,1))
    hx = np.tile(hidden_activity[:par['num_exc_units'], :, :],(par['num_exc_units'],1,1))
    syn_exc_exc = Wx*hx

    # EXC to INH
    Wx = W_rnn[par['num_exc_units']:, :]
    Wx = np.reshape(Wx[:, :par['num_exc_units']], (-1,1,1))
    hx = np.tile(hidden_activity[:par['num_exc_units'], :, :],(par['num_inh_units'],1,1))
    syn_exc_inh = Wx*hx

    # INH to EXC
    Wx = W_rnn[:par['num_exc_units'], :]
    Wx = np.reshape(Wx[:, par['num_exc_units']:], (-1,1,1))
    hx = np.tile(hidden_activity[par['num_exc_units']:, :, :],(par['num_exc_units'],1,1))
    syn_inh_exc = Wx*hx

    # INH to INH
    Wx = np.reshape(W_rnn[par['num_exc_units']:, par['num_exc_units']:], (-1,1,1))
    hx = np.tile(hidden_activity[par['num_exc_units']:, :, :],(par['num_inh_units'],1,1))
    syn_inh_inh = Wx*hx

    synaptic_current_dict = {}
    synaptic_current_dict[synaptic_placeholders[0]] = syn_inp_exc
    synaptic_current_dict[synaptic_placeholders[1]] = syn_inp_inh
    synaptic_current_dict[synaptic_placeholders[2]] = syn_exc_exc
    synaptic_current_dict[synaptic_placeholders[3]] = syn_exc_inh
    synaptic_current_dict[synaptic_placeholders[4]] = syn_inh_exc
    synaptic_current_dict[synaptic_placeholders[5]] = syn_inh_inh

    return synaptic_current_dict, sum_exc_current, sum_inh_current

def append_model_performance(model_performance, accuracy, loss, perf_loss, spike_loss, trial_num, iteration_time):

    model_performance['accuracy'].append(accuracy)
    model_performance['loss'].append(loss)
    model_performance['perf_loss'].append(perf_loss)
    model_performance['spike_loss'].append(spike_loss)
    model_performance['trial'].append(trial_num)
    model_performance['time'].append(iteration_time)

    return model_performance

def eval_weights():

    with tf.variable_scope('rnn_cell', reuse=True):
        W_in = tf.get_variable('W_in')
        W_rnn = tf.get_variable('W_rnn')
        b_rnn = tf.get_variable('b_rnn')

    with tf.variable_scope('output', reuse=True):
        W_out = tf.get_variable('W_out')
        b_out = tf.get_variable('b_out')

    weights = {
        'w_in'  : W_in.eval(),
        'w_rnn' : W_rnn.eval(),
        'w_out' : W_out.eval(),
        'b_rnn' : b_rnn.eval(),
        'b_out'  : b_out.eval()
    }

    return weights

def print_results(iter_num, trials_per_iter, iteration_time, perf_loss, spike_loss, state_hist, accuracy, l2l_loss, w_in_dp, w_rn_dp):

    print('Iter. {:4d}'.format(iter_num) + ' | Accuracy {:0.4f}'.format(np.mean(accuracy)) +
      ' | Perf loss {:0.4f}'.format(np.mean(perf_loss)) + ' | Spike loss {:0.4f}'.format(np.mean(spike_loss)) +
      ' | Grad Loss {:0.4f}'.format(l2l_loss) + ' | Win dp {:0.4f}'.format(w_in_dp) + ' | Wrnn dp {:0.4f}'.format(w_rn_dp))

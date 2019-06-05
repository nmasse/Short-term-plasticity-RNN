"""
Nicolas Masse 2017
Contributions from Gregory Grant, Catherine Lee
"""

import tensorflow as tf
import numpy as np
import stimulus
import analysis
import pickle
import time
from parameters import par
import os, sys

# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

print('Using EI Network:\t', par['EI'])
print('Synaptic configuration:\t', par['synapse_config'], "\n")


class Model:

    def __init__(self, input_data, target_data, mask):

        # Load the input activity, the target data, and the training mask for this batch of trials
        self.input_data = tf.unstack(input_data, axis=0)
        self.target_data = target_data
        self.mask = mask

        self.initialize_weights()

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()

    def initialize_weights(self):

        self.var_dict = {}
        # all keys in par with a suffix of '0' are initial values of trainable variables
        for k, v in par.items():
            if k[-1] == '0':
                name = k[:-1]
                self.var_dict[name] = tf.Variable(par[k], name)

        self.syn_x_init = tf.constant(par['syn_x_init'])
        self.syn_u_init = tf.constant(par['syn_u_init'])
        if par['EI']:
            # ensure excitatory neurons only have postive outgoing weights,
            # and inhibitory neurons have negative outgoing weights
            self.w_rnn = tf.constant(par['EI_matrix']) @ tf.nn.relu(self.var_dict['w_rnn'])
        else:
            self.w_rnn  = self.var_dict['w_rnn']


    def run_model(self):

        self.h = []
        self.syn_x = []
        self.syn_u = []
        self.y = []

        h = tf.tile(self.var_dict['h'], [par['batch_size'], 1])
        syn_x = self.syn_x_init
        syn_u = self.syn_u_init

        """
        Loop through the neural inputs to the RNN, indexed in time
        """
        for rnn_input in self.input_data:
            h, syn_x, syn_u = self.rnn_cell(rnn_input, h, syn_x, syn_u)
            self.h.append(h)
            self.syn_x.append(syn_x)
            self.syn_u.append(syn_u)
            self.y.append(h @ tf.nn.relu(self.var_dict['w_out']) + self.var_dict['b_out'])

        self.h = tf.stack(self.h)
        self.syn_x = tf.stack(self.syn_x)
        self.syn_u = tf.stack(self.syn_u)
        self.y = tf.stack(self.y)


    def rnn_cell(self, rnn_input, h, syn_x, syn_u):

        """
        Update the synaptic plasticity paramaters
        """
        if par['synapse_config'] is not None:
            # implement both synaptic short term facilitation and depression
            syn_x += (par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_u*syn_x*h)*par['dynamic_synapse']
            syn_u += (par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h)*par['dynamic_synapse']
            syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
            syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
            h_post = syn_u*syn_x*h

        else:
            # no synaptic plasticity
            h_post = h

        """
        Update the hidden state
        Only use excitatory projections from input layer to RNN
        All input and RNN activity will be non-negative
        """
        h = tf.nn.relu(h * (1-par['alpha_neuron']) \
            + par['alpha_neuron'] * (rnn_input @ tf.nn.relu(self.var_dict['w_in']) \
            + h_post @ self.w_rnn + self.var_dict['b_rnn']) \
            + tf.random_normal(h.shape, 0, par['noise_rnn'], dtype=tf.float32))

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
        self.perf_loss = tf.reduce_mean(self.mask*tf.nn.softmax_cross_entropy_with_logits_v2(\
            logits = self.y, labels = self.target_data, axis = 2))

        # L2 penalty term on hidden state activity to encourage low spike rate solutions
        #spike_loss = [par['spike_cost']*tf.reduce_mean(tf.square(h), axis=0) for h in self.hidden_state_hist]
        n = 2 if par['spike_regularization'] == 'L2' else 1
        self.spike_loss = tf.reduce_mean(self.h**n)
        self.weight_loss = tf.reduce_mean(tf.nn.relu(self.w_rnn)**n)

        self.loss = self.perf_loss + par['spike_cost']*self.spike_loss + par['weight_cost']*self.weight_loss

        opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        grads_and_vars = opt.compute_gradients(self.loss)

        """
        Apply any applicable weights masks to the gradient and clip
        """
        capped_gvs = []
        for grad, var in grads_and_vars:
            if 'w_rnn' in var.op.name:
                grad *= par['w_rnn_mask']
            elif 'w_out' in var.op.name:
                grad *= par['w_out_mask']
            elif 'w_in' in var.op.name:
                grad *= par['w_in_mask']
            capped_gvs.append((tf.clip_by_norm(grad, par['clip_max_grad_val']), var))

        self.train_op = opt.apply_gradients(capped_gvs)


def main(gpu_id = None):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


    # Print key parameters
    print_important_params()

    # Reset TensorFlow before running anything
    tf.reset_default_graph()

    #Create the stimulus class to generate trial paramaters and input activity
    stim = stimulus.Stimulus()

    # Define all placeholder
    m = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size']], 'mask')
    x = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'], par['n_input']], 'input')
    t = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'], par['n_output']], 'target')

    # enter "config=tf.ConfigProto(log_device_placement=True)" inside Session to check whether CPU/GPU in use
    with tf.Session(config=tf.ConfigProto()) as sess:

        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Model(x, t, m)

        sess.run(tf.global_variables_initializer())

        # keep track of the model performance across training
        model_performance = {'accuracy': [], 'loss': [], 'perf_loss': [], 'spike_loss': [], \
            'weight_loss': [], 'iteration': []}


        for i in range(par['num_iterations']):

            # generate batch of batch_train_size
            trial_info = stim.generate_trial(set_rule = None)

            # Run the model
            _, loss, perf_loss, spike_loss, weight_loss, y, h, syn_x, syn_u = \
                sess.run([model.train_op, model.loss, model.perf_loss, model.spike_loss, \
                model.weight_loss, model.y, model.h, model.syn_x, model.syn_u], \
                {x: trial_info['neural_input'], t: trial_info['desired_output'], m: trial_info['train_mask']})

            accuracy, _, _ = analysis.get_perf(trial_info['desired_output'], y, trial_info['train_mask'])

            model_performance = append_model_performance(model_performance, accuracy, loss, perf_loss, spike_loss, weight_loss, i)

            # Save the network model and output model performance to screen
            if i%par['iters_between_outputs']==0:
                print_results(i, perf_loss, spike_loss, weight_loss, h, accuracy)


        """
        Save model, analyze the network model and save the results
        """
        weights = sess.run(model.var_dict)
        save_results(model_performance, weights)



def save_results(model_performance,weights,  save_fn = None):

    results = {'weights': weights, 'parameters': par}
    for k,v in model_performance.items():
        results[k] = v
    if save_fn is None:
        fn = par['save_dir'] + par['save_fn']
    else:
        fn = par['save_dir'] + save_fn
    pickle.dump(results, open(fn, 'wb'))
    print('Model results saved in ',fn)


def append_model_performance(model_performance, accuracy, loss, perf_loss, spike_loss, weight_loss, iteration):

    model_performance['accuracy'].append(accuracy)
    model_performance['loss'].append(loss)
    model_performance['perf_loss'].append(perf_loss)
    model_performance['spike_loss'].append(spike_loss)
    model_performance['weight_loss'].append(weight_loss)
    model_performance['iteration'].append(iteration)

    return model_performance


def print_results(iter_num, perf_loss, spike_loss, weight_loss, h, accuracy):

    print(par['trial_type'] + ' Iter. {:4d}'.format(iter_num) + ' | Accuracy {:0.4f}'.format(accuracy) +
      ' | Perf loss {:0.4f}'.format(perf_loss) + ' | Spike loss {:0.4f}'.format(spike_loss) +
      ' | Weight loss {:0.4f}'.format(weight_loss) + ' | Mean activity {:0.4f}'.format(np.mean(h)))

def print_important_params():

    important_params = ['num_iterations', 'learning_rate', 'noise_rnn_sd', 'noise_in_sd','spike_cost',\
        'spike_regularization', 'weight_cost','test_cost_multiplier', 'trial_type','balance_EI', 'dt',\
        'delay_time','weight_multiplier', 'connection_prob','synapse_config','tau_slow']
    for k in important_params:
        print(k, ': ', par[k])

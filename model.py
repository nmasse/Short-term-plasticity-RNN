"""
Nicolas Masse 2017
Contributions from Gregory Grant, Catherine Lee
"""

import tensorflow as tf
import numpy as np
import stimulus
import time
import analysis
from stp_cell import STPCell
from collections import namedtuple
from parameters import *
import matplotlib.pyplot as plt

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
        # self.input_data = tf.reshape(input_data, [par['num_time_steps'], par['batch_train_size'], par['n_input']])
        self.input_data = input_data
        self.target_data = target_data
        self.mask = mask

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

        # Create namedtuple when including syn_x, syn_u
        if par['synapse_config'] == 'stf': 
            State_tuple = namedtuple('State', ['hidden', 'syn_u'])
            state = State_tuple(self.hidden_init, self.synapse_u_init)
        elif par['synapse_config'] == 'std':
            State_tuple = namedtuple('State', ['hidden', 'syn_x'])
            state = State_tuple(self.hidden_init, self.synapse_x_init)
        elif par['synapse_config'] == 'std_stf': 
            State_tuple = namedtuple('State', ['hidden', 'syn_x', 'syn_u'])
            state = State_tuple(self.hidden_init, self.synapse_x_init, self.synapse_u_init)
        else:
            state = self.hidden_init


        # EI matrix
        self.W_ei = tf.constant(par['EI_matrix'])

        # variables for history
        self.hidden_state_hist = []
        self.syn_x_hist = []
        self.syn_u_hist = []

        # Create cell from STPCell class in network.py
        cell = STPCell()

        # hidden_state, output = [max_time, batch_size, cell.output_size], [batch_size, cell_state_size]
        self.hidden_state, self.output = tf.nn.dynamic_rnn(cell, self.input_data, initial_state=state, time_major=True)


        # saving data to hist
        if par['synapse_config'] == 'stf': 
            # self.syn_u_hist.append(self.output.syn_u)
            self.syn_u_hist = self.output.syn_u
        elif par['synapse_config'] == 'std':
            # self.syn_x_hist.append(self.output.syn_x)
            self.syn_x_hist = self.output.syn_x
        elif par['synapse_config'] == 'std_stf': 
            # self.syn_x_hist.append(self.output.syn_x)
            # self.syn_u_hist.append(self.output.syn_u)
            self.syn_x_hist = self.output.syn_x
            self.syn_u_hist = self.output.syn_u
        else:
            pass
        # self.hidden_state_hist = self.output.hidden
        self.hidden_state_hist = self.hidden_state


        with tf.variable_scope('output'):
            W_out = tf.get_variable('W_out', initializer = par['w_out0'], trainable=True)
            b_out = tf.get_variable('b_out', initializer = par['b_out0'], trainable=True)


        """
        Network output
        Only use excitatory projections from the RNN to the output layer
        """
        self.y_hat = tf.tensordot(self.hidden_state, tf.nn.relu(W_out), axes=[[2],[1]]) + tf.transpose(tf.expand_dims(b_out, -1))


    def optimize(self):

        """
        Calculate the loss functions and optimize the weights

        perf_loss = [mask*tf.reduce_mean(tf.square(y_hat-desired_output),axis=0)
                     for (y_hat, desired_output, mask) in zip(self.y_hat, self.target_data, self.mask)]
        """
        """
        cross_entropy
        """
        # perf_loss = self.mask * tf.reduce_mean(tf.square(self.y_hat-tf.reshape(self.target_data, [par['num_time_steps'],par['batch_train_size'],par['n_output']])), axis=2)
        #perf_loss = self.mask * tf.reduce_mean(tf.square(self.y_hat-self.target_data), axis=2)
        print(self.y_hat)
        print(self.target_data)
        print(self.mask)
        self.perf_loss = tf.reduce_mean((self.mask * tf.nn.softmax_cross_entropy_with_logits(logits=self.y_hat, \
            labels=self.target_data, dim=2)))/tf.reduce_mean(self.mask)

        # L2 penalty term on hidden state activity to encourage low spike rate solutions
        self.spike_loss = par['spike_cost']*tf.reduce_mean(tf.square(self.hidden_state))

        # self.perf_loss = tf.reduce_mean(tf.stack(perf_loss, axis=0))
        # self.spike_loss = tf.reduce_mean(tf.stack(spike_loss, axis=0))


        self.loss = self.perf_loss + self.spike_loss

        opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        grads_and_vars = opt.compute_gradients(self.loss)

        """
        Apply any applicable weights masks to the gradient and clip
        """
        capped_gvs = []
        for grad, var in grads_and_vars:
            if var.name == "rnn_cell/W_rnn:0":
                grad *= par['w_rnn_mask']
                print('Applied weight mask to w_rnn.')
            elif var.name == "output/W_out:0":
                grad *= par['w_out_mask']
                print('Applied weight mask to w_out.')
            if not str(type(grad)) == "<class 'NoneType'>":
                capped_gvs.append((tf.clip_by_norm(grad, par['clip_max_grad_val']), var))

        self.train_op = opt.apply_gradients(capped_gvs)


def train_and_analyze(gpu_id):

    tf.reset_default_graph()
    main(gpu_id)
    update_parameters(revert_analysis_par)


def main(gpu_id):

    if par['gpu']:
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
    Define all placeholder
    """
    mask = tf.placeholder(tf.float32, shape=[par['num_time_steps'],par['batch_train_size']])
    x = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_train_size'], n_input])  # input data
    y = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_train_size'], n_output]) # target data

    if par['gpu']:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
    else:
        config = tf.ConfigProto(log_device_placement=True)

    # enter "config=tf.ConfigProto(log_device_placement=True)" inside Session to check whether CPU/GPU in use
    with tf.Session(config=config) as sess:

        if par['gpu']:
            with tf.device("/gpu:0"):
                model = Model(x, y, mask)
                init = tf.global_variables_initializer()
        else:
            model = Model(x, y, mask)
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

        for i in range(par['num_iterations']):

            # generate batch of batch_train_size
            trial_info = stim.generate_trial()
            trial_info['neural_input'] = np.transpose(trial_info['neural_input'], (1,2,0))
            trial_info['desired_output'] = np.transpose(trial_info['desired_output'], (1,2,0))

            #plt.imshow(trial_info['neural_input'][:,0,:])
            #plt.show()
            #plt.imshow(trial_info['desired_output'][:,0,:])
            #plt.show()
            #1/0

            """
            Run the model
            """
            _, loss, perf_loss, spike_loss, y_hat, state_hist, syn_x_hist, syn_u_hist = \
                sess.run([model.train_op, model.loss, model.perf_loss, model.spike_loss, model.y_hat, \
                model.hidden_state_hist, model.syn_x_hist, model.syn_u_hist], {x: trial_info['neural_input'], \
                y: trial_info['desired_output'], mask: trial_info['train_mask']})

            accuracy, _, _ = analysis.get_perf(trial_info['desired_output'], y_hat, trial_info['train_mask'])

            iteration_time = time.time() - t_start
            model_performance = append_model_performance(model_performance, accuracy, loss, perf_loss, spike_loss, (i+1)*N, iteration_time)

            """
            Save the network model and output model performance to screen
            """
            if (i+1)%par['iters_between_outputs']==0 or i+1==par['num_iterations']:
                print_results(i, N, iteration_time, perf_loss, spike_loss, state_hist, accuracy)


        """
        Save model, analyze the network model and save the results
        """
        #save_path = saver.save(sess, par['save_dir'] + par['ckpt_save_fn'])
        if par['analyze_model']:
            weights = eval_weights()
            analysis.analyze_model(trial_info, y_hat, state_hist, syn_x_hist, syn_u_hist, model_performance, weights, \
                simulation = True, tuning = False, decoding = False, load_previous_file = False, save_raw_data = False)

            # Generate another batch of trials with decoding_test_mode = True (sample and test stimuli
            # are independently drawn), and then perform tuning and decoding analysis
            update = {'decoding_test_mode': True}
            update_parameters(update)
            trial_info = stim.generate_trial()
            y_hat, state_hist, syn_x_hist, syn_u_hist = \
                sess.run([model.y_hat, model.hidden_state_hist, model.syn_x_hist, model.syn_u_hist], \
                {x: trial_info['neural_input'], y: trial_info['desired_output'], mask: trial_info['train_mask']})
            analysis.analyze_model(trial_info, y_hat, state_hist, syn_x_hist, syn_u_hist, model_performance, weights, \
                simulation = False, tuning = par['analyze_tuning'], decoding = True, load_previous_file = True, save_raw_data = False)

            if par['trial_type'] == 'dualDMS':
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

def print_results(iter_num, trials_per_iter, iteration_time, perf_loss, spike_loss, state_hist, accuracy):

    print('Trial {:7d}'.format((iter_num+1)*trials_per_iter) + ' | Time {:0.2f} s'.format(iteration_time) +
      ' | Perf loss {:0.4f}'.format(np.mean(perf_loss)) + ' | Spike loss {:0.4f}'.format(np.mean(spike_loss)) +
      ' | Mean activity {:0.4f}'.format(np.mean(state_hist)) + ' | Accuracy {:0.4f}'.format(np.mean(accuracy)))

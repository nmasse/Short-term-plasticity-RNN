"""
2017/05/03 Nicolas Masse
Edited: 2017/06/13 Gregory Grant
"""

print("\nRunning model...\n\n")

import tensorflow as tf
import numpy as np
import pickle
import stimulus
import time
import imp
import os
from parameters import *


# Reset TensorFlow before running anythin
tf.reset_default_graph()

# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

print('Using EI Network:\t', par['EI'])
print('Synaptic configuration:\t', par['synapse_config'])

#################################
### Model setup and execution ###
#################################

class Model:

    def __init__(self, input_data, target_data, mask):

        # Load the input activity, the target data, and the training mask for this batch of trials
        self.input_data = tf.unstack(input_data, axis=1)
        self.target_data = tf.unstack(target_data, axis=1)
        self.mask = tf.unstack(mask, axis=0)

        # Load the initial hidden state activity to be used at the start of each trial
        self.hidden_init = tf.constant(par['h_init'])

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
        Run the reccurent network
        History of hidden state activity stored in self.hidden_state_hist
        """
        self.rnn_cell_loop(self.input_data, self.hidden_init, self.synapse_x_init, self.synapse_u_init)

        with tf.variable_scope('output'):
            W_out = tf.get_variable('W_out', initializer = np.float32(par['w_out0']), trainable=True)
            b_out = tf.get_variable('b_out', initializer = np.float32(par['b_out0']), trainable=True)

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
            W_in = tf.get_variable('W_in', initializer = np.float32(par['w_in0']), trainable=True)
            W_rnn = tf.get_variable('W_rnn', initializer = np.float32(par['w_rnn0']), trainable=True)
            b_rnn = tf.get_variable('b_rnn', initializer = np.float32(par['b_rnn0']), trainable=True)
            W_ei = tf.get_variable('EI_matrix', initializer = np.float32(par['EI_matrix']), trainable=False)

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


    def rnn_cell(self, rnn_input, h_soma, syn_x, syn_u):
        """
        Main computation of the recurrent network
        """

        with tf.variable_scope('rnn_cell', reuse=True):
            W_in = tf.get_variable('W_in')
            W_rnn = tf.get_variable('W_rnn')
            b_rnn = tf.get_variable('b_rnn')
            W_ei = tf.get_variable('EI_matrix')

        if par['EI']:
            # ensure excitatory neurons only have postive outgoing weights,
            # and inhibitory neurons have negative outgoing weights
            W_rnn_effective = tf.tensordot(tf.nn.relu(W_rnn), W_ei, ([2],[0]))

        else:
            W_rnn_effective = W_rnn

        # syn_x and syn_u will be capped at 1
        C = tf.constant(np.float32(1))

        """
        Update the synaptic plasticity parameters
        """
        if par['synapse_config'] == 'std_stf':
            # implement both synaptic short term facilitation and depression
            syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_x*h_soma
            syn_u += par['alpha_std']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h_soma
            syn_x = tf.minimum(C, tf.nn.relu(syn_x))
            syn_u = tf.minimum(C, tf.nn.relu(syn_u))
            h_soma = syn_u*syn_x*h_soma
        elif par['synapse_config'] == 'std':
            # implement synaptic short term depression, but no facilitation
            # assume that syn_u remains constant at 1
            syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_x*h_soma
            syn_x = tf.minimum(C, tf.nn.relu(syn_x))
            syn_u = tf.minimum(C, tf.nn.relu(syn_u))
            h_soma = syn_x*h_soma
        elif par['synapse_config'] == 'stf':
            # implement synaptic short term facilitation, but no depression
            # assume that syn_x remains constant at 1
            syn_u += par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h_soma
            syn_u = tf.minimum(C, tf.nn.relu(syn_u))
            h_soma = syn_u*h_soma
        else:
            # no synaptic plasticity
            pass


        """
        Update the hidden state by way of the dendrites
        Only use excitatory projections from input layer to RNN
        All input and RNN activity will be non-negative
        """

        # Sums and multiplies the weights and inputs that are
        # entered into the dendrites
        h_den_in = tf.tensordot(W_in, rnn_input, ([2],[0])) \
                                + tf.tensordot(W_rnn_effective, h_soma, ([2],[0]))

        # Processes each element of h_in to simulate dendrite action
        def dendrite_process(dend_in):
            #dend_out = tf.nn.sigmoid(dend_in)
            dend_out = dend_in
            return dend_out

        h_den_out = dendrite_process(h_den_in)

        # Accumulates the results of the dendrites per neuron and produces
        # the input to the following hidden neuron
        def dendrite_accum(dend_in):
            # Sums along the dendritic dimension and normalizes by the Number
            # of dendrites per neuron
            dend_out = tf.reduce_sum(dend_in,1)/par['den_per_unit']
            return dend_out

        h_soma_in = dendrite_accum(h_den_out)

        # Applies, in order: alpha decay, dendritic input, bias terms,
        # and Gaussian randomness.
        h_soma_out = tf.nn.relu(h_soma*(1-par['alpha_neuron']) \
                            + h_soma_in \
                            + b_rnn \
                            + tf.random_normal([par['n_hidden'], par['batch_train_size']], 0, par['noise_sd'], dtype=tf.float32))

        if par['debug_model']:
            print('\n')
            print('-' * 40)
            print('N Input:\t', par['n_input'])
            print('N Hidden:\t', par['n_hidden'])
            print('N Dendrites:\t', par['den_per_unit'])
            print('N Output:\t', par['n_output'])
            print('Batch Size:\t', par['batch_train_size'])
            print('Alpha Neuron:\t', par['alpha_neuron'])
            print('-' * 40)
            print('W In:\t\t\t', W_in.shape)
            print('W Rnn:\t\t\t', W_rnn_effective.shape)
            print('Neural Input:\t\t', rnn_input.shape)
            print('RNN Input:\t\t', h_soma.shape)
            print('Dendrite Input:\t\t', h_den_in.shape)
            print('Dendrite Output:\t', h_den_out.shape)
            print('Soma Input:\t\t', h_soma_in.shape)
            print('Soma Biases:\t\t', b_rnn.shape)
            print('Soma Output:\t\t', h_soma.shape)
            print('-' * 40)
            quit()

        return h_soma_out, syn_x, syn_u


    def optimize(self):

        """
        Calculate the loss functions and optimize the weights
        """
        perf_loss = [mask*tf.reduce_mean(tf.square(y_hat-desired_output),axis=0)
                     for (y_hat, desired_output, mask) in zip(self.y_hat, self.target_data, self.mask)]

        # L2 penalty term on hidden state activity to encourage low spike rate solutions
        spike_loss = [par['spike_cost']*tf.reduce_mean(tf.square(h), axis=0) for (h, mask)
                            in zip(self.hidden_state_hist, self.mask)]

        self.perf_loss = tf.reduce_mean(tf.stack(perf_loss, axis=0))
        self.spike_loss = tf.reduce_mean(tf.stack(spike_loss, axis=0))

        self.loss = self.perf_loss + self.spike_loss

        opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        grads_and_vars = opt.compute_gradients(self.loss)

        """
        Apply any applicable weights masks to the gradient and clip
        """
        capped_gvs = []
        for grad, var in grads_and_vars:
            if grad.shape == par['w_rec_mask'].shape:
                grad *= par['w_rec_mask']
                print('Applied weight mask to w_rec.')
            elif grad.shape == par['w_out_mask'].shape:
                grad *= par['w_out_mask']
                print('Applied weight mask to w_out.')
            capped_gvs.append((tf.clip_by_norm(grad, par['clip_max_grad_val']), var))
        print("\n")
        self.train_op = opt.apply_gradients(capped_gvs)


def main():


    """
    Create the stimulus class to generate trial paramaters and input activity
    """
    stim = stimulus.Stimulus()

    n_input, n_hidden, n_output = par['shape']
    trial_length = par['num_time_steps']
    batch_size = par['batch_train_size']
    N = par['batch_train_size'] * par['num_batches'] # trials per iteration, calculate gradients after batch_train_size

    """
    Define all placeholder
    """
    mask = tf.placeholder(tf.float32, shape=[trial_length, batch_size])
    x = tf.placeholder(tf.float32, shape=[n_input, trial_length, batch_size])  # input data
    y = tf.placeholder(tf.float32, shape=[n_output, trial_length, batch_size]) # target data

    with tf.Session() as sess:

        model = Model(x, y, mask)
        init = tf.global_variables_initializer()
        sess.run(init)
        t_start = time.time()
        timestr = time.strftime('%H%M%S-%Y%m%d')

        saver = tf.train.Saver()
        # Restore variables from previous model if desired
        if par['load_previous_model']:
            saver.restore(sess, par['save_dir'] + par['ckpt_load_fn'])
            print('Model ' + par['ckpt_load_fn'] + ' restored.')

        # keep track of the model performance across training
        model_performance = {'accuracy': [], 'loss': [], 'trial': [], 'time': []}

        # Write intermittent results to text file
        with open('.\savedir\savefile%s.txt' % timestr, 'w') as f:
            f.write('Trial\tTime\tPerf loss\tSpike loss\tMean activity\tTest Accuracy\n')

        for i in range(par['num_iterations']):

            end_training = False
            save_trial = False

            """
            Save the data if this is the last iteration, if performance threshold has been reached, and once every 500 trials.
            Before saving data, generate trials with a fixed delay period
            """
            if i>0 and (i == par['num_iterations']-1 or np.mean(accuracy) > par['stop_perf_th'] or (i+1)%par['trials_between_outputs']==0):
                var_delay = False
                save_trial = True
                model_results = create_save_dict()
                if np.mean(accuracy) > par['stop_perf_th']:
                    end_training = True

            # generate batch of N (batch_train_size X num_batches) trials
            trial_info = stim.generate_trial(N)

            # keep track of the model performance for this batch
            loss = []
            perf_loss = []
            spike_loss = []
            accuracy = []

            for j in range(par['num_batches']):

                """
                Select batches of size batch_train_size
                """
                ind = range(j*par['batch_train_size'],(j+1)*par['batch_train_size'])
                target_data = trial_info['desired_output'][:,:,ind]
                input_data = trial_info['neural_input'][:,:,ind]
                train_mask = trial_info['train_mask'][:,ind]

                """
                Run the model
                """
                _, loss_temp, perf_loss_temp, spike_loss_temp, y_hat, state_hist = sess.run([model.train_op, model.loss, model.perf_loss, model.spike_loss, model.y_hat, model.hidden_state_hist], {x: input_data, y: target_data, mask: train_mask})

                # append the data before saving
                if save_trial:
                    start_save_time = time.time()
                    model_results = append_hidden_data(model_results, y_hat, state_hist)

                # keep track of performance for each batch
                accuracy.append(get_perf(target_data, y_hat, train_mask))
                perf_loss.append(perf_loss_temp)
                spike_loss.append(spike_loss_temp)

            iteration_time = time.time() - t_start
            model_performance = append_model_performance(model_performance, accuracy, perf_loss, (i+1)*N, iteration_time)

            """
            Save the data and network model
            """

            if save_trial:

                model_results = append_fixed_data(model_results, trial_info, par)
                model_results['performance'] = model_performance
                with open(par['save_dir'] + par['save_fn'], 'wb') as f:
                    pickle.dump(model_results, f)
                    save_path = saver.save(sess,par['save_dir'] + par['ckpt_save_fn'])
                    print(par['save_fn'] + ' pickled!, file save time = ', time.time() - start_save_time)

                with open('.\savedir\savefile%s.txt' % timestr, 'a') as f:
                    # In order, Trial | Time | Perf Loss | Spike Loss | Mean Activity | Accuracy
                    f.write('{:7d}'.format((i+1)*N) \
                            + '\t{:0.2f}'.format(iteration_time) \
                            + '\t{:0.4f}'.format(np.mean(perf_loss)) \
                            + '\t{:0.4f}'.format(np.mean(spike_loss)) \
                            + '\t{:0.4f}'.format(np.mean(state_hist)) \
                            + '\t{:0.4f}'.format(np.mean(accuracy)) \
                            + '\n')

                # output model performance to screen
                print('Trial {:7d}'.format((i+1)*N) + ' | Time {:0.2f} s'.format(iteration_time) +
                  ' | Perf loss {:0.4f}'.format(np.mean(perf_loss)) + ' | Spike loss {:0.4f}'.format(np.mean(spike_loss)) + ' | Mean activity {:0.4f}'.format(np.mean(state_hist)) + ' | Test Accuracy {:0.4f}'.format(np.mean(accuracy)))


            if end_training:
                return None

    print('\nModel execution complete.\n')


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


def append_hidden_data(model_results, y_hat, state):

    model_results['hidden_state'].append(state)
    model_results['model_outputs'].append(y_hat)

    return model_results


def append_model_performance(model_performance, accuracy, loss, trial_num, iteration_time):

    model_performance['accuracy'].append(accuracy)
    model_performance['loss'].append(loss)
    model_performance['trial'].append(trial_num)
    model_performance['time'].append(iteration_time)

    return model_performance


def append_fixed_data(model_results, trial_info, params):

    #model_results['sample_dir'] = trial_info['sample']
    #model_results['test_dir'] = trial_info['test']
    #model_results['match'] = trial_info['match']
    #model_results['catch'] = trial_info['catch']
    #model_results['rule'] = trial_info['rule']
    #model_results['probe'] = trial_info['probe']
    model_results['rnn_input'] = trial_info['neural_input']
    model_results['desired_output'] = trial_info['desired_output']
    model_results['train_mask'] = trial_info['train_mask']
    model_results['params'] = par

    # add extra trial paramaters for the ABBA task
    if 4 in par['possible_rules']:
        print('Adding ABB specific data')
        model_results['num_test_stim'] = trial_info['num_test_stim']
        model_results['repeat_test_stim'] = trial_info['repeat_test_stim']
        model_results['test_stim_code'] = trial_info['test_stim_code']

    with tf.variable_scope('rnn_cell', reuse=True):
        W_in = tf.get_variable('W_in')
        W_rnn = tf.get_variable('W_rnn')
        b_rnn = tf.get_variable('b_rnn')
        W_ei = tf.get_variable('EI_matrix')
            #W_rnn_effective = tf.matmul(tf.nn.relu(W_rnn), W_ei)
    with tf.variable_scope('output', reuse=True):
        W_out = tf.get_variable('W_out')
        b_out = tf.get_variable('b_out')

    model_results['w_in'] = W_in.eval()
    model_results['w_rnn'] = W_rnn.eval()
    model_results['w_out'] = W_out.eval()
    model_results['b_rnn'] = b_rnn.eval()
    model_results['b_out'] = b_out.eval()

    return model_results


def create_save_dict():

    model_results = {
        'hidden_state':    [],
        'desired_output':  [],
        'model_outputs':   [],
        'rnn_input':       [],
        'sample_dir':      [],
        'test_dir':        [],
        'match':           [],
        'rule':            [],
        'catch':           [],
        'probe':           [],
        'train_mask':      [],
        'U':               [],
        'w_in':            [],
        'w_rnn':           [],
        'w_out':           [],
        'b_rnn':           [],
        'b_out':           [],
        'num_test_stim':   [],
        'repeat_test_stim':[],
        'test_stim_code': []}

    return model_results

# Run the model from main
main()

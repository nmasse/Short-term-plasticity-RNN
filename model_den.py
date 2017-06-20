"""
2017/05/03 Nicolas Masse
Edited: 2017/06/13 Gregory Grant
"""

import tensorflow as tf
import numpy as np
import pickle
import stimulus_updated as stimulus
import time


class Model:

    def __init__(self, params, input_data, target_data, mask):

        # Load all variables from the params dictionary
        for key,value in params.items():
            setattr(self, key, value)

        # Load the input activity, the target data, and the training mask for this batch of trials
        self.input_data = tf.unstack(input_data, axis=1)
        self.target_data = tf.unstack(target_data, axis=1)
        self.mask = tf.unstack(mask, axis=0)

        # Load the intial hidden state activity to be used at the start of each trial
        self.hidden_init = tf.constant(self.h_init)

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()


    def run_model(self):

        """
        Run the reccurent network
        History of hidden state activity stored in self.hidden_state_hist
        """
        self.rnn_cell_loop(self.input_data, self.hidden_init)

        with tf.variable_scope('output'):
            W_out = tf.get_variable('W_out', initializer = np.float32(self.w_out0), trainable=True)
            b_out = tf.get_variable('b_out', initializer = np.float32(self.b_out0), trainable=True)

        """
        Network output
        Only use excitatory projections from the RNN to the output layer
        """
        self.y_hat = [tf.matmul(tf.nn.relu(W_out),h)+b_out for h in self.hidden_state_hist]


    def rnn_cell_loop(self, x_unstacked, h):

        """
        Initialize weights and biases
        """
        with tf.variable_scope('rnn_cell'):
            W_in = tf.get_variable('W_in', initializer = np.float32(self.w_in0), trainable=True)
            W_rnn = tf.get_variable('W_rnn', initializer = np.float32(self.w_rnn0), trainable=True)
            b_rnn = tf.get_variable('b_rnn', initializer = np.float32(self.b_rnn0), trainable=True)
            W_ei = tf.get_variable('EI_matrix', initializer = np.float32(self.EI_matrix), trainable=False)

        self.hidden_state_hist = []

        """
        Loop through the neural inputs to the RNN, indexed in time
        """

        for rnn_input in x_unstacked:
            h = self.rnn_cell(rnn_input, h)
            self.hidden_state_hist.append(h)


    def rnn_cell(self, rnn_input, h_soma):
        """
        Main computation of the recurrent network
        """

        with tf.variable_scope('rnn_cell', reuse=True):
            W_in = tf.get_variable('W_in')
            W_rnn = tf.get_variable('W_rnn')
            b_rnn = tf.get_variable('b_rnn')
            W_ei = tf.get_variable('EI_matrix')

        if self.EI:
            # ensure excitatory neurons only have postive outgoing weights,
            # and inhibitory neurons have negative outgoing weights
            W_rnn_effective = tf.tensordot(tf.nn.relu(W_rnn), W_ei, ([2],[0]))

        else:
            W_rnn_effective = W_rnn

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

            dend_out = tf.nn.sigmoid(dend_in)

            """
            t = 0.5 # Threshold

            # Tests if each element is equal to or greater than one
            # Returns 1 if so, returns 0 if less than one
            dend_out = tf.where(tf.less(dend_in, t*tf.ones(dend_in.shape)), \
                                tf.zeros(dend_in.shape), tf.ones(dend_in.shape))
            """
            return dend_out

        h_den_out = dendrite_process(h_den_in)

        # Accumulates the results of the dendrites per neuron and produces
        # the input to the following hidden neuron
        def dendrite_accum(dend_in):

            # Sums along the dendritic dimension and normalizes by the Number
            # of dendrites per neuron
            dend_out = tf.reduce_sum(dend_in,1)/self.den_per_unit

            return dend_out

        h_soma_in = dendrite_accum(h_den_out)

        # Applies, in order: alpha decay, dendritic input, bias terms,
        # and a little bit of randomness.
        h_soma_out = tf.nn.relu(h_soma*(1-self.alpha_neuron) \
                            + h_soma_in \
                            + b_rnn \
                            + tf.random_normal([self.n_hidden, self.batch_train_size], 0, self.noise_sd, dtype=tf.float32))

        if self.debug_model:
            print('\n')
            print('-' * 40)
            print('N Input:\t', self.n_input)
            print('N Hidden:\t', self.n_hidden)
            print('N Dendrites:\t', self.den_per_unit)
            print('N Output:\t', self.n_output)
            print('Batch Size:\t', self.batch_train_size)
            print('Alpha Neuron:\t', self.alpha_neuron)
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

        return h_soma_out


    def optimize(self):

        """
        Calculate the loss functions and optimize the weights
        """
        perf_loss = [mask*tf.reduce_mean(tf.square(y_hat-desired_output),axis=0)
                     for (y_hat, desired_output, mask) in zip(self.y_hat, self.target_data, self.mask)]

        # L2 penalty term on hidden state activity to encourage low spike rate solutions
        spike_loss = [self.spike_cost*tf.reduce_mean(tf.square(h), axis=0) for (h, mask)
                            in zip(self.hidden_state_hist, self.mask)]

        self.perf_loss = tf.reduce_mean(tf.stack(perf_loss, axis=0))
        self.spike_loss = tf.reduce_mean(tf.stack(spike_loss, axis=0))

        self.loss = self.perf_loss + self.spike_loss

        opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        grads_and_vars = opt.compute_gradients(self.loss)

        """
        Apply any applicable weights masks to the gradient and clip
        """
        capped_gvs = []
        for grad, var in grads_and_vars:
            if grad.shape == self.w_rec_mask.shape:
                print('Applying weight mask to w_rec')
                grad *= self.w_rec_mask
            elif grad.shape == self.w_out_mask.shape:
                print('Applying weight mask to w_out')
                grad *= self.w_out_mask
            capped_gvs.append((tf.clip_by_norm(grad, self.clip_max_grad_val), var))

        self.train_op = opt.apply_gradients(capped_gvs)


def main(params):


    """
    Create the stimulus class to generate trial paramaters and input activity
    """
    stim = stimulus.Stimulus(params)

    n_input, n_hidden, n_output = params['shape']
    trial_length = params['trial_length']//params['dt']
    batch_size = params['batch_train_size']
    N = params['batch_train_size']*params['num_batches'] # trials per iteration, calculate gradients after batch_train_size

    """
    Define all placeholder
    """
    mask = tf.placeholder(tf.float32, shape=[trial_length, batch_size])
    x = tf.placeholder(tf.float32, shape=[n_input, trial_length, batch_size])  # input data
    y = tf.placeholder(tf.float32, shape=[n_output, trial_length, batch_size]) # target data

    with tf.Session() as sess:

        model = Model(params, x, y, mask)
        init = tf.global_variables_initializer()
        sess.run(init)
        t_start = time.time()
        timestr = time.strftime('%H%M%S-%Y%m%d')

        saver = tf.train.Saver()
        # Restore variables from previous model if desired
        if params['load_previous_model']:
            saver.restore(sess, params['save_dir'] + params['ckpt_load_fn'])
            print('Model ' +  params['ckpt_load_fn'] + ' restored.')

        # keep track of the model performance across training
        model_performance = {'accuracy': [], 'loss': [], 'trial': [], 'time': []}

        # Write intermittent results to text file
        with open('.\savedir\savefile%s.txt' % timestr, 'w') as f:
            f.write('Trial\tTime\tPerf loss\tSpike loss\tMean activity\tAccuracy\n')

        for i in range(params['num_iterations']):

            end_training = False
            save_trial = False

            """
            Save the data if this is the last iteration, if performance threshold has been reached, and once every 500 trials.
            Before saving data, generate trials with a fixed delay period
            """
            if i>0 and (i == params['num_iterations']-1 or np.mean(accuracy) > params['stop_perf_th'] or (i+1)%5==0): # 500 to 5 trials
                var_delay = False
                save_trial = True
                model_results = create_save_dict()
                if np.mean(accuracy) > params['stop_perf_th']:
                    end_training = True

            # generate batch of N (batch_train_size X num_batches) trials
            trial_info = stim.generate_trial(N)

            # keep track of the model performance for this batch
            loss = []
            perf_loss = []
            spike_loss = []
            accuracy = []

            for j in range(params['num_batches']):

                """
                Select batches of size batch_train_size
                """
                ind = range(j*params['batch_train_size'],(j+1)*params['batch_train_size'])
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
                """
                model_results = append_fixed_data(model_results, trial_info, params)
                model_results['performance'] = model_performance
                with open(params['save_dir'] + params['save_fn'], 'wb') as f:
                    pickle.dump(model_results, f)
                    save_path = saver.save(sess,params['save_dir'] + params['ckpt_save_fn'])
                    print(params['save_fn'] + ' pickled!, file save time = ', time.time() - start_save_time)
                """
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
                  ' | Perf loss {:0.4f}'.format(np.mean(perf_loss)) + ' | Spike loss {:0.4f}'.format(np.mean(spike_loss)) + ' | Mean activity {:0.4f}'.format(np.mean(state_hist)) + ' | Accuracy {:0.4f}'.format(np.mean(accuracy)))


            if end_training:
                return None


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

    model_results['sample_dir'] = trial_info['sample']
    model_results['test_dir'] = trial_info['test']
    model_results['match'] = trial_info['match']
    model_results['catch'] = trial_info['catch']
    model_results['rule'] = trial_info['rule']
    model_results['probe'] = trial_info['probe']
    model_results['rnn_input'] = trial_info['neural_input']
    model_results['desired_output'] = trial_info['desired_output']
    model_results['train_mask'] = trial_info['train_mask']
    model_results['params'] = params

    # add extra trial paramaters for the ABBA task
    if 4 in params['possible_rules']:
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

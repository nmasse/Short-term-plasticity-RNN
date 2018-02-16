"""
Nicolas Masse 2017
Contributions from Gregory Grant, Catherine Lee
"""

import tensorflow as tf
import numpy as np
import stimulus, stimulus3, multistim
import analysis
import AdamOpt
from parameters import *
import pickle
import matplotlib.pyplot as plt
import os, sys, time

# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

"""
Model setup and execution
"""

class AutoModel:

    def __init__(self, input_data, neuron_td, dendrite_td, mask):

        # Load the input activity and gating mechanisms
        self.input_data  = tf.unstack(input_data, axis=1)
        self.mask        = tf.unstack(mask, axis=0)
        self.neuron_td   = neuron_td[:,tf.newaxis]
        self.dendrite_td = dendrite_td

        # Describe network shape
        self.num_layers = 2
        self.lids = list(range(self.num_layers))

        # Initialize internal state
        self.pred_states   = [tf.zeros_like(self.input_data[0], dtype=tf.float32)]*self.num_layers
        self.rnn_states    = [tf.zeros([par['n_hidden'],1], dtype=tf.float32)]*self.num_layers
        self.target_states = [tf.zeros_like(self.input_data[0], dtype=tf.float32)]*self.num_layers

        # Run and optimize
        self.initialize_variables()
        self.run_model()
        self.optimize()


    def initialize_variables(self):
        for lid in self.lids:
            with tf.variable_scope('layer'+str(lid)):
                tf.get_variable('W_err1', shape=[par['n_hidden'],par['n_dendrites'],par['n_input']],
                                initializer=tf.random_uniform_initializer(-1/np.sqrt(par['n_input']), 1/np.sqrt(par['n_input'])),
                                trainable=True)
                tf.get_variable('W_err2', shape=[par['n_hidden'],par['n_dendrites'],par['n_input']],
                                initializer=tf.random_uniform_initializer(-1/np.sqrt(par['n_input']), 1/np.sqrt(par['n_input'])),
                                trainable=True)
                tf.get_variable('W_pred', shape=[par['n_input'],par['n_hidden']],
                                initializer=tf.random_uniform_initializer(-1/np.sqrt(par['n_hidden']), 1/np.sqrt(par['n_hidden'])),
                                trainable=True)
                tf.get_variable('W_rnn', initializer = par['w_rnn0'], trainable=True)
                tf.get_variable('b_rnn', initializer = np.zeros((par['n_hidden'],1), dtype = np.float32), trainable=True)
                tf.get_variable('b_pred', initializer = np.zeros((par['n_input'],1), dtype = np.float32), trainable=True)


    def calc_error(self, target, prediction):
        return tf.nn.relu(prediction - target), tf.nn.relu(target - prediction)


    def layer(self, lid, target):

        # Loading all weights and biases
        with tf.variable_scope('layer'+str(lid), reuse=True):
            W_err1 = tf.get_variable('W_err1') #positive error
            W_err2 = tf.get_variable('W_err2') #negative error
            W_pred = tf.get_variable('W_pred')
            W_rnn  = tf.get_variable('W_rnn')
            b_rnn  = tf.get_variable('b_rnn')
            b_pred  = tf.get_variable('b_pred') # prediction bias

        # Masking certain weights
        W_rnn *= tf.constant(par['w_rnn_mask'], dtype=tf.float32)
        if par['EI']:
            # ensure excitatory neurons only have postive outgoing weights,
            # and inhibitory neurons have negative outgoing weights
            W_rnn = tf.tensordot(tf.nn.relu(W_rnn), tf.constant(par['EI_matrix']), [[2],[0]])

        # Processing data for RNN step
        # Currently, input from R l+1 to R l in not weighted
        err_stim1, err_stim2 = self.calc_error(target, self.pred_states[lid])
        rnn_state = self.rnn_states[lid]
        if lid != self.num_layers-1:
            rnn_next = self.rnn_states[lid+1]
        else:
            rnn_next = tf.zeros_like(rnn_state)

        inp_act = tf.tensordot(W_err1, err_stim1, [[2],[0]]) + tf.tensordot(W_err2, err_stim2, [[2],[0]]) # Error activity
        rnn_act = tf.tensordot(W_rnn, rnn_state, [[2],[0]])       # RNN activity
        tot_act = par['alpha_neuron']*(inp_act + rnn_act)         # Modulating
        #act_eff = tf.reduce_sum(self.dendrite_td*tot_act, axis=1) # Summing dendrites
        act_eff = tf.reduce_sum(tot_act, axis=1) # Summing dendrites

        # Updating RNN state
        #rnn_state = self.neuron_td*tf.nn.relu(rnn_state*(1-par['alpha_neuron']) + act_eff + rnn_next + b_rnn \
        #    + tf.random_normal(rnn_state.shape, 0, par['noise_rnn'], dtype=tf.float32))
        rnn_state = self.neuron_td*tf.nn.relu(rnn_state*(1-par['alpha_neuron']) + act_eff + rnn_next + b_rnn \
            + tf.random_normal(rnn_state.shape, 0, par['noise_rnn'], dtype=tf.float32))

        # Updating prediction state
        pred_state = tf.nn.relu(tf.matmul(W_pred, rnn_state) + b_pred) # A_hat

        # Plugging the RNN and prediction states back into the aggregate model
        self.rnn_states[lid]  = rnn_state
        with tf.control_dependencies([err_stim1, err_stim2]):
            self.pred_states[lid] = pred_state

        return err_stim1 + err_stim2, rnn_state


    def run_model(self):

        # Start recording the error and hidden state histories
        self.error_history = []
        self.hidden_history = []
        self.prediction_history = []

        # Iterate through time via the input data
        for t, input_data in enumerate(self.input_data):

            # Start the state lists
            self.error_states = []
            self.hidden_states = []

            # Iterate over each layer at each time point
            for lid in self.lids:

                # If the first layer, use the actual input
                # Instead of using desired output for layer, we'll use neuronal input
                stim = input_data if lid == 0 else self.error_states[lid-1]

                # Run the current layer and recover the error matrix,
                # then save the error matrix to the NEXT state position,
                # and record the RNN activity
                layer_error, rnn_state = self.layer(lid, stim)
                self.error_states.append(layer_error)
                self.hidden_states.append(rnn_state)

            # Save the current states for later evaluation
            self.error_history.append(self.error_states)
            self.hidden_history.append(self.hidden_states)
            self.prediction_history.append(self.pred_states[0])
            # self.prediction_history is a list of time X layers X (array of neurons and batch size)

    def optimize(self):

        # Use all trainable variables
        variables = [var for var in tf.trainable_variables()]
        adam_optimizer = AdamOpt.AdamOpt(variables, learning_rate = par['learning_rate'])

        previous_weights_mu_minus_1 = {}
        reset_prev_vars_ops = []
        self.big_omega_var = {}
        aux_losses = []

        for var in variables:
            self.big_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            previous_weights_mu_minus_1[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            aux_losses.append(par['omega_c']*tf.reduce_sum(tf.multiply(self.big_omega_var[var.op.name], \
               tf.square(previous_weights_mu_minus_1[var.op.name] - var) )))
            reset_prev_vars_ops.append( tf.assign(previous_weights_mu_minus_1[var.op.name], var ) )

        self.aux_loss = tf.add_n(aux_losses)

        """
        TODO: we only want to reduce the wiring cost for connections weights in which the pre- and post-synaptic partners are active
        for the current task
        """

        self.wiring_loss = tf.constant(0.)
        for var in [var for var in variables if ('W' in var.op.name and not 'td' in var.op.name)]:
            if 'W_in' in var.op.name:
                self.wiring_loss += tf.reduce_sum(tf.nn.relu(var) * tf.constant(par['w_in_pos'], dtype=tf.float32))
            elif 'W_rnn' in var.op.name:
                self.wiring_loss += tf.reduce_sum(tf.nn.relu(var) * tf.constant(par['w_rnn_pos'], dtype=tf.float32))
            elif 'W_out' in var.op.name:
                self.wiring_loss += tf.reduce_sum(tf.nn.relu(var) * tf.constant(par['w_out_pos'], dtype=tf.float32))
        self.wiring_loss *= par['wiring_cost']

        self.error_loss = tf.constant(0.)
        for m, h in zip(self.mask, self.error_history):
            for l in h:
                #self.error_loss += tf.reduce_mean(tf.square(m[tf.newaxis,:]*l))
                self.error_loss += tf.reduce_mean(tf.square(l))

        self.spike_loss = tf.constant(0.)
        for h in self.hidden_history:
            for l in h:
                self.spike_loss += par['spike_cost']*tf.reduce_mean(l)

        # Gradient of the loss+aux function, in order to both perform training and to compute delta_weights
        with tf.control_dependencies([self.error_loss, self.aux_loss, self.wiring_loss, self.spike_loss]):
            self.loss = self.error_loss + self.aux_loss + self.wiring_loss + self.spike_loss
            self.train_op = adam_optimizer.compute_gradients(self.loss)

        if par['stabilization'] == 'pathint':
            # Zenke method
            self.pathint_stabilization(variables, adam_optimizer, previous_weights_mu_minus_1)

        elif par['stabilization'] == 'EWC':
            # Kirkpatrick method
            self.EWC(variables)

        self.reset_prev_vars = tf.group(*reset_prev_vars_ops)
        self.reset_adam_op = adam_optimizer.reset_params()


    def EWC(self, variables):

        # Kirkpatrick method
        epsilon = 1e-5
        fisher_ops = []
        opt = tf.train.GradientDescentOptimizer(1)

        # model results p(y|x, theta)
        p_theta = tf.nn.softmax(self.y, dim = 1)
        # sample label from p(y|x, theta)
        class_ind = tf.multinomial(p_theta, 1)
        class_ind_one_hot = tf.reshape(tf.one_hot(class_ind, par['layer_dims'][-1]), \
            [par['batch_size'], par['layer_dims'][-1]])
        # calculate the gradient of log p(y|x, theta)
        log_p_theta = tf.unstack(class_ind_one_hot*tf.log(p_theta + epsilon), axis = 0)
        for lp in log_p_theta:
            self.gradients = opt.compute_gradients(lp)
            for grad, var in self.gradients:
                fisher_ops.append(tf.assign_add(self.big_omega_var[var.op.name], \
                    grad*grad/par['batch_size']/par['EWC_fisher_num_batches']))

        self.update_big_omega = tf.group(*fisher_ops)


    def pathint_stabilization(self, variables, adam_optimizer, previous_weights_mu_minus_1):
        # Zenke method

        optimizer_task = tf.train.GradientDescentOptimizer(learning_rate =  1.0)
        small_omega_var = {}

        reset_small_omega_ops = []
        update_small_omega_ops = []
        update_big_omega_ops = []
        initialize_prev_weights_ops = []

        for var in variables:

            small_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            reset_small_omega_ops.append( tf.assign( small_omega_var[var.op.name], small_omega_var[var.op.name]*0.0 ) )
            update_big_omega_ops.append( tf.assign_add( self.big_omega_var[var.op.name], tf.div(tf.nn.relu(small_omega_var[var.op.name]), \
            	(par['omega_xi'] + tf.square(var-previous_weights_mu_minus_1[var.op.name])))))


        # After each task is complete, call update_big_omega and reset_small_omega
        self.update_big_omega = tf.group(*update_big_omega_ops)

        # Reset_small_omega also makes a backup of the final weights, used as hook in the auxiliary loss
        self.reset_small_omega = tf.group(*reset_small_omega_ops)

        # This is called every batch
        with tf.control_dependencies([self.train_op]):
            self.delta_grads = adam_optimizer.return_delta_grads()
            self.gradients = optimizer_task.compute_gradients(self.error_loss)
            for grad,var in self.gradients:
                update_small_omega_ops.append(tf.assign_add(small_omega_var[var.op.name], -self.delta_grads[var.op.name]*grad ) )
            self.update_small_omega = tf.group(*update_small_omega_ops) # 1) update small_omega after each train!



def train_and_analyze(gpu_id, save_fn):

    tf.reset_default_graph()
    main(gpu_id, save_fn)
    #update_parameters(revert_analysis_par)


def main(save_fn, gpu_id = None):

    print('\nRunning model.\n')

    ##################
    ### Setting Up ###
    ##################

    """ Set up GPU """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    """ Reset TensorFlow before running anything """
    tf.reset_default_graph()

    """ Create the stimulus object for trial generation """
    #stim = multistim.MultiStimulus()
    stim = stimulus3.Stimulus()

    """ Define all placeholders """
    mask = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_train_size']])
    x = tf.placeholder(tf.float32, shape=[par['n_input'], par['num_time_steps'], par['batch_train_size']])
    td_neur = tf.placeholder(tf.float32, shape=[par['n_hidden']])
    td_dend = tf.placeholder(tf.float32, shape=[par['n_hidden'], par['n_dendrites']])

    """ Set up performance recording """
    accuracy_full = []
    par['n_tasks'] = 19
    accuracy_grid = np.zeros((par['n_tasks'], par['n_tasks']))
    model_performance = {'accuracy': [], 'par': [], 'task_list': []}

    """ Start TensorFlow session """
    with tf.Session() as sess:
        if gpu_id is None:
            model = AutoModel(x, td_neur, td_dend, mask)
        else:
            with tf.device("/gpu:0"):
                model = AutoModel(x, td_neur, td_dend, mask)

        # Initialize session variables
        init = tf.global_variables_initializer()
        sess.run(init)
        t_start = time.time()
        sess.run(model.reset_prev_vars)

        # Restore variables from previous model if desired
        saver = tf.train.Saver()
        if par['load_previous_model']:
            saver.restore(sess, par['save_dir'] + par['ckpt_load_fn'])
            print('Model ' +  par['ckpt_load_fn'] + ' restored.')

        #################
        ### Main Loop ###
        #################

        for j in range(1):          ########## Adjusted to only observe one task

            # Task header
            print('-'*40 + '\nTask {}\n'.format(j)+'-'*40)


            for i in range(par['num_iterations']):

                # make batch of training data
                trial_info = stim.generate_trial()

                if par['stabilization'] == 'pathint':

                    _, _, perf_loss, aux_loss, wiring_loss, h, gradients, big_omegas, pred_hist = sess.run([model.train_op, \
                        model.update_small_omega, model.error_loss, model.aux_loss, model.wiring_loss, \
                        model.hidden_history, model.gradients, model.big_omega_var, model.prediction_history], \
                        feed_dict = {x: trial_info['neural_input'], td_neur: par['neuron_topdown'][j], \
                                     td_dend: par['dendrite_topdown'][j], mask: np.ones_like(trial_info['train_mask'])})

                    #var_order = ['W_in', 'W_rnn', 'b_rnn', 'W_out', 'b_out']
                    #gradients = [gradients[i][0] for i in range(len(gradients))]
                    #omegas    = [big_omegas[k] for k in big_omegas.keys()]
                    """
                    print(len(pred_hist))
                    print(type(pred_hist[0][0]))
                    print(type(pred_hist[0][0][0]))
                    print(pred_hist[0][0][0].shape)
                    print(pred_hist[0][0].shape)
                    print(pred_hist[0].shape)
                    quit()
                    """

                    #quit('Passed')

                elif par['stabilization'] == 'EWC':
                    aux_loss = -1
                    _, acc, perf_loss, h = sess.run([model.train_op, model.accuracy, model.perf_loss, model.hidden_history], \
                        feed_dict = {x: trial_info['neural_input'], td_neur: par['neuron_topdown'][j], td_dend: par['dendrite_topdown'][j], \
                        y: trial_info['desired_output'], mask: trial_info['train_mask'], td_input: td_input_signal, gate_learning:gl})

                if i//par['iters_between_outputs'] == i/par['iters_between_outputs']:
                    #print('Iter ', i, 'Perf Loss ', perf_loss, ' AuxLoss ', aux_loss, ' Mean sr ', np.mean(h), ' WL ', wiring_loss)
                    print('Iter ', str(i).ljust(3), 'Perf Loss ', perf_loss, ' Mean sr ', np.mean(h), ' WL ', wiring_loss)
                    s = np.stack(pred_hist,axis=0)
                    print(s.shape)
                    #s = np.transpose(s,[2,1,0,3])
                    s = np.transpose(s,[1,0,2])
                    f = plt.figure(figsize=(12,8))
                    ax = f.add_subplot(2, 3, 1)
                    ax.imshow(trial_info['neural_input'][:,:,0], interpolation='none', aspect='auto')
                    #plt.colorbar()
                    ax = f.add_subplot(2, 3, 2)
                    ax.imshow(s[:,:,0], interpolation='none', aspect='auto')
                    #plt.colorbar()
                    ax = f.add_subplot(2, 3, 3)
                    ax.imshow(trial_info['neural_input'][:,:,0]-s[:,:,0], interpolation='none', aspect='auto')
                    #plt.colorbar()
                    ax = f.add_subplot(2, 3, 4)
                    ax.imshow(trial_info['neural_input'][:,:,1], interpolation='none', aspect='auto')
                    #plt.colorbar()
                    ax = f.add_subplot(2, 3, 5)
                    ax.imshow(s[:,:,1], interpolation='none', aspect='auto')
                    #plt.colorbar()
                    ax = f.add_subplot(2, 3, 6)
                    ax.imshow(trial_info['neural_input'][:,:,1]-s[:,:,1], interpolation='none', aspect='auto')
                    #plt.colorbar()
                    plt.show()


            # Update big omegaes, and reset other values before starting new task
            if par['stabilization'] == 'pathint':
                big_omegas = sess.run([model.update_big_omega, model.big_omega_var])
            elif par['stabilization'] == 'EWC':
                for n in range(par['EWC_fisher_num_batches']):
                    stim_in, y_hat, mk = stim.make_batch(task, test = False)
                    big_omegas = sess.run([model.update_big_omega,model.big_omega_var], feed_dict = \
                        {x:stim_in, y:y_hat, **gating_dict, mask:mk, droput_keep_pct:1.0, input_droput_keep_pct:1.0})

            # Reset the Adam Optimizer, and set the previous parater values to their current values
            sess.run(model.reset_adam_op)
            sess.run(model.reset_prev_vars)
            if par['stabilization'] == 'pathint':
                sess.run(model.reset_small_omega)

            """
            for k in range(j+1):

                # generate batch of batch_train_size
                _, trial_info = stim.generate_trial(k)
                acc, h, syn_x, syn_u = sess.run([model.accuracy, model.hidden_history, model.syn_x_hist, model.syn_u_hist], \
                    feed_dict = {x: trial_info['neural_input'], td_neur: par['neuron_topdown'][k], td_dend: par['dendrite_topdown'][k], \
                    y: trial_info['desired_output'], mask: trial_info['train_mask'], td_input: td_input_signal})
                print('ACC ',j,k,acc)
            """
            print('')
                #model_performance['accuracy'][j,k] = acc

            #print(model_performance['accuracy'])
            #model_performance['par'] = par
            #model_performance['task_list'] = par['task_list']

        if par['save_analysis']:
            save_results = {'task': task, 'accuracy': accuracy, 'accuracy_full': accuracy_full, \
                            'accuracy_grid': accuracy_grid, 'big_omegas': big_omegas, 'par': par}
            pickle.dump(save_results, open(par['save_dir'] + save_fn, 'wb'))

    print('\nModel execution complete.')


main('testing', str(sys.argv[1]))

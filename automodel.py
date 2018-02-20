"""
Nicolas Masse 2017
Contributions from Gregory Grant, Catherine Lee
"""

import tensorflow as tf
import numpy as np
import stimulus, stimulus2, stimulus3, multistim
import analysis
import AdamOpt
from parameters import *
import pickle
import matplotlib.pyplot as plt
import os, sys, time
from scipy import stats

# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

"""
Model setup and execution
"""

class AutoModel:

    def __init__(self, input_data, mask):

        # Load the input activity and gating mechanisms
        self.input_data  = tf.unstack(input_data, axis=1)
        self.mask        = tf.unstack(mask, axis=0)
        #self.neuron_td   = neuron_td[:,tf.newaxis]
        #self.dendrite_td = dendrite_td

        # Describe network shape
        self.num_layers = len(par['n_hidden'])

        # Initialize internal state
        self.pred_states   = [tf.zeros_like(self.input_data[0], dtype=tf.float32)]*self.num_layers
        self.rnn_states    = [tf.zeros([par['n_hidden'][n],1], dtype=tf.float32) for n in range(self.num_layers)]
        #self.rnn_states2    = [tf.zeros([par['n_hidden'][n],1], dtype=tf.float32) for n in range(self.num_layers)]
        self.target_states = [tf.zeros_like(self.input_data[0], dtype=tf.float32)]*self.num_layers

        # Run and optimize
        self.initialize_variables()
        self.run_model()
        self.optimize()


    def initialize_variables(self):
        c = 0.02
        for lid in range(self.num_layers):
            with tf.variable_scope('layer'+str(lid)):
                tf.get_variable('W_err1', shape=[par['n_hidden'][lid],par['n_dendrites'],par['n_input']],
                                initializer=tf.random_uniform_initializer(0, c), trainable=True)
                tf.get_variable('W_err2', shape=[par['n_hidden'][lid],par['n_dendrites'],par['n_input']],
                                initializer=tf.random_uniform_initializer(0, c), trainable=True)
                tf.get_variable('W_latent_mu', shape=[par['n_hidden'][lid],par['n_hidden'][lid]],
                                initializer=tf.random_uniform_initializer(-c, c), trainable=True)
                tf.get_variable('W_latent_sigma', shape=[par['n_hidden'][lid],par['n_hidden'][lid]],
                                initializer=tf.random_uniform_initializer(-c, c), trainable=True)

                #tf.get_variable('W_latenet_rnn', shape=[par['n_hidden'][lid],par['n_hidden'][lid]],
                                #initializer=tf.random_uniform_initializer(-c, c), trainable=True)
                tf.get_variable('W_pred1', shape=[par['n_input'],par['n_hidden'][lid]],
                                initializer=tf.random_uniform_initializer(-c, c), trainable=True)
                #tf.get_variable('W_pred2', shape=[par['n_input'],par['n_hidden'][lid]],
                                #initializer=tf.random_uniform_initializer(-c, c), trainable=True)

                tf.get_variable('W_rnn', shape = [par['n_hidden'][lid], par['n_dendrites'], par['n_hidden'][lid]], initializer = tf.random_uniform_initializer(-c, c), trainable=True)
                #tf.get_variable('W_rnn2', shape = [par['n_hidden'][lid], par['n_dendrites'], par['n_hidden'][lid]], initializer = tf.random_uniform_initializer(-c, c), trainable=True)
                tf.get_variable('b_rnn', initializer = np.zeros((par['n_hidden'][lid],1), dtype = np.float32), trainable=True)
                #tf.get_variable('b_rnn2', initializer = np.zeros((par['n_hidden'][lid],1), dtype = np.float32), trainable=True)
                tf.get_variable('b_pred1', initializer = np.zeros((par['n_input'],1), dtype = np.float32), trainable=True)
                #tf.get_variable('b_pred2', initializer = np.zeros((par['n_input'],1), dtype = np.float32), trainable=True)

                if lid < self.num_layers - 1 and False:
                    tf.get_variable('W_top_down', shape=[par['n_hidden'][lid],par['n_hidden'][lid+1]],
                                initializer=tf.random_uniform_initializer(-c, c), trainable=True)


    def calc_error(self, target, prediction):
        return tf.nn.relu(prediction - target), tf.nn.relu(target - prediction)
        #return (target - prediction),(prediction - target)


    def layer(self, lid, target, time):

        # Loading all weights and biases
        with tf.variable_scope('layer'+str(lid), reuse=True):
            W_err1 = tf.get_variable('W_err1') #positive error
            W_err2 = tf.get_variable('W_err2') #negative error
            W_latent_mu = tf.get_variable('W_latent_mu')
            W_latent_sigma = tf.get_variable('W_latent_sigma')
            W_pred1 = tf.get_variable('W_pred1')
            b_pred1 = tf.get_variable('b_pred1') # prediction bias
            #W_pred2 = tf.get_variable('W_pred2')
            #b_pred2 = tf.get_variable('b_pred2') # prediction bias
            W_rnn  = tf.get_variable('W_rnn')
            b_rnn  = tf.get_variable('b_rnn')


            if lid < self.num_layers - 1 and False:
                W_top_down = tf.get_variable('W_top_down')

        # Masking certain weights
        #W_rnn *= tf.constant(par['w_rnn_mask'], dtype=tf.float32)
        #if par['EI']:
            # ensure excitatory neurons only have postive outgoing weights,
            # and inhibitory neurons have negative outgoing weights
            #W_rnn = tf.tensordot(tf.nn.relu(W_rnn), tf.constant(par['EI_matrix']), [[2],[0]])

        # Processing data for RNN step
        # Currently, input from R l+1 to R l in not weighted
        err_stim1, err_stim2 = self.calc_error(target, self.pred_states[lid])
        rnn_state = self.rnn_states[lid]
        #rnn_state2 = self.rnn_states2[lid]
        if lid != self.num_layers-1 and False:
            rnn_next = tf.matmul(W_top_down, self.rnn_states[lid+1])
        else:
            rnn_next = tf.zeros_like(rnn_state)

        #print('W_err1',W_err1)
        #print('err_stim1',err_stim1)
        #print('W_rnn',W_rnn)
        #print('rnn_state',rnn_state)
        inp_act = tf.tensordot(W_err1, err_stim1, [[2],[0]]) + tf.tensordot(W_err2, err_stim2, [[2],[0]]) # Error activity
        rnn_act = tf.tensordot(W_rnn, rnn_state, [[2],[0]])       # RNN activity
        tot_act = par['alpha_neuron']*(inp_act + rnn_act)         # Modulating
        act_eff = tf.reduce_sum(tot_act, axis=1) # Summing dendrites

        # Updating RNN state
        #rnn_state = self.neuron_td*tf.nn.relu(rnn_state*(1-par['alpha_neuron']) + act_eff + rnn_next + b_rnn \
        #    + tf.random_normal(rnn_state.shape, 0, par['noise_rnn'], dtype=tf.float32))
        rnn_state = tf.nn.relu(rnn_state*(1-par['alpha_neuron']) + act_eff + rnn_next  + b_rnn \
            + tf.random_normal(rnn_state.shape, 0, par['noise_rnn'], dtype=tf.float32))
        self.rnn_states[lid]  = rnn_state

        # Updating prediction state
        latent_mu = tf.matmul(W_latent_mu, rnn_state) # A_hat
        latent_sigma = tf.matmul(W_latent_sigma, rnn_state)
        latent_sample = latent_mu + tf.exp(latent_sigma)*tf.random_normal([par['n_hidden'][lid], par['batch_train_size']], \
            0, 1 , dtype=tf.float32)

        pred1 = tf.nn.relu(tf.matmul(W_pred1, latent_sample)  + b_pred1)
        self.pred_states[lid] = pred1
        #self.pred_states[lid] = tf.nn.relu(tf.matmul(W_pred2, pred1) + b_pred2)

        if lid > - 1:
            self.latent_loss -= 0.5*par['latent_cost']*tf.reduce_sum(self.mask[time]*(1. + latent_sigma - tf.square(latent_mu) - tf.exp(latent_sigma)))

        return err_stim1 + err_stim2, rnn_state


    def run_model(self):

        # Start recording the error and hidden state histories
        self.error_history = [[] for _ in range(self.num_layers)]
        self.hidden_history = [[] for _ in range(self.num_layers)]
        self.prediction_history = [[] for _ in range(self.num_layers)]
        self.latent_loss = tf.constant(0.)

        # Iterate through time via the input data
        for t, input_data in enumerate(self.input_data):

            # Iterate over each layer at each time point
            for lid in range(self.num_layers):

                # If the first layer, use the actual input
                # Instead of using desired output for layer, we'll use neuronal input
                stim = input_data if lid == 0 else self.error_history[lid-1][-1]

                # Run the current layer and recover the error matrix,
                # then save the error matrix to the NEXT state position,
                # and record the RNN activity
                layer_error, rnn_state = self.layer(lid, stim, t)

                self.hidden_history[lid].append(rnn_state)
                self.prediction_history[lid].append(self.pred_states[lid])
                self.error_history[lid].append(layer_error)



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

        #self.wiring_loss = tf.constant(0.)
        with tf.variable_scope('layer0', reuse=True):
            W_err1 = tf.get_variable('W_err1')
            W_err2 = tf.get_variable('W_err2')
            W_latent_mu = tf.get_variable('W_latent_mu')
            W_latent_sigma = tf.get_variable('W_latent_sigma')
            W_pred1 = tf.get_variable('W_pred1')
            W_rnn  = tf.get_variable('W_rnn')
        self.wiring_loss = tf.reduce_sum(tf.abs(W_rnn)) + tf.reduce_sum(tf.abs(W_pred1)) + \
            tf.reduce_sum(tf.abs(W_latent_sigma)) + tf.reduce_sum(tf.abs(W_latent_mu)) + tf.reduce_sum(tf.abs(W_err1)) + tf.reduce_sum(tf.abs(W_err2))
        self.wiring_loss *= par['wiring_cost']
        """
        for var in [var for var in variables if ('W' in var.op.name and not 'td' in var.op.name)]:
            if 'W_in' in var.op.name:
                self.wiring_loss += tf.reduce_sum(tf.nn.relu(var) * tf.constant(par['w_in_pos'], dtype=tf.float32))
            elif 'W_rnn' in var.op.name:
                self.wiring_loss += tf.reduce_sum(tf.nn.relu(var) * tf.constant(par['w_rnn_pos'], dtype=tf.float32))
            elif 'W_out' in var.op.name:
                self.wiring_loss += tf.reduce_sum(tf.nn.relu(var) * tf.constant(par['w_out_pos'], dtype=tf.float32))
        self.wiring_loss *= par['wiring_cost']
        """

        self.error_loss = tf.constant(0.)
        """
        for m, h in zip(self.mask, self.error_history[1]):
            for l in h:
                #self.error_loss += tf.reduce_mean(tf.square(m[tf.newaxis,:]*l))
                self.error_loss += tf.reduce_mean(tf.square(l))
        """
        for j,eh in enumerate(self.error_history):
            if j>-1:
                for l in eh:
                    self.error_loss += tf.reduce_mean(tf.square(l))

        self.spike_loss = tf.constant(0.)
        for h in self.hidden_history:
            for l in h:
                self.spike_loss += par['spike_cost']*tf.reduce_mean(l)


        # Gradient of the loss+aux function, in order to both perform training and to compute delta_weights
        with tf.control_dependencies([self.error_loss, self.aux_loss,  self.spike_loss, self.latent_loss]):
            self.loss = self.error_loss + self.aux_loss + self.spike_loss + self.latent_loss
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
    #td_neur = tf.placeholder(tf.float32, shape=[par['n_hidden']])
    #td_dend = tf.placeholder(tf.float32, shape=[par['n_hidden'], par['n_dendrites']])

    """ Set up performance recording """
    accuracy_full = []
    par['n_tasks'] = 19
    accuracy_grid = np.zeros((par['n_tasks'], par['n_tasks']))
    model_performance = {'accuracy': [], 'par': [], 'task_list': []}

    """ Start TensorFlow session """
    with tf.Session() as sess:
        if gpu_id is None:
            model = AutoModel(x,  mask)
        else:
            with tf.device("/gpu:0"):
                model = AutoModel(x,  mask)

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

                    _, _, perf_loss, aux_loss, wiring_loss, latent_loss, h, gradients, big_omegas, pred_hist, err_hist = sess.run([model.train_op, \
                        model.update_small_omega, model.error_loss, model.aux_loss, model.wiring_loss, model.latent_loss, \
                        model.hidden_history, model.gradients, model.big_omega_var, model.prediction_history, model.error_history], \
                        feed_dict = {x: trial_info['neural_input'], mask: np.ones_like(trial_info['train_mask'])})

                    #var_order = ['W_in', 'W_rnn', 'b_rnn', 'W_out', 'b_out']
                    #gradients = [gradients[i][0] for i in range(len(gradients))]
                    #omegas    = [big_omegas[k] for k in big_omegas.keys()]


                elif par['stabilization'] == 'EWC':
                    aux_loss = -1
                    _, acc, perf_loss, h = sess.run([model.train_op, model.accuracy, model.perf_loss, model.hidden_history], \
                        feed_dict = {x: trial_info['neural_input'], td_neur: par['neuron_topdown'][j], td_dend: par['dendrite_topdown'][j], \
                        y: trial_info['desired_output'], mask: trial_info['train_mask'], td_input: td_input_signal, gate_learning:gl})

                if i//par['iters_between_outputs'] == i/par['iters_between_outputs']:
                    #print('Iter ', i, 'Perf Loss ', perf_loss, ' AuxLoss ', aux_loss, ' Mean sr ', np.mean(h), ' WL ', wiring_loss)
                    print('Iter ', str(i).ljust(3), 'Perf Loss ', perf_loss, ' Mean sr ', np.mean(h[0]), ' WL ', wiring_loss, ' LL ', latent_loss)

                if i//(10*par['iters_between_outputs']) == i/(10*par['iters_between_outputs']):

                    t0 = (par['dead_time'] + par['fix_time'] + par['sample_time'] + 200)//par['dt']
                    t1 = (par['dead_time'] + par['fix_time'] + par['sample_time'] + par['delay_time'])//par['dt']
                    h_stacked = [np.stack(h[n]) for n in range(len(par['n_hidden']))]
                    mean_h = [np.mean(h_stacked[n][t0:t1, :, :], axis=0)  for n in range(len(par['n_hidden']))]

                    p = [np.zeros((3,par['n_hidden'][n])) for n in range(len(par['n_hidden']))]
                    for m in range(len(par['n_hidden'])):
                        for n in range(par['n_hidden'][m]):
                            p[m][0,n], p[m][1,n], p[m][2,n] = two_way_anova(trial_info['rule'], trial_info['sample'], mean_h[m][n,:])


                    f = plt.figure(figsize=(12,8))
                    for n in range(len(par['n_hidden'])):
                        ax = f.add_subplot(1, len(par['n_hidden']), n+1)
                        ax.plot(np.maximum(-9,np.log10(p[n][0,:])),'b')
                        ax.plot(np.maximum(-9,np.log10(p[n][1,:])),'r')
                        ax.plot(np.maximum(-9,np.log10(p[n][2,:])),'g')

                    plt.show()





                    s = np.stack(pred_hist[0],axis=0)
                    e = np.stack(err_hist[0],axis=0)
                    #s = np.transpose(s,[2,1,0,3])
                    s = np.transpose(s,[1,0,2])
                    e = np.transpose(e,[1,0,2])
                    f = plt.figure(figsize=(12,8))
                    ax = f.add_subplot(2, 3, 1)
                    ax.imshow(trial_info['neural_input'][:,:,0], interpolation='none', aspect='auto')
                    #plt.colorbar()
                    ax = f.add_subplot(2, 3, 2)
                    ax.imshow(s[:,:,0], interpolation='none', aspect='auto')
                    #plt.colorbar()
                    ax = f.add_subplot(2, 3, 3)
                    ax.imshow(e[:,:,0], interpolation='none', aspect='auto')
                    #plt.colorbar()
                    ax = f.add_subplot(2, 3, 4)
                    ax.imshow(trial_info['neural_input'][:,:,1], interpolation='none', aspect='auto')
                    #plt.colorbar()
                    ax = f.add_subplot(2, 3, 5)
                    ax.imshow(s[:,:,1], interpolation='none', aspect='auto')
                    #plt.colorbar()
                    ax = f.add_subplot(2, 3, 6)
                    ax.imshow(e[:,:,1], interpolation='none', aspect='auto')
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


def two_way_anova(x1, x2, y):

    N = len(y)
    df_a = len(np.unique(x1)) - 1
    df_b = len(np.unique(x2)) - 1
    df_axb = df_a*df_b
    df_w = N - (df_a+1)*(df_b+1)

    grand_mean = np.mean(y)
    ssq_a = sum([(np.mean(y[np.where(x1==l)[0]])-grand_mean)**2 for l in x1])
    ssq_b = sum([(np.mean(y[np.where(x2==l)[0]])-grand_mean)**2 for l in x2])
    ssq_t = np.sum((y-grand_mean)**2)

    m = []
    for a in np.unique(x1):
        ind_a = np.where(x1==a)[0]
        x2a = x2[ind_a]
        ya = y[ind_a]
        for j,b in enumerate(x2a):
            ind_b = np.where(x2a == b)[0]
            m.append(np.sum((ya[j] - np.mean(ya[ind_b]))**2))
    ssq_w = np.sum(m)

    ssq_axb = ssq_t-ssq_a-ssq_b-ssq_w
    ms_a = ssq_a/df_a
    ms_b = ssq_b/df_b
    ms_axb = ssq_axb/df_axb
    ms_w = ssq_w/df_w + 1e-9
    f_a = ms_a/ms_w
    f_b = ms_b/ms_w
    f_axb = ms_axb/ms_w
    p_a = stats.f.sf(f_a, df_a, df_w)
    p_b = stats.f.sf(f_b, df_b, df_w)
    p_axb = stats.f.sf(f_axb, df_axb, df_w)

    return p_a, p_b, p_axb



def anova_custom(x, conds):

    m = len(np.unique(conds))
    lx = len(x)
    xr = x - np.mean(x)
    xm = np.zeros((m))
    countx = np.zeros((m))
    for i in range(m):
        ind = np.where(conds==i)[0]
        countx[i] = len(ind)
        xm[i] = np.mean(xr[ind])

    gm = np.mean(xr)
    df1 = np.sum(countx > 0) - 1
    df2 = lx - df1 - 1
    xc = xm - gm
    ind0 = np.where(countx == 0)[0]
    xc[ind0] = 0
    RSS = np.sum(countx*(xc**2))
    TSS = np.sum((xr - gm)**2)
    SSE = TSS - RSS
    if df > 0:
        mse = SSE/df2
    else:
        mse = 1e32

    F = (RSS/df1)/mse

    return F, SSE, TSS, RSS, df1, df2




main('testing', str(sys.argv[1]))

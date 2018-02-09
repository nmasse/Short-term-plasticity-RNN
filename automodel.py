"""
Nicolas Masse 2017
Contributions from Gregory Grant, Catherine Lee
"""

import tensorflow as tf
import numpy as np
import stimulus
import analysis
import AdamOpt
from parameters import *
import pickle
import multistim
import matplotlib.pyplot as plt
import os, sys, time

# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

"""
Model setup and execution
"""

class AutoModel:

    def __init__(self, input_data, neuron_td, dendrite_td):

        # Load the input activity and gating mechanisms
        self.target      = input_data
        self.neuron_td   = neuron_td
        self.dendrite_td = dendrite_td

        # Describe network shape
        self.num_layers = 2
        self.lids = list(range(self.num_layers))

        # Initialize internal state
        init_pred = tf.zeros_like(input_data, dtype=tf.float32)
        init_rnn = tf.zeros(par['n_hidden'], dtype=tf.float32)
        self.pred_states = [prediction]*self.num_layers
        self.rnn_states = [init_rnn]*self.num_layers


    def calc_error(self, target, prediction):
        return prediction - target


    def layer(self, lid, target):

        # Loading all weights and biases
        intlzr = tf.random_uniform_initializer
        with tf.variable_scope('layer_'+str(lid)):
            W_err  = tf.get_variable('W_err', shape=[par['n_hidden'],par['n_dendrites'],par['n_input']],
                                     initializer=intlzr(-1/np.sqrt(par['n_input']), 1/np.sqrt(par['n_input'])), trainable=True)
            W_pred = tf.get_variable('W_pred', shape=[par['n_input'],par['n_hidden']],
                                     initializer=intlzr(-1/np.sqrt(par['n_hidden']), 1/np.sqrt(par['n_hidden'])), trainable=True)
            W_rnn = tf.get_variable('W_rnn', initializer = par['w_rnn0'], trainable=True)
            b_rnn = tf.get_variable('b_rnn', initializer = par['b_rnn0'], trainable=True)
            b_out = tf.get_variable('b_out', initializer = par['b_out0'], trainable=True)

        # Processing data for RNN step
        err_stim  = self.calc_error(target, self.pred_states[lid])
        rnn_state = self.rnn_states[lid]
        if lid != self.num_layers-1:
            rnn_next = self.rnn_states[lid+1]
        else:
            rnn_next = tf.zeros_like(rnn_state)

        inp_act = tf.tensordot(W_err, err_stim, [[2],[0]])
        rnn_act = tf.tensordot(W_rnn, rnn_state, [[2],[0]])
        tot_act = par['alpha_neuron']*(inp_act + rnn_act)
        act_eff = tf.reduce_sum(self.dendrite_td*tot_act, axis=1) # Summing dendrites

        # Updating RNN state
        rnn_state = self.neuron_td*tf.nn.relu(rnn_state*(1-par['alpha_neuron']) + act_eff + rnn_next + b_rnn) \
                  + tf.random.normal(rnn_state.shape, 0, par['noise_rnn'], dtype=tf.float32)

        # Updating prediction state
        pred_state = tf.matmul(W_pred, rnn_state) + b_out

        # Plugging the RNN and prediction states back into the aggregate model
        self.pred_state[lid] = pred_state
        self.rnn_state[lid]  = rnn_state

        return err_stim


    def run_model(self):

        # Access each layer in order from 0 to n,
        # updating each successive layer with the
        # calculated error value
        pass




























    def optimize(self):

        # Use all trainable variables, except those in the convolutional layers
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

        self.task_loss = tf.reduce_mean([mask*tf.nn.softmax_cross_entropy_with_logits(logits = y, \
            labels = target, dim=0) for y, target, mask in zip(self.y_hat, self.target_data, self.mask)])

        self.spike_loss = tf.reduce_mean([par['spike_cost']*tf.reduce_mean(tf.square(h)) for h in self.hidden_state_hist])

        # Gradient of the loss+aux function, in order to both perform training and to compute delta_weights
        with tf.control_dependencies([self.task_loss, self.aux_loss, self.wiring_loss, self.spike_loss]):
            self.loss = self.task_loss + self.aux_loss + self.wiring_loss + self.spike_loss
            self.train_op = adam_optimizer.compute_gradients(self.loss)

        if par['stabilization'] == 'pathint':
            # Zenke method
            self.pathint_stabilization(variables, adam_optimizer, previous_weights_mu_minus_1)

        elif par['stabilization'] == 'EWC':
            # Kirkpatrick method
            self.EWC(variables)

        self.reset_prev_vars = tf.group(*reset_prev_vars_ops)
        self.reset_adam_op = adam_optimizer.reset_params()

        correct_prediction = [tf.reduce_sum(mask*tf.cast(tf.less(tf.argmax(desired_output,0), par['num_motion_dirs']), tf.float32)*tf.cast(tf.equal(tf.argmax(y_hat,0), tf.argmax(desired_output,0)), tf.float32)) \
            for (y_hat, desired_output, mask) in zip(self.y_hat, self.target_data, self.mask)]
        correct_count = [tf.reduce_sum(mask*tf.cast(tf.less(tf.argmax(desired_output,0),par['num_motion_dirs']),tf.float32)) \
            for (desired_output, mask) in zip(self.target_data, self.mask)]

        self.accuracy = tf.reduce_sum(tf.stack(correct_prediction))/tf.reduce_sum(tf.stack(correct_count))

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
            self.gradients = optimizer_task.compute_gradients(self.task_loss)
            for grad,var in self.gradients:
                update_small_omega_ops.append(tf.assign_add(small_omega_var[var.op.name], -self.delta_grads[var.op.name]*grad ) )
            self.update_small_omega = tf.group(*update_small_omega_ops) # 1) update small_omega after each train!



def train_and_analyze(gpu_id, save_fn):

    tf.reset_default_graph()
    main(gpu_id, save_fn)
    #update_parameters(revert_analysis_par)


def main(save_fn, gpu_id = None):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    print('\nRunning model.\n')

    """
    Reset TensorFlow before running anything
    """
    tf.reset_default_graph()
    """
    Create the stimulus class to generate trial paramaters and input activity
    """
    stim = multistim.MultiStimulus()

    n_input, n_hidden, n_output = par['shape']
    N = par['batch_train_size'] # trials per iteration, calculate gradients after batch_train_size

    """
    Define all placeholder
    """
    mask = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_train_size']])
    x = tf.placeholder(tf.float32, shape=[n_input, par['num_time_steps'], par['batch_train_size']])  # input data
    y = tf.placeholder(tf.float32, shape=[n_output, par['num_time_steps'], par['batch_train_size']]) # target data
    td_neur = tf.placeholder(tf.float32, shape=[par['n_hidden']]) # target data
    td_dend = tf.placeholder(tf.float32, shape=[par['n_hidden'], par['n_dendrites']]) # target data
    td_input = tf.placeholder(tf.float32, shape=[par['num_tasks'], par['batch_train_size']]) # top-down input signal
    gate_learning = tf.placeholder(tf.float32)

    accuracy_full = []
    par['n_tasks'] = 19
    accuracy_grid = np.zeros((par['n_tasks'], par['n_tasks']))

    model_performance = {'accuracy': [], 'par': [], 'task_list': []}

    with tf.Session() as sess:
        if gpu_id is None:
            model = Model(x, td_neur, td_dend, y, mask, td_input, gate_learning)
        else:
            with tf.device("/gpu:0"):
                model = Model(x, td_neur, td_dend, y, mask, td_input, gate_learning)

        init = tf.global_variables_initializer()
        sess.run(init)
        t_start = time.time()
        sess.run(model.reset_prev_vars)

        saver = tf.train.Saver()
        # Restore variables from previous model if desired
        if par['load_previous_model']:
            saver.restore(sess, par['save_dir'] + par['ckpt_load_fn'])
            print('Model ' +  par['ckpt_load_fn'] + ' restored.')

        for j in range(2):          ########## Adjusted to only observe first two tasks
            print('-'*40)
            print('Task {}'.format(j))
            print('-'*40)

            td_input_signal = np.zeros((par['num_tasks'], par['batch_train_size']), dtype = np.float32)
            td_input_signal[j, :] = 1   # This ONLY is for dynamic topdown

            for i in range(par['num_iterations']):

                # make batch of training data
                task_name, trial_info = stim.generate_trial(j)
                gl = 0.0

                if par['stabilization'] == 'pathint':

                    _, _, perf_loss, aux_loss, wiring_loss, acc, h, gradients, big_omegas = sess.run([model.train_op, model.update_small_omega, model.task_loss, model.aux_loss,
                        model.wiring_loss, model.accuracy, model.hidden_state_hist, model.gradients, model.big_omega_var], \
                        feed_dict = {x: trial_info['neural_input'], td_neur: par['neuron_topdown'][j], td_dend: par['dendrite_topdown'][j], \
                        y: trial_info['desired_output'], mask: trial_info['train_mask'], td_input: td_input_signal, gate_learning: gl})

                    var_order = ['W_in', 'W_rnn', 'b_rnn', 'W_out', 'b_out']
                    gradients = [gradients[i][0] for i in range(len(gradients))]
                    omegas    = [big_omegas[k] for k in big_omegas.keys()]

                elif par['stabilization'] == 'EWC':
                    aux_loss = -1
                    _, acc, perf_loss, h = sess.run([model.train_op, model.accuracy, model.perf_loss, model.hidden_state_hist], \
                        feed_dict = {x: trial_info['neural_input'], td_neur: par['neuron_topdown'][j], td_dend: par['dendrite_topdown'][j], \
                        y: trial_info['desired_output'], mask: trial_info['train_mask'], td_input: td_input_signal, gate_learning:gl})

                if i//par['iters_between_outputs'] == i/par['iters_between_outputs']:
                    print('Iter ', i, 'Accuracy ', acc , ' AuxLoss ', aux_loss , 'Perf Loss ', perf_loss, ' Mean sr ', np.mean(h), ' WL ', wiring_loss)

                    if j > 0 and i == 0:

                        omega_sum = np.sum(omegas[0][:,0,:],axis=1) + np.sum(omegas[1][:,0,:],axis=0) + np.sum(omegas[3],axis=0)
                        print(omega_sum.shape)
                        omega_th = np.percentile(omega_sum, 100 - par['gate_pct']*100)
                        ind_gate = np.where(omega_sum > omega_th)[0]
                        print('gating ', ind_gate)

                        par['neuron_topdown'][j][ind_gate] = 0

                        print('omega_c', par['omega_c'])
                        f, ax = plt.subplots(2,3)
                        for plot, title, gr, om in zip(range(len(var_order)), var_order, gradients, omegas):
                            print(title, om.shape)
                            om_flat = om.flatten()
                            gr_flat =  np.abs(gr.flatten())
                            ax[plot//3, plot%3].plot(om_flat,gr_flat,'b.')
                            if plot==0: # w_in
                                flat_mask = par['fix_connections_in'].flatten()
                                ind_fix = np.where(flat_mask==1)[0]
                                ax[plot//3, plot%3].plot(om_flat[ind_fix],gr_flat[ind_fix],'r.')
                            elif plot==1: # w_rnn
                                flat_mask = par['fix_connections_rnn'].flatten()
                                ind_fix = np.where(flat_mask==1)[0]
                                ax[plot//3, plot%3].plot(om_flat[ind_fix],gr_flat[ind_fix],'r.')
                            elif plot==3: # w_out
                                flat_mask = par['fix_connections_out'].flatten()
                                ind_fix = np.where(flat_mask==1)[0]
                                ax[plot//3, plot%3].plot(om_flat[ind_fix],gr_flat[ind_fix],'r.')
                            ax[plot//3, plot%3].set_title('Task {} '.format(j) + title)
                            ax[plot//3, plot%3].set_xlabel('Omegas')
                            ax[plot//3, plot%3].set_ylabel('Gradients')
                        plt.show()
                        #plt.savefig('grad_omega_task{}_gating{}.png'.format(j, int(100*par['neuron_gate_pct'])))
                        #plt.clf()
                        #plt.cla()
                        #plt.close()

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

            for k in range(j+1):

                # generate batch of batch_train_size
                _, trial_info = stim.generate_trial(k)
                acc, h, syn_x, syn_u = sess.run([model.accuracy, model.hidden_state_hist, model.syn_x_hist, model.syn_u_hist], \
                    feed_dict = {x: trial_info['neural_input'], td_neur: par['neuron_topdown'][k], td_dend: par['dendrite_topdown'][k], \
                    y: trial_info['desired_output'], mask: trial_info['train_mask'], td_input: td_input_signal})
                print('ACC ',j,k,acc)

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


#main('testing', str(sys.argv[1]))

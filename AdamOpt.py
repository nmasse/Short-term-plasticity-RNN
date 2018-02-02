import numpy as np
import tensorflow as tf
from itertools import product
from parameters import *

class AdamOpt:

    """
    Example of use:

    optimizer = AdamOpt.AdamOpt(variables, learning_rate=self.lr)
    self.train = optimizer.compute_gradients(self.loss, gate=0)
    gvs = optimizer.return_gradients()
    self.g = gvs[0][0]
    self.v = gvs[0][1]
    """

    def __init__(self, variables, learning_rate = 0.001):

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-08
        self.t = 0
        self.variables = variables
        self.learning_rate = learning_rate

        self.m = {}
        self.v = {}
        self.delta_grads = {}
        for var in self.variables:
            self.m[var.op.name]  = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            self.v[var.op.name]  = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            self.delta_grads[var.op.name]  = tf.Variable(tf.zeros(var.get_shape()), trainable=False)

        #self.grad_descent = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
        self.grad_descent = tf.train.GradientDescentOptimizer(learning_rate = 1.0)


    def reset_params(self):

        self.t = 0
        reset_op = []
        for var in self.variables:
            reset_op.append(tf.assign(self.m[var.op.name], tf.zeros(var.get_shape())))
            reset_op.append(tf.assign(self.v[var.op.name], tf.zeros(var.get_shape())))
            reset_op.append(tf.assign(self.delta_grads[var.op.name], tf.zeros(var.get_shape())))

        return tf.group(*reset_op)


    def compute_gradients(self, loss, apply = True):

        self.gradients = self.grad_descent.compute_gradients(loss, var_list = self.variables)

        self.t += 1
        lr = self.learning_rate*np.sqrt(1-self.beta2**self.t)/(1-self.beta1**self.t)
        self.update_var_op = []

        for (grads, vv), var in zip(self.gradients, self.variables):

            print(var.op.name)
            print(grads)
            print(vv)
            print(var)

            if var.op.name == "rnn_cell/W_rnn":
                print('Applying mask to w_rnn gradient')
                grads *= par['w_rnn_mask']
            elif var.op.name == "output/W_out":
                print('Applying mask to w_out gradient')
                grads *= par['w_out_mask']

            grads = tf.clip_by_norm(grads, par['clip_max_grad_val'])
            new_m = self.beta1*self.m[var.op.name] + (1-self.beta1)*grads
            new_v = self.beta2*self.v[var.op.name] + (1-self.beta2)*grads*grads
            self.update_var_op.append(tf.assign(self.m[var.op.name], new_m))
            self.update_var_op.append(tf.assign(self.v[var.op.name], new_v))
            delta_grad = - lr*new_m/(tf.sqrt(new_v) + self.epsilon)
            self.update_var_op.append(tf.assign(self.delta_grads[var.op.name], delta_grad))

            if apply:
                self.update_var_op.append(tf.assign_add(var, delta_grad))

        return tf.group(*self.update_var_op)



    def compute_gradients_with_competition(self, loss, omegas, apply = True):

        self.gradients = self.grad_descent.compute_gradients(loss, var_list = self.variables)
        epsilon = 1e-4

        self.t += 1
        lr = self.learning_rate*np.sqrt(1-self.beta2**self.t)/(1-self.beta1**self.t)
        self.update_var_op = []

        for (grads, _), var in zip(self.gradients, self.variables):
            new_m = self.beta1*self.m[var.op.name] + (1-self.beta1)*grads
            new_v = self.beta2*self.v[var.op.name] + (1-self.beta2)*grads*grads
            self.update_var_op.append(tf.assign(self.m[var.op.name], new_m))
            self.update_var_op.append(tf.assign(self.v[var.op.name], new_v))
            delta_grad = - lr*new_m/(tf.sqrt(new_v) + self.epsilon)

            m = 1*tf.reduce_mean(omegas[var.op.name]) + 1e-9
            delta_grad_clipped = delta_grad*(m - tf.sign(omegas[var.op.name])/2 + 0.5)

            delta_grad_clipped = tf.clip_by_norm(delta_grad_clipped, 1)

            self.update_var_op.append(tf.assign(self.delta_grads[var.op.name], delta_grad_clipped))
            if apply:
                self.update_var_op.append(tf.assign_add(var, delta_grad_clipped))

        return tf.group(*self.update_var_op)


    def compute_gradients_with_weight_stabilization(self, loss, omegas, prev_var, apply = True):
        # currently unsued

        self.gradients = self.grad_descent.compute_gradients(loss, var_list = self.variables)
        epsilon = 1e-4

        self.t += 1
        lr = self.learning_rate*np.sqrt(1-self.beta2**self.t)/(1-self.beta1**self.t)
        self.update_var_op = []

        for (grads, _), var in zip(self.gradients, self.variables):
            new_m = self.beta1*self.m[var.op.name] + (1-self.beta1)*grads
            new_v = self.beta2*self.v[var.op.name] + (1-self.beta2)*grads*grads
            self.update_var_op.append(tf.assign(self.m[var.op.name], new_m))
            self.update_var_op.append(tf.assign(self.v[var.op.name], new_v))
            delta_grad = - lr*new_m/(tf.sqrt(new_v) + self.epsilon)

            omega_vector = - omegas[var.op.name]*(var + delta_grad - prev_var[var.op.name])
            omega_vector_unit = omega_vector/(epsilon + tf.reduce_sum(tf.square(omega_vector)))

            delta_grad_proj = delta_grad - delta_grad*omega_vector_unit

            delta_grad_clipped = tf.clip_by_norm(delta_grad_proj, 1)

            self.update_var_op.append(tf.assign(self.delta_grads[var.op.name], delta_grad_clipped))
            if apply:
                self.update_var_op.append(tf.assign_add(var, delta_grad_clipped))

        return tf.group(*self.update_var_op)

    def dendritic_competition(self, delta_grad, var_name):
        # Currently unused, might delete in future

        corrected_delta_grads = []

        if var_name.find('W') == -1 or var_name.find('2') > 0:
            return delta_grad
        else:
            print(var_name, var_name.find('2'))
            delta_grad_branches = tf.unstack(delta_grad, axis=2)
            corrected_delta_grads = []

            # cycle through dendrites, post-synaptic neurons
            for delta_branch in delta_grad_branches:
                #corrected_delta_grads.append(delta_branch*tf.nn.softmax(tf.abs(3*delta_branch), dim = 0))
                s = tf.exp(tf.abs(delta_branch))
                corrected_delta_grads.append(delta_branch*s/(1e-6+tf.reduce_mean(s)))

            corrected_delta_grads = tf.stack(corrected_delta_grads, axis = 2)
            print(var_name, ' corrected_delta_grads', corrected_delta_grads)
            return corrected_delta_grads

    def return_delta_grads(self):
        return self.delta_grads

    def return_means(self):
        return self.m

    def return_grads_and_vars(self):
        return self.gradients

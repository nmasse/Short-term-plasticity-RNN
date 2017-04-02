import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import functools

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator
      
class weight_initialization:
    
    # create weight and bias values that will be used to initialized the network
    def __init__(self, num_input, num_RNN, num_output, EI_list):
        
        self.num_input = num_input
        self.num_RNN = num_RNN
        self.num_output = num_output
        self.EI_list = EI_list 
        
    def create_output_weights(self, weight_multiplier=0.1, connection_prob=1):
        
        w_out = weight_multiplier*np.random.gamma(shape=0.25, scale=1.0, size=[self.num_RNN, num_rnn_exc])
        
        # will only project from excitatory RNN neurons to the output neurons
        # values in EI_list should be 1 for excitatory, -1 for inhibitory
        ind_exc = self.EI_list == 1
        w_out[ind_exc, :] = 0
        
        w_out = np.float32(w_out)
        mask = np.random.rand(self.num_output, num_rnn_exc)
        mask = mask < connection_prob
        w_out *= mask
        b_out = np.zeros((self.num_output), dtype = np.float32)
        
        return w_out, b_out
        
    def create_RNN_weights(self, connection_prob=1):
        
        w_rnn = np.random.gamma(shape=0.25, scale=1.0, size=[self.num_RNN, self.num_RNN])
        w_rnn = np.float32(w_rnn)
        mask = np.random.rand(self.num_RNN, self.num_RNN)
        mask = mask < connection_prob
        w_rnn *= mask
        
        # make sure neurons don't project onto themselves
        for i in range(self.num_RNN):
            w_rnn[i,i] = 0
            
        rnn_spectral_rad = self.spectral_radius(w_rnn)
        w_rnn /= rnn_spectral_rad
        b_rnn = np.zeros((self.num_RNN), dtype = np.float32)
        
        weight_mask = np.ones((self.num_RNN,self.num_RNN)) - np.identity(self.num_RNN)
        rnn_weight_maks = tf.constant(weight_mask, dtype = tf.float32)
        
        return w_rnn, b_rnn, rnn_weight_mask, rnn_spectral_rad
        
    def create_input_to_RNN_weights(self, weight_multiplier=0.1, connection_prob=1):
        
        w_in = weight_multiplier*np.random.gamma(shape=0.25, scale=1.0, size=[self.num_RNN, self.num_input])
        w_in = np.float32(w_in)
        mask = np.random.rand(self.num_RNN, self.num_input)
        mask = mask < connection_prob
        w_in *= mask
        w_in = np.maximum(0, w_in)
        
        return w_in
        
    def spectral_radius(A):
        
        #Compute the spectral radius of a matrix.
        return np.max(abs(np.linalg.eigvals(A)))
    

class RNN:

    def __init__(self, data, target, params):
        self.data = data
        self.target = target
        # initiate weight initialization class
        self.weight_init = weight_initialization(params['num_input_neurons'], params['num_RNN_neurons'], params['num_classes'], 0.8)
        
        self.prediction
        self.optimize
        self.error

    @lazy_property
    def prediction(self):
        data_size = int(self.data.get_shape()[1])
        target_size = int(self.target.get_shape()[1])
        
        
        
        weight = tf.Variable(tf.truncated_normal([data_size, target_size]))
        bias = tf.Variable(tf.constant(0.1, shape=[target_size]))
        incoming = tf.matmul(self.data, weight) + bias
        return tf.nn.softmax(incoming)

    @lazy_property
    def optimize(self):
        cross_entropy = -tf.reduce_sum(self.target, tf.log(self.prediction))
        optimizer = tf.train.RMSPropOptimizer(0.03)
        return optimizer.minimize(cross_entropy)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))
    

    
def main(trial_params, load_previous_model = False, load_filename, save_filename):
    
    tf.reset_default_graph()

    """
    Placeholders
    """
    input_data = tf.placeholder(tf.float32, shape=[params['num_input_neurons'], trial_params['trial_length'], trial_params['trials_per_batch']])
    target_data = tf.placeholder(tf.float32, shape=[params['num_classes'], trial_params['trial_length'], trial_params['trials_per_batch']])
    target_data = tf.unstack(target_data, axis=1)
    init_state = tf.zeros([params['num_RNN_neurons'], trial_params['trials_per_batch']])

    # initialize network weights
    weight_init = weight_initialization(params['num_input_neurons'], params['num_RNN_neurons'], params['num_classes'], params['EI_list'])
    w_in_start = weight_init.create_input_to_RNN_weights()
    w_rnn_start, b_rnn_start, rnn_weight_mask, _ = weight_init.create_RNN_weights()
    w_out_start, b_out_start = weight_init.create_output_weights()
        
    with tf.variable_scope('rnn_cell'):
        W_in = tf.get_variable('W_in', initializer = w_in_start, trainable=False)
        W_rnn = tf.get_variable('W_rnn', initializer = w_rnn_start, trainable=True)
        b_rnn = tf.get_variable('b_rnn', initializer = b_rnn_start, trainable=True)
        W_ei = tf.diag(EI_list, name = 'EI_diag')
        cell_integration = tf.constant(neuron_integration_const*np.identity(num_RNN), dtype=np.float32, name = 'cell_int')
    
    with tf.variable_scope('output'):
        W_out = tf.get_variable('W_in', initializer = w_out_start, trainable=True)
        b_out = tf.get_variable('b_out', initializer = b_out_start, trainable=True)
        
    
    # run the RNN
    cell_input = init_state
    cell_output = init_state
    rnn_outputs = []
    rnn_inputs = tf.unstack(x, axis=1)
    for rnn_input in rnn_inputs:
        cell_input, cell_output = rnn_cell(rnn_input, cell_input, cell_output)
        rnn_outputs.append(cell_output)
    final_state = rnn_outputs[-1]

       
    # calculate the output values, and the loss function
    outputs = [tf.transpose(tf.matmul(tf.nn.relu(W_out),rnn_output) + b_out) for rnn_output in rnn_outputs]
    losses = [tf.square(tf.transpose(output)-desired_output) for output, desired_output in zip(outputs, desired_outputs)]
    total_loss = tf.reduce_mean(losses)
    error_rate = [tf.reduce_mean(tf.abs(tf.to_float(tf.argmax(output,1)-tf.argmax(desired_output,0)))) for output, desired_output in zip(outputs, desired_outputs)]
        
    # calculate the gradeints
    adam_opt = tf.train.AdamOptimizer(trial_params['learning_rate'])
    grads_vars = adam_opt.compute_gradients(total_loss)
    
    # apply any masks to the gradient
    capped_gvs = []
    for grad, var in grads_vars:
        # apply the weight maks to the RNN weights so that units will not project onto themselves
        if grad.shape == rnn_weight_mask.shape:
            print('Applying mask to ' + str(var))
            capped_gvs.append((tf.multiply(tf.clip_by_norm(grad, params['clip_max_grad_val']),tf_mask), var))
        else:
            capped_gvs.append((tf.clip_by_norm(grad, params['clip_max_grad_val']), var))               
        
    # apply the gradient
    train_op = adam_opt.apply_gradients(capped_gvs)
    
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        if load_previous_model:
            # Restore variables from disk.
            saver.restore(sess, load_filename)
            print('Model ' +  load_filename + ' restored.')

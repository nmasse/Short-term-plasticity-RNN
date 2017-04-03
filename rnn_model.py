import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import functools
import weights
import stimulus
import importlib
import pickle

importlib.reload(weights)
importlib.reload(stimulus)

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator
       

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
    
def rnn_cell(rnn_input, cell_input, cell_output):
    with tf.variable_scope('rnn_cell', reuse=True):
        W_in = tf.get_variable('W_in')
        W_rnn = tf.get_variable('W_rnn')
        b_rnn = tf.get_variable('b_rnn')
        W_ei = tf.get_variable('EI')
        cell_integration = tf.get_variable('cell_int')
        W_rnn_effective = tf.matmul(tf.nn.relu(W_rnn), W_ei)
        cell_input = tf.matmul(cell_integration, cell_input) + tf.matmul(W_in,rnn_input) + tf.matmul(W_rnn_effective, cell_output) + b_rnn
        cell_output = tf.nn.relu(cell_input)
        
    return cell_input, cell_output
    

    
def main(params, stimulus_type, num_epochs, load_previous_model = False, load_filename='model.ckpt', save_filename='model.ckpt'):
    
    num_batches = 10
    trials_per_batch = 250
    tf.reset_default_graph()
    save_dir = '/home/masse/saved_rnn_model_files/'
    
    load_filename = save_dir + load_filename
    save_filename = save_dir + save_filename

    """
    Placeholders
    """
    input_data = tf.placeholder(tf.float32, shape=[params['num_input_neurons'], params['trial_length']//params['delta_t'], trials_per_batch])
    input_data_unstacked = tf.unstack(input_data, axis=1)
    target_data = tf.placeholder(tf.float32, shape=[params['num_classes'], params['trial_length']//params['delta_t'], trials_per_batch])
    target_data_unstacked = tf.unstack(target_data, axis=1)
    init_state = tf.placeholder(tf.float32, shape=[params['num_RNN_neurons'], trials_per_batch])

    # initialize network weights
    weight_init = weights.weight_initialization(params)
    w_in_start = weight_init.create_input_to_RNN_weights()
    w_rnn_start, b_rnn_start, weight_mask = weight_init.create_RNN_weights()
    w_out_start, b_out_start = weight_init.create_output_weights()
    rnn_weight_mask = tf.constant(weight_mask, dtype = tf.float32)
        
    with tf.variable_scope('rnn_cell'):
        W_in = tf.get_variable('W_in', initializer = w_in_start, trainable=True)
        W_rnn = tf.get_variable('W_rnn', initializer = w_rnn_start, trainable=True)
        b_rnn = tf.get_variable('b_rnn', initializer = b_rnn_start, trainable=True)
        W_ei = tf.get_variable('EI', initializer = np.diag(params['EI_list']), trainable=False)
        cell_integration = tf.get_variable('cell_int', initializer = np.float32(params['neuron_integration_const']*np.identity(params['num_RNN_neurons'])), trainable=False)  
    
    # run the RNN
    cell_input = init_state
    cell_output = init_state
    rnn_outputs = []
    for rnn_input in input_data_unstacked:
        cell_input, cell_output = rnn_cell(rnn_input, cell_input, cell_output)
        rnn_outputs.append(cell_output)
    final_state = rnn_outputs[-1]

    with tf.variable_scope('output'):
        W_out = tf.get_variable('W_out', initializer = w_out_start, trainable=True)
        b_out = tf.get_variable('b_out', initializer = b_out_start, trainable=True)
       
    # calculate the output values, and the loss function
    outputs = [tf.transpose(tf.matmul(tf.nn.relu(W_out),rnn_output) + b_out) for rnn_output in rnn_outputs]
    losses = [tf.square(tf.transpose(output)-desired_output) for output, desired_output in zip(outputs, target_data_unstacked)]
    total_loss = tf.reduce_mean(losses)
    error_rate = [tf.reduce_sum(tf.to_float(tf.not_equal(tf.argmax(output,1),tf.argmax(desired_output,0)))) for output, desired_output in zip(outputs, target_data_unstacked)]
    
    #error_rate = [tf.reduce_sum((output[:,1])) for output, desired_output in zip(outputs, target_data_unstacked)]
        
    # calculate the gradeints
    adam_opt = tf.train.AdamOptimizer(params['learning_rate'])
    grads_vars = adam_opt.compute_gradients(total_loss)
    
    # apply any masks to the gradient
    capped_gvs = []
    for grad, var in grads_vars:
        # apply the weight maks to the RNN weights so that units will not project onto themselves
        if grad.shape == rnn_weight_mask.shape:
            print('Applying mask to ' + str(var))
            capped_gvs.append((tf.clip_by_norm(tf.multiply(grad,rnn_weight_mask), params['clip_max_grad_val']), var))
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
            
        training_losses = []
        training_state = np.zeros([params['num_RNN_neurons'], trials_per_batch],dtype=np.float32)
        
        if stimulus_type == 'spatial_cat':
            stim = stimulus.spatial_stimulus(params)
        elif stimulus_type == 'motion_match':
            stim = stimulus.motion_stimulus(params)
        
        for i in range(num_epochs):
            training_loss = 0
            accuracy = 0

            if i%10==0:
                if stimulus_type == 'spatial_cat':
                    trial_info = stim.generate_spatial_cat_trial(num_batches, trials_per_batch)
                elif stimulus_type == 'motion_match':
                    trial_info = stim.generate_motion_working_memory_trial(num_batches, trials_per_batch)

                x = trial_info['neural_input']
                z = trial_info['desired_output']
      
            if i%500==0 and i>1:
                save_path = saver.save(sess,save_filename)
                print("Model saved in file: %s" % save_path)
                
            for j in range(num_batches):
                x_current = np.squeeze(x[j,:,:,:])
                z_current = np.squeeze(z[j,:,:,:])

                tr_losses, training_loss_, fs , grad_vars_batch, to, err_rate = sess.run([losses,total_loss, final_state, grads_vars, train_op, error_rate], feed_dict={input_data:x_current, target_data:z_current, init_state:training_state})
                training_loss += training_loss_
                pc = np.sum(err_rate)/trials_per_batch/(params['trial_length']//params['delta_t'])
                accuracy += pc/num_batches
            
                       
            training_losses.append(training_loss)
            print(i, training_losses[-1], accuracy)
                
                
        # Save the variables to disk.
        save_path = saver.save(sess,save_filename)
        print("Model saved in file: %s" % save_path)
        
        
        # collect output variables 
        num_trials = 40
        if stimulus_type == 'spatial_cat':
            trial_info = stim.generate_spatial_cat_trial(1, num_trials)
        elif stimulus_type == 'motion_match':
            trial_info = stim.generate_motion_working_memory_trial(1, num_trials)
        input_data = np.squeeze(trial_info['neural_input'][0,:,:,:])
        output_data = np.squeeze(trial_info['desired_output'][0,:,:,:])
        rnn_responses, outputs = transform_rnn_data_to_np(input_data,output_data,params, num_trials)
        input_data = []
        output_data = []
        
        print('Creating out dictionary...')
        model_results = {'W_out': W_out.eval(),
                         'b_out': b_out.eval(),
                         'W_rnn' : W_rnn.eval(),
                         'b_rnn' : b_rnn.eval(),
                         'W_in': W_in.eval(),
                         'sample_dir': trial_info['sample_dir'],
                         'test_dir': trial_info['test_dir'],
                         'match': trial_info['match'],
                         'rule': trial_info['rule'],
                         'catch': trial_info['catch'],
                         'rnn_responses': rnn_responses,
                         'outputs': outputs,
                         'training_losses': training_losses,
                         'params': params}
        
        fn = save_dir + 'model_results.pkl'
        output = open(fn, 'wb')
        pickle.dump(model_results, output)
                

    return model_results


def transform_rnn_data_to_np(input_data,output_data,params, num_trials):
    
    trial_length = input_data.shape[1]
    rnn_inputs = np.split(input_data, trial_length, axis=1) 
    rnn_desired_outputs = np.split(output_data, trial_length, axis=1)
    training_state = np.zeros([params['num_RNN_neurons'], num_trials],dtype=np.float32)
    cell_input = training_state
    cell_output = training_state
    #cell_out_hist = training_state
    
    print(trial_length, input_data.shape, cell_input.shape, cell_output.shape)
    count = 0
    rnn_outputs = np.zeros((params['num_RNN_neurons'], params['trial_length'], num_trials),dtype=np.float32)
    for rnn_input in rnn_inputs:
        cell_input, cell_output = rnn_cell(np.float32(np.squeeze(rnn_input)), cell_input, cell_output)
        #rnn_outputs.append(cell_output.eval())
        rnn_outputs[:, count, :] = cell_output.eval()
        count += 1
        print('Converting RNN outputs ', count)
           
    outputs = []
    with tf.variable_scope('output', reuse=True):
        W_out = tf.get_variable('W_out')
        b_out = tf.get_variable('b_out')
        for i in range(trial_length):
            outputs.append((tf.matmul(tf.nn.relu(W_out),np.squeeze(rnn_outputs[:,i,:])) + b_out).eval())
            print('Converting model outputs ', i)
    
    
    return rnn_outputs, outputs

import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/", one_hot=True)


num_tasks_to_run = 10

num_epochs_per_task = 20


# Parameters for the intelligence synapses model.
param_c = 0.1
param_xi = 0.1


minibatch_size = int(64)
learning_rate = 0.001



def weight_variable(input_size, output_size):
	return tf.Variable( tf.random_uniform([input_size,output_size], -1.0/np.sqrt(input_size), 1.0/np.sqrt(input_size)) )



## Network definition -- a simple MLP with 2 hidden layers
x = tf.placeholder(tf.float32, shape=[None, 784])
y_tgt = tf.placeholder(tf.float32, shape=[None, 10])


# Note: the main paper uses a larger network + dropout; both significantly improve the performance of the system.
N1 = 400
N2 = 400

W1 = weight_variable(784,N1)
b1 = tf.Variable(tf.zeros([1,N1]))

W2 = weight_variable(N1,N2)
b2 = tf.Variable(tf.zeros([1,N2]))

Wo = weight_variable(N2,10)
bo = tf.Variable(tf.zeros([1,10]))


h1 = tf.nn.relu( tf.matmul(x,W1) + b1 )
h2 = tf.nn.relu( tf.matmul(h1,W2) + b2 )
y = tf.nn.softmax( tf.matmul(h2,Wo) + bo )


cross_entropy = -tf.reduce_sum( y_tgt*tf.log(y+1e-04) + (1.-y_tgt)*tf.log(1.-y+1e-04) )

optimizer = tf.train.GradientDescentOptimizer(learning_rate)

## Implementation of the intelligent synapses model
variables = [W1, b1, W2, b2, Wo, bo]

small_omega_var = {}
previous_weights_mu_minus_1 = {}
big_omega_var = {}
aux_loss = 0.0

reset_small_omega_ops = []
update_small_omega_ops = []
update_big_omega_ops = []
for var in variables:
	small_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
	previous_weights_mu_minus_1[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
	big_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)

	aux_loss += tf.reduce_sum(tf.multiply( big_omega_var[var.op.name], tf.square(previous_weights_mu_minus_1[var.op.name] - var) ))

	reset_small_omega_ops.append( tf.assign( previous_weights_mu_minus_1[var.op.name], var ) )
	reset_small_omega_ops.append( tf.assign( small_omega_var[var.op.name], small_omega_var[var.op.name]*0.0 ) )

	update_big_omega_ops.append( tf.assign_add( big_omega_var[var.op.name],  tf.div(small_omega_var[var.op.name],(param_xi + tf.square(var-previous_weights_mu_minus_1[var.op.name]) ))   ) )

# After each task is complete, call update_big_omega and reset_small_omega
update_big_omega = tf.group(*update_big_omega_ops)

# Reset_small_omega also makes a backup of the final weights, used as hook in the auxiliary loss
reset_small_omega = tf.group(*reset_small_omega_ops)


# Gradient of the loss function for the current task
gradients = optimizer.compute_gradients(cross_entropy, var_list=variables)

# Gradient of the loss+aux function, in order to both perform training and to compute delta_weights
gradients_with_aux = optimizer.compute_gradients(cross_entropy + param_c*aux_loss, var_list=variables)

for i, (grad,var) in enumerate(gradients_with_aux):
	update_small_omega_ops.append( tf.assign_add( small_omega_var[var.op.name], learning_rate*gradients_with_aux[i][0]*gradients[i][0] ) ) # small_omega -= delta_weight(t)*gradient(t)

update_small_omega = tf.group(*update_small_omega_ops) # 1) update small_omega after each train!



train = optimizer.apply_gradients(gradients_with_aux)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_tgt,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))





## Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())




## Permuted MNIST

# Generate the tasks specifications as a list of random permutations of the input pixels.
task_permutation = []
for task in range(num_tasks_to_run):
	task_permutation.append( np.random.permutation(784) )


avg_performance = []
first_performance = []
last_performance = []
for task in range(num_tasks_to_run):
	print("Training task: ",task+1,"/",num_tasks_to_run)

	for epoch in range(num_epochs_per_task):
		if epoch%5==0:
			print("\t Epoch ",epoch)

		for i in range(int(mnist.train.num_examples/minibatch_size)):
			# Permute batch elements
			batch = mnist.train.next_batch(minibatch_size)
			batch = ( batch[0][:, task_permutation[task]], batch[1] )

			sess.run([train, update_small_omega], feed_dict={x:batch[0], y_tgt:batch[1]})

	sess.run( update_big_omega )
	sess.run( reset_small_omega )

	# Print test set accuracy to each task encountered so far
	avg_accuracy = 0.0
	for test_task in range(task+1):
		test_images = mnist.test.images

		# Permute batch elements
		test_images = test_images[:, task_permutation[test_task]]

		acc = sess.run(accuracy, feed_dict={x:test_images, y_tgt:mnist.test.labels}) * 100.0
		avg_accuracy += acc

		if test_task == 0:
			first_performance.append(acc)
		if test_task == task:
			last_performance.append(acc)

		print("Task: ",test_task," \tAccuracy: ",acc)

	avg_accuracy = avg_accuracy/(task+1)
	print("Avg Perf: ",avg_accuracy)

	avg_performance.append( avg_accuracy )


import matplotlib.pyplot as plt
tasks = range(1,num_tasks_to_run+1)
plt.plot(tasks, first_performance)
plt.plot(tasks, last_performance)
plt.plot(tasks, avg_performance)
plt.legend(["Task 0 (t=i)", "Task i (t=i)", "Avg Task (t=i)"], loc='lower right')
plt.xlabel("Task")
plt.ylabel("Accuracy (%)")
plt.ylim([50, 100])
plt.xticks(tasks)
plt.show()

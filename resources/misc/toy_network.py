import numpy as np
import matplotlib.pyplot as plt


"""
I need to work through a full backpropagation example, once for MSE loss and
another time for omega loss (though that would of course require setting up a
full omega scenario).

To that end, I will simulate a small circuit (2 inputs, 4 hidden, 2 output) to
be a test bed.  I will start with a strictly feed-forward network.

Tasks:

--- XOR ---
 0 0 : 0 1
 1 0 : 1 0
 0 1 : 1 0
 1 1 : 0 1

--- AND ---
 0 0 : 0 1
 1 0 : 0 1
 0 1 : 0 1
 1 1 : 1 0
"""

def sigmoid(x):
    return 1/(1+np.exp(-x))

def print_with_index(x):
    s = x.shape
    i = np.reshape(np.arange(np.product(s)), s)
    if len(s) > 1:
        for n in range(s[1]):
            print(str(x[:,n]).ljust(26), '-', i[:,n])
        print('')
    else:
        print(str(x).ljust(26), '-', i)
        print('')

def tabbed_print(t, s):
    print(t.ljust(26), '|', s)

possible_inputs = [[0,0],[1,0],[0,1],[1,1]]
possible_outputs = [[0,1],[1,0],[1,0],[0,1]]
sample_index = 1

input_weights = 0.1*np.ones([2,4])
output_weights = 0.1*np.ones([4,2])
hidden_biases = 0.2*np.ones([4])
output_biases = 0.2*np.ones([2])

output_weights[2,1] = 0.017

input_activity = np.array(possible_inputs[sample_index])
output_target = np.array(possible_outputs[sample_index])
hidden_activity = np.round(sigmoid(np.matmul(input_activity, input_weights) + hidden_biases), 2)
output_activity = np.round(sigmoid(np.matmul(hidden_activity, output_weights) + output_biases), 2)

print('\n' + '-'*40)
tabbed_print('Input activity', '(2)')
print_with_index(input_activity)
tabbed_print('Input-to-hidden weights', '(2x4)')
print_with_index(input_weights)
tabbed_print('Hidden biases', '(4)')
print_with_index(hidden_biases)
tabbed_print('Hidden activity', '(4)')
print_with_index(hidden_activity)
tabbed_print('Hidden-to-output weights', '(4x2)')
print_with_index(output_weights)
tabbed_print('Output biases', '(2)')
print_with_index(output_biases)
tabbed_print('Output activity', '(2)')
print_with_index(output_activity)
tabbed_print('Output target', '(2)')
print_with_index(output_target)
print('-'*40, '\n')

MSE = np.sum(0.5*(output_target - output_activity)**2)
print('Mean squared error:  {:.4f}'.format(MSE))

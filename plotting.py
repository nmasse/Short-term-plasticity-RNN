"""
2017/06/16 Gregory Grant
"""

import numpy as np
import matplotlib.pyplot as plt
import stimulus
import imp

def import_parameters():
        print('Plotting module:')
        f = open('parameters.py')
        global par
        par = imp.load_source('data', '', f)
        f.close()

import_parameters()
stim = stimulus.Stimulus()
N = par['batch_train_size'] * par['num_batches']

trial_info = stim.generate_trial(N)

def plot_neural_input(trial_info):

    f = plt.figure()
    #ax1 = f.add_subplot(2,1,1)
    #ax2 = f.add_subplot(2,1,2)
    #ax3 = f.add_subplot(4,1,3)
    ax4 = f.add_subplot(1,1,1)
    f.subplots_adjust(hspace=1)
    t = np.arange(0,par['trial_length'],par['dt'])
    #print(np.shape(trial_info['neural_input']))
    #print(np.shape(trial_info['neural_input'][0,:,:]))     # One neuron
    #print(np.shape(trial_info['neural_input'][:,0,:]))     # One time step
    #print(np.shape(trial_info['neural_input'][:,:,0]))     # One trial
    #im1 = ax1.imshow(trial_info['neural_input'][:,:,0], aspect='auto', interpolation='gaussian', cmap='Blues')
    #im2 = ax2.imshow(trial_info['desired_output'][:,:,0], aspect='auto', interpolation='none', cmap='Blues')
    #im3 = ax3.imshow(trial_info['desired_output'][:,:,0], aspect='auto', interpolation='none', cmap='Blues')
    im4 = ax4.imshow(np.transpose(trial_info['train_mask'][:,:]), aspect='auto', interpolation='none', cmap='Blues')

    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(im4, cax=cbar_ax)
    """
    ax1.set_yticks([0, 9, 18, 27])
    ax1.set_yticklabels([0,9,18,27])

    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["no response", "match", "nonmatch"])

    ax1.set_xlabel('Time step (%s ms/step)' % par['dt'])
    ax1.set_ylabel('Neurons')
    ax1.set_title('Motion Input')

    ax2.set_xlabel('Time step (%s ms/step)' % par['dt'])
    ax2.set_title('Motion Output')
    """
    #ax3.set_xlabel('Time step (%s ms/step)' % par['dt'])
    #ax3.set_ylabel('Neurons')
    #ax3.set_title('Motion Output (Non-match)')

    ax4.set_xlabel('Time step (%s ms/step)' % par['dt'])
    ax4.set_ylabel('Trial number')
    ax4.set_title('Train Mask')


    plt.show()



plot_neural_input(trial_info)

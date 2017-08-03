import analysis
import numpy as np
import matplotlib.pyplot as plt
import itertools

import os
os.system('cls' if os.name == 'nt' else 'clear')

### Informal plotting code

data    = analysis.load_data_dir('./savedir/model_multitask_h250_df0009_D17-07-25_T13-14-32')
#weights = model_saver.json_load('./savedir/model_multitask_h250_df0009_D17-07-24_T17-00-57/model_results.json')['weights']['w_rnn_soma']
weight  = model_saver.json_load('./savedir/model_multitask_h100nd_D17-08-01_T10-08-20_m1_without_time/model_results.json')['weights']['w_rnn_soma']
weight = np.maximum(0, weight)
#weight = np.matmul(weight, par['EI_matrix'])
#weight = np.matmul(weight, par['m2_transfer'])
weight = np.sqrt(weight)

fig, axes = plt.subplots(nrows=1, ncols=2)
im1 = axes[0].imshow(weight, cmap='magma')
im2 = axes[1].imshow(np.abs(weight-np.transpose(weight)), cmap='magma')

plt.show()

quit()ss

anova   = data['anova']
roc     = data['roc']
tuning  = data['tuning']

anova_vars  = ['state_hist', 'dend_hist', 'dend_exc_hist', 'dend_inh_hist']
roc_vars    = ['state_hist', 'dend_hist', 'dend_exc_hist', 'dend_inh_hist']
tuning_vars = ['state_hist', 'dend_hist', 'dend_exc_hist', 'dend_inh_hist']

anova_groups    = ['_attn_pval', '_attn_fval', '_no_attn_pval', '_no_attn_fval']
roc_groups      = ['_attn', '_no_attn']
tuning_groups   = ['_attn', '_no_attn']

def view_all():
    i = 0
    n = 186
    d1 = 2
    d2 = 11
    rf = 0
    r = 0
    t = 0
    sig = 0

    # ANOVA = iter num X neuron X (dendrite num) X RF X rule X time
    # ROC = iter num X neuron X (dendrite num) X RF X rule X category (up/down or left/right) time
    # TUNING = iter num X neuron X (dendrite num) X RF X rule X time X num unique samples

    neurons = []
    for n in range(250):
        print()
        neurons.append(np.squeeze(data['tuning']['dend_hist_attn'][:,n,:,:,:,:,:]))
        # Results: i, d, rf, r, t, sig

    all_neurons = np.concatenate(neurons, axis=1)

    f, axarr = plt.subplots(4, 4, sharex=True, sharey=True)
    for r, t, rf in itertools.product(range(4), range(2), range(1)):
        axarr[r, rf*2+t].imshow(all_neurons[i,:,t,:], aspect='auto', interpolation='none')
        axarr[r, rf*2+t].set_title('rf = {}, r = {}, t ={}'.format(rf, r, t), fontsize=6)

    axarr[3,0].set_xlabel("Signal Index          Both")
    axarr[3,0].set_ylabel("Dendrites * Neurons")
    plt.show()

def view_roc():
    # iter num X neuron X (dendrite num) X RF X rule X category (up/down or left/right) X time
    i = 4
    d = 0
    rf = 0
    r = 0
    c = 0
    t = 0

    neurons = []
    for n in range(50):
        neurons.append(np.squeeze(data['roc']['dend_inh_hist_attn'][:,n,:,:,:,:,:]))
        # Results: i, d, rf, r, c, t

    all_neurons = np.concatenate(neurons, axis=1)

    f, axarr = plt.subplots(6, 4, sharex=True, sharey=True)
    for r in range(2):
        for t in range(3):
            for rf in range(4):
                print(np.shape(all_neurons[i,:,rf,r,:,t]))
                axarr[r*3+t, rf].imshow(all_neurons[i,:,rf,r,:,t], aspect='auto', interpolation='none')
                axarr[r*3+t, rf].set_title('rf = {}, r = {}, t = {}'.format(rf, r, t), fontsize=6)

    axarr[5,0].set_xlabel("Category Index")
    axarr[5,0].set_ylabel("Dendrites * Neurons")
    plt.show()

def view_dendrite():
    # ROC = iter num X neuron X (dendrite num) X RF X rule X category (up/down or left/right) time
    i = 2
    rf = 0
    r = 0
    c = 0
    t = 1

    for d in range(4):
        # for rf in range(4):
            # for r in range(2):
                # for c in range(2):
            d1 = data['roc']['dend_exc_hist_attn'][i, :, d, d, 0, 0, t]
            d2 = data['roc']['dend_exc_hist_attn'][i, :, d+4, d, 1, 1, t]

            plt.scatter(d1, d2)
            plt.title("15dendrites_"+str(d)+"_"+str(d+1))
            plt.savefig('./analysis/15dendrites'+str(d)+"_"+str(d+1)+"_rf_"+str(rf)+"_r_"+str(r)+"_c_"+str(c)+'.png')
            plt.clf()




#view_roc()
view_all()
#view_dendrite()

import analysis
import numpy as np
import matplotlib.pyplot as plt

import os
os.system('cls' if os.name == 'nt' else 'clear')

### Informal plotting code

data    = analysis.load_data_dir('./savedir/model_att_h50_df0009_D17-07-17_T15-38-03')

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
    i = 7
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
    for n in range(50):
        neurons.append(np.squeeze(data['tuning']['dend_hist_attn'][:,n,:,:,:,:,:]))
        # Results: i, d, rf, r, t, sig

    all_neurons = np.concatenate(neurons, axis=1)

    f, axarr = plt.subplots(4, 4, sharex=True, sharey=True)
    for r in range(2):
        for t in range(2):
            for rf in range(4):
                axarr[r*2+t, rf].imshow(all_neurons[i,:,rf,r,t,:], aspect='auto', interpolation='none')
                axarr[r*2+t, rf].set_title('rf = {}, r = {}, t ={}'.format(rf, r, t), fontsize=6)

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

#view_roc()
view_all()

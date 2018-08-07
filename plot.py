import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_pev_cross_time(x, num_pulses, cue):
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15,10))
    i = 0
    for ax in axes.flat:
        im = ax.imshow(x['synaptic_pev'][:,i,:],aspect='auto')
        i += 1
    cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
    plt.colorbar(im, cax=cax, **kw)
    plt.savefig("./savedir/synaptic_pev_cross_time_"+str(num_pulses)+"_pulses_"+cue+".png")

def plot_pev_after_stim(x, num_pulses, cue, time_lapse):
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10,12))
    i = 0
    for ax in axes.flat:
        im = ax.imshow(x['synaptic_pev'][:,:,x['timeline'][2*i+1]+time_lapse],aspect='auto')
        i += 1
    cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
    plt.colorbar(im, cax=cax, **kw)
    plt.savefig("./savedir/synaptic_pev_after_stim_"+str(num_pulses)+"_pulses_"+cue+".png")

num_pulses = [8]
cue = ['cue_off', 'cue_on']

for num_pulses in num_pulses:
    for cue in cue:
        x = pickle.load(open('./savedir/analysis_chunking_'+str(num_pulses)+"_"+cue+".pkl", 'rb'))
        plot_pev_cross_time(x, num_pulses, cue)
        plot_pev_after_stim(x, num_pulses, cue, 10)

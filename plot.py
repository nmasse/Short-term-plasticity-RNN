import analysis_new
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import os

def plot(filename,neuronal,synaptic,stability,syn_stability):
    plt.figure()
    plt.plot(np.mean(neuronal[0,0],axis=0), 'r')
    plt.plot(np.mean(synaptic[0,0],axis=0), 'g')
    plt.plot([0,375],[1/8,1/8],"k--")
    plt.savefig('./savedir/plots/'+filename[:-4]+'.png')
    plt.close()

    plt.figure()
    plt.imshow(np.mean(stability[0,0],axis=0))
    plt.savefig('./savedir/plots/'+filename[:-4]+'_heatmap_neuronal.png')
    plt.close()

    plt.figure()
    plt.imshow(np.mean(syn_stability[0,0],axis=0))
    plt.savefig('./savedir/plots/'+filename[:-4]+'_heatmap_synaptic.png')
    plt.close()


if __name__ == "__main__":
    files = os.listdir('./savedir/output')
    files.remove('.DS_Store')
    files2 = np.array(os.listdir('./savedir/test'))
    files = np.setdiff1d(np.array(files), files2)
    for filename in files:
        analysis_new.analyze_model_from_file('./savedir/output/'+filename)
        f = open(('./savedir/test/'+filename), 'rb')
        results = pickle.load(f)
        plot(filename, results['neuronal_sample_decoding'],results['synaptic_sample_decoding'],results['neuronal_sample_decoding_stability'],results['synaptic_sample_decoding_stability'])

import analysis_new
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import os

def plot(filename,neuronal,synaptic):
    print(neuronal.shape)
    print(synaptic.shape)
    quit()
    plt.plot(np.mean(neuronal[0,0],axis=0), 'r')
    plt.plot(np.mean(synaptic[0,0],axis=0), 'g')
    plt.plot([0,375],[1/8,1/8],"k--")
    plt.savefig('./savedir/plots/'+filename[:-4]+'.png')
    plt.show()
    plt.close()

if __name__ == "__main__":
    f = open('./savedir/test/delay_type_2_DMSvar_1.pkl','rb')
    results = pickle.load(f)
    plot('filename', results['neuronal_sample_decoding'],results['synaptic_sample_decoding'])
    # files = os.listdir('./savedir/output')
    # files.remove('.DS_Store')
    # files2 = np.array(os.listdir('./savedir/test'))
    # files = np.setdiff1d(np.array(files), files2)
    # for filename in files:
    #     analysis_new.analyze_model_from_file('./savedir/output/'+filename)
    #     f = open(('./savedir/test/'+filename), 'rb')
    #     results = pickle.load(f)
    #     plot(filename, results['neuronal_sample_decoding'],results['synaptic_sample_decoding'])

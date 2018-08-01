import analysis_new
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import os

def plot(neuronal,synaptic,i):
    print(synaptic.shape)
    plt.plot(np.mean(neuronal[0,0],axis=0), 'r')
    plt.plot(np.mean(synaptic[0,0],axis=0), 'g')
    plt.plot([0,375],[1/8,1/8],"k--")
    plt.savefig('./savedir/'+folder+'_analysis_'+str(i)+'.png')
    plt.show()
    plt.close()

if __name__ == "__main__":
    folders = ['./savedir/delay_type_0','./savedir/delay_type_1','./savedir/delay_type_2']
    for folder in folders:
        #filename = ["./savedir/DMSvar_0_rule_0.pkl", "./savedir/DMSvar_0_rule_1.pkl", "./savedir/DMSvar_0_rule_2.pkl"]
        filename = os.listdir(folder)
        for i in range(len(filename)):
            analysis_new.analyze_model_from_file(filename[i])
            f = open((filename[i][:-4]+'_test.pkl'), 'rb')
            results = pickle.load(f)
            plot(results['neuronal_sample_decoding'],results['synaptic_sample_decoding'],i)

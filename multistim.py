import numpy as np
import matplotlib.pyplot as plt
from parameters import *


# I'm gonna work with 36 units for now, for convenient 10-degree increments
# Note: fixation + modality
par['n_input'] = 1 + 36*2
par['n_output'] = 1 + 36
par['batch_train_size'] = 4
par['num_time_steps'] = 500

class Stimulus:

    def __init__(self):

        self.input_shape = [par['n_input'], par['num_time_steps'], par['batch_train_size']]
        self.output_shape = [par['n_output'], par['num_time_steps'], par['batch_train_size']]

        self.modality_size = 36
        self.pref_theta = np.linspace(0,2*np.pi,self.modality_size)


    def circ_tuning(self, theta):
        von_mises = np.exp(2*np.cos(theta-self.pref_theta[:,np.newaxis]))
        return 0.8*von_mises/np.max(von_mises)


    def task_go(self, variant):
        # Single stimulus in modality 1 or 2
        # Response in direction of stimulus
        # Stimulus appears before fixation cue turns off
        # Response occurs when fixation turns off

        # Task parameters
        if variant == 'go':
            stim_onset = np.random.randint(500, 1500, par['batch_train_size'])//par['dt']
            stim_off = -1
            fixation_end = 1500//par['dt']
            resp_onset = fixation_end
        elif variant == 'rt_go':
            stim_onset = np.random.randint(500, 2500, par['batch_train_size'])//par['dt']
            stim_off = -1
            fixation_end = -1
            resp_onset = stim_onset
        elif variant == 'dly_go':
            stim_onset = 500//par['dt']
            stim_off = 750//par['dt']
            fixation_end = (1500 + np.random.choice([200,400,800,1600], size=par['batch_train_size']))//par['dt']
            resp_onset = fixation_end
        else:
            raise Exception('Bad task variant.')

        # Update parameter compatibility
        if type(stim_onset) != np.ndarray:
            stim_onset = np.array([stim_onset]*par['batch_train_size'])

        if type(fixation_end) != np.ndarray:
            fixation_end = np.array([fixation_end]*par['batch_train_size'])

        if type(resp_onset) != np.ndarray:
            resp_onset = np.array([resp_onset]*par['batch_train_size'])

        # Setting up arrays
        modality_choice = np.random.choice(np.array([0,1], dtype=np.int8), par['batch_train_size'])
        modalities = np.zeros([2, self.modality_size, par['num_time_steps']+1, par['batch_train_size']])
        response   = np.zeros([self.modality_size, par['num_time_steps']+1, par['batch_train_size']]) + 0.05
        fixation   = np.zeros([1, par['num_time_steps']+1, par['batch_train_size']]) + 0.05
        mask       = np.ones([self.modality_size+1, par['num_time_steps'], par['batch_train_size']])

        # Getting tunings
        stim_dir = 2*np.pi*np.random.rand(1, par['batch_train_size'])
        stim = self.circ_tuning(stim_dir)
        resp = self.circ_tuning(stim_dir) + 0.05

        # Setting the mask array
        mask[:,:100//par['dt'],:] = 0

        # Applying tunings to modalities and response arrays
        for b in range(par['batch_train_size']):
            modalities[modality_choice[b],:,stim_onset[b]:stim_off,b] = stim[:,b,np.newaxis]
            response[:,resp_onset[b]:,b] = resp[:,b,np.newaxis]
            mask[:,resp_onset[b]:resp_onset[b]+100//par['dt'],b] = 0
            fixation[:,:fixation_end[b],b] = 0.85

        # Tweak the fixation array
        stim_fix = np.round(fixation)
        resp_fix = fixation

        # Merge activies and fixations into single vectors
        stimulus = np.concatenate([modalities[0], modalities[1], stim_fix], axis=0)
        response = np.concatenate([response, resp_fix], axis=0)

        return stimulus[:,:-1,:], response[:,:-1,:], mask



st = Stimulus()
s, r, m = st.task_go('rt_go')

print(s.shape)
print(r.shape)
print(m.shape)

fig, axarr = plt.subplots(4, 3)
for i in range(4):
    axarr[i,0].imshow(s[:,:,i])
    axarr[i,1].imshow(r[:,:,i])
    axarr[i,2].imshow(m[:,:,i])

plt.show()

import numpy as np
import os, sys

sys.path.append(os.path.abspath(os.path.join('..', '..')))
import model_saver as ms


available_permutations = 100

# Trial generator permutation choices
permutation_template = np.arange(784)
p = [[permutation_template]]
for n in range(1,available_permutations):
    p.append([np.random.permutation(permutation_template)])

ms.json_save(p, savedir='./permutations.json')

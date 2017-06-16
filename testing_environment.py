import numpy as np

events = [[4, 'mask', False, 0], [12, 'mask', True, 0]]

def build_mask(steps, events, start_mask):
    """Creates a [steps]-long mask based on event list
    Creates a [steps]-long mask based on event list"""

    # Clarifies mask and sets up initial value
    mask = np.zeros(steps)
    value = start_mask

    # Starting with the 0th position, edits mask
    n = 0
    for i in range(len(events)):
        mask[n:events[i][0]] = value
        n = events[i][0]
        value = events[i][2]

    # Edits the last portion of the mask
    mask[n:steps] = value

    return mask



fixation = [0,0,0,0,1,0,0,0,0]
stimuli =  [[0,1,0,0,0,1,0,0,0], [0,1,0,1,0,0,0,0,0], \
                    [0,0,0,1,0,0,0,1,0], [0,0,0,0,0,1,0,1,0]]
tests =     [[0,0,1,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], \
                    [0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,0,1]]

batch_train_size = 5
setup = np.random.randint(0,4,size=(2,batch_train_size))

stimulus = []
test = []
output = setup[0] == setup[1]

for i in range(batch_train_size):
    print(i)
    stimulus.append(stimuli[setup[0][i]])
    test.append(tests[setup[1][i]])

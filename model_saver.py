"""
2017/06/27 Gregory Grant
"""

import numpy as np
import json
import copy
import base64
import re

global multi_types
global singlet_types
global other_types

###############################################################################
### This saver assumes that the toplevel structure is a list or dictionary. ###
###############################################################################

# JSON will only handle dictionaries, lists, strings, numbers, booleans, and None/null.
multi_types = [type({}), type([])]
singlet_types = [type(""), type(1), type(True), type(None)]
# Other types that we have written conversions for go here.
other_types = [type(np.array([0])), type(np.float32(0.)), type(np.float64(0.)), \
                type(range(0,1)), type(np.int8(0.)), type(np.int16(0.)), type(np.int32(0.)), \
                type(np.int64(0.))]

def json_save(x, savedir="save.json", toplevel=True):
    x = copy.deepcopy(x)

    def item(x, i):
        s = type(x[i])
        if s in multi_types:
            x[i] = json_save(x[i], toplevel=False)
        elif s in singlet_types:
            pass
        elif s in other_types:
            if s == type(np.array([0])):
                if not x[i].flags['C_CONTIGUOUS']:
                    x[i] = np.ascontiguousarray(x[i])
                x[i] = ["ndarray", str(base64.b64encode(x[i])), str(x[i].shape), str(x[i].dtype)]
            elif (s == type(np.float32(0.)) or s == type(np.float64(0.))):
                x[i] = np.asscalar(x[i])
            elif (s == type(np.int8(0.)) or s == type(np.int16(0.)) \
                                         or s == type(np.int32(0.)) \
                                         or s == type(np.int64(0.))):
                x[i] = np.asscalar(x[i])
            elif (s == type(range(0,1))):
                x[i] = ["range", x[i].start, x[i].stop, x[i].step]
        return x[i]

    if type(x) == multi_types[0]:
        for i in x:
            x[i] = item(x, i)
    elif type(x) == multi_types[1]:
        for i in range(len(x)):
            x[i] = item(x, i)
    else:
        pass

    if toplevel == True:
        with open(savedir, 'w') as f:
            json.dump(x, f)
    else:
        return x


def json_load(savedir="save.json", toplevel=True, a=None):

    if toplevel == True:
        with open(savedir, 'r') as f:
            x = json.load(f)
    else:
        x = copy.deepcopy(a)

    def item(x, i):
        s = type(x[i])
        if s == multi_types[0]:
            x[i] = json_load(toplevel=False, a=x[i])
        elif s == multi_types[1]:
            if len(x[i]) > 0:
                if x[i][0] == "ndarray":
                    b = base64.b64decode(x[i][1][1:])
                    t = x[i][3]
                    sh = x[i][2]
                    st = np.fromstring(b, dtype=t)
                    sh = re.findall('\d+', sh)
                    for j in range(len(sh)):
                        sh[j] = int(sh[j])
                    x[i] = np.reshape(st, sh)
                elif x[i][0] == "range":
                    x[i] = range(*x[i][1:3])
                else:
                    x[i] = json_load(toplevel=False, a=x[i])
            else:
                json_load(toplevel=False, a=x[i])
        elif s in singlet_types:
            pass
        return x[i]

    if type(x) == multi_types[0]:
        for i in list(x):
            x[i] = item(x, i)
    elif type(x) == multi_types[1]:
        for i in range(len(x)):
            x[i] = item(x, i)
    else:
        pass

    return x

# This script contains functions for updating the preference
# parameter distributins

import pandas as pd

def pref_dist(cdat):
    """takes a matrix of preference parameter values"""
    """returns lognormal distribution parameters"""
    
    # First the binomial probabilities
    z = []
    for k in range(31):
        pname = 'p' + str(k + 1)
        frac = sum(cdat[pname] == 0) / float(len(cdat))
        z.append(float(1) - frac)
    
    return z, 1, 1

    




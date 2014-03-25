# This script contains functions for updating the preference
# parameter distributins

import pandas as pd
import math

def bin_probs(cdat):
    '''Calculates binomial probabilities'''

    # Z is the zero probability
    z = []
    for k in range(29):
        pname = 'p' + str(k + 1)
        frac = sum(cdat[pname] == 0) / float(len(cdat))
        z.append(frac)

    return pd.Series(z)

def ln_probs(cdat):
    '''Calculates lognormal dist params'''

    # This whole thing is just maximum likelihood, from wikipedia
    mu, sig2 = [], []
    for k in range(29):
        pname = 'p' + str(k + 1) 
        logp = cdat.loc[cdat[pname] > 0, pname].apply(math.log)
        logp_sum = logp.sum()
        logp_diffs = logp - logp_sum / float(len(logp))
        logp_diffs = logp_diffs.apply(lambda x: x**2)
        logp_sqr_sum = logp_diffs.sum()
        mu.append(logp_sum / float(len(logp)))
        sig2.append(logp_sqr_sum / float(len(logp)))

    return pd.Series(mu), pd.Series(sig2) 

def pref_dist(cdat):
    """takes a matrix of preference parameter values"""
    """returns lognormal distribution parameters"""
    
    # Calculate binomial probabilities
    z = bin_probs(cdat)

    # Calculate lognormal parameters
    mu, sig2 = ln_probs(cdat)
    
    # Return exp of mean, since scipy uses that
    return [z, mu.apply(math.exp),sig2.apply(math.sqrt)]

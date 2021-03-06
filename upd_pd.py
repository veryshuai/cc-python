# This script contains functions for updating the preference
# parameter distributins

import pandas as pd
import math
import numpy as np

def bin_probs(cdat, gn):
    '''Calculates binomial probabilities'''

    # Z is the zero probability
    z = []
    for k in range(gn):
        pname = 'p' + str(k + 1)
        is_zero = cdat[pname] == 0
        frac = sum(is_zero) / float(len(cdat))
        z.append(frac)

    return pd.Series(z)

def get_const(w, logp, N, w_bar, w2_bar):
    '''creates coefficients for lognormal dist update'''

    product = w * logp
    b_t1 = - (np.sum(product) / N) / w2_bar
    b_t2 = w_bar / w2_bar
    b_u1 = np.sum(logp) / N
    b_u2 = w_bar

    return b_t1, b_t2, b_u1, b_u2

def get_dist_params(cdat, w, k, N, w_bar, w2_bar, dparams, calc_t):
    '''calculate the new logn parameters'''

    # old t (initial round comes in as a pandas dataframe)
    try:
        t = dparams[3]
    except:
        t = dparams['3']

    # mu and t hat
    pname = 'p' + str(k + 1) 
    logp = np.log(cdat.loc[cdat[pname] > 0, pname])
    if calc_t:
        b_t1, b_t2, b_u1, b_u2 = get_const(w, logp, N, w_bar, w2_bar)
        new_t = b_t1 + b_t2 * (b_u1 + b_t1 * b_u2) / (1 - b_t2 * b_u2)
        new_mu = (b_u1 + b_t1 * b_u2) / (1 - b_t2 * b_u2)
    else:
        new_mu = logp.sum() / N + w_bar * t[k]
        new_t = t[k]

    # sigma2
    diff = logp + new_t * w - new_mu
    diff2 = diff ** 2
    new_sig2 = np.sum(diff2) / N

    return new_t, new_mu, new_sig2

def ln_probs(cdat, gn, dparams, calc_t):
    '''Calculates max lik lognormal dist params'''

    mu, sig2, t = [], [], []
    pre_w = cdat['exptot'] / float(1000)
    w = np.log(pre_w)
    N = float(len(w))
    w_bar = np.sum(w) / N
    w_bar2 = w ** 2
    w2_bar = np.sum(w_bar2) / N
    for k in range(gn):
        new_t, new_mu, new_sig2 = get_dist_params(cdat, w, k,
                                           N, w_bar, w2_bar, 
                                           dparams, calc_t)
        t.append(new_t)
        mu.append(new_mu)
        sig2.append(new_sig2)

    return pd.Series(mu), pd.Series(sig2), pd.Series(t)

def pref_dist(cdat, gn, dparams, calc_t):
    """takes a matrix of preference parameter values"""
    """returns lognormal distribution parameters"""
    
    # Calculate binomial probabilities
    z = bin_probs(cdat, gn)

    # Calculate lognormal parameters
    mu, sig2, t = ln_probs(cdat, gn, dparams, calc_t)

    return [z, mu, sig2, t]

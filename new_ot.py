# This file contains functions related to updating observation types based on the data, the current alpha, and the type distribution parameters.

import pandas as pd
import calc_gv
from scipy import stats
import numpy as np
import math
from multiprocessing import Pool
import time

def logpdf(arg, t_adj, u, sig2):
    '''returns the log of the pdf for each observation''' 

    diff = arg + t_adj - u 
    diff2 = diff.values ** 2
    res = -arg - t_adj - math.log(math.sqrt(2))\
            - math.log(math.sqrt(sig2)) - diff2 / (2 * sig2)
    res[diff2 == np.inf] = 0
    return(res)
    
def log_non_zero(x):
    '''returns log if argument is not zero, and zero otherwise'''

    if x > 0:
        res = math.log(x)
    else:
        res = 0

    return res

def olik(cdat, dparams, w):
    '''calculates likelihood given observation type'''

    #unpack distribution parameters
    [z, mu, sig2, t] = dparams

    #add likelihoods together
    lik = 0 
    logp = np.log(cdat.filter(regex='^p[0-9]'))
    zeros = cdat.filter(regex='^p[0-9]') == 0
    for k in range(1, 29):
        #this is the likelihood!
        t_adj = w * t[k]
        arg = logp['p' + str(k + 1)]
        addme = math.log(1 - z[k])\
                + logpdf(arg, t_adj, mu[k], sig2[k])

        if k != 0: #avoid math log(0) errors
            lik = lik + addme + zeros['p' + str(k + 1)] * math.log(z[k])
        else:
            lik = addme

    return lik

def obs_type_lik_loop(data_input):
    '''loops through each observation type to get likelihoods'''

    (k, cdat, dparams, alp, r, lw, w) = data_input
    ot_try = pd.Series([k + 1] * len(cdat))
    cdat['ot'] = ot_try # new observation type
    cdat = calc_gv.get_pp(cdat, alp, r, lw) # update parameters
    out = olik(cdat, dparams, w) + cdat['vin' + str(k + 1)] #add vindex

    return out

def ot_step(cdat, dparams, alp, r, lw):
    ''' updates observation types based on dist params'''

    #get wealth term for dividing 
    pre_w = (cdat['exptot'] - 1000) / float(1000)
    w = pre_w.apply(math.log)

    # Create tuple arguments for multiprocessing
    data_input = []
    for k in range(29):
        data_input.append((k, cdat, dparams, alp, r, lw, w))
        
    # Call multiprocessing
    #TEST
    #mp_out = [];
    #for k in range(29):
    #    print(k)
    #    base = time.time()
    #    mp_out.append(obs_type_lik_loop(data_input[k]))
    #    print(time.time() - base)
    #import pdb; pdb.set_trace()
    pool = Pool(processes=3) # process per core
    mp_out = pool.map(obs_type_lik_loop, data_input) 
    pool.close() #proper closing
    pool.join() #proper closing

    # Read results into cdat
    for k in range(29):
        cdat['lik' + str(k + 1)] = mp_out[k]

    #Find new observation types
    cdat_liks = cdat.filter(regex = '^lik') #only liks
    cdat_liks.columns = range(29) #numeric column names
    cdat['ot'] = cdat_liks.idxmax(axis = 1) + 1

    return cdat


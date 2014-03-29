# This file contains functions related to updating observation types based on the data, the current alpha, and the type distribution parameters.

import pandas as pd
import calc_gv
from scipy import stats
import numpy as np
import math
from multiprocessing import Pool

def olik(cdat, dparams):
    '''calculates likelihood given observation type'''

    # NO DEMOGRAPHICS YET!!!

    #unpack distribution parameters
    [z, emu, shape] = dparams

    #add likelihoods together
    lik = 0 
    for k in range(29):
        #this is the likelihood!
        addme = math.log(1 - z[k])\
                + np.nan_to_num(stats.lognorm
                        .logpdf(list(cdat['p' + str(k + 1)]),
                            shape[k], loc=0, scale=emu[k]))
        if k != 0: #avoid math log(0) errors
            lik = lik + addme\
                + (cdat['p' + str(k + 1)] == 0) * (math.log(z[k]))
        else:
            lik = addme
        
    return lik

def obs_type_lik_loop(data_input):
    '''loops through each observation type to get likelihoods'''

    (k, cdat, dparams, alp, r, lw) = data_input
    ot_try = pd.Series([k + 1] * len(cdat))
    cdat['ot'] = ot_try # new observation type
    cdat = calc_gv.get_pp(cdat, alp, r, lw) # update parameters
    out = olik(cdat, dparams) + cdat['vin' + str(k + 1)] #add vindex

    return out

def ot_step(cdat, dparams, alp, r, lw):
    ''' updates observation types based on dist params'''

    # Create tuple arguments for multiprocessing
    data_input = []
    for k in range(1,29):
        data_input.append((k, cdat, dparams, alp, r, lw))
        
    # Call multiprocessing
    pool = Pool(processes=3) # process per core
    mp_out = pool.map(obs_type_lik_loop, data_input) 
    pool.close() #proper closing
    pool.join() #proper closing

    # Read results into cdat
    for k in range(1,29):
        cdat['lik' + str(k + 1)] = mp_out[k - 1]

    #Find new observation types
    cdat_liks = cdat.filter(regex = '^lik') #only liks
    cdat_liks.columns = range(1,29) #numeric column names
    cdat['ot'] = cdat_liks.idxmax(axis = 1) + 1

    return cdat


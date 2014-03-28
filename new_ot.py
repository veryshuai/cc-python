# This file contains functions related to updating observation types based on the data, the current alpha, and the type distribution parameters.

import pandas as pd
import calc_gv
from scipy import stats
import numpy as np
import math

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

def ot_step(cdat, dparams, alp, r, lw):
    ''' updates observation types based on dist params'''

    #Running observation type (note first group not allowed)
    for k in range(1,29):
        ot_try = pd.Series([k + 1] * len(cdat))
        cdat['ot'] = ot_try # new observation type
        cdat = calc_gv.get_pp(cdat, alp, r, lw) # update parameters
        cdat['lik' + str(k + 1)] = olik(cdat, dparams) \
                                    + cdat['vin' + str(k + 1)] #add vindex

    #Find new observation types
    cdat_liks = cdat.filter(regex = '^lik') #only liks
    cdat_liks.columns = range(1,29) #numeric column names
    cdat['ot'] = cdat_liks.idxmax(axis = 1) + 1

    return cdat


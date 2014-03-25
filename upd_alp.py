#This file contains function for updating the alpha parameter in the
#conspicuous consumption model

import pandas as pd
from scipy.optimize import minimize_scalar
import calc_gv
import math
from scipy import stats
import numpy as np
import new_ot

def alp_lik(alp, cdat, dparams, r, lw):
    '''objective function for minimizing alp'''

    #update preference parameters
    cdat = calc_gv.get_pp(cdat, alp, r, lw)

    #get likelihood
    vin = []
    lik = new_ot.olik(cdat, vin, dparams)

    #eliminate the really low guys
    lik = lik.apply(lambda x: max(-1e15,x))
    lik_sum = -lik.sum()

    return lik_sum

def alp_step(cdat, alp, r, lw, dparams):
    '''calls routine for minimizing alp'''
    
    #Call minimize
    result = minimize_scalar(alp_lik, args = (cdat, dparams, r, lw), bounds=[0,0.5], method = 'bounded')
    alp = result['x']

    return alp
    



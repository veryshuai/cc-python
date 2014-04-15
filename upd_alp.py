#This file contains function for updating the alpha parameter in the
#conspicuous consumption model

import pandas as pd
from scipy import optimize
import calc_gv
import math
from scipy import stats
import numpy as np
import new_ot
import upd_pd

def alp_lik(alp_trans, cdat, dparams, r, lw, gn):
    '''objective function for minimizing alp'''

    #get w
    pre_w = (cdat['exptot'] - 1000) / float(1000)
    w = np.log(pre_w)

    #untransform alp
    alp = untrans(alp_trans)

    #update preference parameters
    cdat = calc_gv.get_pp(cdat, alp, r, lw, gn)

    #Update observation types
    cdat = new_ot.ot_step(cdat, dparams, alp, r, lw, gn)

    #update preference parameters
    cdat = calc_gv.get_pp(cdat, alp, r, lw, gn)

    #get likelihood
    lik = new_ot.olik(cdat, dparams, w, gn)
    lik_sum = -lik.sum()

    print(alp)
    print('{0:.6f}'.format(lik_sum))

    return lik_sum

def untrans(alp_trans,forward=False):
    '''undos or does an alp transform'''
    
    #if forward:
    #    res = stats.norm.ppf(alp_trans)
    #else:
    #    res = stats.norm.cdf(alp_trans) * 1.0
    res = alp_trans

    return res

def alp_step(cdat, alp, r, lw, dparams, gn):
    '''calls routine for minimizing alp'''

    #Call minimize
    # result = optimize.minimize(alp_lik, [untrans(alp, True)], bounds=[(0, 1)], method = 'TNC', args = (cdat, dparams, r, lw))
    result = optimize.minimize_scalar(alp_lik, bounds=[0, 0.5], method = 'bounded', args = (cdat, dparams, r, lw, gn))
    alp = untrans(result['x'])

    return alp, result['fun']
    



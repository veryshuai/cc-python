# This file contains functions for numerically solving for gamma, given fraction of spending on the visible good r and ratio of own wealth to minimum wealth.

import pandas as pd
from scipy import optimize 
import run_est
import numpy as np
import math

def err(g, w, r, a):
    '''calculates the error in function'''

    #read parameters
    ft = (1 - r) * (1 + g / a)
    st = (1 - a) * r / a
    tt = (r * (1 + 1 / g)) ** (-g / a) * w ** (1 + g / a)
    res = ft - st - tt

    #warn if nan encountered
    if math.isnan(res):
        print('Warning: NaN encountered in visible parameter estimation (solve_gam, err)')
        res = 0

    return res

def vis_param(tup, a):
    '''solves for visible parameter ratio'''

    #call solve
    try: 
        sol = optimize.brentq(err,1e-15,0.25,args=tup + (a,))
    except Exception as e: 
        #print(e)
        #print('optimization problem!')
        sol = 1e-15 #return almost zero (to flag impossiblity)

    return sol

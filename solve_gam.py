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

    #stop if nan encountered
    if math.isnan(res):
        res = 0

    return res

def vis_param(tup, a):
    '''solves for visible parameter ratio'''

    #call solve
    sol = optimize.newton(err,1e-4,args=tup + (a,), maxiter=int(1e6))

    return sol

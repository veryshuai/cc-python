# This file contains functions for numerically solving for gamma, given fraction of spending on the visible good r and ratio of own wealth to minimum wealth.

import pandas as pd
from scipy import optimize 
import run_est
import numpy as np
import math

def err(g, w, r, a):
    '''calculates the error in function'''

    #exp
    if g < 700: #avoid overflows
        g = math.exp(g)
    else:
        g = 10
    if g > 10: #avoid overflows (we never see solutions this large)
        g = float(10)

    #read parameters
    if g > 1e-6: #stop if things get very small
        ft = (1 - r) * (1 + g / a)
        st = (1 - a) * r / a
        tt = (r * (1 + 1 / g)) ** (-g / a) * w ** (1 + g / a)
        res = ft - st - tt
    else:
        res = 0

    #stop if nan encountered
    if math.isnan(res):
        print('Warning: NaN encountered in visible parameter estimation (solve_gam, err)')
        res = 0

    return res

def vis_param(tup, a):
    '''solves for visible parameter ratio'''

    #call solve
    try: 
        sol = optimize.newton(err,0,args=tup + (a,), maxiter=int(1e5), tol=1e-6)
    except Exception as e: 
        print(e)
        sol = -100 #return a zero

    #routine will always return something positive, let very small numbers be zero
    if sol > -27:
        op = math.exp(sol)
    else:
        op = 0

    return op

# This file holds functions related to calculating the preference parameter gamma v

import pandas as pd
import random as rand
import numpy as np
import upd_pd 
import new_ot
import upd_alp
from copy import deepcopy
import run_est
import fit_spline
import time
    
def non_obs_pp(cdat):
    """creates non-observation-type preference parameters"""

    #normalize by food at home
    cdat['p1'] = 1 
    homefood = cdat['fc1']

    #loop through consumption categories
    for k in range(1,29):
        p = cdat['fc' + str(k+1)] / homefood
        cdat['p' + str(k + 1)] = p

    return cdat

def make_psums(cdat, alp, lw):
    """Sums quantities for use in preference param calculation"""

    # get expenditure on observation good
    ev = cdat['fc1'] * -1 #create arbitrary ev series
    for k in range(1,30):
        addme = (cdat['ot'] == k)\
                * (cdat['fc' + str(int(k))]) #writes type k ev
        ev = addme.add((cdat['ot'] != k) * ev) #preserves old ev's

        #kill the parameter on observation type 
        cdat['p' + str(k)] = cdat['p' + str(k)] * (cdat['ot'] != k)
        
    #sum all preference params
    phat = cdat.filter(regex = '^p[0-9]').sum(axis = 1)

    return phat, ev

def obs_pp(cdat, alp, lw):
    """replace obs type params"""

    # Get parameter sums
    phat, r = make_psums(cdat, alp, lw)

    # Get wealth and expenditure ratios
    exptot = cdat['exptot']
    w = lw / exptot

    # Fit spline
    gp = run_est.make_grid(cdat) #should move this to save on calculation
    spline = fit_spline.fit(gp, alp)

    # Get gammas
    gam = spline.ev(w,r) 
    op = gam * phat
    op[cdat['ot'] == 1] = 1 #in case of observation type food at home
    op[op < 0] = 0 #eliminate negative values (approximation error)

    # Read into consumption data
    cdat['op'] = op
    for i in range(1,29):
        pname = 'p' + str(i + 1)
        addme = (cdat['ot'] == i + 1) * op 
        cdat[pname] = addme.add((cdat['ot'] != i + 1) * cdat[pname])

    cond = cdat['ot'] == 1
    for i in range(1,29):
        pname = 'p' + str(i + 1)
        cdat.loc[cond, pname] = cdat.loc[cond, pname] / (gam[cond] * phat[cond])
    
    return cdat

def get_pp(cdat, alp, r, lw):
    """gets preference parameters from cons data"""

    #create non-observation type pref params
    cdat = non_obs_pp(cdat)

    #replace observation type params
    cdat = obs_pp(cdat, alp, lw)

    return cdat


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
from scipy.interpolate import griddata
    
def non_obs_pp(cdat):
    """creates non-observation-type preference parameters"""

    #loop through consumption categories
    counter = 0
    for k in range(29):
        p = cdat['p' + str(int(k + 1))]
        addme = (cdat['ot'] != k + 1)\
                * (cdat['fc' + str(int(k + 1))]) #writes type k ev
        counter = addme.add(counter)
        newp = addme.add((cdat['ot'] == k + 1) * p) #preserves old ev's
        cdat['p' + str(int(k + 1))] = newp 

    for k in range(29):
        p = cdat['p' + str(int(k + 1))]
        addme = (cdat['ot'] != k + 1) * p / counter * 100 #writes type k ev
        newp = addme.add((cdat['ot'] == k + 1) * p) #preserves old ev's
        cdat['p' + str(int(k + 1))] = newp 

    return cdat

def make_psums(cdat, alp, lw):
    """Sums quantities for use in preference param calculation"""

    # get expenditure on observation good
    ev = cdat['fc1'] * -1 #create arbitrary ev series
    for k in range(1,30):
        addme = (cdat['ot'] == k)\
                * (cdat['fc' + str(int(k))]) #writes type k ev
        ev = addme.add((cdat['ot'] != k) * ev) #preserves old ev's

    return ev

def obs_pp(cdat, alp, lw):
    """replace obs type params"""

    # Get parameter sums
    r = make_psums(cdat, alp, lw)
    phat = 100

    # Get wealth ratios
    exptot = cdat['exptot']
    w = lw / exptot

    # Fit spline
    gp = run_est.make_grid(cdat) #should move this to save on calculation
    #spline = fit_spline.fit(gp, alp)
    w_pts, r_pts, op_pts = fit_spline.fit(gp, alp)

    # Get gammas
    #gam = spline.ev(w,r) 
    gam = np.nan_to_num(griddata((w_pts, r_pts), op_pts, (w,r)))
    op = gam * phat
    op[op <= 1e-10] = 1e3 #eliminate negative and exactly zero values (approximation error)

    # Read into consumption data
    cdat['op'] = op
    for i in range(29):
        pname = 'p' + str(i + 1)
        addme = (cdat['ot'] == i + 1) * op 
        cdat[pname] = addme.add((cdat['ot'] != i + 1) * cdat[pname])

    return cdat

def get_pp(cdat, alp, r, lw):
    """gets preference parameters from cons data"""

    #replace observation type params
    cdat = obs_pp(cdat, alp, lw)

    #create non-observation type pref params
    cdat = non_obs_pp(cdat)

    return cdat


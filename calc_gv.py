# This file holds functions related to calculating the preference parameter gamma v

import pandas as pd
import random as rand
import numpy as np
import upd_pd 
import new_ot

def load_dat():
    """This function reads in data and does very simple cleaning"""

    cdat = pd.read_stata('exp_dat.dta')

    # Drop observations with very low income
    cdat = cdat[cdat['exptot'] >= 1000]

    # Drop observations with no food at home expend
    cdat = cdat[cdat['fc1'] > 0].reset_index().drop('index',1)

    return cdat

def fake_ob_types(cdat):
    """creates fake observation types for debugging before
    main implementation"""

    #create random integer list
    cdat['ot'] = np.random.random_integers(2,29,cdat.shape[0])

    return cdat

def get_pars():
    """creates parameter list"""

    #cons weight
    alp = 0.01

    #prices
    r = [1] * 29

    #w lower bar
    lw = 1000

    return alp, r, lw
    
def non_obs_pp(cdat, r):
    """creates non-observation-type preference parameters"""

    #normalize by food at home
    cdat['p1'] = 1
    homefood = r[1] * cdat['fc1']

    #loop through consumption categories
    for k in range(1,29):
        exp = r[k] * cdat['fc' + str(k+1)]
        p = exp / homefood
        cdat['p' + str(k + 1)] = p

    return cdat

def make_psums(cdat, alp, r, lw):
    """Sums quantities for use in preference param calculation"""

    print('1')
    #sum all preference params
    psum = cdat.filter(regex = '^p').sum(axis = 1)

    print('2')
    #get observation type param
    pobs = cdat.apply(lambda row: 
            row['p' + str(int(row['ot']))], axis=1)

    print('3')
    # get consumption of observation good
    cv = cdat.apply(lambda row: 
            row['fc' + str(int(row['ot']))], axis=1)

    print('4')
    # get price of observation good
    rv = cdat.apply(lambda row: 
            r[int(row['ot']) - 1], axis=1)

    print('5')
    #subtract the observation type from the sum
    phat = psum - pobs

    return psum, pobs, phat, rv, cv


def obs_pp(cdat, alp, r, lw):
    """replace obs type params, currently approx for speed"""

    # Get parameter sums
    psum, pobs, phat, rv, cv = make_psums(cdat, alp, r, lw)

    # Product of rv and cv
    rc = rv * cv

    # Approximation
    ft = phat * rc / (1 - rc)
    st = - phat / (1 - rc)
    op = ft + st * alp
    op[op < 0] = 0
    
    # Read into consumption data
    cdat['op'] = op
    for i in range(1,29):
        cdat.ix[cdat['ot'] == i + 1, 'p' + str(i + 1)] = list(cdat.ix[cdat['ot'] == i + 1, 'op'])
    
    return cdat


def get_pp(cdat, alp, r, lw):
    """gets preference parameters from cons data"""

    #create non-observation type pref params
    cdat = non_obs_pp(cdat, r)

    #replace observation type params
    cdat = obs_pp(cdat, alp, r, lw)

    return cdat

if __name__ == '__main__':
    """Main program, will probably eventually move this"""

    #Load data
    cdat = load_dat()
    cdat = fake_ob_types(cdat) 

    #Get parameters
    alp, r, lw = get_pars()

    #Infer preference parameters
    cdat = get_pp(cdat, alp, r, lw)

    #Calculate distribution parameters
    dparams = upd_pd.pref_dist(cdat)

    #Load vindex data 
    vin = pd.read_pickle('vin_dat.pickle')

    #Update observation types
    cdat = new_ot.ot_step(cdat, vin, dparams, alp, r, lw)

    import pdb; pdb.set_trace() 


# This file holds functions related to calculating the preference parameter gamma v

import pandas as pd
import random as rand
import numpy as np

def load_dat():
    """This function reads in data and does very simple cleaning"""

    cdat = pd.read_stata('exp_dat.dta')

    # Drop observations with very low income
    cdat = cdat[cdat['exptot'] >= 1000]

    # Drop observations with no food at home expend
    cdat = cdat[cdat['fc1'] > 0]

    return cdat

def fake_ob_types(cdat):
    """creates fake observation types for debugging before
    main implementation"""

    #create random integer list
    cdat['ot'] = np.random.random_integers(2,31,cdat.shape[0])

    return cdat

def get_pars():
    """creates parameter list"""

    #cons weight
    alp = 0.5

    #prices
    r = [1] * 31

    #w lower bar
    lw = 1000

    return alp, r, lw
    
def non_obs_pp(cdat, r):
    """creates non-observation-type preference parameters"""

    #normalize by food at home
    cdat['p1'] = 1
    homefood = r[1] * cdat['fc1']

    #loop through consumption categories
    for k in range(1,31):
        exp = r[k] * cdat['fc' + str(k+1)]
        p = exp / homefood
        cdat['p' + str(k + 1)] = p

    return cdat

def make_k(cdat, alp, r, lw):
    """creates K for use in preference param calculation"""

    #sum all preference params
    psum = cdat.filter(regex = '^p').sum(axis = 1)

    #get observation type param
    pobs = cdat['p' + str(ot)]

    #subtract the observation type from the sum
    phat = psum - pobs

    import pdb; pdb.set_trace()

def obs_pp(cdat, alp, r, lw):
    """replace obs type params, currently approx for speed"""

    cdat = make_k(cdat, alp, r, lw)

def get_pp(cdat, alp, r, lw):
    """gets preference parameters from cons data"""

    #create non-observation type pref params
    cdat = non_obs_pp(cdat, r)

    #replace observation type params
    cdat = obs_pp(cdat, alp, r, lw)

if __name__ == '__main__':
    """Main program, will probably eventually move this"""

    #Load data
    cdat = load_dat()
    cdat = fake_ob_types(cdat) 

    #Get parameters
    alp, r, lw = get_pars()

    #Infer preference parameters
    cdat = get_pp(cdat, alp, r, lw)
    




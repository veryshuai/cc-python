# This script contains functions for simulating the conspicuous consumption model

import pandas as pd
import scipy.stats
import numpy as np
import math
import random

def main():
    '''entry point'''

    # load data
    dat, vindat = load_dat()

    # add simulation number 
    dat['sn'] = 1e5

    # add wealth parameters
    dat['wp'] = est_wealth_dist(dat)

    # simulate data
    sim_dat = simulation(dat)
    import pdb; pdb.set_trace()

def simulation(dat):
    '''get simulated data'''

    # simulate wealths
    wealth_sim = sim_wealth(dat)

    # simulate parameters
    pref_sim = sim_pref(dat)

    # simulate obervation types
    ot_sim = sim_ot(dat)

    # get column names for parameters
    col_names = []
    for k in range(dat['pn']):
        col_names.append('p' + str(k + 1))

    np.concatenate(pref_sim, axis=1)
    sim_dat = pd.DataFrame(pref_sim, columns=col_names)

def load_dat():
    '''load data'''

    # get data
    cdat = pd.read_csv('data/cdat2014_04_14_18_53_57.csv')
    dparams = pd.read_csv('data/params2014_04_14_18_53_57.csv')
    vindat = pd.read_pickle('data/vin_dat.pickle')
    pn = len(dparams)
    cn = len(cdat)

    #put into a dictionary object
    dat = {'cd': cdat, 'dp': dparams, 'pn': pn, 'cn': cn}

    return dat, vindat

def est_wealth_dist(dat):
    '''return parameters of wealth distribution'''

    # get expenditure totals
    wealth = dat['cd']['exptot']
    wealth = wealth[wealth > 1000]

    # estimate lognormal parameters
    lw = np.log(wealth)
    mu = np.mean(lw)
    dif = lw - mu
    dif2 = dif ** 2
    sig2 = np.mean(dif2)

    params = {'mu': mu, 'sig2': sig2}

    return params

def sim_wealth(dat):
    '''return simulated wealth levels'''

    wealth = scipy.stats.lognorm.rvs(math.sqrt(dat['wp']['sig2']),
            scale=math.exp(dat['wp']['mu']),size=dat['sn'])

    return wealth

def sim_pref(dat):
    '''return simulated parameters'''
    
    prefs = []
    for k in range(dat['pn']):
        pname = 'p' + str(k)
        mu = dat['dp']['1'].iat[k]
        sig2 = dat['dp']['2'].iat[k]
        plist = scipy.stats.lognorm.rvs(math.sqrt(sig2),
            scale=math.exp(mu),size=dat['sn'])
        prefs.append(plist)

    return prefs

def sim_ot(dat):
    '''return simulated observation types'''

    # The estimated observation type list
    emp_ot = dat['cd']['ot']

    # Random sample
    ot_sim = emp_ot.iloc[[random.randint(0,len(emp_ot) - 1) for k in range(int(dat['sn']))]]

    return ot_sim

def est_dem_type_fracs(real_dat):
    '''return fraction of each dem type'''
    pass

def est_eq(): 
    '''return estimated equilibrium surface for ot'''
    pass

def write_cons(surf, sim_vals):
    '''return a simulated version of cdat'''
    pass

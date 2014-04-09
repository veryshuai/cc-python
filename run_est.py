# This script runs the estimation for my conspicuous consumption paper

import pandas as pd
import numpy as np
from calc_gv import *
import datetime
import time
import math
from math import log
import itertools
import yappi

def load_dat():
    """This function reads in data and does very simple cleaning"""

    cdat = pd.read_stata('exp_dat.dta')

    # Drop observations with very low income
    cdat = cdat[cdat['exptot'] >= 1000]

    # Drop observations with no food at home expend
    cdat = cdat[cdat['fc1'] > 0].reset_index().drop('index',1)

    # Initialize observation types
    cdat = fake_ob_types(cdat) 

    # Derive actually consumption (not necessary currently)
    # cdat = trans_expends(cdat, r)

    # Add demographic specific vins
    cdat = add_vins(cdat)

    return cdat

def make_grid(cdat):
    '''creates grid points'''

    #create min to max wealth on a log scale
    w = np.logspace(log(cdat.exptot.min() / float(1000), 2),
            log(cdat.exptot.max() / float(1000), 2), 10, base=2) ** -1
    r = np.logspace(log(0.01, 2),log(0.5, 2), 10, base=2)

    #create all possible tuples
    tups = []
    for j in w:
        for k in r:
            tups.append((j,k))

    return tups 

def fake_ob_types(cdat):
    """creates fake observation types"""

    #create random integer list
    cdat['ot'] = np.random.random_integers(2,29,cdat.shape[0])

    return cdat

def trans_expends(cdat, r):
    '''create actual consuption before estimation'''

    #Assign R's to cdat based on year
    print(r['hef_ord'])
    for k in range(29):
        cdat['exp' + str(k + 1)] = cdat.apply(lambda row: r.loc[r['hef_ord'] == k + 1, str(int(row['year']))]**-1 * row['fc' + str(k + 1)], axis=1)

    return cdat

def ask_boot():
    '''user decides if we are bootstraping or not'''

    boot = 0
    while boot != 'y' and boot != 'n':
        boot = input('Is this a bootstrap run? (y/n): ')

    #change boot to a boolean
    print(boot)
    if boot == 'y':
        boot_bool = True
    else:
        boot_bool = False

    return boot_bool

def est_loop(cdat, boot):
    '''main estimation loop'''

    cdat_orig = deepcopy(cdat) #for use with bootstrap
    for k in range(runs):
    
        #Get sample from cdat for bootstrap
        if boot:
            randlist = np.random.randint(0,len(cdat),len(cdat))
            cdat = deepcopy(cdat_orig.iloc[randlist].reset_index())

        #Timestamp
        timestamp = datetime.datetime\
                        .fromtimestamp(time.time())\
                        .strftime('%Y_%m_%d_%H_%M_%S')

        #Get parameters
        alp, r, lw = get_pars()

        alp = 0.1
        old_alp = 100

        while (old_alp - alp) ** 2 > 1e-6:

            # Print information
            old_alp = deepcopy(alp)
            print(alp)

            print('params')
            #Infer preference parameters
            cdat = get_pp(cdat, alp, r, lw)

            print('prefs')
            #Calculate distribution parameters
            dparams = upd_pd.pref_dist(cdat)

            print('ob_types')
            #Update observation types
            cdat = new_ot.ot_step(cdat, dparams, alp, r, lw)

            print('params')
            #Infer preference parameters
            cdat = get_pp(cdat, alp, r, lw)

            print('prefs')
            #Calculate distribution parameters
            dparams = upd_pd.pref_dist(cdat)

            print('alp')
            #Update alpha and partial lik
            alp, lik = upd_alp.alp_step(cdat, alp, r, lw, dparams)
            print(alp)
            print(old_alp)

        #get the correct folder
        if boot:
            folder = 'boot'
        else:
            folder = 'results'
        cdat.to_csv(folder + '/cdat' + timestamp + '.csv')
        pd.DataFrame(dparams).T.to_csv(folder + '/params' + timestamp + '.csv')
        f = open(folder + '/alp' + timestamp + '.csv', 'w')
        f.write('{:.6f}'.format(alp))
        f.close()

if __name__ == '__main__':


    boot = ask_boot()

    #How many runs?
    if boot:
        runs = 100 
    else:
        runs = 1

    #Load data
    #cdat = load_dat()
    # cdat.to_pickle('cdat.pickle')
    cdat = pd.read_pickle('cdat.pickle')

    # Run main est loop
    est_loop(cdat, boot)


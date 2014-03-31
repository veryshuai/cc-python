# This script runs the estimation for my conspicuous consumption paper

import pandas as pd
import numpy as np
from calc_gv import *
import datetime
import time

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

if __name__ == '__main__':

    boot = ask_boot()

    #How many runs?
    if boot:
        runs = 100 
    else:
        runs = 1

    #Load data
    # cdat = load_dat()
    # cdat.to_pickle('cdat.pickle')
    cdat = pd.read_pickle('cdat.pickle')

    for k in range(runs):
    
        #Timestamp
        timestamp = datetime.datetime\
                        .fromtimestamp(time.time())\
                        .strftime('%Y_%m_%d_%H_%M_%S')

        #Get parameters
        alp, r, lw = get_pars()

        alp = 0.4
        old_alp = 100

        for k in range(1):
        #while (old_alp - alp) ** 2 > 1e-4:

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

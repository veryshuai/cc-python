# This script runs the estimation for my conspicuous consumption paper

import pandas as pd
import numpy as np
import calc_gv
import datetime
import time
import math
from math import log
import itertools
from copy import deepcopy
import upd_pd 
import new_ot
import upd_alp


def load_dat(dfile, vfile, gn):
    """This function reads in data and does very simple cleaning"""

    cdat = pd.read_stata(dfile)

    # Drop observations with very low income
    cdat = cdat[cdat['exptot'] >= 1000]
    cdat = cdat.reset_index()

    # Initialize observation types
    cdat = fake_ob_types(cdat, gn) 

    # Add demographic specific vins
    cdat = add_vins(cdat, vfile, gn)

    # Add simple initial parameter values
    for k in range(gn):
        cdat['p' + str(int(k + 1))] = cdat['fc' + str(int(k + 1))] * 100

    return cdat

def get_pars():
    """creates parameter list"""

    #cons weight
    alp = 0.1

    #prices
    r = pd.read_csv('price_dat.csv').set_index('cat_name')
    vin = pd.read_pickle('vin_dat.pickle') #get hef order
    r['hef_ord'] = vin.set_index(r.index)['hef_ord']

    #w lower bar
    lw = 1000

    return alp, r, lw

def add_vins(cdat, vname, gn):
    '''adds demographic specific vins to cdat'''

    #Load vindex data 
    vin = pd.read_pickle(vname)

    #Normalize
    col = vin.columns.values
    for k in range(1,len(col)-1):
        vin[col[k]] = vin[col[k]] / vin[col[k]].sum()

    #Add demographic specific vindex
    for k in range(1,gn + 1):
        cdat['vin' + str(k)] = cdat.apply(lambda row: select_vin(row, vin[vin['hef_ord'] == k]), axis = 1)

    return cdat

def select_vin(row, vin):
    '''selects correct vin for particular demographic'''
    try:
        if row.loc['up40'] == 1:
            head = 'up40'
        else:
            head = 'be40'

        if row.loc['ne'] == 1:
            tail = 'northeast'
        elif row.loc['st'] == 1:
            tail = 'south'
        elif row.loc['wt'] == 1:
            tail = 'west'
        else:
            tail = 'midwest'
    except:
        head = ''
        tail = 'vin'

    return log(vin[head + tail].iat[0])

def trans_expends(cdat, r):
    '''create actual consuption before estimation'''

    #Assign R's to cdat based on year
    print(r['hef_ord'])
    for k in range(29):
        cdat['exp' + str(k + 1)] = cdat.apply(lambda row: r.loc[r['hef_ord'] == k + 1, str(int(row['year']))]**-1 * row['fc' + str(k + 1)], axis=1)

    return cdat

def make_grid(cdat):
    '''creates grid points'''

    #create min to max wealth on a log scale
    #w = np.logspace(log(float(1000) / cdat.exptot.min(), 2),
    #        log(float(1000) / cdat.exptot.max(), 2), 40, base=2)
    #r = np.logspace(log(0.001, 2),log(0.5, 2), 40, base=2)
    w = np.linspace(float(1000) / cdat.exptot.max(),
            float(1000) / cdat.exptot.min(), 100)
    r = np.linspace(0.001,0.2, 100)

    #create all possible tuples
    tups = []
    for j in w:
        for k in r:
            tups.append((j,k))

    return tups 

def fake_ob_types(cdat, gn):
    """creates fake observation types"""

    #create random integer list
    cdat['ot'] = np.random.random_integers(1, gn, cdat.shape[0])

    return cdat

def trans_expends(cdat, r):
    '''create actual consuption before estimation'''

    #Assign R's to cdat based on year
    print(r['hef_ord'])
    for k in range(29):
        cdat['exp' + str(k + 1)] = cdat.apply(lambda row: r.loc[r['hef_ord'] == k + 1, str(int(row['year']))]**-1 * row['fc' + str(k + 1)], axis=1)

    return cdat

def ask(question):
    '''user decides something'''

    ans = 0
    while ans != 'y' and ans != 'n':
        ans = input(question)

    #change ans to a boolean
    if ans == 'y':
        ans_bool = True
    else:
        ans_bool = False

    return ans_bool

def est_loop(cdat, boot, runs, gn, prepend):
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

        while (old_alp - alp) ** 2 > 1e-9:

            # Print information
            old_alp = deepcopy(alp)
            print(alp)

            print('params')
            #Infer preference parameters
            cdat = calc_gv.get_pp(cdat, alp, r, lw, gn)

            print('prefs')
            #Calculate distribution parameters
            dparams = upd_pd.pref_dist(cdat, gn)

            print('ob_types')
            #Update observation types
            cdat = new_ot.ot_step(cdat, dparams, alp, r, lw, gn)

            print('params')
            #Infer preference parameters
            cdat = calc_gv.get_pp(cdat, alp, r, lw, gn)

            print('prefs')
            #Calculate distribution parameters
            dparams = upd_pd.pref_dist(cdat, gn)

            print('alp')
            #Update alpha and partial lik
            alp, lik = upd_alp.alp_step(cdat, alp, r, lw, dparams, gn)
            print(alp)
            print(old_alp)

        #get the correct folder
        if boot:
            folder = 'boot'
        else:
            folder = 'results'
        cdat.to_csv(prepend + folder + '/cdat' + timestamp + '.csv')
        pd.DataFrame(dparams).T.to_csv(folder + '/params' + timestamp + '.csv')
        f = open(folder + '/alp' + timestamp + '.csv', 'w')
        f.write('{:.6f}'.format(alp))
        f.close()

def do_u_boot():
    '''gets user input for bootstrap'''

    boot = ask('Is this a bootstrap run?: ')

    #How many runs?
    if boot:
        runs = 100 
    else:
        runs = 1

    return boot, runs

def go(dfile, pfile, vfile, gn, prepend):
    '''runs estimation routine'''

    #Is this a boot run?
    boot, runs = do_u_boot()

    #Load data
    redo_load = ask('Shall we reload data?: ')
    if redo_load:
        cdat = load_dat(dfile, vfile, gn)
        cdat.to_pickle(pfile)
    else:
        cdat = pd.read_pickle(pfile)

    # Run main est loop
    est_loop(cdat, boot, runs, gn, prepend)


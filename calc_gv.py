# This file holds functions related to calculating the preference parameter gamma v

import pandas as pd
import random as rand
import numpy as np
import upd_pd 
import new_ot
import upd_alp
from copy import deepcopy

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

def select_vin(row, vin):
    '''selects correct vin for particular demographic'''
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

    return vin[head + tail].iat[0]

def add_vins(cdat):
    '''adds demographic specific vins to cdat'''

    #Load vindex data 
    vin = pd.read_pickle('vin_dat.pickle')

    #Normalize
    col = vin.columns.values
    for k in range(1,9):
        vin[col[k]] = vin[col[k]] / vin[col[k]].sum()

    #Add demographic specific vindex
    for k in range(1,30):
        cdat['vin' + str(k)] = cdat.apply(lambda row: select_vin(row, vin[vin['hef_ord'] == k]), axis = 1)

    return cdat

def fake_ob_types(cdat):
    """creates fake observation types for debugging before
    main implementation"""

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

    # get expenditure on oservation good
    ev = cdat['fc1'] * -1 #create arbitrary ev series
    for k in range(2,30):
        addme = (cdat['ot'] == k)\
                * (cdat['fc' + str(int(k))]) #writes type k ev
        ev = addme.add((cdat['ot'] != k) * ev) #preserves old ev's

        #kill the parameter on observation type 
        cdat['p' + str(k)] = cdat['p' + str(k)] * (cdat['ot'] != k)
        
    #sum all preference params
    phat = cdat.filter(regex = '^p').sum(axis = 1)
    return phat, ev

def obs_pp(cdat, alp, lw):
    """replace obs type params, currently approx for speed"""

    # Get parameter sums
    phat, rc = make_psums(cdat, alp, lw)

    # Approximation
    ft = phat * rc / (1 - rc)
    st = - phat / (1 - rc)
    op = ft + st * alp
    op[(op < 0) | pd.isnull(op)] = 0

    # Read into consumption data
    cdat['op'] = op
    for i in range(1,29):
        pname = 'p' + str(i + 1)
        addme = (cdat['ot'] == i + 1) * op 
        cdat[pname] = addme.add((cdat['ot'] != i + 1) * cdat[pname])
    
    return cdat

def get_pp(cdat, alp, r, lw):
    """gets preference parameters from cons data"""

    #create non-observation type pref params
    cdat = non_obs_pp(cdat)

    #replace observation type params
    cdat = obs_pp(cdat, alp, lw)

    return cdat

if __name__ == '__main__':
    """Main program, will probably eventually move this"""

    #Get parameters
    alp, r, lw = get_pars()

    #Load data
    # cdat = load_dat()
    # cdat.to_pickle('cdat.pickle')
    cdat = pd.read_pickle('cdat.pickle')

    alp_list = []
    lik_list = []
    for init_ind in range(10):
        alp = np.linspace(0, 0.5, 10)[init_ind] 
        old_alp = 100
        print(init_ind)

        while (old_alp - alp) ** 2 > 1e-4:

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
        
        alp_list.append(alp)
        lik_list.append(lik)
        
        print(alp_list)
        print(lik_list)

    dat = pd.DataFrame([pd.Series(alp_list), pd.Series(lik_list)])
    dat.to_csv('first_results.csv')






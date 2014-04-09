# This file holds functions related to calculating the preference parameter gamma v

import pandas as pd
import random as rand
import numpy as np
import upd_pd 
import new_ot
import upd_alp
from copy import deepcopy

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

    # Get wealth and expenditure ratios
    w = cdat['exptot'] / lw

    # Exact solution 
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


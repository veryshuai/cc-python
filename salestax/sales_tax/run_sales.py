# This script contains functions for estimating the optimal sales tax in my conspicuous consumption project

import pandas as pd
import numpy as np
import math
import scipy.optimize

def load_dat():
    '''imports data'''

    # get data
    cdat = pd.read_csv('sales_tax/data/cdat2014_04_14_18_53_57.csv')
    dparams = pd.read_csv('sales_tax/data/params2014_04_14_18_53_57.csv')
    vindat = pd.read_pickle('sales_tax/data/vin_dat.pickle')
    pn = len(dparams)
    cn = len(cdat)
    vis = 3 

    #put into a dictionary object
    dat = {'cd': cdat, 'dp': dparams, 'pn': pn, 'cn': cn, 'vis': vis}

    return dat, vindat

def get_w(dat):
    '''creates adjusted wealth w'''

    pre_w = (dat['cd']['exptot'] - 1000) / float(1000)
    w = np.log(pre_w)

    return w

def create_adj_gam(dat):
    '''creates gamma parameters adjusted for wealth'''
    
    # get adjusted wealth
    w = get_w(dat)
    
    for k in range(dat['pn']):

        # create scaling factor
        fact = np.exp(- w * dat['dp']['3'].iloc[k])

        # get column names
        ap_name = 'ap' + str(k + 1)
        p_name = 'p' + str(k + 1)

        # write adjusted gamma values
        dat['cd'][ap_name] = dat['cd'][p_name] / fact

    return dat

def get_cons_shares(dat):
    '''creates the share of spending on the vis good'''

    #individual consumptions
    si = dat['cd']['fc' + str(int(dat['vis']))]
    ws = dat['cd']['exptot'] * si 

    #aggregate share
    dat['cd']['s'] = ws.sum() / dat['cd']['exptot'].sum()

    # #individual consumptions
    # dat['cd']['s'] = dat['cd']['fc' + str(int(dat['vis']))]

    return dat

def run_est(dat):
    '''runs estimation of the optimal tax'''

    sol = scipy.optimize.minimize_scalar(eval_wel_change, bounds=(0,1), args=[dat], method='bounded', tol=1e-20)

    return sol

def eval_wel_change(u,dat):
    '''returns welfare change for given tax'''

    rel_cols = dat['cd'].filter(regex = '^ap[0-9]')
    rel_cols = rel_cols.div(rel_cols.sum(axis = 1), axis = 0) #normalize
    rel_cols['sum'] = rel_cols.sum(axis=1)
    rel_cols['s'] = dat['cd']['s']

    ft = -np.log(1 - rel_cols['s'] * float(u)) * rel_cols['sum']
    st = rel_cols['ap' + str(dat['vis'])] * math.log(1 - u)
    wel = (ft + st)
    res = sum(wel)
    return -res

def main():
    '''entry point'''
    
    # load data
    dat, vindat = load_dat()

    # create adjusted gamma
    dat = create_adj_gam(dat)

    for k in range(dat['pn']):
        dat['vis'] = k + 1

        # get expenditure shares on visible good
        dat = get_cons_shares(dat)

        # call tax estimation routine
        fin_res = run_est(dat)

        print(vindat[vindat['hef_ord'] == dat['vis']]
                ['merge_id'].values)
        print(fin_res.x)


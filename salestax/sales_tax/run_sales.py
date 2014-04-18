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
    pn = len(dparams)
    cn = len(cdat)
    vis = 19

    #put into a dictionary object
    dat = {'cd': cdat, 'dp': dparams, 'pn': pn, 'cn': cn, 'vis': vis}

    return dat

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

    sol = scipy.optimize.minimize_scalar(eval_wel_change, bounds=(0,0.01), args=[dat], method='bounded', tol=1e-20)

    return sol

def sing_wel(row, u, dat):
    '''returns welfare change of a single household'''

    #scaling factor
    up = 1 / float(1 - u * row['s'])
    down = (1 - u) / float(1 - u * row['s'])
    
    wel_tot = 0
    for k in range(dat['pn']):
        if k + 1 != dat['vis']:
            aname = 'ap' + str(k + 1)
            addme = row[aname] * math.log(up)
        else:
            aname = 'ap' + str(k + 1)
            addme = row[aname] * math.log(down)
        wel_tot = wel_tot + addme
        
    return wel_tot

def eval_wel_change(u,dat):
    '''returns welfare change for given tax'''

    rel_cols = dat['cd'].filter(regex = '^ap[0-9]')
    rel_cols['sum'] = rel_cols.sum(axis=1)
    rel_cols['s'] = dat['cd']['s']

    #wel = rel_cols.apply(lambda row: sing_wel(row, u, dat), axis = 1)
    ft = -np.log(1 - rel_cols['s'] * float(u)) * rel_cols['sum']
    st = rel_cols['ap' + str(dat['vis'])] * math.log(1 - u)
    wel = ft + st 
    res = sum(wel)

    print(u)
    print(res)

    return -res

def main():
    '''entry point'''
    
    # load data
    dat = load_dat()

    # create adjusted gamma
    dat = create_adj_gam(dat)

    # get expenditure shares on visible good
    dat = get_cons_shares(dat)

    # call tax estimation routine
    fin_res = run_est(dat)
    import pdb; pdb.set_trace() 
    


# This script contains functions for estimating the optimal sales tax in my conspicuous consumption project

import pandas as pd
import numpy as np
import math
import scipy.optimize
import matplotlib.pyplot as plt

def load_dat():
    '''imports data'''

    # get data
    cdat = pd.read_csv('sales_tax/data/cdat2014_04_14_18_53_57.csv')
    cdat = cdat[cdat['ot'] == 10]
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
    
    runsum = 0
    for k in range(dat['pn']):

        # create scaling factor
        fact = np.exp(- w * dat['dp']['3'].iloc[k])

        # get column names
        ap_name = 'ap' + str(k + 1)
        p_name = 'p' + str(k + 1)

        # write adjusted gamma values
        dat['cd'][ap_name] = dat['cd'][p_name] / fact

        # add to normalization factor
        runsum = runsum + dat['cd'][ap_name]

    # normalize
    for k in range(dat['pn']):
        ap_name = 'ap' + str(k + 1)
        p_name = 'p' + str(k + 1)
        dat['cd'][ap_name] = dat['cd'][ap_name] / runsum

    return dat

def get_cons_shares(dat):
    '''creates the share of spending on the vis good'''

    #individual consumptions
    si = dat['cd']['fc' + str(int(dat['vis']))]
    ws = dat['cd']['exptot'] * si 

    #aggregate share
    dat['cd']['s'] = ws.sum() / dat['cd']['exptot'].sum()

    return dat

def return_x(x, dat):
    '''returns x'''
    return x[0]

def return_1_x(x, dat):
    '''returns 1 - x'''
    return 1 - x[0]

def min_wel(u,dat):
    '''get min from eval_wel_change'''

    wel = eval_wel_change(u,dat,True)
    
    return wel.quantile(q=0)

def run_est(dat):
    '''runs estimation of the optimal tax'''

    #sol = scipy.optimize.minimize_scalar(eval_wel_change, bounds=(0,0.9), args=[dat], method='bounded', tol=1e-12)
    sol = scipy.optimize.fmin_slsqp(eval_wel_change, 0.3,ieqcons=[min_wel], args=[dat], bounds=[(0,0.9999999)], iprint = 0, acc = 1e-15)

    return sol

def eval_wel_change(u,dat,vec=False):
    '''returns welfare change for given tax'''

    #prepare data
    rel_cols = dat['cd'].filter(regex = '^ap[0-9]')
    rel_cols['sum'] = rel_cols.sum(axis=1)
    rel_cols['s'] = dat['cd']['s']

    # calculate welfare
    ft = -np.log(1 - rel_cols['s'] * float(u)) * rel_cols['sum']
    st = rel_cols['ap' + str(dat['vis'])] * math.log(1 - float(u))
    wel = np.log((ft + st + dat['cd']['bu']) / dat['cd']['bu'])
    res = wel.sum()
    if vec:
        return wel
    else:
        return -res

def ind_util(row, dat):
    '''returns an individuals pre-tax utility'''
    addme = 0
    for k in range(dat['pn']):
        pname = 'p' + str(k + 1)
        fname = 'fc' + str(k + 1)
        if row[fname] > 0:
            addme = addme + row[pname] * math.log(row[fname]
                * row['exptot'])
    return addme

def get_base_util(dat):
    '''returns the pre-tax utility for each consumer'''

    dat['cd']['bu'] = dat['cd'].apply(lambda row: 
            ind_util(row, dat), axis = 1)
    return dat
    
def main():
    '''entry point'''
    
    # load data
    dat, vindat = load_dat()

    # create adjusted gamma
    dat = create_adj_gam(dat)

    # base utility
    dat = get_base_util(dat)

    for k in range(dat['pn']):
        dat['vis'] = k + 1

        # get expenditure shares on visible good
        dat = get_cons_shares(dat)

        # call tax estimation routine
        fin_res = run_est(dat)

        # get welfare distribution at optimum
        wel = eval_wel_change(fin_res[0], dat, True)

        if fin_res[0] > 1e-5:
            print(vindat[vindat['hef_ord'] == dat['vis']]
                ['merge_id'].values)
            print(fin_res[0])
            print('tax:')
            print(fin_res[0] / (1 - fin_res[0]))
            print(wel.describe()['mean'])
            print(wel.describe()['std'])

            #10% tax
            print('Now for the difference from 10% tax')
            wel = eval_wel_change(0.1, dat, True)
            print(wel.describe()['mean'])
            print(wel.describe()['std'])

            #plot gains
            grid = np.linspace(1e-12,.9999,100)
            pp1 = []
            pp2 = []
            pp3 = []
            for k in grid:
                wel = eval_wel_change(k, dat, True)
                pp1.append(wel.describe()['min'])
                pp2.append(wel.describe()['mean'])
                pp3.append(0)
            plt.plot(grid, pp1)
            plt.plot(grid, pp2)
            plt.plot(grid, pp3)
            plt.show()


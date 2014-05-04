# This script contains functions for estimating the optimal sales tax in my conspicuous consumption project

import pandas as pd
import numpy as np
import math
import scipy.optimize
import matplotlib.pyplot as plt
import pickle

def load_dat():
    '''imports data'''

    # get data
    cdat = pd.read_pickle('sales_tax/data/sim_dat2014_05_03_12_38_51.pickle')
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

    #aggregate share spent on visible good
    dat['cd']['sv'] = ws.sum() / dat['cd']['exptot'].sum()

    return dat

def return_x(x, dat):
    '''returns x'''
    return x[0]

def return_1_x(x, dat):
    '''returns 1 - x'''
    return 1 - x[0]

def min_wel(u, dat):
    '''get min from eval_wel_change'''

    wel = eval_wel_change(u, dat, True)
    try:
        quant = wel.quantile(q=0.0005)
    except Exception as e:
        print('WARNING: error in welfare quantile calculation')
        print(e)
        quant = -1 

    return quant

def run_est(dat):
    '''runs estimation of the optimal tax'''

    # Constraints
    cons = {'type' : 'ineq', 'fun' : min_wel, 'args': [dat]}

    # Solver
    sol = scipy.optimize.minimize(eval_wel_change, 0.3, bounds=[(0,0.9999999)], args=(dat,), constraints=cons, method = 'SLSQP', tol=1e-6)

    return sol

def get_subs(rel_cols, u):
    '''calculates balanced budget government subsidy'''

    #subsidy
    num = rel_cols['sv'].iat[0] * u
    s = num / float(1 - num)
    rel_cols['s'] = s[0]


    return rel_cols

def eval_wel_change(u, dat, vec=False):
    '''returns welfare change for given tax'''

    #prepare data (relevant columns)
    rel_cols = dat['cd'].filter(regex = '^ap[0-9]')
    rel_cols['sum'] = rel_cols.sum(axis=1)
    rel_cols['sv'] = dat['cd']['sv'] #aggregate spending on visible good
    rel_cols['wealth'] = dat['cd']['exptot'] #aggregate spending on visible good

    # calculate welfare
    try:
        # get subsidy
        rel_cols = get_subs(rel_cols, u)

        # calculate welfare
        ft = np.log(1 + rel_cols['s']) * rel_cols['sum']
        st = rel_cols['ap' + str(dat['vis'])] * math.log(1 - float(u))
        wel = ft + st
        res = wel.sum()
    except Exception as e:
        print(e)
        print('WARNING: Error in welfare calculation')
        wel = pd.Series([-np.inf])
        res = np.inf
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
    
def plot_tax(grid, plt_num, tit, dat, f, axarr):
    '''plots minimum and mean welfare gains'''

    #plot 
    pp1 = []
    pp2 = []
    pp3 = []
    pp4 = []
    ten = eval_wel_change(np.array([0.1 / float(1 + 0.1)]), dat, True)
    for k in grid:
        frac = k / float(100)
        wel = eval_wel_change(np.array([frac / float(1 + frac)]), dat, True)
        pp4.append(wel[dat['cd']['ot'] == 19].describe()['mean'])
        pp3.append(wel.quantile(q=0.0005))
        pp2.append(wel.describe()['mean'])
        pp1.append(0)
        #pp4.append(ten.describe()['mean'])

    # set up axes
    #axarr[plt_num % 2, plt_num % 3].plot(grid, pp1)
    #axarr[plt_num % 2, plt_num % 3].plot(grid, pp2)
    #axarr[plt_num % 2, plt_num % 3].plot(grid, pp3)
    #axarr[plt_num % 2, plt_num % 3].set_title(tit)
    axarr.plot(grid, pp1)
    axarr.plot(grid, pp2)
    axarr.plot(grid, pp3)
    axarr.plot(grid, pp4, 'o')
    axarr.set_title(tit)
    import pdb; pdb.set_trace() 

def main():
    '''entry point'''
    
    # load data
    #dat, vindat = load_dat()

    # create adjusted gamma
    #dat = create_adj_gam(dat)

    # base utility
    #dat = get_base_util(dat)

    #pickle.dump(dat, open('sales_tax/data/dat.pickle','wb'))
    dat = pickle.load(open('sales_tax/data/dat.pickle','rb'))
    vindat = pd.read_pickle('sales_tax/data/vin_dat.pickle')

    plt_num = 0
    f, axarr = plt.subplots(1, 1)
    #for k in range(dat['pn']):
    for k in [18]:

        dat['vis'] = k + 1
        
        # Print current category
        name = vindat[vindat['hef_ord']
                         == dat['vis']]['merge_id'].values
        print(name)

        # get expenditure shares on visible good
        dat = get_cons_shares(dat)

        # call tax estimation routine
        fin_res = run_est(dat)

        # get welfare distribution at optimum
        wel = eval_wel_change(fin_res.x, dat, True)

        # print only non-zero optimal taxes
        if fin_res.x[0] > 1e-6:
            print(fin_res.x[0])
            print('tax:')
            print(fin_res.x[0] / (1 - fin_res.x[0]))
            print('mean welfare')
            print(wel.describe())

            #10% tax
            print('Now for the difference from 10% tax')
            wel = eval_wel_change(np.array([0.1 / 1.1]), dat, True)
            print('10% mean welfare')
            print(wel.describe())

            #180% tax
            print('Now for the difference from 180% tax')
            wel = eval_wel_change(np.array([1.80 / 2.80]), dat, True)
            print('180% mean welfare')
            print(wel.describe())

            #plot gains
            grid = np.linspace(1e-12,100,100)
            plot_tax(grid, plt_num, name, dat, f, axarr)
            plt_num = plt_num + 1

    plt.show()


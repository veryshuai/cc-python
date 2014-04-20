# This script contains functions for simulating the conspicuous consumption model

import pandas as pd
import scipy.stats
import numpy as np
import math
import random
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import time, datetime

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

    # get surface
    surf = est_eq(dat)

    # calc simulated consumption fracs
    sim_dat = cons_fracs(sim_dat, surf, dat)

    # write the data
    timestamp = datetime.datetime\
                    .fromtimestamp(time.time())\
                    .strftime('%Y_%m_%d_%H_%M_%S')
    sim_dat['exptot'] = dat['cd']['exptot'] #need this to actual data
    sim_dat.to_csv('results/sim_dat' + timestamp + '.csv')

    print('Success!')

    return 0


def get_params(sim_dat, dat):
    '''creates gamma and wealth parameters needed for 
    consumption fraction calculation'''

    # create gammas
    only_prefs = sim_dat.filter(regex = '^p[0-9]')
    phat_plus = only_prefs.sum(axis=1)
    pt = 0 
    for k in range(dat['pn']):
        pt = pt + only_prefs['p' + str(k + 1)] * (sim_dat['ot'] == k + 1)
    sim_dat['phat'] = phat_plus - pt
    g = pt / sim_dat['phat']
         
    # create wealth
    w = 1000 / sim_dat['w']

    return g, w, sim_dat

def goodwise_cons_fracs(sim_dat, dat):
    '''returns good by good consumption fractions'''

    for k in range(dat['pn']): 
        fname = 'fc' + str(k + 1)
        pname = 'p'+ str(k + 1)
        sim_dat[fname] = (sim_dat['ot'] == k + 1) * sim_dat['r']\
                        + (sim_dat['ot'] != k + 1) * (1 - sim_dat['r'])\
                        * sim_dat[pname] / sim_dat['phat']

    return sim_dat

def cons_fracs(sim_dat, surf, dat):
    '''takes simulated preference params, wealth, and ot
    and returns consumption fractions'''

    # get parameters
    g, w, sim_dat = get_params(sim_dat, dat)

    # get obs consumption fractions
    sim_dat['r'] = np.nan_to_num(griddata((surf['w'], surf['g']), surf['r'], (w,g)))

    # get good-wise consumption fractions 
    sim_dat = goodwise_cons_fracs(sim_dat, dat)

    return sim_dat
     
def simulation(dat):
    '''get simulated data'''

    # simulate wealths
    wealth_sim = sim_wealth(dat)

    # simulate parameters
    pref_sim = sim_pref(dat)

    # simulate obervation types
    ot_sim = sim_ot(dat).reset_index()['ot']

    # get column names for parameters
    col_names = []
    for k in range(dat['pn']):
        col_names.append('p' + str(k + 1))

    #add parameters to sim_dat
    stacked = np.vstack(pref_sim)
    sim_dat = pd.DataFrame(stacked).T
    sim_dat.columns = col_names

    # add other simulation results to sim_dat
    sim_dat['ot'] = ot_sim
    sim_dat['w'] = wealth_sim

    return sim_dat

def load_dat():
    '''load data'''

    # get data
    cdat = pd.read_csv('data/cdat2014_04_14_18_53_57.csv')
    alp = pd.read_csv('data/alp2014_04_14_18_53_57.csv', header=None).values[0]
    dparams = pd.read_csv('data/params2014_04_14_18_53_57.csv')
    vindat = pd.read_pickle('data/vin_dat.pickle')
    pn = len(dparams)
    cn = len(cdat)

    #put into a dictionary object
    dat = {'cd': cdat, 'dp': dparams, 'pn': pn, 'cn': cn, 'alp': alp}

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

def err(r, w, g, a):
    '''calculates the error in function'''

    #read parameters
    ft = (1 - r) * (1 + g / a)
    st = (1 - a) * r / a
    if r == 0:
        tt = 0
    else:
        tt = (r * (1 + 1 / g)) ** (-g / a) * w ** (1 + g / a)
    res = ft - st - tt

    #warn if nan encountered
    if math.isnan(res):
        print('Warning: NaN encountered in visible parameter estimation (solve_gam, err)')
        res = np.inf

    return res

def vis_param(tup, a):
    '''solves for visible consumption ratio r'''

    #call solve
    try: 
        sol = scipy.optimize.brentq(err,0,1,args=tup + (a,))
    except Exception as e: 
        print(e)
        #print('optimization problem!')
        sol = 1e-15 #return almost zero (to flag impossiblity)

    return sol

def make_grid(cdat):
    '''creates grid points'''

    w = np.linspace(float(1000) / cdat.exptot.max(),
            float(1000) / cdat.exptot.min(), 100)
    g = np.linspace(0.001, 3, 100)

    #create all possible tuples
    tups = []
    for j in w:
        for k in g:
            tups.append((j,k))

    return tups 

def est_eq(dat): 
    '''return estimated equilibrium surface for ot'''

    # get grid
    gp = make_grid(dat['cd']) #should move this to save on calculation

    sols = []
    for tup in gp: 
        sol = vis_param(tup, dat['alp'])
        sols.append(sol)

    # #Sanity check for weird values 
    # pg = 0 
    # while pg != 'y' and pg != 'n':
    #     pg = input('plot gamma? (y/n): ')
    # if pg == 'y':
    #     plot_check(gp, sols)

    # Return lists of wealth, gamma ratios, and cons fractions
    w = [k[0] for k in gp]
    g = [k[1] for k in gp]
    surf = {'w': w, 'g': g, 'r': sols}
    return surf
    

def elim_weirds(group):
    '''even a zero param will have some consumption of the obs good
    which means that some consumptions are impossible, and confuse my
    solver.  This script eliminates those weird values'''

    minind = group[2].idxmin()
    group[group.index < minind] = 1e-12

    return group

def plot_check(gp, sols):
    '''plot solution to gamma problem'''

    # Scatter
    x = [k[0] for k in gp]
    y = [k[1] for k in gp]
    df = pd.DataFrame(np.array([x,y,sols]).T)
    #clean = df.groupby(0).apply(lambda row: elim_weirds(row))
    clean = df

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #spline = SmoothBivariateSpline(x,y,sols,kx=5,ky=5)
    #spline = SmoothBivariateSpline(clean[0],clean[1],clean[2],kx=2,ky=2)
    #ax.scatter(x,y,spline.ev(x,y))
    #ax.scatter(x,y,sols)
    ax.scatter(clean[0],clean[1],clean[2])
    xLabel = ax.set_xlabel('w')
    yLabel = ax.set_ylabel('g')
    zLabel = ax.set_zlabel('op')
    #ax.scatter(x,y,spline.ev(x,y))
    
    # Surface
    # xi = np.linspace(min(x), max(x), 100)
    # yi = np.linspace(min(y), max(y), 100)
    # zi = griddata(gp, sols, (xi, yi))
    # xim, yim = np.meshgrid(xi,yi)
    # ax.plot_surface(xim,yim,zi)

    plt.show()

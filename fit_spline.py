# This file contains functions which fit a 
# spline to model solutions

import numpy as np
import solve_gam
import run_est
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import SmoothBivariateSpline

def plot_check(gp, sols):
    '''plot solution to gamma problem'''

    # Scatter
    x = [k[0] for k in gp]
    y = [k[1] for k in gp]
    df = pd.DataFrame(np.array([x,y,sols]).T)
    clean = df.groupby(0).apply(lambda row: elim_weirds(row))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #spline = SmoothBivariateSpline(x,y,sols,kx=5,ky=5)
    #spline = SmoothBivariateSpline(clean[0],clean[1],clean[2],kx=2,ky=2)
    #ax.scatter(x,y,spline.ev(x,y))
    #ax.scatter(x,y,sols)
    ax.scatter(clean[0],clean[1],clean[2])
    xLabel = ax.set_xlabel('w')
    yLabel = ax.set_ylabel('r')
    zLabel = ax.set_zlabel('op')
    #ax.scatter(x,y,spline.ev(x,y))
    
    # Surface
    # xi = np.linspace(min(x), max(x), 100)
    # yi = np.linspace(min(y), max(y), 100)
    # zi = griddata(gp, sols, (xi, yi))
    # xim, yim = np.meshgrid(xi,yi)
    # ax.plot_surface(xim,yim,zi)

    plt.show()

def fit(gp, alp):
    '''fits spline polynomial '''

    #Solve on the grid
    sols = []
    for tup in gp: 
        sol = solve_gam.vis_param(tup, alp)
        sols.append(sol)

    #Sanity check for weird values 
    # pg = 0 
    # while pg != 'y' and pg != 'n':
    #     pg = input('plot gamma? (y/n): ')
    # if pg == 'y':
    #     plot_check(gp, sols)

    # Remove weird values 
    x = [k[0] for k in gp]
    y = [k[1] for k in gp]
    df = pd.DataFrame(np.array([x,y,sols]).T)
    clean = df.groupby(0).apply(lambda row: elim_weirds(row))

    return clean[0], clean[1], clean[2]

def elim_weirds(group):
    '''even a zero param will have some consumption of the obs good
    which means that some consumptions are impossible, and confuse my
    solver.  This script eliminates those weird values'''

    minind = group[2].idxmin()
    group[group.index < minind] = 1e-12

    return group
    
if __name__ == '__main__':

    cdat = pd.read_pickle('cdat.pickle')

    #get grid points in form of (w, r)
    gp = run_est.make_grid(cdat)

    alp = 0.3
    spline = fit(gp, alp)

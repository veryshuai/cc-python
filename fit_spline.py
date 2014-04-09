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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    spline = SmoothBivariateSpline(x,y,sols,kx=4,ky=4)
    ax.scatter(x,y,sols - spline.ev(x,y))
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
        sols.append(max(sol,0)) #note sometimes negative numbers are found

    #Sanity check for weird values 
    #plot_check(gp, sols)

    # Fit Spline to solutions
    x = [k[0] for k in gp]
    y = [k[1] for k in gp]
    spline = SmoothBivariateSpline(x,y,sols,kx=4,ky=4)

    return spline

if __name__ == '__main__':

    cdat = pd.read_pickle('cdat.pickle')

    #get grid points in form of (w, r)
    gp = run_est.make_grid(cdat)

    alp = 0.3
    spline = fit(gp, alp)

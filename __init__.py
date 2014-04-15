# This program creates and calls a simple class meant to wrap
# my procedural conspicuous consumption code

import run_est

class Consp_est():
    '''creates and runs consp estimation'''

    def __init__(self, dfile, pfile, vinfile, gsize, prepend):
        self.__dfile = dfile
        self.__vinfile = vinfile
        self.__gsize = gsize
        self.__pfile = pfile
        self.__prepend = prepend

    def estimate(self):
        run_est.go(self.__dfile, self.__pfile,
                   self.__vinfile, self.__gsize,
                   self.__prepend)

if __name__=='__main__':

    est = Consp_est('exp_dat.dta', 'cdat.pickle', 'vin_dat.pickle', 29, 'china_')
    est.estimate()

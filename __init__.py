# This program creates and calls a simple class meant to wrap
# my procedural conspicuous consumption code

import run_est

class Consp_est():
    '''creates and runs consp estimation'''

    def __init__(self, dfile, pfile, vinfile, 
                 gsize, prepend, dpfile, calc_t):
        self.__dfile = dfile
        self.__vinfile = vinfile
        self.__gsize = gsize
        self.__pfile = pfile
        self.__prepend = prepend
        self.__dpfile = dpfile
        self.__calc_t = calc_t

    def estimate(self):
        run_est.go(self.__dfile, self.__pfile,
                   self.__vinfile, self.__gsize,
                   self.__prepend, self.__dpfile,
                   self.__calc_t)

if __name__=='__main__':

    est = Consp_est('china/china_dat.dta', 'china/china_cdat.pickle', 
                    'china/china_vin.pickle', 14, 'china_',
                    'china/china_dparams.csv', False)
    est.estimate()

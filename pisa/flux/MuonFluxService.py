#
# MuonFluxService.py
#
# This flux service provides flux values for a grid of energy / cos(zenith)
# bins. It loads values from a .d file which were obtained from outputing values# for a certain model from  flux splines in MuonGun (for now only using 
# azimuth-averaged data) and uses spline interpolation to provide the integrated
# flux per bin to be consistent with the neutrino flux method.
#
# author: Melanie Day 
#         melanie.day@icecube.wisc.edu 
#
# date:   2014-10-28

import numpy as np
from scipy.interpolate import bisplrep, bisplev
from pisa.utils.log import logging
from pisa.utils.utils import get_bin_centers, get_bin_sizes
from pisa.resources.resources import open_resource

class MuonFluxService():
    '''Load a flux from CR muon flux tables in
       units of [GeV^-1 m^-2 s^-1 sr^-1] and
       return a 2D spline interpolated function.
       For now only supports azimuth-averaged input files.
    '''

    def __init__(self, flux_file=None, smooth=0.05, **params):
        logging.info("Loading muon flux table %s" %flux_file)

        #Load the data table
        table = np.loadtxt(open_resource(flux_file)).T

        #columns in files are in the same order
        cols = ['energy','muons']
        flux_dict = dict(zip(cols, table))
        for key in flux_dict.iterkeys():
            #There are 20 lines per zenith range
            flux_dict[key] = np.array(np.split(flux_dict[key], 20))
            if not key=='energy':
                flux_dict[key] = flux_dict[key].T

        #Set the zenith and energy range
        flux_dict['energy'] = flux_dict['energy'][0]
        flux_dict['coszen'] = np.linspace(0.95, -0.95, 20)

        #Now get a spline representation of the flux table.
        logging.debug('Make spline representation of muon flux')
        # do this in log of energy and log of flux (more stable)
        logE, C = np.meshgrid(np.log10(flux_dict['energy']), flux_dict['coszen'])
        #Make splines
        self.spline_dict = {}
        #Get the logarithmic flux
        log_flux = np.log10(flux_dict['muons']).T
        #Get a spline representation
        spline =  bisplrep(logE, C, log_flux, s=smooth)
        #and store
        self.spline_dict['muons'] = spline

    def get_flux(self, ebins, czbins, prim='muons'):
        '''Get the flux in units [m^-2 s^-1] for the given
           bin edges in energy and cos(zenith).'''
        #Currently there is no use for prim except to make this function behave like the others 
        #Evaluate the flux at the bin centers
        evals = get_bin_centers(ebins)
        czvals = get_bin_centers(czbins)
    
        # Get the spline interpolation, which is in
        # log(flux) as function of log(E), cos(zenith)
        return_table = bisplev(np.log10(evals), czvals, self.spline_dict['muons'])
        return_table = np.power(10., return_table).T
    
        #Flux is given per sr and GeV, so we need to multiply
        #by bin width in both dimensions
        #Get the bin size in both dimensions
        ebin_sizes = get_bin_sizes(ebins)
        czbin_sizes = 2.*np.pi*get_bin_sizes(czbins)
        bin_sizes = np.meshgrid(ebin_sizes, czbin_sizes)
    
        return_table *= np.abs(bin_sizes[0]*bin_sizes[1])
    
        return return_table.T


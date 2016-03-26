#
# HondaFluxService.py
#
# This flux service provides flux values for a grid of energy / cos(zenith)
# bins. It loads a flux table as provided by Honda (for now only able to use
# azimuth-averaged data) and uses spline interpolation to provide the integrated
# flux per bin.
#
# If desired, this will create a .json output file with the results of
# the current stage of processing.
#
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#
# date:   2014-01-27

import os
import numpy as np
from scipy.interpolate import bisplrep, bisplev
from pisa.utils.log import logging, profile
from pisa.utils.utils import get_bin_centers, get_bin_sizes, oversample_binning
from pisa.resources.resources import open_resource

#Global definition of primaries for which there is a neutrino flux
primaries = ['numu', 'numu_bar', 'nue', 'nue_bar']

class myHondaFluxService():
    """Load a neutrino flux from Honda-styles flux tables in units of
       [GeV^-1 m^-2 s^-1 sr^-1] and return a 2D spline interpolated
       function per flavour.  For now only supports azimuth-averaged
       input files.
    """

    def __init__(self, flux_file=None, smooth=0.05, oversample_e=1, oversample_cz=1,**params):

        self.final_tablesT = {}
        self.ebins = None
        self.czbins = None
        self.oversample_e = oversample_e
        self.oversample_cz = oversample_cz
        logging.info("Loading atmospheric flux table %s" %flux_file)

        #Load the data table
        table = np.loadtxt(open_resource(flux_file)).T

        #columns in Honda files are in the same order
        cols = ['energy']+primaries

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
        logging.debug('Make spline representation of flux')
        # do this in log of energy and log of flux (more stable)
        logE, C = np.meshgrid(np.log10(flux_dict['energy']), flux_dict['coszen'])

        self.spline_dict = {}
        for nutype in primaries:
            #Get the logarithmic flux
            log_flux = np.log10(flux_dict[nutype]).T
            #Get a spline representation
            spline =  bisplrep(logE, C, log_flux, s=smooth)
            #and store
            self.spline_dict[nutype] = spline

    def get_flux(self, ebins, czbins, prim):
        """Get the flux in units [m^-2 s^-1] for the given
           bin edges in energy and cos(zenith) and the primary."""

        if self.final_tablesT.has_key(prim):
            # check if this is the table we want
            if all(ebins == self.ebins) and all(self.czbins == czbins):
                # if so, return it
                profile.debug('Reusing flux table %s'%prim)
                return self.final_tablesT[prim]

        # otherwise continue here, update chached values
        profile.debug('Calculating flux table %s'%prim)
        self.ebins = ebins
        self.czbins = czbins

        # do it once, but with much finer steps for 'integrating' the spline interpolation
        # this is now handlet externaly....controlled by the actual_oversampling parameter in the template settings file

        s_ebins = oversample_binning(self.ebins, self.oversample_e)
        s_czbins = oversample_binning(self.czbins, self.oversample_cz) 

        #Evaluate the flux at the bin centers
        evals = get_bin_centers(s_ebins)
        czvals = get_bin_centers(s_czbins)

        # Get the spline interpolation, which is in
        # log(flux) as function of log(E), cos(zenith)
        return_table = bisplev(np.log10(evals), czvals, self.spline_dict[prim])
        return_table = np.power(10., return_table).T

        #Flux is given per sr and GeV, so we need to multiply
        #by bin width in both dimensions
        #Get the bin size in both dimensions
        ebin_sizes = get_bin_sizes(s_ebins)
        czbin_sizes = 2.*np.pi*get_bin_sizes(s_czbins)
        bin_sizes = np.meshgrid(ebin_sizes, czbin_sizes)

        return_table *= np.abs(bin_sizes[0]*bin_sizes[1])

        final_table_e = np.zeros((len(s_czbins)-1,len(ebins)-1))
        
        # sum up energy bins
        for i in xrange(len(ebins)-1):
            for j in xrange(self.oversample_e):
                final_table_e.T[i] += return_table.T[i*self.oversample_e +j]
        
        final_table = np.zeros((len(czbins)-1,len(ebins)-1))
        
        # sum up cz bins
        for i in xrange(len(czbins)-1):
            for j in xrange(self.oversample_cz):
                final_table[i] += final_table_e[i*self.oversample_cz +j]

        self.final_tablesT[prim] = final_table.T

        return self.final_tablesT[prim]


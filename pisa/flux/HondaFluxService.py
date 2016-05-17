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
from pisa.utils.log import logging
from pisa.utils.utils import get_bin_centers, get_bin_sizes
from pisa.resources.resources import open_resource

#Global definition of primaries for which there is a neutrino flux
primaries = ['numu', 'numu_bar', 'nue', 'nue_bar']

class HondaFluxService():
    """Load a neutrino flux from Honda-styles flux tables in units of
       [GeV^-1 m^-2 s^-1 sr^-1] and return a 2D spline interpolated
       function per flavour.
    """

    def __init__(self, flux_file=None, smooth=0.05, **params):
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
        logging.debug('Doing quick bivariate spline interpolation')
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

        logging.debug('Evaluating the bsplines directly')
        
        #Evaluate the flux at the bin centers
        evals = get_bin_centers(ebins)
        czvals = get_bin_centers(czbins)

        # Get the spline interpolation, which is in
        # log(flux) as function of log(E), cos(zenith)
        return_table = bisplev(np.log10(evals), czvals, self.spline_dict[prim])
        return_table = np.power(10., return_table).T

        #Flux is given per sr and GeV, so we need to multiply
        #by bin width in both dimensions
        #Get the bin size in both dimensions
        ebin_sizes = get_bin_sizes(ebins)
        czbin_sizes = 2.*np.pi*get_bin_sizes(czbins)
        bin_sizes = np.meshgrid(ebin_sizes, czbin_sizes)

        return_table *= np.abs(bin_sizes[0]*bin_sizes[1])

        return return_table.T

class Honda3DFluxService():
    """Load a neutrino flux from Honda-styles flux tables in units of
       [GeV^-1 m^-2 s^-1 sr^-1] and return a 3D spline interpolated
       function per flavour. This is actually 12 lots of 2D splines
       (one for each azimuth).
    """

    def __init__(self, flux_file=None, smooth=0.05, **params):
        logging.info("Loading atmospheric flux table %s" %flux_file)

        #Load the data table
        table = np.loadtxt(open_resource(flux_file)).T

        #columns in Honda files are in the same order
        cols = ['energy']+primaries

        flux_dict = dict(zip(cols, table))
        for key in flux_dict.iterkeys():

            #There are 20 lines per zenith range
            coszenith_lists = np.array(np.split(flux_dict[key], 20))
            azimuth_lists = []
            for coszenith_list in coszenith_lists:
                azimuth_lists.append(np.array(np.split(coszenith_list,12)).T)
            flux_dict[key] = np.array(azimuth_lists)
            if not key=='energy':
                flux_dict[key] = flux_dict[key].T

        #Set the zenith and energy range
        flux_dict['energy'] = flux_dict['energy'][0].T[0]
        flux_dict['coszen'] = np.linspace(0.95, -0.95, 20)
        flux_dict['azimuth'] = np.linspace(15,345,12)

        #Now get a spline representation of the flux table.
        logging.debug('Make spline representation of flux')
        logging.debug('Doing quick bsplrep spline interpolation in 3D')
        # do this in log of energy and log of flux (more stable)
        logE, C = np.meshgrid(np.log10(flux_dict['energy']), flux_dict['coszen'])

        self.spline_dict = {}
        
        for nutype in primaries:
            self.spline_dict[nutype] = {}
            #Make 1 2D bsplrep in E,CZ for each azimuth value
            for az, f in zip(flux_dict['azimuth'],flux_dict[nutype]):
                #Get the logarithmic flux
                log_flux = np.log10(f.T)
                #Get a spline representation
                spline =  bisplrep(logE, C, log_flux, s=smooth*4.)
                #and store
                self.spline_dict[nutype][az] = spline

    def get_flux(self, ebins, czbins, prim):
        """Get the flux in units [m^-2 s^-1] for the given
           bin edges in energy and cos(zenith) and the primary."""

        logging.debug('Evaluating the bsplines directly')
        
        #Evaluate the flux at the bin centers
        evals = get_bin_centers(ebins)
        czvals = get_bin_centers(czbins)

        # Get the spline interpolation, which is in
        # log(flux) as function of log(E), cos(zenith)
        # Return one value for every table value of azimuth
        all_az_table = []
        for az in np.linspace(15,345,12):
            log_table = bisplev(np.log10(evals), czvals, self.spline_dict[prim][az])
            all_az_table.append(np.power(10., log_table).T)

        all_az_table = np.array(all_az_table)

        # Integrate along az
        return_table = np.sum(all_az_table,axis=0)
        azbin_sizes = np.pi / 6. #15 degrees
        return_table *= azbin_sizes

        #Flux is given per sr and GeV, so we need to multiply
        #by bin width in both dimensions
        #Get the bin size in both dimensions
        ebin_sizes = get_bin_sizes(ebins)
        czbin_sizes = get_bin_sizes(czbins)
        
        bin_sizes = np.meshgrid(ebin_sizes, czbin_sizes)

        return_table *= np.abs(bin_sizes[0]*bin_sizes[1])
        
        return return_table.T

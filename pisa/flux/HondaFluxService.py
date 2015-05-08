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
from pisa.utils.plot import product_map
from pisa.flux.ShapeMod import modify_shape
from pisa.resources.resources import open_resource

#Global definition of primaries for which there is a neutrino flux
primaries = ['numu', 'numu_bar', 'nue', 'nue_bar']

class HondaFluxService():
    '''Load a neutrino flux from Honda-styles flux tables in
       units of [GeV^-1 m^-2 s^-1 sr^-1] and
       return a 2D spline interpolated function per flavour.
       For now only supports azimuth-averaged input files.
    '''

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

    def get_flux(self, ebins, czbins, prim, flux_hadronic_A, flux_hadronic_B, flux_hadronic_C, flux_hadronic_D, flux_hadronic_E, flux_hadronic_F, flux_hadronic_G, flux_hadronic_H, flux_hadronic_I, flux_hadronic_W, flux_hadronic_X, flux_hadronic_Y, flux_hadronic_Z,  flux_prim_norm_a, flux_prim_exp_norm_b, flux_prim_exp_factor_c, flux_spectral_index_d, flux_pion_chargeratio_Chg, UNC_FILES, **params):
        '''Get the flux in units [m^-2 s^-1] for the given
           bin edges in energy and cos(zenith) and the primary.'''
        
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
      
        ### FORM A TABLE FROM THE UNCERTAINTY WEIGHTS AND THE SPLINED MAPS CORRESPONDING TO THEM - WE DISCUSSED THIS SHOUD BE DONE EXPLICITLY FOR EASIER UNDERSTANDING###
        return_table = return_table \
                       + return_table * modify_shape(ebins, czbins, flux_hadronic_A, UNC_FILES['UNCF_A']) \
                       + return_table * modify_shape(ebins, czbins, flux_hadronic_B, UNC_FILES['UNCF_B']) \
                       + return_table * modify_shape(ebins, czbins, flux_hadronic_C, UNC_FILES['UNCF_C']) \
                       + return_table * modify_shape(ebins, czbins, flux_hadronic_D, UNC_FILES['UNCF_D']) \
                       + return_table * modify_shape(ebins, czbins, flux_hadronic_E, UNC_FILES['UNCF_E']) \
                       + return_table * modify_shape(ebins, czbins, flux_hadronic_F, UNC_FILES['UNCF_F']) \
                       + return_table * modify_shape(ebins, czbins, flux_hadronic_G, UNC_FILES['UNCF_G']) \
                       + return_table * modify_shape(ebins, czbins, flux_hadronic_H, UNC_FILES['UNCF_H']) \
                       + return_table * modify_shape(ebins, czbins, flux_hadronic_I, UNC_FILES['UNCF_I']) \
                       + return_table * modify_shape(ebins, czbins, flux_hadronic_W, UNC_FILES['UNCF_W']) \
                       + return_table * modify_shape(ebins, czbins, flux_hadronic_X, UNC_FILES['UNCF_X']) \
                       + return_table * modify_shape(ebins, czbins, flux_hadronic_Y, UNC_FILES['UNCF_Y']) \
                       + return_table * modify_shape(ebins, czbins, flux_hadronic_Z, UNC_FILES['UNCF_Z']) \
                       + return_table * modify_shape(ebins, czbins, flux_prim_norm_a, UNC_FILES['UNCF_a']) \
                       + return_table * modify_shape(ebins, czbins, flux_prim_exp_norm_b, UNC_FILES['UNCF_b']) \
                       + return_table * modify_shape(ebins, czbins, flux_prim_exp_factor_c, UNC_FILES['UNCF_c']) \
                       + return_table * modify_shape(ebins, czbins, flux_spectral_index_d, UNC_FILES['UNCF_d']) \
                       + return_table * modify_shape(ebins, czbins, flux_pion_chargeratio_Chg, UNC_FILES['UNCF_Chg'])
                
        return return_table.T


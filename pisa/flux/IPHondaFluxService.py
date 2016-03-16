#
# IPHondaFluxService.py
#
# This flux service provides flux values for a grid of energy / cos(zenith)
# bins. It loads a flux table as provided by Honda (for now only able to use
# azimuth-averaged data) and uses spline interpolation to provide the integrated
# flux per bin.
#
# If desired, this will create a .json output file with the results of
# the current stage of processing.
#
# I have copied Sebastian's HondaFluxService.py and just changed how it deals
# with the Honda tables once they are read in. Otherwise it is equivalent.
#
# author: Steven Wren
#         steven.wren@icecube.wisc.edu
#
# date:   2016-03-16

import os
import numpy as np
from scipy.interpolate import splrep, splev
from pisa.flux.HondaFluxService import primaries
from pisa.utils.log import logging
from pisa.utils.utils import get_bin_centers, get_bin_sizes
from pisa.resources.resources import open_resource

class IPHondaFluxService():
    """Load a neutrino flux from Honda-styles flux tables in units of
       [GeV^-1 m^-2 s^-1 sr^-1] and return a 2D spline interpolated
       function per flavour.  For now only supports azimuth-averaged
       input files.
    """

    def __init__(self, flux_file=None, **params):
        logging.info("Loading atmospheric flux table %s" %flux_file)

        #Load the data table
        table = np.loadtxt(open_resource(flux_file)).T

        #columns in Honda files are in the same order
        cols = ['energy']+primaries

        flux_dict = dict(zip(cols, table))
        for key in flux_dict.iterkeys():

            #There are 20 lines per zenith range
            flux_dict[key] = np.array(np.split(flux_dict[key], 20))

        #Set the zenith and energy range
        flux_dict['energy'] = flux_dict['energy'][0]
        flux_dict['coszen'] = np.linspace(0.95, -0.95, 20)

        #Now get a spline representation of the flux table.
        logging.debug('Make spline representation of flux')
        logging.debug('Doing this integral-preserving. Will take longer')

        self.spline_dict = {}

        # Do integral-preserving method as in IceCube's NuFlux
        # This one will be based purely on SciPy rather than ROOT
        # Stored splines will be 1D in integrated flux over energy
        int_flux_dict = {}
        # Energy and CosZenith bins needed for integral-preserving
        # method must be the edges of those of the normal tables
        int_flux_dict['logenergy'] = np.linspace(-1.025,4.025,102)
        int_flux_dict['coszen'] = np.linspace(-1,1,21)
        for nutype in primaries:
            # spline_dict now wants to be a set of splines for
            # every table cosZenith value.
            splines = {}
            CZiter = 1
            for energyfluxlist in flux_dict[nutype]:
                int_flux = []
                tot_flux = 0.0
                int_flux.append(tot_flux)
                for energyfluxval, energyval in zip(energyfluxlist, flux_dict['energy']):
                    # Spline works best if you integrate flux * energy
                    tot_flux += energyfluxval*energyval
                    int_flux.append(tot_flux)

                spline = splrep(int_flux_dict['logenergy'],int_flux,s=0)
                CZvalue = '%.2f'%(1.05-CZiter*0.1)
                splines[CZvalue] = spline
                CZiter += 1
                    
            self.spline_dict[nutype] = splines

    def get_flux(self, ebins, czbins, prim):
        """Get the flux in units [m^-2 s^-1] for the given
           bin edges in energy and cos(zenith) and the primary."""

        # Integral-preserving mode is more involved.
        # Requires evaluating differential of splines at the
        # chosen energy value for each table cosZen value.
        # These values are the integrated, splined and the differential
        # is evaluated at the required cosZen value

        logging.debug('Evaluating the derivatives of the splines for integral-preserving method.')
            
        #Evaluate the flux at the bin centers
        evals = get_bin_centers(ebins)
        czvals = get_bin_centers(czbins)

        return_table = []

        for energyval in evals:
            logenergyval = np.log10(energyval)
            spline_vals = []
            for czkey in np.linspace(-0.95,0.95,20):
                # Have to multiply by bin widths to get correct derivatives
                # Here the bin width is in log energy, is 0.05
                spline_vals.append(splev(logenergyval,self.spline_dict[prim]['%.2f'%czkey],der=1)*0.05)
            int_spline_vals = []
            tot_val = 0.0
            int_spline_vals.append(tot_val)
            for val in spline_vals:
                tot_val += val
                int_spline_vals.append(tot_val)

            spline = splrep(np.linspace(-1,1,21),int_spline_vals,s=0)
                
            # Have to multiply by bin widths to get correct derivatives
            # Here the bin width is in cosZenith, is 0.1
            czfluxes = splev(czvals,spline,der=1)*0.1/energyval
            return_table.append(czfluxes)

        return_table = np.array(return_table).T

        #Flux is given per sr and GeV, so we need to multiply
        #by bin width in both dimensions
        #Get the bin size in both dimensions
        ebin_sizes = get_bin_sizes(ebins)
        czbin_sizes = 2.*np.pi*get_bin_sizes(czbins)
        bin_sizes = np.meshgrid(ebin_sizes, czbin_sizes)

        return_table *= np.abs(bin_sizes[0]*bin_sizes[1])

        return return_table.T

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
from pisa.utils.log import logging
from pisa.utils.jsons import from_json
from pisa.utils.utils import get_bin_centers, get_bin_sizes
from pisa.resources.resources import open_resource, find_resource


#Global definition of primaries for which there is a neutrino flux
primaries = ['numu', 'numu_bar', 'nue', 'nue_bar']

class HondaFluxService():
    """Load a neutrino flux from presaved json file where flux were 
       calculated via NuFlux IP method at the bin centers, with actual_over_sample_e
       or actual_over_sample_cz applied.  Flux unit: [GeV^-1 m^-2 s^-1 sr^-1]
    """

    def __init__(self, NuFluxIP_file=None, smooth=0.05, **params):
        logging.info("Loading atmospheric flux table %s" %NuFluxIP_file)

        # Read from json file
        self.flux_NuFluxIP = from_json(find_resource(NuFluxIP_file))

    def get_flux(self, ebins, czbins, prim):
        """Get the flux in units [m^-2 s^-1] for the given
           bin edges in energy and cos(zenith) and the primary."""

        #Evaluate the flux at the bin centers
        evals = get_bin_centers(ebins)
        czvals = get_bin_centers(czbins)

        # Get the flux from NuFluxIP_file 
        return_table = self.flux_NuFluxIP[prim]
        return_table *= 1e4         # change from cm^-2 to m^-2

        #Flux is given per sr and GeV, so we need to multiply
        #by bin width in both dimensions
        #Get the bin size in both dimensions
        ebin_sizes = get_bin_sizes(ebins)
        czbin_sizes = 2.*np.pi*get_bin_sizes(czbins)
        bin_sizes = np.meshgrid(ebin_sizes, czbin_sizes)

        return_table *= np.abs(bin_sizes[0]*bin_sizes[1])

        return return_table.T


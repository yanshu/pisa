#! /usr/bin/env python
#
# PreCalculateFlux.py
#
# Sample the atmospheric neutrino flux from a FluxService with a given
# binning in cos(zenith) and energy.
#
# This will create a .json file in the analysis binning to be read directly
# in template maker to save calculating every time
#
# This is most useful when doing the calculations integral-preserving since
# that slows template down by a factor of 3
#
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   2014-01-27

import os
import sys
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.analysis.stats.Maps import apply_ratio_scale
from pisa.flux.HondaFluxService import HondaFluxService, primaries
from pisa.utils.jsons import from_json, to_json, json_string
from pisa.utils.log import logging, physics, set_verbosity
from pisa.utils.proc import report_params, get_params, add_params
from pisa.utils.utils import get_bin_centers

def get_median_energy(flux_map):
    """Returns the median energy of the flux_map-expected to be a dict
    with keys 'map', 'ebins', 'czbins'
    """

    ecen = get_bin_centers(flux_map['ebins'])
    energy = ecen[len(ecen)/2]

    return energy

def get_flux_maps(flux_service, ebins, czbins, **kwargs):
    """
    Get a set of flux maps for the different primaries.

    \params:
      * flux_service -
      * ebins/czbins - energy/coszenith bins to calculate flux
    """

    # Be verbose on input
    params = get_params()
    report_params(params, units = [''])

    # Initialize return dict
    maps = {'params': params}

    for prim in primaries:

        # Get the flux for this primary
        maps[prim] = {'ebins': ebins,
                      'czbins': czbins,
                      'map': flux_service.get_flux(ebins,czbins,prim)}

        # be a bit verbose
        logging.trace("Total flux of %s is %u [s^-1 m^-2]"%
                      (prim,maps[prim]['map'].sum()))

    return maps


if __name__ == '__main__':

    # parser
    parser = ArgumentParser(description='Take a settings file '
        'as input and write out a set of flux maps',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--template_settings', type=str,
                        metavar='JSONFILE', required=True,
                        help='''settings for the template generation''')
    parser.add_argument('--flux_calc', metavar='STRING', type=str,
                        help='''Type of flux interpolation to perform''',
                        default='bisplrep')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE',
                        type=str, action='store', default='flux.json',
                        help='file to store the output')
    parser.add_argument('-v', '--verbose', action='count', default=None,
                        help='set verbosity level')
    args = parser.parse_args()

    #Set verbosity level
    set_verbosity(args.verbose)

    template_settings = from_json(args.template_settings)
    ebins = template_settings['binning']['ebins']
    czbins = template_settings['binning']['czbins']

    flux_file = template_settings['params']['flux_file']['value']
    flux_calc = args.flux_calc

    logging.debug("Using %u bins in energy from %.2f to %.2f GeV"%
                                (len(ebins)-1,ebins[0],ebins[-1]))
    logging.debug("Using %u bins in cos(zenith) from %.2f to %.2f"%
                                (len(czbins)-1,czbins[0],czbins[-1]))

    #Instantiate a flux model
    if flux_calc.lower() == 'bisplrep':
        flux_model = HondaFluxService(flux_file, IP=False)
    elif flux_calc.lower() == 'integral-preserving':
        flux_model = HondaFluxService(flux_file, IP=True)
    else:
        'Your flux calculation preference was not recognised. Please choose from [\'bisplrep\',\'integral-preserving\']'
        flux_model = None

    if flux_model is not None:
        #get the flux
        flux_maps = get_flux_maps(flux_model, ebins, czbins)

        #write out to a file
        logging.info("Saving output to: %s"%args.outfile)
        to_json(flux_maps, args.outfile)

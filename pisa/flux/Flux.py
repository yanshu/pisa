#! /usr/bin/env python
#
# Flux.py
#
# Sample the atmospheric neutrino flux from a FluxService with a given
# binning in cos(zenith) and energy.
#
# If desired, this will create a .json output file with the results of
# the current stage of processing.
#
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#
# date:   2014-01-27

import os
import sys
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pisa.utils.log import logging, physics, set_verbosity
from pisa.utils.jsons import from_json, to_json, json_string
from pisa.utils.proc import report_params, get_params, add_params
from pisa.flux.HondaFluxService import HondaFluxService, primaries
from pisa.flux.MuonFluxService import MuonFluxService

def get_flux_maps(nu_flux_service, muon_flux_service, ebins, czbins, **params):
    '''Get a set of flux maps for the different primaries'''

    #Be verbose on input
    params = get_params()
    report_params(params, units = [])

    #Initialize return dict
    maps = {'params': params}
    
    p_services = dict.fromkeys(primaries,nu_flux_service) 
    p_services['muons'] = muon_flux_service

    for prim, service in p_services.items():
        #Get the flux for this primary
        maps[prim] = {'ebins': ebins,
                    'czbins': czbins,
                    'map': service.get_flux(ebins,czbins,prim)}
 
        #be a bit verbose
        physics.trace("Total flux of %s is %u [s^-1 m^-2]"%
                     (prim,maps[prim]['map'].sum()))

    #return this map
    return maps


if __name__ == '__main__':

    # parser
    parser = ArgumentParser(description='Take a settings file '
        'as input and write out a set of flux maps',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ebins', metavar='[1.0,2.0,...]', type=json_string,
        help= '''Edges of the energy bins in units of GeV, default is '''
              '''40 edges (39 bins) from 1.0 to 80 GeV in logarithmic spacing.''',
              default = np.logspace(np.log10(1.0),np.log10(80.0),41) )

    parser.add_argument('--czbins', metavar='[-1.0,-0.8.,...]', type=json_string,
        help= '''Edges of the cos(zenith) bins, default is '''
              '''21 edges (20 bins) from -1. (upward) to 0. horizontal in linear spacing.''',
        default = np.linspace(-1.,1.,21))
    parser.add_argument('--nu_flux_file', metavar='FILE', type=str,
        help= '''Input flux file in Honda format. ''',
        default = 'flux/spl-solmin-aa.d')

    parser.add_argument('--muon_flux_file', metavar='FILE', type=str,
        help= '''Input flux file for CR muons. ''',
        default = 'flux/GaisserH4a_atmod12_SIBYLL.d')

    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE', 
                        type=str, action='store', default='flux.json',
                        help='file to store the output')    

    parser.add_argument('-v', '--verbose', action='count', default=None,
                        help='set verbosity level')
    
    args = parser.parse_args()

    #Set verbosity level
    set_verbosity(args.verbose)

    logging.debug("Using %u bins in energy from %.2f to %.2f GeV"%
                                (len(args.ebins)-1,args.ebins[0],args.ebins[-1]))
    logging.debug("Using %u bins in cos(zenith) from %.2f to %.2f"%
                                (len(args.czbins)-1,args.czbins[0],args.czbins[-1]))

    #Instantiate a flux model
    nu_flux_model = HondaFluxService(args.nu_flux_file)
    muon_flux_model = MuonFluxService(args.muon_flux_file)
    
    #get the flux 
    flux_maps = get_flux_maps(nu_flux_model,muon_flux_model,args.ebins,args.czbins)

    #write out to a file
    logging.info("Saving output to: %s"%args.outfile)
    to_json(flux_maps, args.outfile)

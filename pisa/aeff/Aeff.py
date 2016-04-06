#! /usr/bin/env python
#
# Aeff.py
#
# This module is the implementation of the stage2 analysis. The main
# purpose of stage2 is to combine the "oscillated Flux maps" with the
# effective areas to create oscillated event rate maps, using the true
# information. This signifies what the "true" event rate would be for
# a detector with our effective areas, but with perfect PID and
# resolutions.
#
# If desired, this will create a .json output file with the results of
# the current stage of processing.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   April 8, 2014
#

import os,sys
import numpy as np
from scipy.constants import Julian_year
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.resources.resources import find_resource
from pisa.utils.jsons import from_json, to_json
from pisa.utils.log import logging, set_verbosity
from pisa.utils.proc import report_params, get_params, add_params
from pisa.utils.utils import check_binning, get_binning
from pisa.utils.params import get_values

from pisa.aeff.AeffServiceMC import AeffServiceMC
from pisa.aeff.AeffServicePar import AeffServicePar

def get_event_rates(osc_flux_maps,aeff_service,livetime=None,
                    aeff_scale=None,nutau_norm=None,**kwargs):
    '''
    Main function for this module, which returns the event rate maps
    for each flavor and interaction type, using true energy and zenith
    information. The content of each bin will be the weighted aeff
    multiplied by the oscillated flux, so that the returned dictionary
    will be of the form:
    {'nue': {'cc':map,'nc':map},
     'nue_bar': {'cc':map,'nc':map}, ...
     'nutau_bar': {'cc':map,'nc':map} }
    \params:
      * osc_flux_maps - maps containing oscillated fluxes
      * aeff_service - the effective area service to use
      * livetime - detector livetime for which to calculate event counts
      * aeff_scale - systematic to be a proxy for the realistic effective area
    '''

    #Get parameters used here
    params = get_params()
    report_params(params,units = ['','yrs',''])

    #Initialize return dict
    event_rate_maps = {'params': add_params(params,osc_flux_maps['params'])}

    #add 'nutau_norm' parameter to params
    event_rate_maps = {'params': add_params(params,{'nutau_norm':nutau_norm})}

    #Get effective area
    aeff_dict = aeff_service.get_aeff()

    ebins, czbins = get_binning(osc_flux_maps)

    # apply the scaling for nu_xsec_scale and nubar_xsec_scale...
    flavours = ['nue','numu','nutau','nue_bar','numu_bar','nutau_bar']
    for flavour in flavours:
        osc_flux_map = osc_flux_maps[flavour]['map']
        int_type_dict = {}
        for int_type in ['cc','nc']:
            scale = 1.0
            if flavour == 'nutau' or flavour == 'nutau_bar':
                if int_type == 'cc':
                    scale = nutau_norm 
            event_rate = scale*osc_flux_map*aeff_dict[flavour][int_type]*aeff_scale
            event_rate *= (livetime*Julian_year)
            int_type_dict[int_type] = {'map':event_rate,
                                       'ebins':ebins,
                                       'czbins':czbins}
            logging.debug("  Event Rate before reco for %s/%s: %.2f"
                          %(flavour,int_type,np.sum(event_rate)))
        event_rate_maps[flavour] = int_type_dict

    # else: no scaling to be applied
    return event_rate_maps

if __name__ == '__main__':

    parser = ArgumentParser(description='Take an oscillated flux file '
                          'as input & write out a set of oscillated event counts. ',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--osc_flux_maps',metavar='FLUX',type=str,default ='',
                     help='''JSON osc flux input file with the following parameters:
      {"nue": {'czbins':[], 'ebins':[], 'map':[]},
       "numu": {...},
       "nutau": {...},
       "nue_bar": {...},
       "numu_bar": {...},
       "nutau_bar": {...} }''')
    parser.add_argument('-t', '--template-settings', dest='ts', metavar='FILE', type=str,
                        action='store')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE', type=str,
                        action='store',default="aeff.json",
                        help='''file to store the output''')
    parser.add_argument('-v', '--verbose', action='count', default=None,
                        help='''set verbosity level''')
    args = parser.parse_args()

    #Set verbosity level
    set_verbosity(args.verbose)
    template_settings = get_values(from_json(args.ts)['params'])
    ebins = from_json(args.ts)['binning']['ebins']
    czbins = from_json(args.ts)['binning']['czbins']


    aeff_mode = template_settings['aeff_mode']
    if aeff_mode == 'param':
        logging.debug(" Using effective area from PARAMETRIZATION...")
        aeff_service = AeffServicePar(ebins, czbins,
                                           **template_settings)
    elif aeff_mode == 'MC':
        logging.debug(" Using effective area from MC EVENT DATA...")
        aeff_service = AeffServiceMC(ebins, czbins,
                                          **template_settings)

    if args.osc_flux_maps:
        osc_flux_maps=from_json(args.osc_flux_maps)
    else:
        osc_flux_maps = {}
        osc_flux_maps['params'] = {'bla':'bla'}
        for flav in ['nue','numu','nutau','nue_bar','numu_bar','nutau_bar']:
            osc_flux_maps[flav] = {'czbins':czbins, 'ebins':ebins, 'map':np.ones((len(ebins)-1,len(czbins)-1))}

    event_rate_maps = get_event_rates(osc_flux_maps,aeff_service,template_settings['livetime'],template_settings['aeff_scale'],
                                      template_settings['nutau_norm'])

    logging.info("Saving output to: %s"%args.outfile)
    to_json(event_rate_maps,args.outfile)



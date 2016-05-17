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
from pisa.utils.shape import SplineService
from pisa.utils.params import construct_genie_dict

from pisa.aeff.AeffServiceMC import AeffServiceMC
from pisa.aeff.AeffServicePar import AeffServicePar

def apply_shape_mod(aeff_dict, ebins, czbins, **params):
    '''
    Taking Joakim's shape mod functionality and applying it generally
    to the aeff_dict, regardless of the aeff_service used
    '''

    ### make dict of genie parameters ###
    GENSYS = construct_genie_dict(params)

    ### make spline service for genie parameters ###
    genie_spline_service = SplineService(ebins, dictFile = params['GENSYS_files'])
        
    return_dict = {}
    ### modify the aeff for NC interactions ###
    ### here the aeff is gotten for NC interactions###
    map_shape = [[0 for x in range(len(czbins))] for x in range(len(ebins))]
    mod_table = np.zeros_like(map_shape)
    mod_table_nc = np.zeros_like(map_shape)
    mod_table_nc_bar = np.zeros_like(map_shape)
        
    ### AND FOR CC INTERACTIONS BY FLAVOUR ###
    for flavor in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
        flavor_dict = {}
        logging.debug("Working on %s effective areas"%flavor)

        # POPULATE DICT WITH FLAVORS #
        aeff2d_mod = aeff_dict[flavor]['cc'].T
        aeff2d_nc_mod = aeff_dict[flavor]['nc'].T
        aeff2d_nc_bar_mod = aeff_dict[flavor]['nc'].T

        mod_table = np.zeros_like(aeff2d_nc_bar_mod)
        mod_table_nc = np.zeros_like(aeff2d_nc_bar_mod)
        mod_table_nc_bar = np.zeros_like(aeff2d_nc_bar_mod)

        # FILL THE SHAPE MODIFICATION TABLES
        for entry in GENSYS:
            if GENSYS[entry] != 0.0:
                #print "we are now passing onto modify shape: ", GENSYS[entry]
                mod_table += genie_spline_service.modify_shape(ebins, czbins, GENSYS[entry], str(entry)+"_"+str(flavor))
                if entry == "MaCCQE": continue
                mod_table_nc += genie_spline_service.modify_shape(ebins, czbins, GENSYS[entry], str(entry)+"_nuall_nc")
                mod_table_nc_bar += genie_spline_service.modify_shape(ebins, czbins, GENSYS[entry], str(entry)+"_nuallbar_nc") 

        # GENERATE MOD TABLES DICT#
        mod_tables = { 'mod_table': mod_table,
                       'mod_table_nc': mod_table_nc,
                       'mod_table_nc_bar': mod_table_nc_bar
        }
                    
        ### THIS FOLLOWING SECTION HAS DELIBERATE BEEN MADE THIS COMPLICATED - THE ASIMOV METHOD BREAKS THINGS OTHERWISE (ask me about it if interested) ###
        for table_name in mod_tables:
            if mod_tables[table_name][mod_tables[table_name] == 0.0].all(): continue
            if mod_tables[table_name][mod_tables[table_name]<0].any():
                for i in range(len(mod_tables[table_name])):
                    for f in range(len(mod_tables[table_name][i])):
                        if mod_tables[table_name][i][f] < 0.0:
                            mod_tables[table_name][i][f] = -1.0 * (np.sqrt(-1.0 * mod_tables[table_name][i][f]))
                        else:
                            mod_tables[table_name][i][f] = np.sqrt(mod_tables[table_name][i][f])
            else:
                mod_tables[table_name] = np.sqrt(mod_tables[table_name])
            
        aeff2d_mod = aeff2d_mod * (mod_tables["mod_table"] + 1.0)
        aeff2d_nc_mod = aeff2d_nc_mod * (mod_tables["mod_table_nc"] + 1.0)
        aeff2d_nc_bar_mod = aeff2d_nc_bar_mod * (mod_tables["mod_table_nc_bar"] + 1.0)
        
        #TEST FOR 0 AND OR NEGATIVE VALEUS #
        if aeff2d_mod[aeff2d_mod == 0.0].any() or aeff2d_nc_mod[aeff2d_nc_mod == 0.0].any() or aeff2d_nc_bar_mod[aeff2d_nc_bar_mod == 0.0].any():
            raise ValueError("Aeff Templates must have all bins > 0")
            
        # LOAD INTO FLAVOR DICT & SET RETURN DICT#
        flavor_dict['cc'] = aeff2d_mod.T
        flavor_dict['nc'] = aeff2d_nc_bar_mod.T if 'bar' in flavor else aeff2d_nc_mod.T
        return_dict[flavor] = flavor_dict

    return return_dict

def get_event_rates(osc_flux_maps,aeff_service,livetime=None,
                    aeff_scale=None,nutau_norm=None,
                    GENSYS_files=None,
                    GENSYS_AhtBY=None,GENSYS_BhtBY=None,
                    GENSYS_CV1uBY=None,GENSYS_CV2uBY=None,
                    GENSYS_MaCCQE=None,GENSYS_MaRES=None,**kwargs):
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
      * GENSYS_files - location of files needed for genie systematics
      * GENSYS_* - Names of genie scaling parameters for shape-dependent modifications
    '''

    #Get parameters used here
    params = get_params()
    report_params(params,units = ['','','','','','','','','yrs','',''])

    #Initialize return dict, add 'nutau_norm' parameter to params
    tmp_params = add_params(params,osc_flux_maps['params'])
    event_rate_maps = {'params': add_params(tmp_params,{'nutau_norm':nutau_norm})}

    #Get effective area - unmodified
    aeff_dict_bare = aeff_service.get_aeff()

    ebins, czbins = get_binning(osc_flux_maps)
    
    #Apply GENIE uncertainties
    aeff_dict = apply_shape_mod(aeff_dict_bare, ebins, czbins, **params)

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
    parser.add_argument('-ts', '--template-settings', dest='ts', metavar='FILE', type=str,
                        action='store')
    parser.add_argument('--weighted_aeff_file',metavar='WEIGHTFILE',type=str,
                        default='events/V15_weighted_aeff.hdf5',
                        help='''HDF5 File containing event data for each flavours for a particular
instrumental geometry. The effective area is calculated from the event
weights in this file. Only applies in non-parametric mode.''')
    parser.add_argument('--settings_file',metavar='SETTINGS',type=str,
                        default='aeff/V36_aeff.json',
                        help='''json file containing parameterizations of the effective
area and its cos(zenith) dependence. Only applies in parametric mode.''')
    parser.add_argument('--coszen_par',metavar='JSON',type=str,
                        default='aeff/V36/V36_aeff_cz.json')
    parser.add_argument('--livetime',type=float,default=1.0,
                        help='''livetime in years to re-scale by.''')
    parser.add_argument('--aeff_scale',type=float,default=1.0,
                        help='''Overall scale on aeff''')
    parser.add_argument('--mc_mode',action='store_true', default=False,
                        help='''Use MC-based effective areas instead of using the parameterized versions.''')
    parser.add_argument('--GENSYS_files', metavar='DICT', type=str,
                        help='''Dictionary of files containing the shapes of uncertainties corresponding to the below parameters''')
    parser.add_argument('--GENSYS_AhtBY',metavar='FLOAT',type=float,
                        help='''Factor to scale aeff dict by''',default=0)
    parser.add_argument('--GENSYS_BhtBY',metavar='FLOAT',type=float,
                        help='''Factor to scale aeff dict by''',default=0)
    parser.add_argument('--GENSYS_CV1uBY',metavar='FLOAT',type=float,
                        help='''Factor to scale aeff dict by''',default=0)
    parser.add_argument('--GENSYS_CV2uBY',metavar='FLOAT',type=float,
                        help='''Factor to scale aeff dict by''',default=0)
    parser.add_argument('--GENSYS_MaCCQE',metavar='FLOAT',type=float,
                        help='''Factor to scale aeff dict by''',default=0)
    parser.add_argument('--GENSYS_MaRES',metavar='FLOAT',type=float,
                        help='''Factor to scale aeff dict by''',default=0)
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
                                      template_settings['nutau_norm'],
                                      args.GENSYS_files,args.GENSYS_AhtBY,args.GENSYS_BhtBY,
                                      args.GENSYS_CV1uBY,args.GENSYS_CV2uBY,
                                      args.GENSYS_MaCCQE,args.GENSYS_MaRES)

    logging.info("Saving output to: %s"%args.outfile)
    to_json(event_rate_maps,args.outfile)



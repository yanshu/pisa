#! /usr/bin/env python
#
# NutauAnalysis.py
#
# Runs the LLR optimizer-based analysis for a nutau appearance search
#
# author: Feifei Huang - fxh140@psu.edu
#         Tim Arlen - tca3@psu.edu
#
# date:   31-March-2015
#

import numpy as np
import copy
import random as rnd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.utils.log import logging, profile, physics, set_verbosity
from pisa.utils.jsons import from_json,to_json
from pisa.analysis.llr.LLHAnalysis_nutau import find_max_llh_bfgs
from pisa.analysis.stats.Maps import get_seed, flatten_map
from pisa.analysis.stats.Maps_nutau_noDOMIce import get_pseudo_data_fmap, get_true_template
from pisa.analysis.stats.Maps_nutau import get_flipped_map, get_combined_map, get_up_map, get_flipped_down_map
from pisa.analysis.TemplateMaker import TemplateMaker
from pisa.analysis.GetMCError import GetMCError
from pisa.utils.params import get_values, select_hierarchy_and_nutau_norm,change_nutau_norm_settings, select_hierarchy

parser = ArgumentParser(description='''Runs the LLR optimizer-based analysis varying a number of systematic parameters
defined in settings.json file and saves the likelihood values for all
combination of hierarchies.''',
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-t','--template_settings',type=str,
                    metavar='JSONFILE', required = True,
                    help='''Settings related to the template generation and systematics.''')
parser.add_argument('-pd','--pseudo_data_settings',type=str,
                    metavar='JSONFILE',default=None,
                    help='''Settings for pseudo data templates, if desired to be different from template_settings.''')
parser.add_argument('-n','--ntrials',type=int, default = 1,
                    help="Number of trials to run")
parser.add_argument('-npoints','--npoints',type=int, default = 30,
                    help="Number of points in x axis ")
parser.add_argument('-s','--save-steps',action='store_true',default=False,
                    dest='save_steps',
                    help="Save all steps the optimizer takes.")
parser.add_argument('-o','--outfile',type=str,default='llh_data.json',metavar='JSONFILE',
                    help="Output filename.")
parser.add_argument('-v', '--verbose', action='count', default=None,
                    help='set verbosity level')
args = parser.parse_args()

set_verbosity(args.verbose)

#Read in the settings
template_settings = from_json(args.template_settings)
czbins = template_settings['binning']['czbins']

up_template_settings = copy.deepcopy(template_settings)
up_template_settings['params']['reco_vbwkde_evts_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_10_percent_up.hdf5'}
up_template_settings['params']['reco_mc_wt_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_100_percent_up.hdf5'}

down_template_settings = copy.deepcopy(template_settings)
down_template_settings['params']['pid_paramfile'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/pid/1X60_pid_down.json'}
down_template_settings['params']['reco_vbwkde_evts_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_10_percent_down.hdf5'}
down_template_settings['params']['reco_mc_wt_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_100_percent_down.hdf5'}

pseudo_data_settings = from_json(args.pseudo_data_settings) if args.pseudo_data_settings is not None else template_settings

# make sure that both pseudo data and template are using the same
# channel. Raise Exception and quit otherwise
channel = template_settings['params']['channel']['value']
if channel != pseudo_data_settings['params']['channel']['value']:
    error_msg = "Both template and pseudo data must have same channel!\n"
    error_msg += " pseudo_data_settings channel: '%s', template channel: '%s' "%(pseudo_data_settings['params']['channel']['value'],channel)
    raise ValueError(error_msg)

template_maker_down = TemplateMaker(get_values(down_template_settings['params']),
                                             **down_template_settings['binning'])
template_maker_up = TemplateMaker(get_values(up_template_settings['params']),
                               **up_template_settings['binning'])
template_maker = [template_maker_up,template_maker_down]


#store results from all the trials
trials = []
npoints = args.npoints
for itrial in xrange(1,args.ntrials+1):
    profile.info("start trial %d"%itrial)
    logging.info(">"*10 + "Running trial: %05d"%itrial + "<"*10)

    true_templates = []
    results = {}

    #for data_tag, data_nutau_norm in [('data_tau',1.0),('data_notau',0.0)]:
    for data_tag, data_nutau_norm in [('data_tau',1.0)]:

        results[data_tag] = {}
        results['MCErr'] = {}
        results['E_scale'] = {}

        # Get templates for different values of energy scale

        xval_escale = np.linspace(0.8,1.2,npoints)
        for e_scale_val in xval_escale:
            #print "e_scale_val = ", e_scale_val
            up_template_settings_Escale = copy.deepcopy(up_template_settings)
            up_template_settings_Escale['params']['energy_scale']['value'] = e_scale_val
            tmap = get_true_template(get_values(select_hierarchy(up_template_settings_Escale['params'],True)),template_maker)
            #print tmap , " "
            results['E_scale'][str(e_scale_val)] = tmap

        # Get templates from 8 MC sets
        for run_num in [50,60,61,64,65,70,71,72]:
            results[data_tag][str(run_num)] = {}
            results['MCErr'][str(run_num)] = {}
            print "run_num = ", run_num
            aeff_mc_file = '~/pisa/pisa/resources/aeff/1X%i_aeff_mc.hdf5' % run_num
            reco_mc_file_up = '~/pisa/pisa/resources/events/1X%i_weighted_aeff_joined_nu_nubar_100_percent_up.hdf5' % run_num
            reco_mc_file_down = '~/pisa/pisa/resources/events/1X%i_weighted_aeff_joined_nu_nubar_100_percent_down.hdf5' % run_num
            pid_param_file_up = '~/pisa/pisa/resources/pid/1X%i_pid.json' % run_num
            pid_param_file_down = '~/pisa/pisa/resources/pid/1X%i_pid_down.json' % run_num
            DH_up_template_settings = copy.deepcopy(up_template_settings)
            DH_up_template_settings['params']['aeff_weight_file']['value'] = aeff_mc_file
            DH_up_template_settings['params']['reco_mc_wt_file']['value'] = reco_mc_file_up
            DH_up_template_settings['params']['pid_paramfile']['value'] = pid_param_file_up 

            DH_down_template_settings = copy.deepcopy(down_template_settings)
            DH_down_template_settings['params']['aeff_weight_file']['value'] = aeff_mc_file
            DH_down_template_settings['params']['reco_mc_wt_file']['value'] = reco_mc_file_down
            DH_down_template_settings['params']['pid_paramfile']['value'] = pid_param_file_down

            DH_template_maker_down = TemplateMaker(get_values(DH_down_template_settings['params']), **DH_down_template_settings['binning'])
            DH_template_maker_up = TemplateMaker(get_values(DH_up_template_settings['params']), **DH_up_template_settings['binning'])
            DH_template_maker = [DH_template_maker_up,DH_template_maker_down]
        
            tmap = get_true_template(get_values(change_nutau_norm_settings(DH_up_template_settings['params'], 1.0 ,True)),DH_template_maker)

            #print "get_values = " , get_values(change_nutau_norm_settings(DH_up_template_settings['params'], 1.0 ,True))

            DH_MCMap_up = GetMCError(get_values(DH_up_template_settings['params']),DH_up_template_settings['binning']['ebins'],DH_up_template_settings['binning']['czbins'],reco_mc_file_up)
            DH_MCMap_down = GetMCError(get_values(DH_down_template_settings['params']),DH_down_template_settings['binning']['ebins'],DH_down_template_settings['binning']['czbins'],reco_mc_file_down)
            tmap_MC_up = DH_MCMap_up.get_mc_events_map(get_values(DH_up_template_settings['params']),reco_mc_file_up)
            tmap_MC_down = DH_MCMap_down.get_mc_events_map(get_values(DH_down_template_settings['params']),reco_mc_file_down)

            #print "tmap_MC_up = ", tmap_MC_up
            #print "tmap_MC_down = ", tmap_MC_down

            tmap_MC_up_down_combined = get_combined_map(tmap_MC_up,tmap_MC_down, channel= channel)
            template_MC_up = get_up_map(tmap_MC_up_down_combined, channel= channel)
            reflected_template_MC_down = get_flipped_down_map(tmap_MC_up_down_combined, channel= channel)

            flat_tmap_MC_up = flatten_map(template_MC_up,channel=channel)
            #print "flat_tmap_MC_up = ", flat_tmap_MC_up
            flat_tmap_MC_down = flatten_map(reflected_template_MC_down,channel=channel)
            tmap_MC_err = np.append(np.sqrt(flat_tmap_MC_up),np.sqrt(flat_tmap_MC_down))
            print "template_map = ", tmap
            #print "template_map_MC_events = ", tmap_MC_err
            true_templates.append(tmap)
            results[data_tag][str(run_num)] = tmap
            results['MCErr'][str(run_num)] = tmap_MC_err

            #tmap_up = DH_template_maker_up.get_template(get_values(change_nutau_norm_settings(DH_up_template_settings['params'], 1.0 ,True)))
            #tmap_down = DH_template_maker_down.get_template(get_values(change_nutau_norm_settings(DH_up_template_settings['params'], 1.0 ,True)))
            #print "template_map_up = ", tmap_up
            #print "template_map_down = ", tmap_down
            #results[data_tag][str(run_num)]['trck']['up'] = tmap_up['trck']['map']
            #results[data_tag][str(run_num)]['cscd']['up'] = tmap_up['cscd']['map']
            #results[data_tag][str(run_num)]['trck']['down'] = tmap_down['trck']['map']
            #results[data_tag][str(run_num)]['cscd']['down'] = tmap_down['cscd']['map']

    trials += [results]
    profile.info("stop trial %d"%itrial)

#Assemble output dict
output = {'trials' : trials,
          'template_settings_up' : up_template_settings,
          'template_settings_down' : down_template_settings}
if args.pseudo_data_settings is not None:
    output['pseudo_data_settings'] = pseudo_data_settings

    #And write to file
to_json(output,args.outfile)

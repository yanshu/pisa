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
import h5py
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.utils.log import logging, profile, physics, set_verbosity
from pisa.utils.jsons import from_json,to_json
from pisa.analysis.llr.LLHAnalysis_nutau import find_max_llh_bfgs
from pisa.analysis.stats.Maps import get_seed
import pisa.analysis.stats.Maps as Maps
from pisa.resources.resources import find_resource
from pisa.analysis.stats.Maps_nutau import get_up_map, get_flipped_down_map, get_burn_sample
from pisa.analysis.TemplateMaker_nutau import TemplateMaker
from pisa.utils.params import get_values, select_hierarchy_and_nutau_norm,change_nutau_norm_settings

parser = ArgumentParser(description='''Runs the LLR optimizer-based analysis varying a number of systematic parameters
defined in settings.json file and saves the likelihood values for all
combination of hierarchies.''',
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-t','--template_settings',type=str,
                    metavar='JSONFILE', required = True,
                    help='''Settings related to the template generation and systematics.''')
parser.add_argument('--burn_sample_file',metavar='FILE',type=str,
                    default='burn_sample/Matt_L5b_burn_sample_IC86_2_to_4.hdf5',
                    help='''HDF5 File containing burn sample.'
                    inverted corridor cut data''')
parser.add_argument('-m','--minimizer_settings',type=str,
                    metavar='JSONFILE', required = True,
                    help='''Settings related to the optimizer used in the LLR analysis.''')
parser.add_argument('-s','--save-steps',action='store_true',default=False,
                    dest='save_steps',
                    help="Save all steps the optimizer takes.")
parser.add_argument('-o','--outfile',type=str,default='llh_data.json',metavar='JSONFILE',
                    help="Output filename.")
parser.add_argument('-v', '--verbose', action='count', default=None,
                    help='set verbosity level')
parser.add_argument('--check_octant',action='store_true',default=False,
                    help="When theta23 LLH is multi-modal, check both octants for global minimum.")

args = parser.parse_args()

set_verbosity(args.verbose)

#Read in the settings
template_settings = from_json(args.template_settings)
ebins = template_settings['binning']['ebins']
czbins = template_settings['binning']['czbins']

up_template_settings = copy.deepcopy(template_settings)
up_template_settings['params']['reco_vbwkde_evts_file'] = {u'fixed': True, u'value': '~/work/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_10_percent_up.hdf5'}
up_template_settings['params']['reco_mc_wt_file'] = {u'fixed': True, u'value': '~/work/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_100_percent_up.hdf5'}

down_template_settings = copy.deepcopy(template_settings)
down_template_settings['params']['pid_paramfile'] = {u'fixed': True, u'value': '~/work/pisa/pisa/resources/pid/1X60_pid_down.json'}
down_template_settings['params']['reco_vbwkde_evts_file'] = {u'fixed': True, u'value': '~/work/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_10_percent_down.hdf5'}
down_template_settings['params']['reco_mc_wt_file'] = {u'fixed': True, u'value': '~/work/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_100_percent_down.hdf5'}

minimizer_settings  = from_json(args.minimizer_settings)

#Workaround for old scipy versions
import scipy
if scipy.__version__ < '0.12.0':
    logging.warn('Detected scipy version %s < 0.12.0'%scipy.__version__)
    if 'maxiter' in minimizer_settings:
      logging.warn('Optimizer settings for \"maxiter\" will be ignored')
      minimizer_settings.pop('maxiter')


template_maker_down = TemplateMaker(get_values(down_template_settings['params']),
                                 **down_template_settings['binning'])
template_maker_up = TemplateMaker(get_values(up_template_settings['params']),
                               **up_template_settings['binning'])
template_maker = [template_maker_up,template_maker_down]

anlys_ebins = template_settings['binning']['anlys_ebins']
anlys_bins = (anlys_ebins, czbins)

print "anlys_bins " , anlys_bins


# /////////////////////////////////////////////////////////////////////////
# Use burn sample, find the best matching template in the four combinations
# of hierarchy hypothesis and tau (notau). 
# /////////////////////////////////////////////////////////////////////////
results = {}

data_tag = 'burn_sample'
results[data_tag] = {}

# 1) get burn sample
burn_sample_maps = get_burn_sample(burn_sample_file= args.burn_sample_file, anlys_ebins= anlys_ebins, czbins= czbins, output_form ='map', cut_level='L6', channel=template_settings['params']['channel']['value'])

burn_sample_in_array = get_burn_sample(burn_sample_file= args.burn_sample_file, anlys_ebins = anlys_ebins, czbins = czbins, output_form = 'array', cut_level='L6', channel=template_settings['params']['channel']['value'])

print "     total no. of events in burn sample :", np.sum(burn_sample_in_array) 

print "burn_sample_in_array = ", burn_sample_in_array

# 2) find max llh (and best fit free params) from matching burn sample to templates.
for hypo_MH_tag, hypo_normal in [('hypo_N', True), ('hypo_I', False)]:
    results[data_tag][hypo_MH_tag] = {}
    rnd.seed(get_seed())
    init_nutau_norm = 1.0
    for hypo_tag, hypo_nutau_norm, nutau_norm_fix in [('hypo_free',init_nutau_norm, False)]:
    #for hypo_tag, hypo_nutau_norm, nutau_norm_fix in [('hypo_notau',0, True),('hypo_tau',1, True)]:

        physics.info("Finding best fit for %s under %s %s assumption" % (data_tag, hypo_MH_tag, hypo_tag))
        profile.info("start optimizer")
        llh_data = find_max_llh_bfgs(burn_sample_in_array,template_maker,change_nutau_norm_settings(template_settings['params'],
                                     hypo_nutau_norm,nutau_norm_fix, hypo_normal),
                                     minimizer_settings,args.save_steps,
                                     normal_hierarchy=hypo_normal,
                                     check_octant = args.check_octant)
        profile.info("stop optimizer")
        print "init_nutau_norm = ", init_nutau_norm

        #Store the LLH data
        results[data_tag][hypo_MH_tag][hypo_tag] = llh_data


#Assemble output dict
output = {'results' : results,
          'template_settings_up' : up_template_settings,
          'template_settings_down' : down_template_settings,
          'minimizer_settings' : minimizer_settings}

    #And write to file
to_json(output,args.outfile)

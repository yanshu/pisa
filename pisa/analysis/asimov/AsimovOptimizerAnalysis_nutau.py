#! /usr/bin/env python
#
# AsimovOptimizerAnalysis.py
#
# Runs the Asimov optimizer-based analysis. Based on
# LLROptimizerAnalysis, but the primary difference is that it only
# uses the one fiducial model template of the "pseudo data set" and
# fits to the templates finding the best fit template by maximizing
# the LLH / or minimizing the chisquare using the optimizer.
#
# author: Tim Arlen - tca3@psu.edu
#         Sebatian Boeser - sboeser@uni-mainz.de
#
# date:   02-July-2014
#

import numpy as np
import copy
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.utils.log import logging, tprofile, physics, set_verbosity
from pisa.utils.jsons import from_json,to_json
from pisa.analysis.llr.LLHAnalysis_nutau import find_max_llh_bfgs
from pisa.analysis.stats.Maps_nutau import get_asimov_data_fmap_up_down
from pisa.analysis.TemplateMaker_nutau import TemplateMaker
from pisa.utils.params import get_values, select_hierarchy_and_nutau_norm, change_nutau_norm_settings, fix_param, fix_all_params

parser = ArgumentParser(description='''Runs the Asimov optimizer-based analysis varying a number of systematic parameters
defined in settings.json file and saves the likelihood (or chisquare) values for all
combinations of hierarchies.''',
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-t','--template_settings',type=str,
                    metavar='JSONFILE', required = True,
                    help='''Settings related to the template generation and systematics.''')
parser.add_argument('-m','--minimizer_settings',type=str,
                    metavar='JSONFILE', required = True,
                    help='''Settings related to the optimizer used in the LLR analysis.''')
parser.add_argument('-pd','--pseudo_data_settings',type=str,
                    metavar='JSONFILE',default=None,
                    help='''Settings for pseudo data templates, if desired to be different from template_settings.''')
parser.add_argument('-s','--save-steps',action='store_true',default=False,
                    dest='save_steps',
                    help="Save all steps the optimizer takes.")
parser.add_argument('-o','--outfile',type=str,default='llh_data.json',metavar='JSONFILE',
                    help="Output filename.")
parser.add_argument('-v', '--verbose', action='count', default=None,
                    help='set verbosity level')
parser.add_argument('--check_octant',action='store_true',default=False,
                    help="When theta23 LLH is multi-modal, check both octants for global minimum.")
parser.add_argument('-f', default='',help='parameter to be fixed',dest='f_param')
args = parser.parse_args()

set_verbosity(args.verbose)

#Read in the settings
template_settings = from_json(args.template_settings)

if not args.f_param == '':
    template_settings['params'] = fix_param(template_settings['params'], args.f_param)
    print 'fixed param %s'%args.f_param
    logging.info('fixed param %s'%args.f_param)

czbins = template_settings['binning']['czbins']

up_template_settings = copy.deepcopy(template_settings)
up_template_settings['params']['reco_vbwkde_evts_file'] = {u'fixed': True, u'value': 'pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_10_percent_up.hdf5'}
up_template_settings['params']['reco_mc_wt_file'] = {u'fixed': True, u'value': 'pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_100_percent_up.hdf5'}

down_template_settings = copy.deepcopy(template_settings)
down_template_settings['params']['pid_paramfile'] = {u'fixed': True, u'value': 'pisa/resources/pid/1X60_pid_down.json'}
down_template_settings['params']['reco_vbwkde_evts_file'] = {u'fixed': True, u'value': 'pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_10_percent_down.hdf5'}
down_template_settings['params']['reco_mc_wt_file'] = {u'fixed': True, u'value': 'pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_100_percent_down.hdf5'}

minimizer_settings  = from_json(args.minimizer_settings)
pseudo_data_settings = from_json(args.pseudo_data_settings) if args.pseudo_data_settings is not None else template_settings

# Parse the metric to be used, fixed here
metric_name = 'llh'

# Workaround for old scipy versions
import scipy
if scipy.__version__ < '0.12.0':
    logging.warn('Detected scipy version %s < 0.12.0'%scipy.__version__)
    if 'maxiter' in minimizer_settings:
      logging.warn('Optimizer settings for \"maxiter\" will be ignored')
      minimizer_settings.pop('maxiter')


# Make sure that both pseudo data and template are using the same
# channel. Raise Exception and quit otherwise
channel = template_settings['params']['channel']['value']
if channel != pseudo_data_settings['params']['channel']['value']:
    error_msg = "Both template and pseudo data must have same channel!\n"
    error_msg += " pseudo_data_settings chan: '%s', template chan: '%s' "%(pseudo_data_settings['params']['channel']['value'],channel)
    raise ValueError(error_msg)


template_maker_down = TemplateMaker(get_values(down_template_settings['params']),
                                 **down_template_settings['binning'])
template_maker_up = TemplateMaker(get_values(up_template_settings['params']),
                               **up_template_settings['binning'])
template_maker = [template_maker_up,template_maker_down]
pseudo_data_template_maker = [template_maker_up,template_maker_down]

# //////////////////////////////////////////////////////////////////////
# Generate two pseudo-data experiments (one for each hierarchy),
# and for each experiment, find the best matching template in each
# of the hierarchy hypotheses.
# //////////////////////////////////////////////////////////////////////
results = {}
data_normal = True
hypo_normal = True
data_tag, data_nutau_norm = ('data_tau',1.0)

results[data_tag] = {}
# 1) get "Asimov" average fiducial template:
asimov_fmap = get_asimov_data_fmap_up_down(pseudo_data_template_maker,
                            get_values(select_hierarchy_and_nutau_norm(pseudo_data_settings['params'],
                            normal_hierarchy=data_normal,nutau_norm_value=data_nutau_norm)),
                            channel=channel)

# 2) find max llh or min chisquare (and best fit free params) from matching pseudo data
#    to templates, fixing mu to 0
hypo_tag, hypo_nutau_norm, nutau_norm_fix = ('hypo_notau',0, True)

physics.info("Finding best fit for %s under %s assumption"%(data_tag,hypo_tag))
tprofile.info("start optimizer")
llh_data = find_max_llh_bfgs(asimov_fmap,template_maker,change_nutau_norm_settings(template_settings['params'],
                             hypo_nutau_norm,nutau_norm_fix, hypo_normal),
                             minimizer_settings,args.save_steps,
                             normal_hierarchy=hypo_normal,
                             check_octant = args.check_octant)
tprofile.info("stop optimizer")

#Store the LLH data
results[data_tag][hypo_tag] = llh_data

# just calculate llh for denominator which doesnt require a minimization, because the input values of the nuisance paras are equal to their best fit value
hypo_tag, hypo_nutau_norm, nutau_norm_fix = ('hypo_tau',1, True)
physics.info("Calculating llh without fitting for ")
template_settings['params'] = fix_all_params(template_settings['params'])
llh_data = find_max_llh_bfgs(asimov_fmap,template_maker,change_nutau_norm_settings(template_settings['params'],
                             hypo_nutau_norm,nutau_norm_fix, hypo_normal),
                             minimizer_settings,args.save_steps,
                             normal_hierarchy=hypo_normal,
                             check_octant = args.check_octant)
#Store the LLH data
results[data_tag][hypo_tag] = llh_data

# Assemble output dict
output = {'results' : results,
          'template_settings_up' : up_template_settings,
          'template_settings_down' : down_template_settings,
          'minimizer_settings' : minimizer_settings}

if args.pseudo_data_settings is not None:
    output['pseudo_data_settings'] = pseudo_data_settings

# And write to file
to_json(output,args.outfile)

print results
# Finally, report on what we have
q0 = 2*(np.min(results['data_tau']['hypo_notau']['llh']) - np.min(results['data_tau']['hypo_tau']))
print 'sqrt(q0) = %.4f'%(np.sqrt(q0))

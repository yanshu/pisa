#! /usr/bin/env python
#
# LLROptimizerAnalysis.py
#
# Runs the LLR optimizer-based LLR analysis
#
# author: Tim Arlen - tca3@psu.edu
#         Sebatian Boeser - sboeser@uni-mainz.de
#
# date:   02-July-2014
#

import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.utils.log import logging, profile, physics, set_verbosity
from pisa.utils.jsons import from_json,to_json
from pisa.analysis.llr.LLHAnalysis import find_max_llh_bfgs
from pisa.analysis.stats.Maps import get_pseudo_data_fmap, get_seed
from pisa.analysis.TemplateMaker import TemplateMaker
from pisa.utils.params import get_values, select_hierarchy

parser = ArgumentParser(description='''Runs the LLR optimizer-based analysis varying a number of systematic parameters
defined in settings.json file and saves the likelihood values for all
combination of hierarchies.''',
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-t_f_free','--template_settings_f_floating',type=str,
                    metavar='JSONFILE', required = True,
                    default='~/pisa/pisa/resources/settings/template_settings/nutau/V36_par_2syst_e39_cz20_quick_f_floating.json',
                    help='''Settings (f floating) related to the template generation and systematics.''')
parser.add_argument('-t_f_fixed_0','--template_settings_f_fixed_0',type=str,
                    metavar='JSONFILE', required = True,
                    default='~/pisa/pisa/resources/settings/template_settings/nutau/V36_par_2syst_e39_cz20_quick_f_fixed_0.json',
                    help='''Settings (f fixed to 0) related to the template generation and systematics.''')
parser.add_argument('-m','--minimizer_settings',type=str,
                    metavar='JSONFILE', required = True,
                    help='''Settings related to the optimizer used in the LLR analysis.''')
parser.add_argument('-pd_tau','--pseudo_data_tau_settings',type=str,
                    metavar='JSONFILE', required = True,
                    default='~/pisa/pisa/resources/settings/template_settings/nutau/V36_mc_2syst_e39_cz20_quick_tau.json',
                    help='''Settings for pseudo data templates(tau,f=1), if desired to be different from template_settings.''')
parser.add_argument('-pd_notau','--pseudo_data_notau_settings',type=str,
                    metavar='JSONFILE', required = True,
                    default='~/pisa/pisa/resources/settings/template_settings/nutau/V36_mc_2syst_e39_cz20_quick_notau.json',
                    help='''Settings for pseudo data templates(notau,f=0), if desired to be different from template_settings.''')
parser.add_argument('-n','--ntrials',type=int, default = 1,
                    help="Number of trials to run")
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
template_settings_f_floating = from_json(args.template_settings_f_floating)
template_settings_f_fixed_0 = from_json(args.template_settings_f_fixed_0)
minimizer_settings  = from_json(args.minimizer_settings)
pseudo_data_tau_settings = from_json(args.pseudo_data_tau_settings) if args.pseudo_data_tau_settings is not None else template_settings_f_floating
pseudo_data_notau_settings = from_json(args.pseudo_data_notau_settings) if args.pseudo_data_notau_settings is not None else template_settings_f_floating
## pseudo_data_settings should be different with template_settings.. need to change?

#Workaround for old scipy versions
import scipy
if scipy.__version__ < '0.12.0':
    logging.warn('Detected scipy version %s < 0.12.0'%scipy.__version__)
    if 'maxiter' in minimizer_settings:
      logging.warn('Optimizer settings for \"maxiter\" will be ignored')
      minimizer_settings.pop('maxiter')


# make sure that both pseudo data and template are using the same
# channel. Raise Exception and quit otherwise
channel = template_settings_f_floating['params']['channel']['value']
channel_f_fixed_0 = template_settings_f_fixed_0['params']['channel']['value']
if channel != channel_f_fixed_0:
    error_msg = "template_settings_f_floating and template_settings_f_fixed_0 must have the same channel!\n"
    error_msg += " template_settings_f_floating: '%s', template_settings_f_fixed_0: '%s' " %(channel,channel_f_fixed_0)
    raise ValueError(error_msg)

if channel != pseudo_data_tau_settings['params']['channel']['value'] or channel != pseudo_data_notau_settings['params']['channel']['value']:
    error_msg = "Both template and pseudo data must have same channel!\n"
    error_msg += " pseudo_data_tau_settings chan: '%s', template chan: '%s' "%(pseudo_data_tau_settings['params']['channel']['value'],channel)
    error_msg += " pseudo_data_notau_settings chan: '%s', template chan: '%s' "%(pseudo_data_notau_settings['params']['channel']['value'],channel)
    raise ValueError(error_msg)


template_maker_f_floating = TemplateMaker(get_values(template_settings_f_floating['params']),
                               **template_settings_f_floating['binning'])
template_maker_f_fixed_0 = TemplateMaker(get_values(template_settings_f_fixed_0['params']),
                               **template_settings_f_fixed_0['binning'])
if args.pseudo_data_notau_settings:
    pseudo_data_notau_template_maker = TemplateMaker(get_values(pseudo_data_notau_settings['params']),
                                               **pseudo_data_notau_settings['binning'])
else:
    pseudo_data_notau_template_maker = template_maker_f_floating

if args.pseudo_data_tau_settings:
    pseudo_data_tau_template_maker = TemplateMaker(get_values(pseudo_data_tau_settings['params']),
                                               **pseudo_data_tau_settings['binning'])
else:
    pseudo_data_tau_template_maker = template_maker_f_floating

#store results from all the trials
trials = []
for itrial in xrange(1,args.ntrials+1):
    profile.info("start trial %d"%itrial)
    logging.info(">"*10 + "Running trial: %05d"%itrial + "<"*10)


    # //////////////////////////////////////////////////////////////////////
    # For each trial, generate two pseudo-data experiemnts (one for each
    # hierarchy), and for each find the best matching template in each of the
    # hierarchy hypothesis.
    # //////////////////////////////////////////////////////////////////////
    results = {}

    data_normal = True
    hypo_normal = True

    results['data_tau'] = {}
    # 0) get a random seed and store with the data
    results['data_tau']['seed'] = get_seed()
    logging.info("  RNG seed: %ld"%results['data_tau']['seed'])
    # 1) get a pseudo data fmap (f=1) from fiducial model (best fit vals of params).
    fmap_tau = get_pseudo_data_fmap(pseudo_data_tau_template_maker,
                    get_values(select_hierarchy(pseudo_data_tau_settings['params'],
                                                normal_hierarchy=data_normal)),
                                seed=results['data_tau']['seed'],chan=channel)
    results['data_notau'] = {}
    # 0') get a random seed and store with the data
    results['data_notau']['seed'] = get_seed()
    logging.info("  RNG seed: %ld"%results['data_notau']['seed'])
    # 1') get a pseudo data fmap (f=0) from fiducial model (best fit vals of params).
    fmap_notau = get_pseudo_data_fmap(pseudo_data_notau_template_maker,
                    get_values(select_hierarchy(pseudo_data_notau_settings['params'],
                                                normal_hierarchy=data_normal)),
                                seed=results['data_notau']['seed'],chan=channel)

    # 2) find max llh (and best fit free params) from matching pseudo data
        #    to templates.
    profile.info("start optimizer")
    physics.info("Finding best fit for data_tau(f=1) under template(f floating) ")
    llh_data_1 = find_max_llh_bfgs(fmap_tau,template_maker_f_floating,template_settings_f_floating['params'],
                                    minimizer_settings,args.save_steps,
                                    normal_hierarchy=hypo_normal)
    physics.info("Finding best fit for data_notau (f=0) under template(f floating) ")
    llh_data_2 = find_max_llh_bfgs(fmap_notau,template_maker_f_floating,template_settings_f_floating['params'],
                                    minimizer_settings,args.save_steps,
                                    normal_hierarchy=hypo_normal)
    physics.info("Finding best fit for data_tau (f=1) under template(f=0,fixed) ")
    llh_data_3 = find_max_llh_bfgs(fmap_tau,template_maker_f_fixed_0,template_settings_f_fixed_0['params'],
                                    minimizer_settings,args.save_steps,
                                    normal_hierarchy=hypo_normal)
    physics.info("Finding best fit for data_notau (f=0) under template(f=0,fixed) ")
    llh_data_4 = find_max_llh_bfgs(fmap_notau,template_maker_f_fixed_0,template_settings_f_fixed_0['params'],
                                    minimizer_settings,args.save_steps,
                                    normal_hierarchy=hypo_normal)
    profile.info("stop optimizer")
    #Store the LLH data
    results['data_tau']['hypo_free'] = llh_data_1
    results['data_notau']['hypo_free'] = llh_data_2
    results['data_tau']['hypo_notau'] = llh_data_3
    results['data_notau']['hypo_notau'] = llh_data_4

    #Store this trial
    trials += [results]
    profile.info("stop trial %d"%itrial)

#Assemble output dict
output = {'trials' : trials,
          'template_settings_f_floating' : template_settings_f_floating,
          'template_settings_f_fixed_0' : template_settings_f_fixed_0,
          'minimizer_settings' : minimizer_settings}
if args.pseudo_data_tau_settings is not None:
    output['pseudo_data_tau_settings'] = pseudo_data_tau_settings
if args.pseudo_data_notau_settings is not None:
    output['pseudo_data_notau_settings'] = pseudo_data_notau_settings

    #And write to file
to_json(output,args.outfile)

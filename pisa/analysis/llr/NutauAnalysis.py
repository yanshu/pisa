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
import random as rnd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.utils.log import logging, profile, physics, set_verbosity
from pisa.utils.jsons import from_json,to_json
from pisa.analysis.llr.LLHAnalysis import find_max_llh_bfgs
from pisa.analysis.stats.Maps import get_pseudo_data_fmap, get_seed
from pisa.analysis.TemplateMaker import TemplateMaker
from pisa.utils.params import get_values, select_hierarchy_and_nutau_norm,change_nutau_norm_settings

parser = ArgumentParser(description='''Runs the LLR optimizer-based analysis varying a number of systematic parameters
defined in settings.json file and saves the likelihood values for all
combination of hierarchies.''',
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
template_settings = from_json(args.template_settings)
minimizer_settings  = from_json(args.minimizer_settings)
pseudo_data_settings = from_json(args.pseudo_data_settings) if args.pseudo_data_settings is not None else template_settings

#Workaround for old scipy versions
import scipy
if scipy.__version__ < '0.12.0':
    logging.warn('Detected scipy version %s < 0.12.0'%scipy.__version__)
    if 'maxiter' in minimizer_settings:
      logging.warn('Optimizer settings for \"maxiter\" will be ignored')
      minimizer_settings.pop('maxiter')


# make sure that both pseudo data and template are using the same
# channel. Raise Exception and quit otherwise
channel = template_settings['params']['channel']['value']
if channel != pseudo_data_settings['params']['channel']['value']:
    error_msg = "Both template and pseudo data must have same channel!\n"
    error_msg += " pseudo_data_settings chan: '%s', template chan: '%s' "%(pseudo_data_settings['params']['channel']['value'],channel)
    raise ValueError(error_msg)


template_maker = TemplateMaker(get_values(template_settings['params']),
                               **template_settings['binning'])
if args.pseudo_data_settings:
    pseudo_data_template_maker = TemplateMaker(get_values(pseudo_data_settings['params']),
                                               **pseudo_data_settings['binning'])
else:
    pseudo_data_template_maker = template_maker

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
    for data_tag, data_nutau_norm in [('data_tau',1.0),('data_notau',0.0)]:

        results[data_tag] = {}
        # 0) get a random seed and store with the data
        results[data_tag]['seed'] = get_seed()
        logging.info("  RNG seed: %ld"%results[data_tag]['seed'])
        # 1) get a pseudo data fmap from fiducial model (best fit vals of params).
        fmap = get_pseudo_data_fmap(pseudo_data_template_maker,
                        get_values(select_hierarchy_and_nutau_norm(pseudo_data_settings['params'],
                                   normal_hierarchy=data_normal,nutau_norm_value=data_nutau_norm)),
                                    seed=results[data_tag]['seed'],chan=channel)

        # 2) find max llh (and best fit free params) from matching pseudo data
        #    to templates.
        rnd.seed(get_seed())
        init_nutau_norm = rnd.uniform(-0.7,3)
        for hypo_tag, hypo_nutau_norm, nutau_norm_fix in [('hypo_free',init_nutau_norm, False),('hypo_notau',0, True)]:

            physics.info("Finding best fit for %s under %s assumption"%(data_tag,hypo_tag))
            profile.info("start optimizer")
            llh_data = find_max_llh_bfgs(fmap,template_maker,change_nutau_norm_settings(template_settings['params'],
                                         hypo_nutau_norm,nutau_norm_fix),
                                         minimizer_settings,args.save_steps,
                                         normal_hierarchy=hypo_normal)
            print "injected initial nutau_norm: ",init_nutau_norm
            profile.info("stop optimizer")

            #Store the LLH data
            results[data_tag][hypo_tag] = llh_data


    #Store this trial
    trials += [results]
    profile.info("stop trial %d"%itrial)

#Assemble output dict
output = {'trials' : trials,
          'template_settings' : template_settings,
          'minimizer_settings' : minimizer_settings}
if args.pseudo_data_settings is not None:
    output['pseudo_data_settings'] = pseudo_data_settings

    #And write to file
to_json(output,args.outfile)

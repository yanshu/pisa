#! /usr/bin/env python
#
# ScanAnalysisNutau_ratio.py
#
# Runs a brute-force scan LLR analysis
#
# authors: Feifei Huang - fxh140@psu.edu
#          Sebatian Boeser - sboeser@uni-mainz.de
#

import numpy as np
import copy
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.utils.log import logging, profile, physics, set_verbosity
from pisa.utils.jsons import from_json,to_json
from pisa.analysis.stats.Maps import get_seed
from pisa.analysis.stats.Maps_nutau import get_pseudo_data_fmap, get_true_template
from pisa.analysis.scan.Scan_nutau import find_max_grid
from pisa.analysis.TemplateMaker_nutau import TemplateMaker
from pisa.utils.params import get_values, select_hierarchy_and_nutau_norm, change_nutau_norm_settings
import random as rnd

parser = ArgumentParser(
    description='''Runs a brute-force scan analysis varying a number of systematic parameters
    defined in settings.json file and saves the likelihood values for all
    combination of hierarchies.''',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-t', '--template_settings', type=str,
    metavar='JSONFILE', required = True,
    help='''Settings (f floating) related to the template generation and systematics.''')
parser.add_argument(
    '-pd', '--pseudo_data_settings', type=str,
    metavar='JSONFILE', required = False,
    help='''Settings for pseudo data templates(tau,f=0), if desired to be different from template_settings.''')
parser.add_argument(
    '-g', '--grid_settings',type=str,
    metavar='JSONFILE', required=True,
    help='''Settings for the grid search.''')
parser.add_argument(
    '-n', '--ntrials', type=int,
    default=1, help="Number of trials to run")
sselect = parser.add_mutually_exclusive_group(required=False)
sselect.add_argument(
    '--save-steps', action='store_true', default=True,
    dest='save_steps', help="Save all steps")
sselect.add_argument(
    '--no-save-steps', action='store_false',
    default=False, dest='save_steps',
   help="Save just the maximum")
parser.add_argument(
    '-o', '--outfile', type=str,
    default='llh_data.json', metavar='JSONFILE',
    help="Output filename.")
parser.add_argument(
    '-v', '--verbose', action='count',
    default=None, help='set verbosity level')
args = parser.parse_args()

set_verbosity(args.verbose)

# Read in the settings
template_settings = from_json(args.template_settings)
czbins = template_settings['binning']['czbins']
pseudo_data_settings = from_json(args.pseudo_data_settings) if args.pseudo_data_settings is not None else template_settings
grid_settings  = from_json(args.grid_settings)

#store results from all the trials
trials = []

if args.pseudo_data_settings:
    pseudo_data_template_maker = TemplateMaker(get_values(pseudo_data_settings['params']),
                                               **pseudo_data_settings['binning'])
else:
    pseudo_data_template_maker = template_maker

# Make sure that both pseudo data and template are using the same channel.
# Raise Exception and quit otherwise
channel = template_settings['params']['channel']['value']
if channel != pseudo_data_settings['params']['channel']['value']:
    error_msg = "template_settings and pseudo_data_settings must have the same channel!\n"
    error_msg += " template_settings: '%s', pseudo_data_settings: '%s' " %(channel,pseudo_data_settings['params']['channel']['value'])
    raise ValueError(error_msg)

# perform n trials
trials = []
for itrial in xrange(1,args.ntrials+1):
    results = {}

    profile.info("start trial %d"%itrial)
    logging.info(">"*10 + "Running trial: %05d"%itrial + "<"*10)

    # --- Get the data, or psudeodata, and store it in fmap

    # if read fmap from json
    if args.read_fmap_from_json != '':
        file = from_json(args.read_fmap_from_json)
        fmap = file['fmap']
    else:
        # Asimov dataset (exact expecation values)
        if args.t_stat == 'asimov':
            fmap = get_true_template(get_values(pseudo_data_settings['params']),
                                                pseudo_data_template_maker,
                                                num_data_events = None
                    )
            
        # Real data
        elif args.data_file:
            logging.info('Running on real data! (%s)'%args.data_file)
            physics.info('Running on real data! (%s)'%args.data_file)
            fmap = get_burn_sample_maps(file_name=args.data_file, anlys_ebins = anlys_ebins, czbins = czbins, output_form = 'array', channel=channel, pid_remove=template_settings['params']['pid_remove']['value'], pid_bound=template_settings['params']['pid_bound']['value'], sim_version=template_settings['params']['sim_ver']['value'])
        # Randomly sampled (poisson) data
        else:
            if args.seed:
                results['seed'] = int(args.seed)
            else:
                results['seed'] = get_seed()
            logging.info("  RNG seed: %ld"%results['seed'])
            if args.fluct == 'poisson':
                fmap = get_pseudo_data_fmap(pseudo_data_template_maker,
                                            get_values(pseudo_data_settings['params']),
                                            seed=results['seed'],
                                            channel=channel
                                            )
            elif args.fluct == 'model_stat':
                fmap = get_stat_fluct_map(pseudo_data_template_maker,
                                            get_values(pseudo_data_settings['params']),
                                            seed=results['seed'],
                                            channel=channel
                                            )
            else:
                raise Exception('psudo data fluctuation method not implemented!')

    # if want to save fmap to json
    if args.save_fmap_to_json != '':
        to_json({'fmap':fmap}, args.save_fmap_to_json)

    # get the total no. of events in fmap
    num_data_events = np.sum(fmap)

    # 2) find max llh (and best fit free params) from matching pseudo data
    #    to templates.
    profile.info("start scan")
    hypo_params = change_nutau_norm_settings(
        template_settings['params'],
        hypo_nutau_norm,nutau_norm_fix,hypo_normal
    )
    # common setings
    kwargs = {'normal_hierarchy':not(args.inv_h_hypo),'check_octant':args.check_octant, 'save_steps':args.save_steps}
    largs = [fmap, template_maker, None , grid_settings]
    largs[2] = change_settings(template_settings['params'],scan_param,pseudo_data_settings['params'][scan_param]['value'], False)
    res, chi2, chi2_p, dof = find_max_grid(blind_fit, num_data_events, use_chi2=args.use_chi2, use_rnd_init=args.use_rnd_init, *largs, **kwargs)
    res['chi2'] = [chi2]
    res['chi2_p'] = [chi2_p]
    res['dof'] = [dof]
    fit_results.append(res)
    print "chi2, chi2_p, dof = ", chi2, " ", chi2_p , " ", dof
    fit_results = find_max_grid(fmap=fmap,
                             template_maker=template_maker,
                             params=hypo_params,
                             grid_settings=grid_settings,
                             save_steps=args.save_steps,
                             normal_hierarchy=hypo_normal)
    profile.info("stop scan")

    # store fit results
    results['fit_results'] = fit_results
    # store the value of interest, q = -2log(lh[0]/lh[1]) , llh here is already negative, so no need for the minus sign
    if not any([args.on, args.od]):
        results['q'] = np.array([2*(llh-fit_results[1]['llh'][0]) for llh in fit_results[0]['llh']])
        physics.info('found q values %s'%results['q'])
        physics.info('sqrt(q) = %s'%np.sqrt(results['q']))

    # save minimizer settings info
    if args.use_chi2:
        logging.info('Using chi2 for minimizer')
        results['use_chi2_in_minimizing'] = 'True'
    else:
        logging.info('Using -llh for minimizer')
        results['use_chi2_in_minimizing'] = 'False'
    if args.use_rnd_init:
        logging.info('Using random initial sys values for minimizer')
        results['use_rnd_init'] = 'True'
    else:
        logging.info('Using always nominal values as initial values for minimizer')
        results['use_rnd_init'] = 'False'

    # save PISA settings info
    if args.use_hist_PISA:
        results['PISA'] = 'hist'
    else:
        results['PISA'] = 'MC'

    # Store this trial
    trials += [results]
    profile.info("stop trial %d"%itrial)

# Assemble output dict
output = {'trials' : trials,
          'template_settings' : template_settings,
          'grid_settings' : grid_settings}
output['pseudo_data_settings'] = pseudo_data_settings
# And write to file
to_json(output, args.outfile)

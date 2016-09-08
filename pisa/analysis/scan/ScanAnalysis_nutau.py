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
from pisa.analysis.stats.Maps_nutau import get_pseudo_data_fmap, get_burn_sample_maps, get_true_template, get_stat_fluct_map
from pisa.analysis.scan.Scan_nutau import find_max_grid
from pisa.utils.params import get_values, select_hierarchy, select_hierarchy_and_nutau_norm, change_nutau_norm_settings, change_settings, float_param
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
parser.add_argument('-ts', '--test-statistics',choices=['llr', 'profile', 'asimov'], default='llr', dest='t_stat', help='''Choose test statistics from llh, profile or asimov''')
parser.add_argument('--data','--data_file',metavar='FILE',type=str, dest='data_file',
                    default='', help='HDF5 File containing real data, can be only be burn sample file for this scan!!!')
parser.add_argument('--read_fmap_from_json', default='', dest='read_fmap_from_json', help='''Read fmap from json.''')
parser.add_argument('--save_fmap_to_json', default='', dest='save_fmap_to_json', help='''Save fmap to json.''')
parser.add_argument('--seed', default='',help='provide a fixed seed for pseudo data sampling',dest='seed')
parser.add_argument('--mu-data', default=1.0, dest='mu_data', help='''nu tau normalization for the psudodata''')
parser.add_argument('--mu-hypo', default=0.0, dest='mu_hypo', help='''nu tau normalization for the test hypothesis''')
parser.add_argument('--float_param', default='', dest='float_param', help='''make a niusance parameter float''')

parser.add_argument('--inv-mh-data', action='store_true', default=False, dest='inv_h_data', help='''invert mass hierarchy in psudodata''')
parser.add_argument('--inv-mh-hypo', action='store_true', default=False, dest='inv_h_hypo', help='''invert mass hierarchy in test hypothesis''')
parser.add_argument('--fluct', default='poisson', help='''What random sampling to be used for psudo data, this is usually just poisson, but can also be set to model_stat to gaussian fluctuate the model expectations by theiruncertainty''')
parser.add_argument('--use_hist_PISA',action='store_true',default=False, help='''Use event-by-event PISA; otherwise, use histogram-based PISA''') 
parser.add_argument('--use_chi2',action='store_true',default=False, help='''Use chi2 instead of -llh for the minimizer.''') 
parser.add_argument('--use_rnd_init',action='store_true',default=False, help='''Use random initial values for the minimizer.''') 
parser.add_argument('--check_octant',action='store_true',default=False,
                    help="When theta23 LLH is multi-modal, check both octants for global minimum.")
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

# the below cases do not make much sense, therefor complain if the user tries to use them
if args.t_stat == 'asimov':
    assert(args.data_file == '')
if args.data_file:
    assert(args.pseudo_data_settings == None)
    assert(args.mu_data == 1.0)
    template_settings['params']['livetime']['value'] = 0.045 

if not args.float_param == '':
    template_settings['params'] = float_param(template_settings['params'], args.float_param)
    print 'make param %s float'%(args.float_param)

ebins = template_settings['binning']['ebins']
anlys_ebins = template_settings['binning']['anlys_ebins']
czbins = template_settings['binning']['czbins']
anlys_bins = (anlys_ebins, czbins)
# one sanity check for background scale
if template_settings['params']['use_atmmu_f']['value'] == False:
    assert(template_settings['params']['atmmu_f']['fixed'] == True)
else:
    assert(template_settings['params']['atmos_mu_scale']['fixed'] == True)

if args.use_chi2:
    logging.info('Using chi2 for minimizer')
else:
    logging.info('Using -llh for minimizer')

if args.use_hist_PISA:
    logging.info('Using pisa.analysis.TemplateMaker_nutau, i.e. hist-based PISA')
    from pisa.analysis.TemplateMaker_nutau import TemplateMaker
    pisa_mode = 'hist'
else:
    logging.info('Using pisa.analysis.TemplateMaker_MC, i.e. MC-based PISA')
    from pisa.analysis.TemplateMaker_MC import TemplateMaker
    pisa_mode = 'event'
#store results from all the trials
trials = []

template_settings['params'] = select_hierarchy(template_settings['params'],normal_hierarchy=not(args.inv_h_hypo))
pseudo_data_settings['params'] = select_hierarchy_and_nutau_norm(pseudo_data_settings['params'],normal_hierarchy=not(args.inv_h_data),nutau_norm_value=float(args.mu_data))

template_maker = TemplateMaker(get_values(template_settings['params']),
                                    **template_settings['binning'])
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
    fit_results = []

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
            fmap = get_burn_sample_maps(file_name=args.data_file, anlys_ebins = anlys_ebins, czbins = czbins, output_form = 'array', channel=channel, pid_remove=template_settings['params']['pid_remove']['value'], pid_bound=template_settings['params']['pid_bound']['value'], further_bdt_cut=template_settings['params']['further_bdt_cut']['value'], sim_version=template_settings['params']['sim_ver']['value'])
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
    # common setings
    kwargs = {'normal_hierarchy':not(args.inv_h_hypo),'check_octant':args.check_octant, 'save_steps':args.save_steps}
    largs = [fmap, template_maker, template_settings['params'] , grid_settings]
    result = find_max_grid(num_data_events, use_chi2=args.use_chi2, use_rnd_init=args.use_rnd_init, *largs, **kwargs)
    #res['chi2'] = [chi2]
    #res['chi2_p'] = [chi2_p]
    #res['dof'] = [dof]
    fit_results.append(result)
    profile.info("stop scan")

    # store fit results
    results['fit_results'] = fit_results

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

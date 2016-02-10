#! /usr/bin/env python
#
# NutauAnalysis.py
#
# Computes q for different test statistics for the nutau appearance search analysis
#
# author: Philipp Eller - pde3@psu.edu
#         Feifei Huang - fxh140@psu.edu
#
# date:   8-Feb-2016

import numpy as np
import copy
import random as rnd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pisa.utils.log import logging, profile, physics, set_verbosity
from pisa.utils.jsons import from_json,to_json
from pisa.analysis.llr.LLHAnalysis_nutau import find_max_llh_bfgs
from pisa.analysis.stats.Maps import get_seed
from pisa.analysis.stats.Maps_nutau import get_pseudo_data_fmap, get_asimov_data_fmap_up_down
from pisa.analysis.TemplateMaker_nutau import TemplateMaker
from pisa.utils.params import get_values, select_hierarchy_and_nutau_norm, change_nutau_norm_settings, fix_param, fix_all_params

# --- parse command line arguments ---
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
parser.add_argument('--check_octant',action='store_true',default=False,
                    help="When theta23 LLH is multi-modal, check both octants for global minimum.")
parser.add_argument('-ts', '--test-statistics',choices=['llr', 'profile', 'asimov'], default='llr', dest='t_stat')
parser.add_argument('--mu-data', default=1, dest='mu_data')
parser.add_argument('--mu-hypo', default=0, dest='mu_hypo')
parser.add_argument('--inv-mh-data', action='store_true', default=False, dest='inv_h_data')
parser.add_argument('--inv-mh-hypo', action='store_true', default=False, dest='inv_h_hypo')
parser.add_argument('-f', default='',help='parameter to be fixed',dest='f_param')
args = parser.parse_args()
set_verbosity(args.verbose)

# Read in the settings
template_settings = from_json(args.template_settings)

# fix a nuisance parameter if requested
if not args.f_param == '':
    template_settings['params'] = fix_param(template_settings['params'], args.f_param)
    print 'fixed param %s'%args.f_param
    logging.info('fixed param %s'%args.f_param)

up_template_settings = copy.deepcopy(template_settings)
up_template_settings['params']['reco_vbwkde_evts_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_10_percent_up.hdf5'}
up_template_settings['params']['reco_mc_wt_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_100_percent_up.hdf5'}
down_template_settings = copy.deepcopy(template_settings)
down_template_settings['params']['pid_paramfile'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/pid/1X60_pid_down.json'}
down_template_settings['params']['reco_vbwkde_evts_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_10_percent_down.hdf5'}
down_template_settings['params']['reco_mc_wt_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_100_percent_down.hdf5'}
minimizer_settings  = from_json(args.minimizer_settings)
pseudo_data_settings = from_json(args.pseudo_data_settings) if args.pseudo_data_settings is not None else template_settings

# Workaround for old scipy versions
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


template_maker_down = TemplateMaker(get_values(down_template_settings['params']),
                                    **down_template_settings['binning'])
template_maker_up = TemplateMaker(get_values(up_template_settings['params']),
                                    **up_template_settings['binning'])
template_maker = [template_maker_up, template_maker_down]
pseudo_data_template_maker = [template_maker_up, template_maker_down]

# store results from all the trials
trials = []
for itrial in xrange(1,args.ntrials+1):

    profile.info("start trial %d"%itrial)
    logging.info(">"*10 + "Running trial: %05d"%itrial + "<"*10)

    results = {}
    
    # save settings
    results['test_statistics'] = args.t_stat
    if not args.t_stat == 'asimov':
        results['mu_data'] = float(args.mu_data)
    if not args.t_stat == 'llr':
        results['mu_hypo'] = float(args.mu_hypo)
    results['data_mass_hierarchy'] = 'inverted' if args.inv_h_data else 'normal'
    results['hypo_mass_hierarchy'] = 'inverted' if args.inv_h_hypo else 'normal'


    if args.t_stat == 'asimov':
        # 1) get "Asimov" data fmap, the exact bin expectations
        fmap = get_asimov_data_fmap_up_down(pseudo_data_template_maker,
                                                get_values(select_hierarchy_and_nutau_norm(pseudo_data_settings['params'],
                                                            normal_hierarchy=not(args.inv_h_data),
                                                            nutau_norm_value=1.0)
                                                ),
                                                channel=channel
                                            )
    else:
        results['seed'] = get_seed()
        # 1) get a pseudo data fmap, randomly sampled from a poisson distribution around the exact bin expecatations
        logging.info("  RNG seed: %ld"%results['seed'])
        fmap = get_pseudo_data_fmap(pseudo_data_template_maker,
                                    get_values(select_hierarchy_and_nutau_norm(pseudo_data_settings['params'],
                                                normal_hierarchy=not(args.inv_h_data),
                                                nutau_norm_value=float(args.mu_data))),
                                    seed=results['seed'],
                                    channel=channel
                                    )


    fit_results = []
    # did the optimizer set mu < 0?
    negative_mu = False
    if args.t_stat == 'llr':
        # perfomr two fits, for H0 and H1 (background only and signal + background hypothese)
        for mu_hypo in [0.0, 1.0]:    
            physics.info("Finding best fit for hypothesis mu_tau = %s"%mu_hypo)
            profile.info("start optimizer")
            fit_results.append(find_max_llh_bfgs(   fmap,
                                            template_maker,
                                            change_nutau_norm_settings( template_settings['params'],
                                                                        mu_hypo,
                                                                        # mu fixed
                                                                        True,
                                                                        not(args.inv_h_hypo)
                                                                      ),
                                            minimizer_settings,
                                            args.save_steps,
                                            normal_hierarchy=not(args.inv_h_hypo),
                                            check_octant = args.check_octant
                                         ))
            profile.info("stop optimizer")

    # profile LLR/Asimov
    else:
        # first perform a fit for fixed mu
        physics.info("Finding best fit for hypothesis mu_tau = %s"%args.mu_hypo)
        profile.info("start optimizer")
        fit_results.append(find_max_llh_bfgs(   fmap,
                                        template_maker,
                                        change_nutau_norm_settings( template_settings['params'],
                                                                    float(args.mu_hypo),
                                                                    # mu fixed
                                                                    True,
                                                                    not(args.inv_h_hypo)
                                                                  ),
                                        minimizer_settings,
                                        args.save_steps,
                                        normal_hierarchy=not(args.inv_h_hypo),
                                        check_octant = args.check_octant
                                     ))
        profile.info("stop optimizer")
        
        # now fit while profiling mu
        if args.t_stat == 'profile':
            physics.info("Finding best fit while profiling mu_tau")
            profile.info("start optimizer")
            fit_results.append(find_max_llh_bfgs(   fmap,
                                            template_maker,
                                            change_nutau_norm_settings( template_settings['params'],
                                                                        float(args.mu_hypo),
                                                                        # mu fixed
                                                                        False,
                                                                        not(args.inv_h_hypo),
                                                                        pos_def = False
                                                                      ),
                                            minimizer_settings,
                                            args.save_steps,
                                            normal_hierarchy=not(args.inv_h_hypo),
                                            check_octant = args.check_octant
                                         ))
            profile.info("stop optimizer")
            if fit_results[1]['nutau_norm'][0] < 0: negative_mu = True
        # in case of the asimov dataset the MLE for the parameters are simply their input values, so we can save time by not performing the actual fit
        elif args.t_stat == 'asimov':
            profile.info("clculate llh without fitting")
            fit_results.append(find_max_llh_bfgs(   fmap,
                                            template_maker,
                                            change_nutau_norm_settings( fix_all_params(template_settings['params']),
                                                                        float(args.mu_data),
                                                                        # mu fixed
                                                                        True,
                                                                        not(args.inv_h_hypo)
                                                                      ),
                                            minimizer_settings,
                                            args.save_steps,
                                            normal_hierarchy=not(args.inv_h_hypo),
                                            check_octant = args.check_octant
                                         ))
    # store fit results
    results['fit_results'] = fit_results
    llh = []
    for res in fit_results:
        llh.append(res['llh'][0])
    # store the value of interest, q = -2log(lh[0]/lh[1]) , llh here is already negative, so no need for the minus sign
    results['llh'] = llh
    # truncate the cases with negative mu in case of profile llh
    if not(negative_mu):
        results['q'] = 2*(llh[0]-llh[1])
    else:
        results['q'] = 0
    physics.info('found q value %.2f'%results['q'])

    # Store this trial
    trials += [results]
    profile.info("stop trial %d"%itrial)

# Assemble output dict
output = {}
output['trials'] = trials
output['template_settings'] = template_settings
output['minimizer_settings'] = minimizer_settings
if args.pseudo_data_settings is not None:
    output['pseudo_data_settings'] = pseudo_data_settings

# And write to file
to_json(output,args.outfile)

#! /usr/bin/env python
#
# NutauAnalysis.py
#
# Computes q for different test statistics for the nutau appearance search analysis
#
# the data is represented as b + s*mu_data
# the hypo is represented as b + s*mu_hypo
# theta denote nuisance parameters
# ^ denotes quantities that are MLEs
#
# in case of the llh method, q is defined as:
# q = -2*log(p(data|mu_hypo=0,theta^) / p(data|mu_hypo=1,theta^))
#
# in case of the profile llh method (including asimov), q is defined as:
# q = -2*log(p(data|mu_hypo,theta^) / p(data|mu_hypo^,theta^))
#
# psudo data is produced by randomly sampling from a poisson deistribution with lambda = b + s*mu_data
# the asimov dataset is the exact expecation values b + s*mu_data
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
from pisa.analysis.stats.Maps_nutau import get_pseudo_data_fmap, get_asimov_data_fmap_up_down, get_burn_sample, get_true_template
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
parser.add_argument('-bs','--burn_sample_file',metavar='FILE',type=str, dest='bs',
                    default='', help='HDF5 File containing burn sample.')
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
parser.add_argument('-ts', '--test-statistics',choices=['llr', 'profile', 'asimov'], default='llr', dest='t_stat', help='''Choose test statistics from llh, profile or asimov''')
parser.add_argument('--mu-data', default=1.0, dest='mu_data', help='''nu tau normalization for the psudodata''')
parser.add_argument('--mu-hypo', default=0.0, dest='mu_hypo', help='''nu tau normalization for the test hypothesis''')
parser.add_argument('--inv-mh-data', action='store_true', default=False, dest='inv_h_data', help='''invert mass hierarchy in psudodata''')
parser.add_argument('--inv-mh-hypo', action='store_true', default=False, dest='inv_h_hypo', help='''invert mass hierarchy in test hypothesis''')
parser.add_argument('-f', default='', dest='f_param', help='''fix a niusance parameter''')
parser.add_argument('--seed', default='',help='provide a fixed seed for pseudo data sampling',dest='seed')
parser.add_argument('--only-numerator',action='store_true',default=False, dest='on', help='''only calculate numerator''')
parser.add_argument('--only-denominator',action='store_true',default=False, dest='od', help='''only calculate denominator''')
args = parser.parse_args()
set_verbosity(args.verbose)
# -----------------------------------

# --- do some checks and asseble all necessary parameters/settings

# the below cases do not make much sense, therefor complain if the user tries to use them
if args.t_stat == 'asimov':
    assert(args.mu_data == 1.0)
    assert(args.bs == '')
if args.t_stat == 'llr': 
    assert(args.mu_hypo == 0.0)
if args.bs:
    assert(args.pseudo_data_settings == None)
    assert(args.mu_data == 1.0)

# Read in the settings
template_settings = from_json(args.template_settings)
ebins = template_settings['binning']['ebins']
anlys_ebins = template_settings['binning']['anlys_ebins']
czbins = template_settings['binning']['czbins']
anlys_bins = (anlys_ebins, czbins)

# fix a nuisance parameter if requested
if not args.f_param == '':
    template_settings['params'] = fix_param(template_settings['params'], args.f_param)
    print 'fixed param %s'%args.f_param
    logging.info('fixed param %s'%args.f_param)

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


template_maker = TemplateMaker(get_values(template_settings['params']),
                                    **template_settings['binning'])
pseudo_data_template_maker = TemplateMaker(get_values(pseudo_data_settings['params']),
                                    **pseudo_data_settings['binning'])

# -----------------------------------



# perform n trials
trials = []
for itrial in xrange(1,args.ntrials+1):

    profile.info("start trial %d"%itrial)
    logging.info(">"*10 + "Running trial: %05d"%itrial + "<"*10)

    results = {}
    
    # save settings
    results['test_statistics'] = args.t_stat
    if not args.t_stat == 'asimov' or args.bs:
        results['mu_data'] = float(args.mu_data)
    if not args.t_stat == 'llr':
        results['mu_hypo'] = float(args.mu_hypo)
    results['data_mass_hierarchy'] = 'inverted' if args.inv_h_data else 'normal'
    results['hypo_mass_hierarchy'] = 'inverted' if args.inv_h_hypo else 'normal'


    # --- Get the data, or psudeodata, and store it in fmap

    # Asimov dataset (exact expecation values)
    if args.t_stat == 'asimov':
        fmap = get_true_template(get_values(select_hierarchy_and_nutau_norm(pseudo_data_settings['params'],
                                            normal_hierarchy=not(args.inv_h_data),
                                            nutau_norm_value=float(args.mu_data))
                                            ),
                                            pseudo_data_template_maker
                )
        
       # get_asimov_data_fmap_up_down(pseudo_data_template_maker,
       #                                         get_values(select_hierarchy_and_nutau_norm(pseudo_data_settings['params'],
       #                                                     normal_hierarchy=not(args.inv_h_data),
       #                                                     nutau_norm_value=float(args.mu_data))
       #                                         ),
       #                                         channel=channel
       #                                     )
    # Real data
    elif args.bs:
        logging.info('Running on real data! (%s)'%args.bs)
        physics.info('Running on real data! (%s)'%args.bs)
        fmap = get_burn_sample(burn_sample_file=args.bs, anlys_ebins = anlys_ebins, czbins = czbins, output_form = 'array', cut_level='L6', channel=channel)
    # Randomly sampled (poisson) data
    else:
        if args.seed:
            results['seed'] = int(args.seed)
        else:
            results['seed'] = get_seed()
        logging.info("  RNG seed: %ld"%results['seed'])
        fmap = get_pseudo_data_fmap(pseudo_data_template_maker,
                                    get_values(select_hierarchy_and_nutau_norm(pseudo_data_settings['params'],
                                                normal_hierarchy=not(args.inv_h_data),
                                                nutau_norm_value=float(args.mu_data))),
                                    seed=results['seed'],
                                    channel=channel
                                    )

    # -----------------------------------


    # --- perform the fits for the LLR: first numerator, then denominator

    fit_results = []
    # common seetings
    kwargs = {'normal_hierarchy':not(args.inv_h_hypo),'check_octant':args.check_octant, 'save_steps':args.save_steps}
    largs = [fmap, template_maker, None , minimizer_settings]

    # - numerator (first fir for ratio)
    if not args.od:
        # LLH
        if args.t_stat == 'llr':
            physics.info("Finding best fit for hypothesis mu_tau = 0.0")
            profile.info("start optimizer")
            largs[2] = change_nutau_norm_settings( template_settings['params'], 0.0, True, not(args.inv_h_hypo))

        # profile LLH/Asimov
        else:
            physics.info("Finding best fit for hypothesis mu_tau = %s"%args.mu_hypo)
            profile.info("start optimizer")
            largs[2] = change_nutau_norm_settings( template_settings['params'],float(args.mu_hypo),True, not(args.inv_h_hypo))
        
        # execute optimizer
        fit_results.append(find_max_llh_bfgs(*largs, **kwargs))
        profile.info("stop optimizer")
            
    # - denominator (second fit for ratio)

    # LLR method 
    if not args.on:
        if args.t_stat == 'llr':
            physics.info("Finding best fit for hypothesis mu_tau = 1.0")
            profile.info("start optimizer")
            largs[2] = change_nutau_norm_settings( template_settings['params'], 1.0, True, not(args.inv_h_hypo))
        # profile LLH
        elif args.t_stat == 'profile':
            physics.info("Finding best fit while profiling mu_tau")
            profile.info("start optimizer")
            largs[2] = change_nutau_norm_settings(template_settings['params'], float(args.mu_hypo), False, not(args.inv_h_hypo))
        # in case of the asimov dataset the MLE for the parameters are simply their input values, so we can save time by not performing the actual fit
        elif args.t_stat == 'asimov':
            profile.info("clculate llh without fitting")
            largs[2] = change_nutau_norm_settings(template_settings['params'], float(args.mu_data), True, not(args.inv_h_hypo))
            kwargs['no_optimize']=True

        # execute optimizer
        fit_results.append(find_max_llh_bfgs(*largs, **kwargs))
        profile.info("stop optimizer")

    # -----------------------------------


    # store fit results
    results['fit_results'] = fit_results
    llh = []
    for res in fit_results:
        llh.append(res['llh'][0])
    # store the value of interest, q = -2log(lh[0]/lh[1]) , llh here is already negative, so no need for the minus sign
    results['llh'] = llh
    if not any([args.on, args.od]):
        results['q'] = 2*(llh[0]-llh[1])
        physics.info('found q value %.2f'%results['q'])
        physics.info('sqrt(q) = %.2f'%np.sqrt(results['q']))

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

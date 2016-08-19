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
#from pympler.tracker import SummaryTracker
#tracker = SummaryTracker()
import gc
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pisa.utils.log import logging, profile, physics, set_verbosity
from pisa.utils.jsons import from_json,to_json
from pisa.analysis.llr.LLHAnalysis_nutau import find_max_llh_bfgs
from pisa.analysis.stats.Maps import get_seed
from pisa.analysis.stats.Maps_nutau import get_pseudo_data_fmap, get_burn_sample_maps, get_true_template, get_stat_fluct_map
from pisa.utils.params import get_values, select_hierarchy_and_nutau_norm, select_hierarchy, fix_param, change_settings, float_param

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
parser.add_argument('--data','--data_file',metavar='FILE',type=str, dest='data_file',
                    default='', help='HDF5 File containing real data, can be all data or burn sample file.')
parser.add_argument('--bs',action='store_true', default=False, help='Input data is burn sample.')
parser.add_argument('--blind_fit',action='store_true', default=False, help='Do blind fit.')
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
parser.add_argument('--scan', default='', dest='scan', help='''parameter to be scanned, e.g. hole_ice=[0.,0.1,0.2,0.3]i, not available for llr''')
parser.add_argument('--inv-mh-data', action='store_true', default=False, dest='inv_h_data', help='''invert mass hierarchy in psudodata''')
parser.add_argument('--inv-mh-hypo', action='store_true', default=False, dest='inv_h_hypo', help='''invert mass hierarchy in test hypothesis''')
parser.add_argument('--fluct', default='poisson', help='''What random sampling to be used for psudo data, this is usually just poisson, but can also be set to model_stat to gaussian fluctuate the model expectations by theiruncertainty''')
parser.add_argument('-f', default='', dest='f_param', help='''fix a niusance parameter and if needed set to a value by e.g. -f nuisance_p=1.2''')
parser.add_argument('--float_param', default='', dest='float_param', help='''make a niusance parameter float''')
parser.add_argument('-fd', default='', dest='f_param_data', help='''fix a niusance parameter for the psudo data to a value by e.g. -f nuisance_p=1.2''')
parser.add_argument('-fs', default='', dest='f_param_scan', help='''fix a niusance parameter to a value by e.g. -f nuisance_p=1.2 for grid point calculations''')
parser.add_argument('--read_fmap_from_json', default='', dest='read_fmap_from_json', help='''Read fmap from json.''')
parser.add_argument('--save_fmap_to_json', default='', dest='save_fmap_to_json', help='''Save fmap to json.''')
parser.add_argument('--seed', default='',help='provide a fixed seed for pseudo data sampling',dest='seed')
parser.add_argument('--only-numerator',action='store_true',default=False, dest='on', help='''only calculate numerator''')
parser.add_argument('--only-denominator',action='store_true',default=False, dest='od', help='''only calculate denominator''')
parser.add_argument('--use_hist_PISA',action='store_true',default=False, help='''Use event-by-event PISA; otherwise, use histogram-based PISA''') 
parser.add_argument('--use_chi2',action='store_true',default=False, help='''Use chi2 instead of -llh for the minimizer.''') 
parser.add_argument('--use_rnd_init',action='store_true',default=False, help='''Use random initial values for the minimizer.''') 
args = parser.parse_args()
set_verbosity(args.verbose)
# -----------------------------------

# --- do some checks and asseble all necessary parameters/settings

# the below cases do not make much sense, therefor complain if the user tries to use them
if args.t_stat == 'asimov':
    assert(args.data_file == '')
if args.data_file:
    assert(args.pseudo_data_settings == None)
    assert(args.mu_data == 1.0)
if args.bs:
    assert (args.data_file != '')

# Read in the settings
template_settings = from_json(args.template_settings)
minimizer_settings  = from_json(args.minimizer_settings)
if args.bs:
    template_settings['params']['livetime']['value'] = 0.045 
ebins = template_settings['binning']['ebins']
anlys_ebins = template_settings['binning']['anlys_ebins']
czbins = template_settings['binning']['czbins']
anlys_bins = (anlys_ebins, czbins)
blind_fit = args.blind_fit
# one sanity check for background scale
if template_settings['params']['use_atmmu_f']['value'] == False:
    assert(template_settings['params']['atmmu_f']['fixed'] == True)
else:
    assert(template_settings['params']['atmos_mu_scale']['fixed'] == True)

pseudo_data_settings = from_json(args.pseudo_data_settings) if args.pseudo_data_settings is not None else template_settings
print 'livetime in pseudo_data_settings =', pseudo_data_settings['params']['livetime']['value']
pseudo_data_settings['params'] = select_hierarchy_and_nutau_norm(pseudo_data_settings['params'],normal_hierarchy=not(args.inv_h_data),nutau_norm_value=float(args.mu_data))
template_settings['params'] = select_hierarchy(template_settings['params'],normal_hierarchy=not(args.inv_h_hypo))

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

# make a nuisance parameter float
if not args.float_param == '':
    template_settings['params'] = float_param(template_settings['params'], args.float_param)
    print 'make param %s float'%(args.float_param)

# fix a nuisance parameter if requested
fix_param_name = None
fix_param_val = None
if not args.f_param == '':
    f_param = args.f_param.split('=')
    fix_param_name = f_param[0]
    if len(f_param) == 1:
        template_settings['params'] = fix_param(template_settings['params'], fix_param_name)
        print 'fixed param %s'%fix_param_name
    elif len(f_param) == 2:
        fix_param_val = float(f_param[1])
        template_settings['params'] = change_settings(template_settings['params'],fix_param_name,fix_param_val,True)
        print 'fixed param %s to %s'%(fix_param_name,fix_param_val)

if not args.f_param_data == '':
    f_param_data = args.f_param_data.split('=')
    assert(len(f_param_data)==2)
    fix_param_name = f_param_data[0]
    fix_param_val = float(f_param_data[1])
    pseudo_data_settings['params'] = change_settings(pseudo_data_settings['params'],fix_param_name,fix_param_val,True)
    print 'fixed param %s to %s'%(fix_param_name,fix_param_val)

fix_param_scan_name = None
fix_param_scan_val = None
if not args.f_param_scan == '':
    fix_param_scan_name,val = args.f_param_scan.split('=')
    fix_param_scan_val = float(val)

# list of hypos to be scanned
if not args.scan == '':
    assert(args.t_stat != 'llr')
    scan = args.scan.split('=')
    scan_param = scan[0]
    scan_list = eval(scan[1])
else:
    scan_param = 'nutau_norm'
    scan_list = [float(args.mu_hypo)]

print "scan = ", args.scan
print "scan_list = ", scan_list
print "scan_param = ", scan_param


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

    # -----------------------------------

    #keep results, with common denominator in second list
    fit_results = []
    
    # save settings
    results['test_statistics'] = args.t_stat
    if not args.t_stat == 'asimov' or args.data_file:
        results['mu_data'] = float(args.mu_data)
    if not args.t_stat == 'llr':
        results[scan_param] = scan_list
    if fix_param_scan_name:
        results[fix_param_scan_name] = [fix_param_scan_val]*len(scan_list)

    results['data_mass_hierarchy'] = 'inverted' if args.inv_h_data else 'normal'
    results['hypo_mass_hierarchy'] = 'inverted' if args.inv_h_hypo else 'normal'
    # test all hypos
    for value in scan_list:
        physics.info('Scan point %s for %s'%(value,scan_param))


        # --- perform the fits for the LLR: first numerator, then denominator

        # common seetings
        kwargs = {'normal_hierarchy':not(args.inv_h_hypo),'check_octant':args.check_octant, 'save_steps':args.save_steps}
        largs = [fmap, template_maker, None , minimizer_settings]

        # - numerator (first fir for ratio)
        if not args.od:
            # LLH
            if args.t_stat == 'llr':
                physics.info("Finding best fit for hypothesis mu_tau = 0.0")
                profile.info("start optimizer")
                largs[2] = change_settings( template_settings['params'], 'nutau_norm', 0.0, True)

            # profile LLH/Asimov
            else:
                physics.info("Finding best fit for hypothesis %s = %s"%(scan_param, value))
                profile.info("start optimizer")
                largs[2] = change_settings(template_settings['params'],scan_param, value,True)
                if fix_param_scan_name:
                    largs[2] = change_settings(largs[2],fix_param_scan_name,fix_param_scan_val,True)

            
            res, chi2, chi2_p, dof = find_max_llh_bfgs(blind_fit, num_data_events, use_chi2=args.use_chi2, use_rnd_init=args.use_rnd_init, *largs, **kwargs)
            res['chi2'] = [chi2]
            res['chi2_p'] = [chi2_p]
            res['dof'] = [dof]
            # execute optimizer
            if len(fit_results) == 0:
                fit_results.append(res)
            else:
                for key,value in res.items():
                    fit_results[0][key].append(value[0])
            print "chi2, chi2_p, dof = ", chi2, " ", chi2_p , " ", dof
            profile.info("stop optimizer")
                
        # - denominator (second fit for ratio)

        #if not args.on and len(fit_results) == 1:
        if not args.on:
            # LLR method 
            if args.t_stat == 'llr':
                physics.info("Finding best fit for hypothesis mu_tau = 1.0")
                profile.info("start optimizer")
                largs[2] = change_settings(template_settings['params'], 'nutau_norm', 1.0, True)
            # profile LLH, and temporarily also asimov. since the convolution method alters the expecation value of the p.d.f
            elif args.t_stat == 'profile' or args.t_stat == 'asimov':
                print "len(fit_results) = ", len(fit_results)
                if scan_param =='nutau_norm' and len(fit_results)>1:
                    print "One fit for profiling nutau_norm is done already."
                    continue
                else:
                    physics.info("Finding best fit while profiling %s"%scan_param)
                    profile.info("start optimizer")
                    largs[2] = change_settings(template_settings['params'],scan_param,pseudo_data_settings['params'][scan_param]['value'], False)
                    # in case of the asimov dataset the MLE for the parameters are simply their input values, so we can save time by not performing the actual fit
                    #elif args.t_stat == 'asimov':
                    #    profile.info("clculate llh without fitting")
                    #    largs[2] = change_settings(template_settings['params'],scan_param,pseudo_data_settings['params'][scan_param]['value'], False)
                    #    kwargs['no_optimize']=True

                    # execute optimizer
                    res, chi2, chi2_p, dof = find_max_llh_bfgs(blind_fit, num_data_events, use_chi2=args.use_chi2, use_rnd_init=args.use_rnd_init, *largs, **kwargs)
                    res['chi2'] = [chi2]
                    res['chi2_p'] = [chi2_p]
                    res['dof'] = [dof]
                    fit_results.append(res)
                    print "chi2, chi2_p, dof = ", chi2, " ", chi2_p , " ", dof
            profile.info("stop optimizer")

        del largs
        del kwargs

        # -----------------------------------


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
    trials.append(results)
    profile.info("stop trial %d"%itrial)
    gc.collect()

# Assemble output dict
output = {}
output['trials'] = trials
output['template_settings'] = template_settings
output['minimizer_settings'] = minimizer_settings
if args.pseudo_data_settings is not None:
    output['pseudo_data_settings'] = pseudo_data_settings

# And write to file
to_json(output,args.outfile)
#tracker.print_diff()

#
# LLHAnalysis.py
#
# Helper module for running the Optimizer-based LLR analysis.
#
# author: Tim Arlen
#         tca3@psu.edu
#
# date:   2 July 2014
#

import sys
from copy import deepcopy
import numpy as np
import scipy.optimize as opt

from pisa.utils.jsons import to_json
from pisa.utils.log import logging, physics, tprofile
from pisa.utils.params import get_values, select_hierarchy, get_fixed_params, get_free_params, get_param_values, get_param_scales, get_param_bounds, get_param_priors
from pisa.utils.utils import Timer
from pisa.analysis.stats.LLHStatistics import get_binwise_llh, get_binwise_chisquare
from pisa.analysis.stats.Maps import get_channel_template

#@profile
def find_alt_hierarchy_fit(asimov_data_set, template_maker,hypo_params,hypo_normal,
                           minimizer_settings,only_atm_params=True,check_octant=False):
    """
    For the hypothesis of the mass hierarchy being NMH
    ('normal_hierarchy'=True) or IMH ('normal_hierarchy'=False), finds the
    best fit for the free params in 'param_values' for the alternative
    (opposite) hierarchy that maximizes the LLH of the Asimov data set.

    \params:
      * asimov_data_set - asimov data set to find best fit llh
      * template_maker - instance of class pisa.analysis.TemplateMaker, used to
        generate the asimov data set.
      * hypo_params - parameters for template generation
      * hypo_normal - boolean for NMH (True) or IMH (False)
      * minimizer_settings - settings for minimization
      * only_atm_params - boolean to denote whether the fit will be over the
        atmospheric oscillation parameters only or over all the free params
        in params
    """

    # Find best fit of the alternative hierarchy to the
    # hypothesized asimov data set.
    #hypo_types = [('hypo_IMH',False)] if data_normal else  [('hypo_NMH',True)]
    hypo_params = select_hierarchy(hypo_params, normal_hierarchy=hypo_normal)

    with Timer() as t:
        llh_data, opt_flags = find_opt_scipy(
            fmap=asimov_data_set,
            template_maker=template_maker,
            params=hypo_params,
            minim_settings=minimizer_settings,
            normal_hierarchy=hypo_normal,
            check_octant=check_octant)
    tprofile.info("==> elapsed time for optimizer: %s sec"%t.secs)

    return llh_data, opt_flags

def display_optimizer_settings(free_params, names, init_vals, bounds, priors,
                               minim_settings):
    """
    Displays parameters and optimization settings that minimizer will run.
    """
    physics.info('%d parameters to be optimized'%len(free_params))
    for name, init_val, bound, prior in zip(names, init_vals, bounds, priors):
        physics.info(('%20s : init = %6.4f, bounds = [%6.4f,%6.4f], prior = %s')
                     %(name, init_val, bound[0], bound[1], prior))

    physics.debug("Optimizer settings:")
    physics.debug("  %s -> '%s' = %s"%(minim_settings['method']['desc'],
                                       'method',
                                       minim_settings['method']['value']))
    for key,item in minim_settings['options']['value'].items():
        physics.debug("  %s -> `%s` = %.02f"%(minim_settings['options']['desc'][key],
                                           key,
                                           item))

    return

def find_opt_scipy(fmap, template_maker, params, minim_settings,
                   save_steps=False, normal_hierarchy=None,
                   check_octant=False, metric_name='llh'):
    """
    Finds the template (and free systematic params) that maximize
    likelihood that the data came from the chosen template of
    true params (or minimize chisquare), using several methods 
    available through scipy.optimize.

    returns a dictionary of llh/chisquare data and best fit params, in the format:
      {'llh/chisquare': [...],
       'param1': [...],
       'param2': [...],
       ...}
    where 'param1', 'param2', ... are the free params varied by
    optimizer, and they hold a list of all the values tested in
    optimizer algorithm, unless save_steps is False, in which case
    they are one element in length-the best fit params and best fit llh/chi2.
    """

    # Get params dict which will be optimized (free_params) and which
    # won't be (fixed_params) but are still needed for get_template()
    fixed_params = get_fixed_params(select_hierarchy(params, normal_hierarchy))
    free_params = get_free_params(select_hierarchy(params, normal_hierarchy))

    if len(free_params) == 0:
	logging.warn("NO FREE PARAMS, returning %s"%metric_name)
	true_template = template_maker.get_template(get_values(fixed_params))
	channel = params['channel']['value']
	if metric_name=='chisquare':
            return {'chisquare':
			[get_binwise_chisquare(fmap, true_template, channel)]}
	elif metric_name=='llh':
            return {'llh': [-get_binwise_llh(fmap, true_template, channel)]}

    init_vals = get_param_values(free_params)
    scales = get_param_scales(free_params)
    bounds = get_param_bounds(free_params)
    priors = get_param_priors(free_params)
    names  = sorted(free_params.keys())

    # Scale init-vals and bounds to work with optimizer:
    init_vals = np.array(init_vals)*np.array(scales)
    bounds = [bounds[i]*scales[i] for i in range(len(bounds))]

    opt_steps_dict = {key:[] for key in names}
    opt_steps_dict[metric_name] = []

    const_args = (names, scales, fmap, fixed_params, template_maker,
                  opt_steps_dict, priors, metric_name)

    display_optimizer_settings(free_params, names, init_vals, bounds, priors,
                               minim_settings)

    minim_result = opt.minimize(fun=minim_metric, 
                                x0=init_vals, 
                                args=const_args, 
                                bounds=bounds,
                                **get_values(minim_settings))

    best_fit_vals = minim_result.x
    metric_val = minim_result.fun
    dict_flags = {}
    dict_flags['warnflag'] = minim_result.status
    dict_flags['task'] = minim_result.message
    if minim_result.has_key('jac'):
        dict_flags['grad'] = minim_result.jac
    dict_flags['funcalls'] = minim_result.nfev
    dict_flags['nit'] = minim_result.nit

    # If needed, run optimizer again, checking for second octant solution:
    if check_octant and ('theta23' in free_params.keys()):
        free_params_copy = deepcopy(free_params)
        physics.info("Checking alternative octant solution")
        old_th23_val = free_params_copy['theta23']['value']

        # Reflect across pi/4.0:
        delta = (np.pi/4.0) - old_th23_val
        free_params_copy['theta23']['value'] = (np.pi/4.0) + delta

        init_vals = get_param_values(free_params_copy)
        # Also scale init-vals, bounds are unchanged
        init_vals = np.array(init_vals)*np.array(scales)

        alt_opt_steps_dict = {key:[] for key in names}
        alt_opt_steps_dict[metric_name] = []
        const_args = (names, scales, fmap, fixed_params, template_maker,
                      alt_opt_steps_dict, priors, metric_name)
        display_optimizer_settings(free_params=free_params_copy,
                                   names=names,
                                   init_vals=init_vals,
                                   bounds=bounds,
                                   priors=priors,
                                   minim_settings=minim_settings)

        alt_minim_result = opt.minimize(fun=minim_metric, 
                                        x0=init_vals, 
                                        args=const_args,
                                        bounds=bounds, 
                                        **get_values(minim_settings))

        alt_fit_vals = alt_minim_result.x
        alt_metric_val = alt_minim_result.fun
        alt_dict_flags = {}
        alt_dict_flags['warnflag'] = alt_minim_result.status
        alt_dict_flags['task'] = alt_minim_result.message
        if minim_result.has_key('jac'):
            alt_dict_flags['grad'] = alt_minim_result.jac
        alt_dict_flags['funcalls'] = alt_minim_result.nfev
        alt_dict_flags['nit'] = alt_minim_result.nit

        # Alternative octant solution is optimal:
        if alt_metric_val < metric_val:
            best_fit_vals = alt_fit_vals
            metric_val = alt_metric_val
            dict_flags = alt_dict_flags
            opt_steps_dict = alt_opt_steps_dict

    best_fit_params = { name: value for name, value in zip(names, best_fit_vals) }

    # Report best fit (approximately)
    physics.info('Found best %s = %.2f in %d calls at:'
        %(metric_name, metric_val, dict_flags['funcalls']))
    for name, val in best_fit_params.items():
        physics.info('  %20s = %6.4f'%(name,val))

    # Report any warnings if there are
    lvl = logging.WARN if (dict_flags['warnflag'] != 0) else logging.DEBUG
    for name, val in dict_flags.items():
        physics.log(lvl," %s : %s"%(name,val))

    # insert a 'total' entry: sum of channels + prior
    for (i,val) in enumerate(opt_steps_dict[metric_name]):
        opt_steps_dict[metric_name][i]['total'] = \
	        sum([ val[chan] for chan in val.keys() ])

    if not save_steps:
	# Last step not necessarily exactly minimum, so find minimum first
	min_ind = np.argmin([ val['total']
				  for val in opt_steps_dict[metric_name] ])
        # Do not store the extra history of opt steps
        for key in opt_steps_dict.keys():
            opt_steps_dict[key] = [opt_steps_dict[key][min_ind]]

    return opt_steps_dict, dict_flags


def minim_metric(opt_vals, names, scales, fmap, fixed_params, template_maker,
                 opt_steps_dict, priors, metric_name='llh'):
    """
    Function that the scipy.optimize.minimize tries to minimize: 
    wraps get_template() and get_binwise_llh() (or 
    get_binwise_chisquare()), and returns the negative log likelihood (the chisquare).

    This function is set up this way because the fmin_l_bfgs_b algorithm must
    take a function with two inputs: params & *args, where 'params' are the
    actual VALUES to be varied, and must correspond to the limits in 'bounds',
    and 'args' are arguments which are not varied and optimized, but needed by
    the get_template() function here.

    Parameters
    ----------
    opt_vals : sequence of scalars
        Systematics varied in the optimization.
        Format: [param1, param2, ... , paramN]
    names : sequence of str
        Dictionary keys corresponding to param1, param2, ...
    scales : sequence of float
        Scales to be applied before passing to get_template
        [IMPORTANT! In the optimizer, all parameters must be ~ the same order.
        Here, we keep them between 0.1,1 so the "epsilon" step size will vary
        the parameters with roughly the same precision.]
    fmap : sequence of float
        Pseudo data flattened map
    fixed_params : dict
        Other paramters needed by the get_template() function.
    template_maker : template maker object
    opt_steps_dict: dict
        Dictionary recording information regarding the steps taken for each
        trial of the optimization process.
    priors : sequence of pisa.utils.params.Prior objects
        Priors corresponding to opt_vals list.
    metric_name : string
	Returns chisquare instead of negative llh if metric_name is 'chisquare'.
	Note: this string has to be present as a key in opt_steps_dict

    Returns
    -------
    metric_val : float
        either minimum negative llh or chisquare found by BFGS minimizer

    """
    # free parameters being "optimized" by minimizer re-scaled to their true
    # values.
    unscaled_opt_vals = [opt_vals[i]/scales[i] for i in xrange(len(opt_vals))]

    unscaled_free_params = { names[i]: val for i,val in enumerate(unscaled_opt_vals) }
    template_params = dict(unscaled_free_params.items() +
                           get_values(fixed_params).items())

    # Now get true template, and compute metric
    with Timer() as t:
        if template_params['theta23'] == 0.0:
            logging.info("Zero theta23, so generating no oscillations template...")
            true_template = template_maker.get_template_no_osc(template_params)
        else:
            true_template = template_maker.get_template(template_params)

    tprofile.info("==> elapsed time for template maker: %s sec"%t.secs)

    if metric_name=='chisquare':
	metric_val = get_binwise_chisquare(fmap, true_template,
					   template_params['channel'])
	metric_val['prior'] = \
	    sum([prior.chi2(opt_val)
                   for (opt_val, prior) in zip(unscaled_opt_vals, priors)])

    elif metric_name=='llh':
	binwise_llh = get_binwise_llh(fmap, true_template,
				      template_params['channel'])
	# NOTE: The minus sign is present on the next lines
	# because the optimizer finds a minimum rather than maximum, so we
	# have to minimize the negative of the log likelhood.
	metric_val = { k: -llh for (k, llh) in binwise_llh.items() }
	metric_val['prior'] = \
	    -sum([prior.llh(opt_val)
                   for (opt_val, prior) in zip(unscaled_opt_vals, priors)])

    # Save all optimizer-tested values to opt_steps_dict, to see
    # optimizer history later
    for key in names:
        opt_steps_dict[key].append(template_params[key])
    opt_steps_dict[metric_name].append(metric_val)

    physics.debug("%s is %.2f at: " %
	(metric_name, sum([ metric_val[key] for key in metric_val.keys() ])))

    for name, val in zip(names, opt_vals):
        physics.debug(" %20s = %6.4f" %(name,val))

    return sum([ metric_val[key] for key in metric_val.keys() ])

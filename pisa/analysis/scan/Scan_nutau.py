#
# Scan.py
#
# Helper module for brute-force scanning analysis over the whole parameter space
#
# author: Sebastian Boeser <sboeser@uni-mainz.de>
#

import sys
import numpy as np
from itertools import product

from pisa.utils.log import logging, physics, profile
from pisa.utils.params import get_values, select_hierarchy, get_fixed_params, get_free_params, get_param_values, get_param_scales, get_param_bounds, get_param_priors
from pisa.analysis.stats.Maps import flatten_map
from pisa.analysis.stats.Maps_nutau import get_true_template
from pisa.analysis.stats.LLHStatistics_nutau import get_binwise_llh, get_chi2
from pisa.analysis.scan.Scan import calc_steps


def find_max_grid(num_data_events, fmap, template_maker, params,
                  grid_settings, save_steps=True, normal_hierarchy=True,
                  check_octant=False, use_chi2=False, use_rnd_init=False):
    '''
    Finds the template (and free systematic params) that maximize
    likelihood that the data came from the chosen template of true
    params, using a brute force grid scan over the whole parameter space.

    returns a dictionary of llh data and best fit params, in the format:
      {'llh': [...],
       'param1': [...],
       'param2': [...],
       ...}
    where 'param1', 'param2', ... are the free params that are varied in the
    scan. If save_steps is False, all lists only contain the best-fit parameters
    and llh values.
    '''

    #print "NOW INSIDE find_max_grid:"
    #print "After fixing to their true values, params dict is now: "
    #for key in params.keys():
    #    try: print "  >>param: %s value: %s"%(key,str(params[key]['best']))
    #    except: continue


    # Get params dict which will be optimized (free_params) and which
    # won't be (fixed_params) but are still needed for get_template()
    fixed_params = get_fixed_params(select_hierarchy(params, normal_hierarchy))
    free_params = get_free_params(select_hierarchy(params, normal_hierarchy))

    #Obtain just the priors
    priors = get_param_priors(free_params)

    # Calculate steps [(prior,value),...] for all free parameters
    calc_steps(free_params, grid_settings['steps'])

    #Build a list from all parameters that holds a list of (name, step) tuples
    steplist = [ [(name,step) for step in param['steps']] for name, param in sorted(free_params.items())]
    print "steplist = ", steplist

    #Prepare to store all the steps
    steps = {key:[] for key in free_params.keys()}
    steps['llh'] = []
    steps['chi2'] = []
    steps['chi2_p'] = []
    steps['dof'] = []

    #Iterate over the cartesian product
    for pos in product(*steplist):

        #Get a dict with all parameter values at this position
        #including the fixed parameters
        template_params = dict(list(pos) + get_values(fixed_params).items())

        #print "   >> NOW IN LOOP: "
        #for key in template_params.keys():
        #    try: print "  >>param: %s value: %s"%(key,str(template_params[key]['value']))
        #    except: continue

        # Now get true template
        profile.info('start template calculation')
        # normal
        #true_fmap = get_true_template(template_params,template_maker, num_data_events) 
        # smeared
        true_fmap, true_fmap_sumw2 = get_true_template(template_params,template_maker,num_data_events,error=True) 
        # barlow
        #map_nu, map_mu, sumw2_nu, sumw2_mu = get_true_template(template_params, template_maker, num_data_events, error=True, both=True)

        profile.info('stop template calculation')

        # and calculate the likelihood
        # normal
        llh = -get_binwise_llh(fmap,true_fmap)
        # smeared
        #llh = -get_binwise_smeared_llh(fmap, true_fmap, sumw2_map)
        # barlow
        #llh = get_barlow_llh(fmap, map_nu, sumw2_nu, map_mu, sumw2_mu)

        # get chi2
        fmap_sumw2 = true_fmap
        #free_params_vals = get_param_values(free_params)
        free_params_vals = [v for k,v in sorted(pos)]
        print "free_params_vals = ", free_params_vals
        chi2, chi2_p, dof = get_chi2(fmap, true_fmap, fmap_sumw2, true_fmap_sumw2, free_params_vals, priors)


        # get sorted vals to match with priors
        vals = [ v for k,v in sorted(pos) ]
        llh -= sum([prior.llh(val) for val, prior in zip(vals, priors)])

        # Save all values to steps and report
        steps['llh'].append(llh)
        steps['chi2'].append(chi2)
        steps['chi2_p'].append(chi2_p)
        physics.debug("LLH is %.2f at: "%llh)
        for key, val in pos:
            steps[key].append(val)
            physics.debug(" %20s = %6.4f" %(key, val))
    steps['dof']= [dof]

    #Find best fit value
    maxllh = min(steps['llh'])
    maxpos_llh = steps['llh'].index(maxllh)
    minchi2 = min(steps['chi2'])
    minpos_chi2 = steps['chi2'].index(minchi2)
    maxchi2p = max(steps['chi2_p'])
    maxpos_chi2p = steps['chi2_p'].index(maxchi2p)

    print "steps.items() = ", steps.items()
    #Report best fit
    if use_chi2:
        physics.info('Found best chi2 = %.2f in %d calls at:' %(minchi2,len(steps['chi2'])))
        for name, vals in steps.items():
            if name == 'dof':
                continue
            physics.info('  %20s = %6.4f'%(name,vals[minpos_chi2]))
        physics.info('Found best chi2_p = %.2f in %d calls at:' %(maxchi2p,len(steps['chi2_p'])))
        for name, vals in steps.items():
            if name == 'dof':
                continue
            physics.info('  %20s = %6.4f'%(name,vals[maxpos_chi2p]))
    else:
        physics.info('Found best LLH = %.2f in %d calls at:' %(maxllh,len(steps['llh'])))
        for name, vals in steps.items():
            if name == 'dof':
                continue
            physics.info('  %20s = %6.4f'%(name,vals[maxpos_llh]))

        #only save this maximum if asked for
        if not save_steps:
            steps[name]=vals[maxpos]

    return steps

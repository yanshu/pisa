#
# Maps_nutau_noDOMIce.py
#
# Utilities augmenting and/or replacing those in Maps.py for dealing with event
# rate maps in analysis
#
# author: Feifei Huang <fxh140@psu.edu>
# date:   2015-06-11
#

import os
import numpy as np
from pisa.analysis.stats.LLHStatistics import get_random_map
from pisa.utils.log import logging
import pisa.analysis.stats.Maps as Maps
from pisa.analysis.stats.Maps_nutau import get_up_map, get_combined_map, get_flipped_down_map

def get_pseudo_data_fmap(template_maker, fiducial_params, channel, seed=None):
    """
    Creates a true template from fiducial_params, then uses Poisson statistics
    to vary the expected counts per bin to create a pseudo data set.
    If seed is provided, the random state is seeded with seed before the map is
    created.

    IMPORTANT: returns a SINGLE flattened map of trck/cscd combined
    \params:
      * channel = channel of flattened fmap to use.
        if 'all': returns a single flattened map of trck/cscd combined.
        if 'cscd' or 'trck' only returns the channel requested.
    """

    if fiducial_params['residual_up_down'] or fiducial_params['simp_up_down'] or fiducial_params['ratio_up_down']:
        template_maker_up = template_maker[0]
        template_maker_down = template_maker[1]
        template_up = template_maker_up.get_template(fiducial_params)  
        template_down = template_maker_down.get_template(fiducial_params)  

        template_up_down_combined = get_combined_map(template_up,template_down, channel=fiducial_params['channel'])
        template_up = get_up_map(template_up_down_combined, channel=fiducial_params['channel'])
        reflected_template_down = get_flipped_down_map(template_up_down_combined, channel=fiducial_params['channel'])

        true_fmap_up = Maps.flatten_map(template_up, channel=fiducial_params['channel'])
        true_fmap_down = Maps.flatten_map(reflected_template_down, channel=fiducial_params['channel'])
        fmap_up = get_random_map(true_fmap_up, seed=Maps.get_seed())
        fmap_down = get_random_map(true_fmap_down, seed=Maps.get_seed())
        # if we want to recreate the same template, then use the input seed for both
        #fmap_up = get_random_map(true_fmap_up, seed=seed)
        #fmap_down = get_random_map(true_fmap_down, seed=seed)
        if fiducial_params['residual_up_down']:
            fmap = fmap_up-fmap_down
        elif fiducial_params['ratio_up_down']:
            fmap = np.array([fmap_up, fmap_down])
        else:
            fmap = np.append(fmap_up, fmap_down)
    else:
        true_template = template_maker.get_template(fiducial_params)
        true_fmap = Maps.flatten_map(true_template, channel=channel)
        fmap = get_random_map(true_fmap, seed=seed)
    return fmap

def get_true_template(template_params, template_maker):
    if template_params['theta23'] == 0.0:
        logging.info("Zero theta23, so generating no oscillations template...")
        true_template = template_maker.get_template_no_osc(template_params)
        true_fmap = Maps.flatten_map(true_template, channel=template_params['channel'])
    elif type(template_maker)==list and len(template_maker)==2:
        template_maker_up = template_maker[0]
        template_maker_down = template_maker[1]
        template_up = template_maker_up.get_template(template_params)  
        template_down = template_maker_down.get_template(template_params)  

        template_up_down_combined = get_combined_map(template_up,template_down, channel=template_params['channel'])
        template_up = get_up_map(template_up_down_combined, channel=template_params['channel'])
        reflected_template_down = get_flipped_down_map(template_up_down_combined, channel=template_params['channel'])

        true_fmap_up = Maps.flatten_map(template_up, channel=template_params['channel'])
        true_fmap_down = Maps.flatten_map(reflected_template_down, channel=template_params['channel'])
        if template_params['residual_up_down'] or template_params['ratio_up_down']:
            true_fmap = np.array([true_fmap_up, true_fmap_down])
        else:
            true_fmap = np.append(true_fmap_up, true_fmap_down)
    else:
        true_template = template_maker.get_template(template_params)  
        true_fmap = Maps.flatten_map(true_template, channel=template_params['channel'])

    return true_fmap

def get_pseudo_tau_fmap(template_maker, fiducial_params, channel=None, seed=None):
    '''
    Creates a true template from fiducial_params, then uses Poisson statistics
    to vary the expected counts per bin to create a pseudo data set.
    If seed is provided, the random state is seeded with seed before the map is
    created.

    IMPORTANT: returns a SINGLE flattened map of trck/cscd combined
    \params:
      * channel = channel of flattened fmap to use.
        if 'all': returns a single flattened map of trck/cscd combined.
        if 'cscd' or 'trck' only returns the channel requested.
    ''' 
    if fiducial_params['residual_up_down'] or fiducial_params['simp_up_down'] or fiducial_params['ratio_up_down']:
        template_maker_up = template_maker[0]
        template_maker_down = template_maker[1]
        template_up = template_maker_up.get_tau_template(fiducial_params)  
        template_down = template_maker_down.get_tau_template(fiducial_params)  

        template_up_down_combined = get_combined_map(template_up,template_down, channel=fiducial_params['channel'])
        template_up = get_up_map(template_up_down_combined, channel=fiducial_params['channel'])
        reflected_template_down = get_flipped_down_map(template_up_down_combined, channel=fiducial_params['channel'])

        true_fmap_up = Maps.flatten_map(template_up, channel=fiducial_params['channel'])
        true_fmap_down = Maps.flatten_map(reflected_template_down, channel=fiducial_params['channel'])
        fmap_up = get_random_map(true_fmap_up, seed=Maps.get_seed())
        fmap_down = get_random_map(true_fmap_down, seed=Maps.get_seed())
        # if we want to recreate the same template, then use the input seed for both
        #fmap_up = get_random_map(true_fmap_up, seed=seed)
        #fmap_down = get_random_map(true_fmap_down, seed=seed)
        if fiducial_params['residual_up_down']:
            fmap = fmap_up-fmap_down
        elif fiducial_params['ratio_up_down']:
            fmap = np.array([fmap_up, fmap_down])
        else:
            fmap = np.append(fmap_up, fmap_down)
    else:
        true_template = template_maker.get_tau_template(fiducial_params)  
        true_fmap = Maps.flatten_map(true_template, channel=channel)
        fmap = get_random_map(true_fmap, seed=seed)
    return fmap

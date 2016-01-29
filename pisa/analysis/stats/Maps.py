#
# Maps.py
#
# Utilities for dealing with event rate maps in analysis
#
# author: Tim Arlen   <tca3@psu.edu>
#

import os
import numpy as np
from scipy.stats import poisson

from pisa.utils.log import logging
from pisa.utils.utils import get_all_channel_names, get_channels

def apply_ratio_scale(orig_maps, key1, key2, ratio_scale, is_flux_scale, int_type=None):
    """
    Scales the ratio of the entries of two maps, conserving the total.
    """

    if is_flux_scale: log_str = 'flux'
    else: log_str = 'event rate (%s)'%int_type

    if not is_flux_scale:
	# we have maps of event counts of a certain interaction type
        orig_sum = orig_maps[key1][int_type]['map'] + orig_maps[key2][int_type]['map']
        orig_total1 = orig_maps[key1][int_type]['map'].sum()
        orig_total2 = orig_maps[key2][int_type]['map'].sum()
        orig_ratio = orig_maps[key1][int_type]['map'] / orig_maps[key2][int_type]['map']
    else:
        # we have flux_maps
        orig_sum = orig_maps[key1]['map'] + orig_maps[key2]['map']
        orig_total1 = orig_maps[key1]['map'].sum()
        orig_total2 = orig_maps[key2]['map'].sum()
        orig_ratio = orig_maps[key1]['map'] / orig_maps[key2]['map']

    # conserved total:
    scaled_map2 = orig_sum / (1 + ratio_scale*orig_ratio)
    scaled_map1 = ratio_scale*orig_ratio*scaled_map2

    logging.trace(' %s / %s %s ratio before scaling: %.3f'%(key1, key2, log_str,
                    orig_total1/orig_total2))
    logging.trace(' %s / %s %s ratio after scaling with %.2f: %.3f'%(key1, key2, log_str,
                    ratio_scale, scaled_map1.sum()/scaled_map2.sum()))

    return scaled_map1, scaled_map2

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

    true_template = template_maker.get_template(fiducial_params)
    true_fmap = flatten_map(true_template, channel=channel)
    fmap = get_random_map(true_fmap, seed=seed)
    return fmap

def get_channel_template(template, channel='all'):
    """
    Takes a final level true (expected) template, and returns template for
    the selected channel, with zero bins removed.
    """
    channels = get_channels(channel)
    logging.trace("Creating map of channel: %s"%channel)
    channel_template = {}
    for chan in channels:
        channel_template[chan] = template[chan]['map'].flatten()

    return channel_template


def get_seed():
    """
    Returns a random seed from /dev/urandom that can be used to seed the random
    state, e.g. for the poisson random variates.
    """

    return int(os.urandom(4).encode('hex'), 16)

def get_random_map(template, seed=None):
    """
    Gets an event map with integer entries from non-integer entries
    (in general) in the template, varied according to Poisson
    statistics.
    """
    # Set the seed if given
    if not seed is None:
	np.random.seed(seed=seed)

    return { chan: {'map': poisson.rvs(tmplt['map'])}
	for (chan, tmplt) in template.items() if chan in get_all_channel_names() }

#
# Maps.py
#
# Utilities for dealing with event rate maps in analysis
#
# author: Tim Arlen   <tca3@psu.edu>
#

import os
import copy
import numpy as np
from pisa.analysis.stats.LLHStatistics import get_random_map
from pisa.utils.log import logging


def apply_ratio_scale(orig_maps, key1, key2, ratio_scale, is_flux_scale, int_type = None):
    '''
    Scales the ratio of the entries of two maps, conserving the total.
    '''

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


def get_pseudo_data_fmap(template_maker,fiducial_params,seed=None,chan=None):
    '''
    Creates a true template from fiducial_params, then uses Poisson statistics
    to vary the expected counts per bin to create a pseudo data set.
    If seed is provided, the random state is seeded with seed before the map is
    created.

    IMPORTANT: returns a SINGLE flattened map of trck/cscd combined
    \params:
      * chan = channel of flattened fmap to use.
        if 'all': returns a single flattened map of trck/cscd combined.
        if 'cscd' or 'trck' only returns the channel requested.
    ''' 
    if 'residual_up_down' in fiducial_params and fiducial_params['residual_up_down']:
        # get a up and down-going combined template first, change 'residual_up_down' to false
        combined_fiducial_params = copy.deepcopy(fiducial_params)
        combined_fiducial_params['residual_up_down']=False
        true_template = template_maker.get_template(combined_fiducial_params)  
        # get two separate templates: up-going and downgoing
        true_template_up = get_up_map(true_template,chan=fiducial_params['channel']) 
        true_template_down = get_flipped_down_map(true_template,fiducial_params['channel']) 
        true_fmap_up = flatten_map(true_template_up,chan=fiducial_params['channel'])
        true_fmap_down = flatten_map(true_template_down,chan=fiducial_params['channel'])
        fmap_up = get_random_map(true_fmap_up, seed=seed)
        fmap_down = get_random_map(true_fmap_down, seed=seed)
        # return the residual of the two arrays 
        fmap = fmap_up-fmap_down
    else:
        true_template = template_maker.get_template(fiducial_params)  
        if 'ratio_up_down' in fiducial_params:
            true_fmap = flatten_map(true_template,chan=chan,ratio_up_down=fiducial_params['ratio_up_down'])
        else:
            true_fmap = flatten_map(true_template,chan=chan)
        fmap = get_random_map(true_fmap, seed=seed)
    return fmap

def get_asimov_fmap(template_maker,fiducial_params,chan=None):
    '''
    Creates a true template from fiducial_params, then converts the true_template
    expected counts to an integer number of counts to simulate an experiment at the
    average expected value.
    '''

    true_template = template_maker.get_template(fiducial_params)
    fmap = flatten_map(true_template,chan=chan)
    fmap = np.int32(fmap+0.5)

    return fmap

def flatten_map(template,chan='all',ratio_up_down=False):
    '''
    Takes a final level true (expected) template of trck/cscd, and returns a
    single flattened map of trck appended to cscd, with all zero bins
    removed.
    '''

    logging.trace("Getting flattened map of chan: %s"%chan)
    # if ratio_up_down is true, return an array of two arrays (one upgoing, one downgoing map).
    if ratio_up_down:
        if chan == 'all':
            cscd_0 = template['cscd']['map'][0].flatten()
            trck_0 = template['trck']['map'][0].flatten()
            fmap_0 = np.append(cscd_0,trck_0)
            cscd_1 = template['cscd']['map'][1].flatten()
            trck_1 = template['trck']['map'][1].flatten()
            fmap_1 = np.append(cscd_1,trck_1)
        elif chan == 'trck':
            trck_0 = template[chan]['map'][0].flatten()
            fmap_0 = np.array(trck_0)
            trck_1 = template[chan]['map'][1].flatten()
            fmap_1 = np.array(trck_1)
        elif chan == 'cscd':
            cscd_0 = template[chan]['map'][0].flatten()
            fmap_0 = np.array(cscd_0)
            cscd_1 = template[chan]['map'][1].flatten()
            fmap_1 = np.array(cscd_1)
        elif chan == 'no_pid':
            cscd_0 = template['cscd']['map'][0].flatten()
            trck_0 = template['trck']['map'][0].flatten()
            fmap_0 = cscd_0 + trck_0
            cscd_1 = template['cscd']['map'][1].flatten()
            trck_1 = template['trck']['map'][1].flatten()
            fmap_1 = cscd_1 + trck_1
        else:
            raise ValueError("chan: '%s' not implemented! Allowed: ['all', 'trck', 'cscd','no_pid']")

        fmap = np.array([fmap_0,fmap_1])
        return fmap

    if chan == 'all':
        cscd = template['cscd']['map'].flatten()
        trck = template['trck']['map'].flatten()
        fmap = np.append(cscd,trck)
    elif chan == 'trck':
        trck = template[chan]['map'].flatten()
        fmap = np.array(trck)
        #fmap = np.array(fmap)[np.nonzero(fmap)]
    elif chan == 'cscd':
        cscd = template[chan]['map'].flatten()
        fmap = np.array(cscd)
        #fmap = np.array(fmap)[np.nonzero(fmap)]
    elif chan == 'no_pid':
        cscd = template['cscd']['map'].flatten()
        trck = template['trck']['map'].flatten()
        fmap = cscd + trck
        #fmap = np.array(fmap)[np.nonzero(fmap)]
    else:
        raise ValueError("chan: '%s' not implemented! Allowed: ['all', 'trck', 'cscd','no_pid']")

    fmap = np.array(fmap)[np.nonzero(fmap)]
    return fmap

def get_seed():
    '''
    Returns a random seed from /dev/urandom that can be used to seed the random
    state, e.g. for the poisson random variates.
    '''
    return int(os.urandom(4).encode('hex'),16)

def get_up_map(map,chan):
    if chan=='all':
        flavs=['trck','cscd']
    elif chan=='trck':
        flavs=['trck']
    elif chan=='cscd':
        flavs=['cscd']
    elif chan == 'no_pid':
        return {'no_pid':{
            'map': map['trck']['map'][:,0:czbin_mid_idx]+map['cscd']['map'][:,0:czbin_mid_idx],
            'ebins':map['trck']['ebins'],
            'czbins': map['trck']['czbins'][0:czbin_mid_idx+1] }}
    else:
        raise ValueError("chan: '%s' not implemented! Allowed: ['all', 'trck', 'cscd','no_pid']")
    czbin_edges = len(map['cscd']['czbins'])
    czbin_mid_idx = (czbin_edges-1)/2
    return {flav:{
        'map': map[flav]['map'][:,0:czbin_mid_idx],
        'ebins':map[flav]['ebins'],
        'czbins': map[flav]['czbins'][0:czbin_mid_idx+1] }
            for flav in flavs}

def get_flipped_down_map(map,chan):
    ''' Gets the downgoing map and flip it.'''
    if chan=='all':
        flavs=['trck','cscd']
    elif chan=='trck':
        flavs=['trck']
    elif chan=='cscd':
        flavs=['cscd']
    elif chan == 'no_pid':
        return {'no_pid':{
            'map': map['trck']['map'][:,czbin_mid_idx:]+map['cscd']['map'][:,czbin_mid_idx:],
            'ebins':map['trck']['ebins'],
            'czbins': map['trck']['czbins'][czbin_mid_idx:] }}
    else:
        raise ValueError("chan: '%s' not implemented! Allowed: ['all', 'trck', 'cscd','no_pid']")
    czbin_edges = len(map['cscd']['czbins'])
    czbin_mid_idx = (czbin_edges-1)/2
    return {flav:{
        'map': np.fliplr(map[flav]['map'][:,czbin_mid_idx:]),
        'ebins':map[flav]['ebins'],
        'czbins': np.sort(-map['trck']['czbins'][czbin_mid_idx:]) }
            for flav in flavs}


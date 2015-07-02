#
# Maps_nutau.py
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
from pisa.utils.jsons import from_json,to_json
import pisa.analysis.stats.Maps as Maps


def get_asimov_data_fmap_up_down(template_maker, fiducial_params, channel=None):
    if fiducial_params['residual_up_down'] or fiducial_params['simp_up_down'] or fiducial_params['ratio_up_down']:
        template_maker_up = template_maker[0]
        template_maker_down = template_maker[1]
        template_up = template_maker_up.get_template(fiducial_params)  
        template_down = template_maker_down.get_template(fiducial_params)  
        reflected_template_down = get_flipped_map(template_down, channel=fiducial_params['channel'])
        true_fmap_up = Maps.flatten_map(template_up, channel=fiducial_params['channel'])
        true_fmap_down = Maps.flatten_map(reflected_template_down, channel=fiducial_params['channel'])
        fmap_up = np.int32(true_fmap_up+0.5)
        fmap_down = np.int32(true_fmap_down+0.5)
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
        # add domeff and/or hole ice effects
        [template_up_dh,template_down_dh] = apply_domeff_holeice([template_up,template_down],fiducial_params)
        reflected_template_down_dh = get_flipped_map(template_down_dh, channel=fiducial_params['channel'])
        true_fmap_up = Maps.flatten_map(template_up_dh, channel=fiducial_params['channel'])
        true_fmap_down = Maps.flatten_map(reflected_template_down_dh, channel=fiducial_params['channel'])
        fmap_up = get_random_map(true_fmap_up, seed=Maps.get_seed())
        fmap_down = get_random_map(true_fmap_down, seed=Maps.get_seed())
        if fiducial_params['residual_up_down']:
            fmap = fmap_up-fmap_down
        elif fiducial_params['ratio_up_down']:
            fmap = np.array([fmap_up, fmap_down])
        else:
            fmap = np.append(fmap_up, fmap_down)
    else:
        true_template = template_maker.get_template(fiducial_params)
        # add domeff and/or hole ice effects
        true_template_dh = apply_domeff_holeice(true_template,fiducial_params)
        true_fmap = Maps.flatten_map(true_template_dh, channel=channel)
        fmap = get_random_map(true_fmap, seed=seed)
    return fmap

def apply_domeff_holeice(template, params):
    # Apply DOMeff and HoleIce changes to template 
    channel=params['channel']
    if channel=='all':
        flavs=['trck', 'cscd']
    elif channel=='trck':
        flavs=['trck']
    elif channel=='cscd':
        flavs=['cscd']
    elif channel == 'no_pid':
        print "To do"
        #TODO
    else:
        raise ValueError(
                "channel: '%s' not implemented! Allowed: ['all', 'trck', 'cscd', 'no_pid']"%channel)
    if isinstance(template,list) and len(template)==2:
        template_up = template[0]
        template_down = template[1]
        dom_eff_val = params['dom_eff']
        hole_ice_val = params['hole_ice']
        slope = from_json('DH_slopes_up_down.json')
        for flav in flavs:
            slope_DE_up = slope['slopes']['data_tau']['k_DomEff'][flav]['up']
            slope_DE_down = slope['slopes']['data_tau']['k_DomEff'][flav]['down']
            slope_HI_up = slope['slopes']['data_tau']['k_HoleIce'][flav]['up']
            slope_HI_down = slope['slopes']['data_tau']['k_HoleIce'][flav]['down']
            scale_up = slope_DE_up*(dom_eff_val-0.91)+ slope_HI_up*(hole_ice_val-0.2) + 1
            scale_down = slope_DE_down*(dom_eff_val-0.91)+ slope_HI_down*(hole_ice_val-0.2) + 1
            #print "scale_up ", flav, " ", scale_up
            #print "scale_down ", flav, " ", scale_down
            #print "slope_DE_up , ", flav ," ", slope_DE_up
            #print "slope_HI_up , ", flav ," ", slope_HI_up
            #print "slope_DE_down ", slope_DE_down
            output_map_up = {flav:{
                'map': (template_up[flav]['map'])*scale_up,
                'ebins':template_up[flav]['ebins'],
                'czbins': template_up[flav]['czbins'] }
                    for flav in flavs}
            output_map_down = {flav:{
                'map': (template_down[flav]['map'])*scale_down,
                'ebins':template_down[flav]['ebins'],
                'czbins': template_down[flav]['czbins'] }
                    for flav in flavs}
        return [output_map_up,output_map_down]
    elif isinstance(template,dict):
        #TODO
        print "to do"
    else:
       raise TypeError("The type of input template is wrong!")


def get_true_template(template_params, template_maker):
    if template_params['theta23'] == 0.0:
        logging.info("Zero theta23, so generating no oscillations template...")
        true_template = template_maker.get_template_no_osc(template_params)
        # add domeff and/or hole ice effects
        true_template_dh = apply_domeff_holeice(true_template,template_params)
        true_fmap = Maps.flatten_map(true_template_dh, channel=template_params['channel'])
    elif type(template_maker)==list and len(template_maker)==2:
        template_maker_up = template_maker[0]
        template_maker_down = template_maker[1]
        template_up = template_maker_up.get_template(template_params)  
        template_down = template_maker_down.get_template(template_params)  
        # add domeff and/or hole ice effects
        [template_up_dh,template_down_dh] = apply_domeff_holeice([template_up,template_down],template_params)
        reflected_template_down_dh = get_flipped_map(template_down_dh, channel=template_params['channel'])
        true_fmap_up = Maps.flatten_map(template_up_dh, channel=template_params['channel'])
        true_fmap_down = Maps.flatten_map(reflected_template_down_dh, channel=template_params['channel'])
        if template_params['residual_up_down'] or template_params['ratio_up_down']:
            true_fmap = np.array([true_fmap_up, true_fmap_down])
        else:
            true_fmap = np.append(true_fmap_up, true_fmap_down)
    else:
        true_template = template_maker.get_template(template_params)  
        # add domeff and/or hole ice effects
        true_template_dh = apply_domeff_holeice(true_template,template_params)
        true_fmap = Maps.flatten_map(true_template_dh, channel=template_params['channel'])

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
        reflected_template_down = get_flipped_map(template_down, channel=fiducial_params['channel'])
        true_fmap_up = Maps.flatten_map(template_up, channel=fiducial_params['channel'])
        true_fmap_down = Maps.flatten_map(reflected_template_down, channel=fiducial_params['channel'])
        fmap_up = get_random_map(true_fmap_up, seed=Maps.get_seed())
        fmap_down = get_random_map(true_fmap_down, seed=Maps.get_seed())
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

def get_up_map(map, channel):
    ''' Gets the upgoing map from a full sky map.'''
    if channel =='all':
        flavs=['trck', 'cscd']
    elif channel =='trck':
        flavs=['trck']
    elif channel =='cscd':
        flavs=['cscd']
    elif channel == 'no_pid':
        return {'no_pid':{
            'map': map['trck']['map'][:,0:czbin_mid_idx]+map['cscd']['map'][:,0:czbin_mid_idx],
            'ebins':map['trck']['ebins'],
            'czbins': map['trck']['czbins'][0:czbin_mid_idx+1] }}
    else:
        raise ValueError("channel: '%s' not implemented! Allowed: ['all', 'trck', 'cscd', 'no_pid']"%channel)
    czbin_edges = len(map['cscd']['czbins'])
    czbin_mid_idx = (czbin_edges-1)/2
    return {flav:{
        'map': map[flav]['map'][:,0:czbin_mid_idx],
        'ebins':map[flav]['ebins'],
        'czbins': map[flav]['czbins'][0:czbin_mid_idx+1] }
            for flav in flavs}

def get_flipped_down_map(map, channel):
    ''' Gets the downgoing map from a full sky map and flip it around cz = 0.'''
    czbin_edges = len(map['cscd']['czbins'])
    czbin_mid_idx = (czbin_edges-1)/2
    if  channel=='all':
        flavs=['trck', 'cscd']
    elif  channel=='trck':
        flavs=['trck']
    elif  channel=='cscd':
        flavs=['cscd']
    elif  channel == 'no_pid':
        return {'no_pid':{
            'map': np.fliplr(map['trck']['map'][:,czbin_mid_idx:]+map['cscd']['map'][:,czbin_mid_idx:]),
            'ebins':map['trck']['ebins'],
            'czbins': np.sort(-map['trck']['czbins'][czbin_mid_idx:]) }}
    else:
        raise ValueError(" channel: '%s' not implemented! Allowed: ['all', 'trck', 'cscd', 'no_pid']"%channel)
    return {flav:{
        'map': np.fliplr(map[flav]['map'][:,czbin_mid_idx:]),
        'ebins':map[flav]['ebins'],
        'czbins': np.sort(-map['trck']['czbins'][czbin_mid_idx:]) }
            for flav in flavs}

def get_flipped_map(map, channel):
    ''' Flip the down-going map around cz = 0'''
    if not np.alltrue(map['cscd']['czbins']>=0):
        raise ValueError("This map has to be down-going!")
    if channel=='all':
        flavs=['trck', 'cscd']
    elif channel=='trck':
        flavs=['trck']
    elif channel=='cscd':
        flavs=['cscd']
    elif channel == 'no_pid':
        return {'no_pid':{
            'map': np.fliplr(map['trck']['map']+map['cscd']['map']),
            'ebins':map['trck']['ebins'],
            'czbins': np.sort(-map['cscd']['czbins']) }}
    else:
        raise ValueError("channel: '%s' not implemented! Allowed: ['all', 'trck', 'cscd', 'no_pid']"%channel)
    return {flav:{
        'map': np.fliplr(map[flav]['map']),
        'ebins':map[flav]['ebins'],
        'czbins': np.sort(-map[flav]['czbins']) }
            for flav in flavs}

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
from scipy.optimize import curve_fit

def get_asimov_data_fmap_up_down(template_maker,fiducial_params,chan=None):
    if fiducial_params['residual_up_down'] or fiducial_params['simp_up_down'] or fiducial_params['ratio_up_down']:
        template_maker_up = template_maker[0]
        template_maker_down = template_maker[1]
        template_up = template_maker_up.get_template(fiducial_params)  
        template_down = template_maker_down.get_template(fiducial_params)  
        reflected_template_down = get_flipped_map(template_down,chan=fiducial_params['channel'])
        true_fmap_up = flatten_map(template_up,chan=fiducial_params['channel'])
        true_fmap_down = flatten_map(reflected_template_down,chan=fiducial_params['channel'])
        fmap_up = np.int32(true_fmap_up+0.5)
        fmap_down = np.int32(true_fmap_down+0.5)
        if fiducial_params['residual_up_down']:
            fmap = fmap_up-fmap_down
        elif fiducial_params['ratio_up_down']:
            fmap = np.array([fmap_up,fmap_down])
        else:
            fmap = np.append(fmap_up,fmap_down)
    else:
        true_template = template_maker.get_template(fiducial_params)  
        true_fmap = flatten_map(true_template,chan=chan)
        fmap = get_random_map(true_fmap, seed=seed)
    return fmap

def apply_ratio_scale(orig_maps, key1, key2, ratio_scale, is_flux_scale, int_type = None):
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
        y =  orig_maps[key2][int_type]['map']==0
        if (True in y):
            print "key2 int_type = ", key2 , " " , int_type
            print orig_maps[key2][int_type]['map']
        orig_ratio = orig_maps[key1][int_type]['map'] / orig_maps[key2][int_type]['map']
    else:
        # we have flux_maps
        orig_sum = orig_maps[key1]['map'] + orig_maps[key2]['map']
        orig_total1 = orig_maps[key1]['map'].sum()
        orig_total2 = orig_maps[key2]['map'].sum()
        y =  orig_maps[key2]['map']==0
        if (True in y):
            print "key2 = ", key2
            print orig_maps[key2]['map']
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
    """
    Creates a true template from fiducial_params, then uses Poisson statistics
    to vary the expected counts per bin to create a pseudo data set.
    If seed is provided, the random state is seeded with seed before the map is
    created.

    IMPORTANT: returns a SINGLE flattened map of trck/cscd combined
    \params:
      * chan = channel of flattened fmap to use.
        if 'all': returns a single flattened map of trck/cscd combined.
        if 'cscd' or 'trck' only returns the channel requested.
    """

    #if fiducial_params['dom_eff']!= 1.0 or fiducial_params['hole_ice']!= 0.2:
    if fiducial_params['residual_up_down'] or fiducial_params['simp_up_down'] or fiducial_params['ratio_up_down']:
        tmap_up = []
        tmap_down = []
        for i in range(0,8):
            true_up_and_down_map = get_up_and_down_template(template_maker[1][i],fiducial_params)
            tmap_up.append(true_up_and_down_map[0])
            tmap_down.append(true_up_and_down_map[1])
        #tmap_up_fit = get_map_plane_fit(np.array(tmap_up),fiducial_params)
        #tmap_down_fit = get_map_plane_fit(np.array(tmap_down),fiducial_params)
        tmap_up_fit = get_map_two_linear_fits(np.array(tmap_up),fiducial_params)
        tmap_down_fit = get_map_two_linear_fits(np.array(tmap_down),fiducial_params)
        fmap_up = get_random_map(tmap_up_fit, seed=get_seed())
        fmap_down = get_random_map(tmap_down_fit, seed=get_seed())
        if fiducial_params['residual_up_down']:
            fmap = fmap_up-fmap_down
        elif fiducial_params['ratio_up_down']:
            fmap = np.array([fmap_up,fmap_down])
        else:
            fmap = np.append(fmap_up,fmap_down)
    else:
        tmap = []
        for i in range(0,8):
            true_template = template_maker[1][i].get_template(fiducial_params)  
            true_fmap = flatten_map(true_template,chan=chan)
            tmap.append(true_fmap)
        #tmap_fit = get_map_plane_fit(np.array(tmap),fiducial_params)
        tmap_fit = get_map_two_linear_fits(np.array(tmap),fiducial_params)
        fmap = get_random_map(tmap_fit, seed=get_seed())
    #else:
    #    if fiducial_params['residual_up_down'] or fiducial_params['simp_up_down'] or fiducial_params['ratio_up_down']:
    #        true_up_and_down_map = get_up_and_down_template(template_maker[0],fiducial_params)
    #        true_fmap_up = true_up_and_down_map[0]
    #        true_fmap_down = true_up_and_down_map[1]
    #        fmap_up = get_random_map(true_fmap_up, seed=get_seed())
    #        fmap_down = get_random_map(true_fmap_down, seed=get_seed())
    #        if fiducial_params['residual_up_down']:
    #            fmap = fmap_up-fmap_down
    #        elif fiducial_params['ratio_up_down']:
    #            fmap = np.array([fmap_up,fmap_down])
    #        else:
    #            fmap = np.append(fmap_up,fmap_down)
    #    else:
    #        true_template = template_maker.get_template(fiducial_params)  
    #        true_fmap = flatten_map(true_template,chan=chan)
    #        fmap = get_random_map(true_fmap, seed=seed)
    return fmap

def get_up_and_down_template(template_maker,params):
    assert(type(template_maker)==list and len(template_maker)==2)
    template_maker_up = template_maker[0]
    template_maker_down = template_maker[1]
    template_up = template_maker_up.get_template(params)  
    template_down = template_maker_down.get_template(params)  
    #print " in get_up_and_down_template(), params = ", params 
    reflected_template_down = get_flipped_map(template_down,chan=params['channel'])
    true_fmap_up = flatten_map(template_up,chan=params['channel'])
    true_fmap_down = flatten_map(reflected_template_down,chan=params['channel'])
    return [true_fmap_up,true_fmap_down]

def get_true_template(template_params,template_maker):
    #if template_params['dom_eff']!= 1.0 or template_params['hole_ice']!= 0.2:
    if template_params['residual_up_down'] or template_params['simp_up_down'] or template_params['ratio_up_down']:
        tmap_up = []
        tmap_down = []
        for i in range(0,8):
            true_up_and_down_map = get_up_and_down_template(template_maker[1][i],template_params)
            tmap_up.append(true_up_and_down_map[0])
            tmap_down.append(true_up_and_down_map[1])
            #print "true_up_and_down_map[0] = ", true_up_and_down_map[0]
            #print "true_up_and_down_map[1] = ", true_up_and_down_map[1]
        tmap_up_fit = get_map_plane_fit(np.array(tmap_up),template_params)
        tmap_down_fit = get_map_plane_fit(np.array(tmap_down),template_params)
        if template_params['residual_up_down'] or template_params['ratio_up_down']:
            true_fmap = np.array([tmap_up_fit,tmap_down_fit])
        else:
            true_fmap = np.append(tmap_up_fit,tmap_down_fit)
    else:
        tmap = []
        for i in range(0,8):
            true_template = template_maker[1][i].get_template(template_params)  
            true_fmap = flatten_map(true_template,chan=template_params['channel'])
            tmap.append(true_fmap)
        true_fmap = get_map_plane_fit(np.array(tmap),template_params)
    #else:
    #    if template_params['residual_up_down'] or template_params['simp_up_down'] or template_params['ratio_up_down']:
    #        true_up_and_down_map = get_up_and_down_template(template_maker[0],template_params)
    #        true_fmap_up = true_up_and_down_map[0]
    #        true_fmap_down = true_up_and_down_map[1]
    #        if template_params['residual_up_down'] or template_params['ratio_up_down']:
    #            true_fmap = np.array([true_fmap_up,true_fmap_down])
    #        else:
    #            true_fmap = np.append(true_fmap_up,true_fmap_down)
    #    else:
    #        true_template = template_maker[0].get_template(template_params)  
    #        true_fmap = flatten_map(true_template,chan=template_params['channel'])
    return true_fmap

def hole_ice_line_func(x,k):
    # line goes through point (0.2,1)
    return k*x+1-k*0.2  

def dom_eff_line_func(x,k):
    # line goes through point (0.91,1)
    return k*x+1-k*0.91

def plane_func(param,a,b,c,d):
    x = param[0]
    y = param[1]
    return -(a/c)*x - (b/c)*y + d/c 

def get_map_plane_fit(tmaps,template_params):
    dom_eff_val = template_params['dom_eff']
    hole_ice_val = template_params['hole_ice']
    dom_eff = np.array([0.91,1.0,0.95,1.1,1.05,0.91,0.91,0.91])
    hole_ice = 10*np.array([1.0/50,1.0/50,1.0/50,1.0/50,1.0/50,0.0,1.0/30,1.0/100])     # unit : dm^-1
    output_maps = np.empty(len(tmaps[0]))
    tmaps_t = tmaps.transpose()
    for i in range(0,len(tmaps_t)):
        bin_val = tmaps_t[i]/tmaps_t[i][1]
        popt, pcov = curve_fit(plane_func,np.array([dom_eff,hole_ice]),bin_val)
        a = popt[0]
        b = popt[1]
        c = popt[2]
        d = popt[3]
        val = plane_func(np.array([dom_eff_val,hole_ice_val]),a,b,c,d)
        if val <= 0.0:
            print "all 8 values = ", bin_val
            print "a, b, c, d = ", a, " ", b , " ", c, " ", d
            print "fit val = ", val
        output_maps[i] = tmaps_t[i][1]*val
    #print "At dom_eff = ", dom_eff_val , " , hole ice = ", hole_ice_val, "output_maps = ", output_maps
    return output_maps

def get_map_two_linear_fits(tmaps,template_params):
    dom_eff_val = template_params['dom_eff']
    hole_ice_val = template_params['hole_ice']
    output_maps = np.empty(len(tmaps[0]))

    dom_eff = np.array([0.91,1.0,0.95,1.1,1.05])
    tmaps_dom_eff = np.array([tmaps[0],tmaps[1],tmaps[2],tmaps[3],tmaps[4]])
    tmaps_t_dom_eff = tmaps_dom_eff.transpose()

    hole_ice = 10*np.array([1.0/50,0,1.0/100,1.0/30])       # unit : dm^-1
    tmaps_hole_ice = np.array([tmaps[0],tmaps[5],tmaps[7],tmaps[6]])
    tmaps_t_hole_ice = tmaps_hole_ice.transpose()
    for i in range(0,len(tmaps_t_hole_ice)):
        bin_val_1 = tmaps_t_dom_eff[i]/tmaps_t_dom_eff[i][0]
        popt_1, pcov_1 = curve_fit(dom_eff_line_func,dom_eff,bin_val_1)
        k1 = popt_1[0]
        bin_val_2 = tmaps_t_hole_ice[i]/tmaps_t_hole_ice[i][0]
        popt_2, pcov_2 = curve_fit(hole_ice_line_func,hole_ice,bin_val_2)
        k2 = popt_2[0]
        val = k1*(dom_eff_val-0.91)+ k2*(hole_ice_val-0.2) + 1
        output_maps[i] = tmaps_t_dom_eff[i][0]*val
    return output_maps

def get_pseudo_tau_fmap(template_maker,fiducial_params,seed=None,chan=None):
    if fiducial_params['residual_up_down'] or fiducial_params['simp_up_down'] or fiducial_params['ratio_up_down']:
        template_maker_up = template_maker[0]
        template_maker_down = template_maker[1]
        template_up = template_maker_up.get_tau_template(fiducial_params)  
        template_down = template_maker_down.get_tau_template(fiducial_params)  
        reflected_template_down = get_flipped_map(template_down,chan=fiducial_params['channel'])
        true_fmap_up = flatten_map(template_up,chan=fiducial_params['channel'])
        true_fmap_down = flatten_map(reflected_template_down,chan=fiducial_params['channel'])
        fmap_up = get_random_map(true_fmap_up, seed=get_seed())
        fmap_down = get_random_map(true_fmap_down, seed=get_seed())
        if fiducial_params['residual_up_down']:
            fmap = fmap_up-fmap_down
        elif fiducial_params['ratio_up_down']:
            fmap = np.array([fmap_up,fmap_down])
        else:
            fmap = np.append(fmap_up,fmap_down)
    else:
        true_template = template_maker.get_tau_template(fiducial_params)  
        true_fmap = flatten_map(true_template,chan=chan)
        fmap = get_random_map(true_fmap, seed=seed)
    return fmap

def get_asimov_fmap(template_maker,fiducial_params,chan=None):
    """Creates a true template from fiducial_params"""

    true_template = template_maker.get_template(fiducial_params)
    return flatten_map(true_template,chan=chan)

def flatten_map(template,chan='all'):
    """
    Takes a final level true (expected) template of trck/cscd, and returns a
    single flattened map of trck appended to cscd, with all zero bins
    removed.
    """

    logging.trace("Getting flattened map of chan: %s"%chan)

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
        raise ValueError(
            "chan: '%s' not implemented! Allowed: ['all', 'trck', 'cscd','no_pid']")

    fmap = np.array(fmap)[np.nonzero(fmap)]
    return fmap

def get_seed():
    """
    Returns a random seed from /dev/urandom that can be used to seed the random
    state, e.g. for the poisson random variates.
    """

    return int(os.urandom(4).encode('hex'),16)

def get_up_map(map,chan):
    ''' Gets the upgoing map from a full sky map and flip it.'''
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
    ''' Gets the downgoing map from a full sky map and flip it.'''
    czbin_edges = len(map['cscd']['czbins'])
    czbin_mid_idx = (czbin_edges-1)/2
    if chan=='all':
        flavs=['trck','cscd']
    elif chan=='trck':
        flavs=['trck']
    elif chan=='cscd':
        flavs=['cscd']
    elif chan == 'no_pid':
        return {'no_pid':{
            'map': np.fliplr(map['trck']['map'][:,czbin_mid_idx:]+map['cscd']['map'][:,czbin_mid_idx:]),
            'ebins':map['trck']['ebins'],
            'czbins': np.sort(-map['trck']['czbins'][czbin_mid_idx:]) }}
    else:
        raise ValueError("chan: '%s' not implemented! Allowed: ['all', 'trck', 'cscd','no_pid']")
    return {flav:{
        'map': np.fliplr(map[flav]['map'][:,czbin_mid_idx:]),
        'ebins':map[flav]['ebins'],
        'czbins': np.sort(-map['trck']['czbins'][czbin_mid_idx:]) }
            for flav in flavs}

def get_flipped_map(map,chan):
    ''' Flip a map.'''
    if not np.alltrue(map['cscd']['czbins']>=0):
        raise ValueError("This map has to be down-going neutrinos!")
    if chan=='all':
        flavs=['trck','cscd']
    elif chan=='trck':
        flavs=['trck']
    elif chan=='cscd':
        flavs=['cscd']
    elif chan == 'no_pid':
        return {'no_pid':{
            'map': np.fliplr(map['trck']['map']+map['cscd']['map']),
            'ebins':map['trck']['ebins'],
            'czbins': np.sort(-map['cscd']['czbins']) }}
    else:
        raise ValueError("chan: '%s' not implemented! Allowed: ['all', 'trck', 'cscd','no_pid']")
    return {flav:{
        'map': np.fliplr(map[flav]['map']),
        'ebins':map[flav]['ebins'],
        'czbins': np.sort(-map[flav]['czbins']) }
            for flav in flavs}


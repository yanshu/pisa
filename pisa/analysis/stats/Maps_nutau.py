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
import h5py
from pisa.analysis.stats.LLHStatistics import get_random_map
from pisa.utils.log import logging
from pisa.utils.jsons import from_json,to_json
from pisa.resources.resources import find_resource
import pisa.analysis.stats.Maps as Maps

def get_burn_sample(burn_sample_file, ebins, czbins, get_map, get_array, channel):
    burn_sample_file = h5py.File(find_resource(burn_sample_file),'r')

    dLLH = np.array(burn_sample_file['IC86_Dunkman_L6']['delta_LLH'])
    L6_result = np.array(burn_sample_file['IC86_Dunkman_L6']['result'])
    reco_energy_all = np.array(burn_sample_file['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['energy'])
    reco_coszen_all = np.array(np.cos(burn_sample_file['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['zenith']))

    print "before L6 cut, no. of burn sample = ", len(reco_coszen_all)

    dLLH_L6 = dLLH[L6_result==1]
    reco_energy_L6 = reco_energy_all[L6_result==1]
    reco_coszen_L6 = reco_coszen_all[L6_result==1]
    print "after L6 cut, no. of burn sample = ", len(reco_coszen_L6)
   
    # throw away dLLH < -3
    #reco_energy_L6_cut1 = reco_energy_L6[dLLH_L6>=-3]
    #reco_coszen_L6_cut1 = reco_coszen_L6[dLLH_L6>=-3]
    #dLLH_L6_cut1 = dLLH_L6[dLLH_L6>=-3]
    # don't throw away dLLH < -3
    reco_energy_L6_cut1 = reco_energy_L6
    reco_coszen_L6_cut1 = reco_coszen_L6
    dLLH_L6_cut1 = dLLH_L6

    # write burn sample data to dictionary
    burn_sample_dict = {}
    for flavor in ['cscd','trck']:
        if flavor == 'cscd':
            cut = dLLH_L6_cut1 < 3.0 
        if flavor == 'trck':
            cut = dLLH_L6_cut1 >= 3.0 
        reco_energy_L6_final = reco_energy_L6_cut1[cut]
        reco_coszen_L6_final = reco_coszen_L6_cut1[cut]

        bins = (ebins, czbins)
        burn_sample_hist,_,_ = np.histogram2d(reco_energy_L6_final,reco_coszen_L6_final,bins=bins)
        burn_sample_dict[flavor] = burn_sample_hist

    # get the burn sample maps (cz in [-1, 1])
    burn_sample_maps={}
    for flav in ['trck','cscd']:
        burn_sample_maps[flav] = {'map':burn_sample_dict[flav],
                                 'ebins':ebins,
                                 'czbins':czbins}
    if get_map:
        return burn_sample_maps

    if get_array:
        # return a 1D array (used for the fit in the LLR analysis)
        burn_sample_map_up = get_up_map(burn_sample_maps, channel=channel)
        burn_sample_map_flipped_down = get_flipped_down_map(burn_sample_maps, channel=channel)
        flattend_burn_sample_map_up = Maps.flatten_map(burn_sample_map_up, channel=channel)
        flattend_burn_sample_map_flipped_down = Maps.flatten_map(burn_sample_map_flipped_down, channel=channel)
        burn_sample_in_array = np.append(flattend_burn_sample_map_up, flattend_burn_sample_map_flipped_down)
        return burn_sample_in_array


def get_asimov_data_fmap_up_down(template_maker, fiducial_params, channel=None):
    if fiducial_params['residual_up_down'] or fiducial_params['simp_up_down'] or fiducial_params['ratio_up_down']:
        template_maker_up = template_maker[0]
        template_maker_down = template_maker[1]
        template_up = template_maker_up.get_template(fiducial_params)  
        template_down = template_maker_down.get_template(fiducial_params)  

        template_up_down_combined = get_combined_map(template_up,template_down, channel=fiducial_params['channel'])
        template_up = get_up_map(template_up_down_combined, channel=fiducial_params['channel'])
        reflected_template_down = get_flipped_down_map(template_up_down_combined, channel=fiducial_params['channel'])

        fmap_up = Maps.flatten_map(template_up, channel=fiducial_params['channel'])
        fmap_down = Maps.flatten_map(reflected_template_down, channel=fiducial_params['channel'])
        #fmap_up = np.int32(true_fmap_up+0.5)
        #fmap_down = np.int32(true_fmap_down+0.5)
        if fiducial_params['residual_up_down']:
            fmap = fmap_up-fmap_down
        elif fiducial_params['ratio_up_down']:
            fmap = np.array([fmap_up, fmap_down])
        else:
            fmap = np.append(fmap_up, fmap_down)
    else:
        true_template = template_maker.get_template(fiducial_params)  
        true_fmap = Maps.flatten_map(true_template, channel=channel)
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

        template_up_down_combined = get_combined_map(template_up,template_down, channel=fiducial_params['channel'])
        template_up = get_up_map(template_up_down_combined, channel=fiducial_params['channel'])
        reflected_template_down = get_flipped_down_map(template_up_down_combined, channel=fiducial_params['channel'])

        # add domeff and/or hole ice effects
        [template_up_dh,reflected_template_down_dh] = apply_domeff_holeice([template_up,reflected_template_down],fiducial_params)
        [template_up_dh_prcs,reflected_template_down_dh_prcs] = apply_reco_precisions([template_up_dh,reflected_template_down_dh],fiducial_params)
        true_fmap_up = Maps.flatten_map(template_up_dh_prcs, channel=fiducial_params['channel'])
        true_fmap_down = Maps.flatten_map(reflected_template_down_dh_prcs, channel=fiducial_params['channel'])
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
        # add domeff and/or hole ice effects
        true_template_dh = apply_domeff_holeice(true_template, fiducial_params)
        true_template_dh_prcs = apply_reco_precisions(true_template_dh, fiducial_params)
        true_fmap = Maps.flatten_map(true_template_dh_prcs, channel=channel)
        fmap = get_random_map(true_fmap, seed=seed)
    return fmap

def apply_domeff_holeice(template, params):
    # Apply DOMeff and HoleIce changes to template 
    channel=params['channel']
    dom_eff_val = params['dom_eff']
    hole_ice_val = params['hole_ice']
    slope = from_json(find_resource(params['domeff_holeice_slope_file']))
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
        for flav in flavs:
            assert(np.all(template_up[flav]['czbins']) <= 0)
            assert(np.all(template_down[flav]['czbins']) >= 0)
        output_map_up = {}
        output_map_down = {}
        for flav in flavs:
            output_map_up[flav] = {}
            output_map_down[flav] = {}
            slope_DE_up = slope['slopes']['data_tau']['k_DomEff'][flav]['up']
            slope_DE_down = slope['slopes']['data_tau']['k_DomEff'][flav]['down']
            slope_HI_up = slope['slopes']['data_tau']['k_HoleIce'][flav]['up']
            slope_HI_down = slope['slopes']['data_tau']['k_HoleIce'][flav]['down']
            scale_up = slope_DE_up*(dom_eff_val-0.91)+ slope_HI_up*(hole_ice_val-0.2) + 1
            scale_down = slope_DE_down*(dom_eff_val-0.91)+ slope_HI_down*(hole_ice_val-0.2) + 1
            output_map_up[flav] = { 'map': (template_up[flav]['map'])*scale_up,
                                    'ebins':template_up[flav]['ebins'],
                                    'czbins': template_up[flav]['czbins'] }
            output_map_down[flav]={ 'map': (template_down[flav]['map'])*scale_down,
                                    'ebins':template_down[flav]['ebins'],
                                    'czbins': template_down[flav]['czbins'] }
        return [output_map_up,output_map_down]
    elif isinstance(template,dict):
        if flavs == ['trck', 'cscd']:
            assert(np.all(template['cscd']['czbins'] == template['trck']['czbins']))
        if np.all(template['cscd']['czbins']) <= 0:
            direction = 'up'
        elif np.all(template['cscd']['czbins']) >= 0:
            direction = 'down'
        else:
            raise ValueError( "czbins should be either >=0 or <=0! ")
        output_map = {}
        for flav in flavs:
            output_map[flav] = {}
            slope_DE = slope['slopes']['data_tau']['k_DomEff'][flav][direction]
            slope_HI = slope['slopes']['data_tau']['k_HoleIce'][flav][direction]
            scale = slope_DE*(dom_eff_val-0.91)+ slope_HI*(hole_ice_val-0.2) + 1
            output_map[flav]={ 'map': (template[flav]['map'])*scale,
                               'ebins':template[flav]['ebins'],
                               'czbins': template[flav]['czbins']}
        return output_map
    else:
       raise TypeError("The type of input template is wrong!")

def apply_reco_precisions(template, params):
    # Apply reconstruction precision changes to template 
    channel=params['channel']
    e_precision_up_val = params['e_reco_precision_up']
    e_precision_down_val = params['e_reco_precision_down']
    cz_precision_up_val = params['cz_reco_precision_up']
    cz_precision_down_val = params['cz_reco_precision_down']
    cubic_coeff = from_json(find_resource(params['reco_prcs_coeff_file']))
    if e_precision_up_val == 1.0 and e_precision_down_val == 1.0 and cz_precision_up_val == 1.0 and cz_precision_down_val == 1.0:
        return template
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
        for flav in flavs:
            assert(np.all(template_up[flav]['czbins']) <= 0)
            assert(np.all(template_down[flav]['czbins']) >= 0)
        output_map_up = {}
        output_map_down = {}
        for flav in flavs:
            output_map_up[flav] = {}
            output_map_down[flav] = {}
            for direction in ['up', 'down']:
                cubic_coeff_e_precision_up = cubic_coeff['coeffs']['data_tau']['e_reco_precision_up'][flav][direction]
                cubic_coeff_e_precision_down = cubic_coeff['coeffs']['data_tau']['e_reco_precision_down'][flav][direction]
                cubic_coeff_cz_precision_up = cubic_coeff['coeffs']['data_tau']['cz_reco_precision_up'][flav][direction]
                cubic_coeff_cz_precision_down = cubic_coeff['coeffs']['data_tau']['cz_reco_precision_down'][flav][direction]
                #scale = cubic_coeff_e_precision_up[:,:,0] * (e_precision_up_val**3 - 1.0) + cubic_coeff_e_precision_up[:,:,1] * (e_precision_up_val**2 - 1.0) + cubic_coeff_e_precision_up[:,:,2] * (e_precision_up_val - 1.0) + cubic_coeff_e_precision_down[:,:,0] * (e_precision_down_val**3 - 1.0) + cubic_coeff_e_precision_down[:,:,1] * (e_precision_down_val**2 - 1.0) + cubic_coeff_e_precision_down[:,:,2] * (e_precision_down_val - 1.0)  + cubic_coeff_cz_precision_up[:,:,0] * (cz_precision_up_val**3 - 1.0 ) + cubic_coeff_cz_precision_up[:,:,1] * (cz_precision_up_val**2 - 1.0 ) + cubic_coeff_cz_precision_up[:,:,2] * (cz_precision_up_val - 1.0 ) + cubic_coeff_cz_precision_down[:,:,0] * (cz_precision_down_val**3 - 1.0 ) + cubic_coeff_cz_precision_down[:,:,1] * (cz_precision_down_val**2 - 1.0 ) + cubic_coeff_cz_precision_down[:,:,2] * (cz_precision_down_val - 1.0 ) + 1.0
                scale = (cubic_coeff_e_precision_up[:,:,0] * (e_precision_up_val**3 - 1.0) + cubic_coeff_e_precision_up[:,:,1] * (e_precision_up_val**2 - 1.0) + cubic_coeff_e_precision_up[:,:,2] * (e_precision_up_val - 1.0) + 1.0) * (cubic_coeff_e_precision_down[:,:,0] * (e_precision_down_val**3 - 1.0) + cubic_coeff_e_precision_down[:,:,1] * (e_precision_down_val**2 - 1.0) + cubic_coeff_e_precision_down[:,:,2] * (e_precision_down_val - 1.0)  + 1.0) * (cubic_coeff_cz_precision_up[:,:,0] * (cz_precision_up_val**3 - 1.0 ) + cubic_coeff_cz_precision_up[:,:,1] * (cz_precision_up_val**2 - 1.0 ) + cubic_coeff_cz_precision_up[:,:,2] * (cz_precision_up_val - 1.0 ) + 1.0) * (cubic_coeff_cz_precision_down[:,:,0] * (cz_precision_down_val**3 - 1.0 ) + cubic_coeff_cz_precision_down[:,:,1] * (cz_precision_down_val**2 - 1.0 ) + cubic_coeff_cz_precision_down[:,:,2] * (cz_precision_down_val - 1.0 ) + 1.0)

                if direction == 'up':
                    output_map_up[flav] = { 'map': (template_up[flav]['map'])*scale,
                                            'ebins':template_up[flav]['ebins'],
                                            'czbins': template_up[flav]['czbins'] }
                else:
                    output_map_down[flav] = { 'map': (template_down[flav]['map'])*scale,
                                              'ebins':template_down[flav]['ebins'],
                                              'czbins': template_down[flav]['czbins'] }
        return [output_map_up,output_map_down]
    elif isinstance(template,dict):
        if flavs == ['trck', 'cscd']:
            assert(np.all(template['cscd']['czbins'] == template['trck']['czbins']))
        if np.all(template['cscd']['czbins']) <= 0:
            direction = 'up'
        elif np.all(template['cscd']['czbins']) >= 0:
            direction = 'down'
        else:
            raise ValueError( "czbins should be either >=0 or <=0! ")
        output_map = {}
        for flav in flavs:
            output_map[flav] = {}
            cubic_coeff_e_precision_up = cubic_coeff['coeffs']['data_tau']['e_reco_precision_up'][flav][direction]
            cubic_coeff_e_precision_down = cubic_coeff['coeffs']['data_tau']['e_reco_precision_down'][flav][direction]
            cubic_coeff_cz_precision_up = cubic_coeff['coeffs']['data_tau']['cz_reco_precision_up'][flav][direction]
            cubic_coeff_cz_precision_down = cubic_coeff['coeffs']['data_tau']['cz_reco_precision_down'][flav][direction]
            scale = cubic_coeff_e_precision_up[:,:,0] * (e_precision_up_val**3 - 1.0) + cubic_coeff_e_precision_up[:,:,1] * (e_precision_up_val**2 - 1.0) + cubic_coeff_e_precision_up[:,:,2] * (e_precision_up_val - 1.0) + cubic_coeff_e_precision_down[:,:,0] * (e_precision_down_val**3 - 1.0) + cubic_coeff_e_precision_down[:,:,1] * (e_precision_down_val**2 - 1.0) + cubic_coeff_e_precision_down[:,:,2] * (e_precision_down_val - 1.0)  + cubic_coeff_cz_precision_up[:,:,0] * (cz_precision_up_val**3 - 1.0 ) + cubic_coeff_cz_precision_up[:,:,1] * (cz_precision_up_val**2 - 1.0 ) + cubic_coeff_cz_precision_up[:,:,2] * (cz_precision_up_val - 1.0 ) + cubic_coeff_cz_precision_down[:,:,0] * (cz_precision_down_val**3 - 1.0 ) + cubic_coeff_cz_precision_down[:,:,1] * (cz_precision_down_val**2 - 1.0 ) + cubic_coeff_cz_precision_down[:,:,2] * (cz_precision_down_val - 1.0 ) + 1.0
            output_map[flav]={ 'map': (template[flav]['map'])*scale,
                               'ebins':template[flav]['ebins'],
                               'czbins': template[flav]['czbins']}
        return output_map
    else:
       raise TypeError("The type of input template is wrong!")


def get_true_template(template_params, template_maker):
    if template_params['theta23'] == 0.0:
        logging.info("Zero theta23, so generating no oscillations template...")
        true_template = template_maker.get_template_no_osc(template_params)
        # add domeff and/or hole ice effects
        true_template_dh = apply_domeff_holeice(true_template,template_params)
        true_template_dh_prcs = apply_reco_precisions(true_template_dh, template_params)
        true_fmap = Maps.flatten_map(true_template_dh_prcs, channel=template_params['channel'])
    elif type(template_maker)==list and len(template_maker)==2:
        template_maker_up = template_maker[0]
        template_maker_down = template_maker[1]
        template_up = template_maker_up.get_template(template_params)  
        template_down = template_maker_down.get_template(template_params)  

        template_up_down_combined = get_combined_map(template_up,template_down, channel= template_params['channel'])
        template_up = get_up_map(template_up_down_combined, channel= template_params['channel'])
        reflected_template_down = get_flipped_down_map(template_up_down_combined, channel= template_params['channel'])

        # add domeff and/or hole ice effects
        [template_up_dh,reflected_template_down_dh] = apply_domeff_holeice([template_up,reflected_template_down],template_params)
        [template_up_dh_prcs,reflected_template_down_dh_prcs] = apply_reco_precisions([template_up_dh,reflected_template_down_dh],template_params)
        true_fmap_up = Maps.flatten_map(template_up_dh_prcs, channel=template_params['channel'])
        true_fmap_down = Maps.flatten_map(reflected_template_down_dh_prcs, channel=template_params['channel'])
        if template_params['residual_up_down'] or template_params['ratio_up_down']:
            true_fmap = np.array([true_fmap_up, true_fmap_down])
        else:
            true_fmap = np.append(true_fmap_up, true_fmap_down)
    else:
        true_template = template_maker.get_template(template_params)  
        # add domeff and/or hole ice effects
        true_template_dh = apply_domeff_holeice(true_template,template_params)
        true_template_dh_prcs = apply_reco_precisions(true_template_dh, template_params)
        true_fmap = Maps.flatten_map(true_template_dh_prcs, channel=template_params['channel'])

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

        # add domeff and/or hole ice effects
        [template_up_dh,reflected_template_down_dh] = apply_domeff_holeice([template_up,reflected_template_down],fiducial_params)
        [template_up_dh_prcs,reflected_template_down_dh_prcs] = apply_reco_precisions([template_up_dh,reflected_template_down_dh],fiducial_params)
        true_fmap_up = Maps.flatten_map(template_up_dh_prcs, channel=fiducial_params['channel'])
        true_fmap_down = Maps.flatten_map(reflected_template_down_dh_prcs, channel=fiducial_params['channel'])

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
        true_template_dh = apply_domeff_holeice(true_template,fiducial_params)
        true_template_dh_prcs = apply_reco_precisions(true_template_dh, fiducial_params)
        true_fmap = Maps.flatten_map(true_template_dh_prcs, channel=channel)
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
    len_czbin_edges = len(map['cscd']['czbins'])
    assert(len_czbin_edges%2 == 1)    # length of cz_bin_edges has to be odd
    czbin_mid_idx = (len_czbin_edges-1)/2
    return {flav:{
        'map': map[flav]['map'][:,0:czbin_mid_idx],
        'ebins':map[flav]['ebins'],
        'czbins': map[flav]['czbins'][0:czbin_mid_idx+1] }
            for flav in flavs}

def get_flipped_down_map(map, channel):
    ''' Gets the downgoing map from a full sky map and flip it around cz = 0.'''
    len_czbin_edges = len(map['cscd']['czbins'])
    assert(len_czbin_edges %2 == 1)    # length of cz_bin_edges has to be odd
    czbin_mid_idx = (len_czbin_edges-1)/2
    if channel=='all':
        flavs=['trck', 'cscd']
    elif channel=='trck':
        flavs=['trck']
    elif channel=='cscd':
        flavs=['cscd']
    elif channel == 'no_pid':
        return {'no_pid':{
            'map': np.fliplr(map['trck']['map'][:,czbin_mid_idx:]+map['cscd']['map'][:,czbin_mid_idx:]),
            'ebins':map['trck']['ebins'],
            'czbins': np.sort(-map['trck']['czbins'][czbin_mid_idx:]) }}
    else:
        raise ValueError(" channel: '%s' not implemented! Allowed: ['all', 'trck', 'cscd', 'no_pid']"%channel)
    return {flav:{
        'map': np.fliplr(map[flav]['map'][:,czbin_mid_idx:]),
        'ebins':map[flav]['ebins'],
        'czbins': np.sort(-map[flav]['czbins'][czbin_mid_idx:]) }
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

def get_combined_map(amap, bmap, channel):
    ''' Sum the up-going and the down-going map.'''
    if not (np.all(amap['cscd']['czbins'] == bmap['cscd']['czbins']) and np.all(amap['trck']['czbins'] == bmap['trck']['czbins'])):
        raise ValueError("These two maps should have the same cz binning!")
    if channel=='all':
        flavs=['trck', 'cscd']
    elif channel=='trck':
        flavs=['trck']
    elif channel=='cscd':
        flavs=['cscd']
    elif channel == 'no_pid':
        return {'no_pid':{
            'map': amap['trck']['map']+ amap['cscd']['map']+ bmap['trck']['map']+bmap['cscd']['map'],
            'ebins':amap['trck']['ebins'],
            'czbins': amap['cscd']['czbins'] }}
    else:
        raise ValueError("channel: '%s' not implemented! Allowed: ['all', 'trck', 'cscd', 'no_pid']"%channel)
    return {flav:{
        'map': amap[flav]['map'] + bmap[flav]['map'],
        'ebins':amap[flav]['ebins'],
        'czbins': amap[flav]['czbins'] }
            for flav in flavs}

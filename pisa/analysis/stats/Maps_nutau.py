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
from pisa.utils.log import logging, profile
from pisa.utils.utils import Timer
from pisa.utils.jsons import from_json,to_json
from pisa.resources.resources import find_resource
import pisa.analysis.stats.Maps as Maps
from scipy.stats import poisson, norm
import pisa.utils.mcSimRunSettings as MCSRS
import pisa.utils.dataProcParams as DPP

def get_low_level_quantities(file_name, file_type, anlys_ebins, czbins, fields, sim_version, cuts='analysis',
                              run_setting_file='events/mc_sim_run_settings.json', det='deepcore',
                              data_proc_file='events/data_proc_params.json'):
    if sim_version=='dima':
        proc_version = '5digit'
    else:
        proc_version = '4digit'
    data_proc_params = DPP.DataProcParams(
            detector=det,
            proc_ver=proc_version,
            data_proc_params=find_resource(data_proc_file))
    run_settings = MCSRS.DetMCSimRunsSettings(find_resource(run_setting_file), detector=det)
    data = data_proc_params.getData(find_resource(file_name), run_settings=run_settings, file_type=file_type)
    cut_data = data_proc_params.applyCuts(data,
                    cuts=cuts,
                    return_fields=fields)
    return cut_data

def get_burn_sample_maps(file_name, anlys_ebins, czbins, output_form, channel, pid_remove, pid_bound, sim_version='4digit'):
    # right now only use burn sample with sim_version = '4digit'
    print "sim_version == ", sim_version
    if sim_version == "4digit":
        Reco_Neutrino_Name = 'IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino'
        Reco_Track_Name = 'IC86_Dunkman_L6_MultiNest8D_PDG_Track'
    elif sim_version == "5digit" or sim_version=="dima":
        Reco_Neutrino_Name = 'IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC'
        Reco_Track_Name = 'IC86_Dunkman_L6_PegLeg_MultiNest8D_Track'
    else:
        raise ValueError('only allow 4digit, 5digit(H2 model for hole ice) or dima (dima p1 and p2 for hole ice)!') 

    burn_sample_file = h5py.File(find_resource(file_name),'r')
    L6_result = np.array(burn_sample_file['IC86_Dunkman_L6']['result'])
    dLLH = np.array(burn_sample_file['IC86_Dunkman_L6']['delta_LLH'])
    reco_energy_all = np.array(burn_sample_file[Reco_Neutrino_Name]['energy'])
    reco_coszen_all = np.array(np.cos(burn_sample_file[Reco_Neutrino_Name]['zenith']))
    reco_trck_len_all = np.array(burn_sample_file[Reco_Track_Name]['length'])
    burn_sample_file.close()
    #print "before L6 cut, no. of burn sample = ", len(reco_coszen_all)

    dLLH_L6 = dLLH[L6_result==1]
    reco_energy_L6 = reco_energy_all[L6_result==1]
    reco_coszen_L6 = reco_coszen_all[L6_result==1]

    #print "after L6 cut, no. of burn sample = ", len(reco_coszen_L6)
   
    # throw away dLLH < -3
    reco_energy_L6_cut1 = reco_energy_L6[dLLH_L6>=pid_remove]
    reco_coszen_L6_cut1 = reco_coszen_L6[dLLH_L6>=pid_remove]
    dLLH_L6_cut1 = dLLH_L6[dLLH_L6>=pid_remove]

    # don't throw away dLLH < -3, only use this when using param service for PID in PISA
    #reco_energy_L6_cut1 = reco_energy_L6
    #reco_coszen_L6_cut1 = reco_coszen_L6
    #dLLH_L6_cut1 = dLLH_L6

    # get cscd array and trck array
    reco_energy = {}
    reco_coszen = {}

    # write burn sample data to dictionary
    burn_sample_dict = {}
    for flav in ['cscd','trck']:
        if flav == 'cscd':
            cut_pid = dLLH_L6_cut1 < pid_bound 
        if flav == 'trck':
            cut_pid = dLLH_L6_cut1 >= pid_bound 
        reco_energy_L6_pid = reco_energy_L6_cut1[cut_pid]
        reco_coszen_L6_pid = reco_coszen_L6_cut1[cut_pid]

        reco_energy[flav] = reco_energy_L6_pid
        reco_coszen[flav] = reco_coszen_L6_pid

        bins = (anlys_ebins, czbins)
        burn_sample_hist,_,_ = np.histogram2d(reco_energy_L6_pid,reco_coszen_L6_pid,bins=bins)
        burn_sample_dict[flav] = burn_sample_hist

    # get the burn sample maps (cz in [-1, 1])
    burn_sample_maps={}
    for flav in ['trck','cscd']:
        burn_sample_maps[flav] = {'map':burn_sample_dict[flav],
                                 'ebins':anlys_ebins,
                                 'czbins':czbins}

    if output_form == 'map':
        return burn_sample_maps

    if output_form == 'side_map':
        f_select = from_json('sgnl_side_region_selection.json')
        burn_sample_maps_side = {}
        for flav in ['trck','cscd']:
            burn_sample_maps_side[flav] = {'map':burn_sample_dict[flav]* f_select['side'][flav],
                                     'ebins':anlys_ebins,
                                     'czbins':czbins}
        return burn_sample_maps_side

    if output_form == 'sgnl_map':
        f_select = from_json('sgnl_side_region_selection.json')
        burn_sample_maps_sgnl = {}
        for flav in ['trck','cscd']:
            burn_sample_maps_sgnl[flav] = {'map':burn_sample_dict[flav]* f_select['sgnl'][flav],
                                     'ebins':anlys_ebins,
                                     'czbins':czbins}
        return burn_sample_maps_sgnl

    if output_form == 'array':
        # return a 1D array (used for the fit in the LLR analysis)
        flattend_burn_sample_map = Maps.flatten_map(burn_sample_maps, channel=channel)
        return flattend_burn_sample_map 


def get_asimov_data_fmap_up_down(template_maker, fiducial_params, channel=None):
    true_template = template_maker.get_template(fiducial_params, num_data_events=None)
    true_fmap = Maps.flatten_map(true_template, channel=channel)
    return true_fmap

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

    true_template = template_maker.get_template(fiducial_params, num_data_events=None)
    true_fmap = Maps.flatten_map(true_template, channel=channel)
    if seed:
        fmap = get_random_map(true_fmap, seed=seed)
    else:
        fmap = get_random_map(true_fmap, seed=Maps.get_seed())
    return fmap

def get_stat_fluct_map(template_maker, fiducial_params, channel, seed=None):
    """
    Get a map that is fluctuated by the statistical uncertainty of the model
    """
    true_template = template_maker.get_template(fiducial_params, num_data_events=None)
    true_fmap = Maps.flatten_map(true_template, channel=channel)
    sumw2 = Maps.flatten_map(true_template, channel=channel, mapname='sumw2')
    sigma = np.sqrt(sumw2)
    if not seed is None:
        np.random.seed(seed=seed)
    fmap = np.array([norm.rvs(m,s) for m,s in zip(true_fmap,sigma)])
    fmap.clip(0,out=fmap)
    return fmap

def get_true_template(template_params, template_maker, num_data_events, no_sys_maps=False, error=False, both=False):
    if template_params['theta23'] == 0.0:
        logging.info("Zero theta23, so generating no oscillations template...")
        true_template = template_maker.get_template(template_params, num_data_events=num_data_events, no_osc_maps=True, no_sys_maps = no_sys_maps)
    else:
        true_template = template_maker.get_template(template_params, num_data_events=num_data_events, no_osc_maps=False, no_sys_maps= no_sys_maps)  

    if not both:
        true_fmap = Maps.flatten_map(true_template, channel=template_params['channel'])
        if error:
            error_map = Maps.flatten_map(true_template, channel=template_params['channel'],mapname='sumw2')
            return true_fmap, error_map
        return true_fmap
    else:
        map_nu = Maps.flatten_map(true_template, channel=template_params['channel'],mapname = 'map_nu')
        map_mu = Maps.flatten_map(true_template, channel=template_params['channel'],mapname = 'map_mu')
        if error:
            sumw2_nu = Maps.flatten_map(true_template, channel=template_params['channel'],mapname='sumw2_nu')
            sumw2_mu = Maps.flatten_map(true_template, channel=template_params['channel'],mapname='sumw2_mu')
            return map_nu, map_mu, sumw2_nu, sumw2_mu
        return map_nu, map_mu


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
    true_template = template_maker.get_tau_template(fiducial_params)  
    true_fmap = Maps.flatten_map(true_template, channel=channel)
    if seed:
        fmap = get_random_map(true_fmap, seed=seed)
    else:
        fmap = get_random_map(true_fmap, seed=Maps.get_seed())
    return fmap

def get_half_map(map, channel, section):
    ''' Gets the upgoing map from a full sky map.'''
    assert(section == 'up' or section == 'down')
    len_czbin_edges = len(map['cscd']['czbins'])
    assert(len_czbin_edges%2 == 1)    # length of cz_bin_edges has to be odd
    czbin_mid_idx = (len_czbin_edges-1)/2
    if channel =='all':
        flavs=['trck', 'cscd']
    elif channel =='trck':
        flavs=['trck']
    elif channel =='cscd':
        flavs=['cscd']
    elif channel == 'no_pid':
        return {'no_pid':{
            'map': map['trck']['map'][:,0:czbin_mid_idx]+map['cscd']['map'][:,0:czbin_mid_idx] if section =='up' else map['trck']['map'][:,czbin_mid_idx:]+map['cscd']['map'][:,czbin_mid_idx:],
                'ebins':map['trck']['ebins'],
                'czbins': map['trck']['czbins'][0:czbin_mid_idx+1] if section =='up' else map['trck']['czbins'][czbin_mid_idx:] }}
    else:
        raise ValueError("channel: '%s' not implemented! Allowed: ['all', 'trck', 'cscd', 'no_pid']"%channel)
    return {flav:{
        'map': map[flav]['map'][:,0:czbin_mid_idx] if section =='up' else map[flav]['map'][:,czbin_mid_idx:],
        'ebins':map[flav]['ebins'],
        'czbins': map[flav]['czbins'][0:czbin_mid_idx+1] if section =='up' else map[flav]['czbins'][czbin_mid_idx:] }
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

def get_concatenated_map(up_map, down_map, channel):
    ''' Sum the up-going and the down-going map.'''
    if not (np.all(up_map['cscd']['czbins']<=0) and np.all(down_map['cscd']['czbins']>=0) ):
        raise ValueError("These two maps have wrong cz binnings!")
    if channel=='all':
        flavs=['trck', 'cscd']
    elif channel=='trck':
        flavs=['trck']
    elif channel=='cscd':
        flavs=['cscd']
    elif channel == 'no_pid':
        return {'no_pid':{
            'map': up_map['trck']['map']+ up_map['cscd']['map']+ down_map['trck']['map']+down_map['cscd']['map'],
            'ebins':up_map[flav]['ebins'],
            'czbins': np.hstack((up_map[flav]['czbins'][:-1], down_map[flav]['czbins'][:])) }}
    else:
        raise ValueError("channel: '%s' not implemented! Allowed: ['all', 'trck', 'cscd', 'no_pid']"%channel)
    return {flav:{
        'map': np.hstack((up_map[flav]['map'], down_map[flav]['map'])),
        'ebins':up_map[flav]['ebins'],
        'czbins': np.hstack((up_map[flav]['czbins'][:-1], down_map[flav]['czbins'][:])) }
            for flav in flavs}

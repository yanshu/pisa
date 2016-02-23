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

class Maps_nutau_cls():

    def __init__(self):
        self.reco_prcs_coeff_file = None
        self.cubic_coeff = None
        self.domeff_holeice_slope_file = None
        self.slope = None

    def get_burn_sample(self,burn_sample_file, anlys_ebins, czbins, output_form, cut_level, channel):

        burn_sample_file = h5py.File(find_resource(burn_sample_file),'r')

        dLLH = np.array(burn_sample_file['IC86_Dunkman_L6']['delta_LLH'])
        L6_result = np.array(burn_sample_file['IC86_Dunkman_L6']['result'])

        reco_x_all = burn_sample_file['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['x']
        reco_y_all = burn_sample_file['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['y']
        reco_z_all = burn_sample_file['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['z']
        reco_t_all = burn_sample_file['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['time']

        reco_energy_all = np.array(burn_sample_file['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['energy'])
        reco_coszen_all = np.array(np.cos(burn_sample_file['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['zenith']))
        reco_trck_len_all = np.array(burn_sample_file['IC86_Dunkman_L6_MultiNest8D_PDG_Track']['length'])

        #print "before L6 cut, no. of burn sample = ", len(reco_coszen_all)

        dLLH_L6 = dLLH[L6_result==1]
        reco_energy_L6 = reco_energy_all[L6_result==1]
        reco_coszen_L6 = reco_coszen_all[L6_result==1]

        reco_x_L6 = reco_x_all[L6_result==1]
        reco_y_L6 = reco_y_all[L6_result==1]
        reco_z_L6 = reco_z_all[L6_result==1]
        reco_t_L6 = reco_t_all[L6_result==1]
        reco_trck_len_L6 = reco_trck_len_all[L6_result==1]

        #print "after L6 cut, no. of burn sample = ", len(reco_coszen_L6)
       
        # throw away dLLH < -3
        #reco_energy_L6_cut1 = reco_energy_L6[dLLH_L6>=-3]
        #reco_coszen_L6_cut1 = reco_coszen_L6[dLLH_L6>=-3]
        #dLLH_L6_cut1 = dLLH_L6[dLLH_L6>=-3]

        # don't throw away dLLH < -3
        reco_energy_L6_cut1 = reco_energy_L6
        reco_coszen_L6_cut1 = reco_coszen_L6
        dLLH_L6_cut1 = dLLH_L6

        # get cscd array and trck array
        reco_energy = {}
        reco_coszen = {}

        # write burn sample data to dictionary
        burn_sample_dict = {}
        for flav in ['cscd','trck']:
            if flav == 'cscd':
                cut_pid = dLLH_L6_cut1 < 3.0 
            if flav == 'trck':
                cut_pid = dLLH_L6_cut1 >= 3.0 
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

        if output_form == 'reco_info':
            return (reco_x_L6, reco_y_L6, reco_z_L6, reco_t_L6, reco_energy_L6, reco_coszen_L6, reco_trck_len_L6)

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
            burn_sample_map_up = self.get_up_map(burn_sample_maps, channel='all')
            burn_sample_map_flipped_down = self.get_flipped_down_map(burn_sample_maps, channel='all')
            flattend_burn_sample_map_up = Maps.flatten_map(burn_sample_map_up, channel=channel)
            flattend_burn_sample_map_flipped_down = Maps.flatten_map(burn_sample_map_flipped_down, channel=channel)
            burn_sample_in_array = np.append(flattend_burn_sample_map_up, flattend_burn_sample_map_flipped_down)
            return burn_sample_in_array


    def get_asimov_data_fmap_up_down(self,template_maker, fiducial_params, channel=None):
        if fiducial_params['residual_up_down'] or fiducial_params['simp_up_down'] or fiducial_params['ratio_up_down']:
            template_maker_up = template_maker[0]
            template_maker_down = template_maker[1]
            template_up = template_maker_up.get_template(fiducial_params)  
            template_down = template_maker_down.get_template(fiducial_params)  

            template_up_down_combined = self.get_combined_map(template_up,template_down, channel='all')
            template_up = self.get_up_map(template_up_down_combined, channel='all')
            reflected_template_down = self.get_flipped_down_map(template_up_down_combined, channel='all')

            [template_up_dh,reflected_template_down_dh] = self.apply_domeff_holeice([template_up,reflected_template_down],fiducial_params,channel= 'all')
            [template_up_dh_prcs,reflected_template_down_dh_prcs] = self.apply_reco_precisions([template_up_dh,reflected_template_down_dh],fiducial_params,channel= 'all')
            fmap_up = Maps.flatten_map(template_up_dh_prcs, channel=fiducial_params['channel'])
            fmap_down = Maps.flatten_map(reflected_template_down_dh_prcs, channel=fiducial_params['channel'])
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
            # add domeff and/or hole ice effects
            true_template_dh = self.apply_domeff_holeice(true_template, fiducial_params)
            true_template_dh_prcs = self.apply_reco_precisions(true_template_dh, fiducial_params)
            true_fmap = Maps.flatten_map(true_template_dh_prcs, channel=channel)
        return fmap

    def get_pseudo_data_fmap(self,template_maker, fiducial_params, channel, seed=None):
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

            template_up_down_combined = self.get_combined_map(template_up,template_down, channel='all')
            template_up = self.get_up_map(template_up_down_combined, channel='all')
            reflected_template_down = self.get_flipped_down_map(template_up_down_combined, channel='all')

            # add domeff and/or hole ice effects
            [template_up_dh,reflected_template_down_dh] = self.apply_domeff_holeice([template_up,reflected_template_down],fiducial_params,channel= 'all')
            [template_up_dh_prcs,reflected_template_down_dh_prcs] = self.apply_reco_precisions([template_up_dh,reflected_template_down_dh],fiducial_params,channel= 'all')
            true_fmap_up = Maps.flatten_map(template_up_dh_prcs, channel=fiducial_params['channel'])
            true_fmap_down = Maps.flatten_map(reflected_template_down_dh_prcs, channel=fiducial_params['channel'])
            # if we want to recreate the same template, then use the input seed for both
            if seed:
                fmap_up = get_random_map(true_fmap_up, seed=seed)
                fmap_down = get_random_map(true_fmap_down, seed=seed)
            else:
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
            true_template_dh = self.apply_domeff_holeice(true_template, fiducial_params,channel= 'all')
            true_template_dh_prcs = self.apply_reco_precisions(true_template_dh, fiducial_params,channel= 'all')
            true_fmap = Maps.flatten_map(true_template_dh_prcs, channel=channel)
            if seed:
                fmap = get_random_map(true_fmap, seed=seed)
            else:
                fmap = get_random_map(true_fmap, seed=Maps.get_seed())
        return fmap

    def apply_domeff_holeice(self,template, params, channel):
        # to be deleted
        return template

    def apply_reco_precisions(self,template, params, channel):
        # to be deleted
        return template

    def get_template_for_plot(self,template_params, template_maker):
        flavs=['trck', 'cscd']
        if template_params['theta23'] == 0.0:
            #TODO
            logging.info("Zero theta23, so generating no oscillations template...")
            true_template = template_maker.get_template_no_osc(template_params)
            # add domeff and/or hole ice effects
            true_template_dh = self.apply_domeff_holeice(true_template,template_params,channel= 'all')
            true_template_dh_prcs = self.apply_reco_precisions(true_template_dh, template_params, channel= 'all')
        elif type(template_maker)==list and len(template_maker)==2:
            template_maker_up = template_maker[0]
            template_maker_down = template_maker[1]
            template_up = template_maker_up.get_template(template_params)  
            template_down = template_maker_down.get_template(template_params)  

            template_up_down_combined = self.get_combined_map(template_up,template_down, channel= 'all')
            template_up = self.get_up_map(template_up_down_combined, channel= 'all')
            reflected_template_down = self.get_flipped_down_map(template_up_down_combined, channel= 'all')

            # add domeff and/or hole ice effects
            [template_up_dh,reflected_template_down_dh] = self.apply_domeff_holeice([template_up,reflected_template_down],template_params, channel= 'all')
            [template_up_dh_prcs,reflected_template_down_dh_prcs] = self.apply_reco_precisions([template_up_dh,reflected_template_down_dh],template_params, channel= 'all')

            template_down_dh_prcs = {flav:{
                'map': np.fliplr(reflected_template_down_dh_prcs[flav]['map']),
                'ebins':reflected_template_down_dh_prcs[flav]['ebins'],
                'czbins': np.sort(-reflected_template_down_dh_prcs[flav]['czbins']) }
                for flav in flavs}
            output_map= self.get_concatenated_map(template_up_dh_prcs, template_down_dh_prcs, channel = 'all')
            return output_map

        else:
            #TODO
            true_template = template_maker.get_template(template_params)  
            # add domeff and/or hole ice effects
            #true_template_dh = self.apply_domeff_holeice(true_template,template_params,channel= 'all')
            #reflected_true_template_dh_prcs = self.apply_reco_precisions(true_template_dh, template_params,channel= 'all')
            #true_template_dh_prcs = {flav:{
            #    'map': np.fliplr(reflected_true_template_dh_prcs[flav]['map']),
            #    'ebins':reflected_true_template_dh_prcs[flav]['ebins'],
            #    'czbins': np.sort(-reflected_true_template_dh_prcs[flav]['czbins']) }
            #    for flav in flavs}
            return true_template_dh_prcs

    def get_true_template(self,template_params, template_maker):
        if template_params['theta23'] == 0.0:
            logging.info("Zero theta23, so generating no oscillations template...")
            true_template = template_maker.get_template_no_osc(template_params)
            # add domeff and/or hole ice effects
            with Timer() as t:
                true_template_dh = self.apply_domeff_holeice(true_template,template_params,channel= 'all')
            profile.debug("==> elapsed time for dom eff/hole ice: %s sec"%t.secs)
            with Timer() as t:
                true_template_dh_prcs = self.apply_reco_precisions(true_template_dh, template_params, channel= 'all')
            profile.debug("==> elapsed time for reco precision: %s sec"%t.secs)
            true_fmap = Maps.flatten_map(true_template_dh_prcs, channel=template_params['channel'])
        elif type(template_maker)==list and len(template_maker)==2:
            template_maker_up = template_maker[0]
            template_maker_down = template_maker[1]
            template_up = template_maker_up.get_template(template_params)  
            template_down = template_maker_down.get_template(template_params)  

            template_up_down_combined = self.get_combined_map(template_up,template_down, channel= 'all')
            template_up = self.get_up_map(template_up_down_combined, channel= 'all')
            reflected_template_down = self.get_flipped_down_map(template_up_down_combined, channel= 'all')

            # add domeff and/or hole ice effects
            with Timer() as t:
                [template_up_dh,reflected_template_down_dh] = self.apply_domeff_holeice([template_up,reflected_template_down],template_params, channel= 'all')
            profile.debug("==> elapsed time for dom eff/hole ice: %s sec"%t.secs)
            with Timer() as t:
                [template_up_dh_prcs,reflected_template_down_dh_prcs] = self.apply_reco_precisions([template_up_dh,reflected_template_down_dh],template_params, channel= 'all')
            profile.debug("==> elapsed time for reco precision: %s sec"%t.secs)
            true_fmap_up = Maps.flatten_map(template_up_dh_prcs, channel=template_params['channel'])
            true_fmap_down = Maps.flatten_map(reflected_template_down_dh_prcs, channel=template_params['channel'])
            if template_params['residual_up_down'] or template_params['ratio_up_down']:
                true_fmap = np.array([true_fmap_up, true_fmap_down])
            else:
                true_fmap = np.append(true_fmap_up, true_fmap_down)
        else:
            true_template = template_maker.get_template(template_params)  
            # add domeff and/or hole ice effects
            with Timer() as t:
                true_template_dh = self.apply_domeff_holeice(true_template,template_params,channel= 'all')
            profile.debug("==> elapsed time for dom eff/hole ice: %s sec"%t.secs)
            with Timer() as t:
                true_template_dh_prcs = self.apply_reco_precisions(true_template_dh, template_params,channel= 'all')
            profile.debug("==> elapsed time for reco precision: %s sec"%t.secs)
            true_fmap = Maps.flatten_map(true_template_dh_prcs, channel=template_params['channel'])

        return true_fmap

    def get_pseudo_tau_fmap(self,template_maker, fiducial_params, channel=None, seed=None):
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

            template_up_down_combined = self.get_combined_map(template_up,template_down, channel='all')
            template_up = self.get_up_map(template_up_down_combined, channel='all')
            reflected_template_down = self.get_flipped_down_map(template_up_down_combined, channel='all')

            # add domeff and/or hole ice effects
            [template_up_dh,reflected_template_down_dh] = self.apply_domeff_holeice([template_up,reflected_template_down],fiducial_params,channel= 'all')
            [template_up_dh_prcs,reflected_template_down_dh_prcs] = self.apply_reco_precisions([template_up_dh,reflected_template_down_dh],fiducial_params,channel= 'all')
            true_fmap_up = Maps.flatten_map(template_up_dh_prcs, channel=fiducial_params['channel'])
            true_fmap_down = Maps.flatten_map(reflected_template_down_dh_prcs, channel=fiducial_params['channel'])

            # if we want to recreate the same template, then use the input seed for both
            if seed:
                fmap_up = get_random_map(true_fmap_up, seed=seed)
                fmap_down = get_random_map(true_fmap_down, seed=seed)
            else:
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
            true_template_dh = self.apply_domeff_holeice(true_template,fiducial_params,channel= 'all')
            true_template_dh_prcs = self.apply_reco_precisions(true_template_dh, fiducial_params,channel= 'all')
            true_fmap = Maps.flatten_map(true_template_dh_prcs, channel=channel)
            if seed:
                fmap = get_random_map(true_fmap, seed=seed)
            else:
                fmap = get_random_map(true_fmap, seed=Maps.get_seed())
        return fmap

    def get_up_map(self,map, channel):
        ''' Gets the upgoing map from a full sky map.'''
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
                'map': map['trck']['map'][:,0:czbin_mid_idx]+map['cscd']['map'][:,0:czbin_mid_idx],
                'ebins':map['trck']['ebins'],
                'czbins': map['trck']['czbins'][0:czbin_mid_idx+1] }}
        else:
            raise ValueError("channel: '%s' not implemented! Allowed: ['all', 'trck', 'cscd', 'no_pid']"%channel)
        return {flav:{
            'map': map[flav]['map'][:,0:czbin_mid_idx],
            'ebins':map[flav]['ebins'],
            'czbins': map[flav]['czbins'][0:czbin_mid_idx+1] }
                for flav in flavs}

    def get_flipped_down_map(self,map, channel):
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

    def get_flipped_map(self,map, channel):
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

    def get_combined_map(self,amap, bmap, channel):
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

    def get_concatenated_map(self,up_map, down_map, channel):
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

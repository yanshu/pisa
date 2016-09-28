#! /usr/bin/env python
#
# TemplateMaker.py
#
# Class to implement template-making procedure and to store as much data
# as possible to avoid re-running stages when not needed.
#
# author: Timothy C. Arlen - tca3@psu.edu
#         Sebastian Boeser - sboeser@uni-mainz.de
#
# date:   7 Oct 2014
#

import sys
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.constants import year
import h5py

from pisa.analysis.TemplateMaker_MC_functions import apply_reco_sys, get_osc_probs, apply_flux_ratio, apply_spectral_index
from pisa.analysis.TemplateMaker_MC_functions import apply_GENIE_mod, apply_Barr_mod, apply_Barr_flux_ratio, apply_GENIE_mod_oscFit

from pisa.utils.log import physics, profile, set_verbosity, logging
from pisa.resources.resources import find_resource
from pisa.utils.params import get_fixed_params, get_free_params, get_values, select_hierarchy
from pisa.utils.jsons import from_json, to_json, json_string
from pisa.utils.utils import Timer, oversample_binning
import pisa.utils.flavInt as flavInt
import pisa.utils.events as events

from pisa.oscillations.Oscillation import get_osc_flux

from pisa.reco.RecoServiceMC import RecoServiceMC
from pisa.reco.RecoServiceParam import RecoServiceParam
from pisa.reco.RecoServiceKernelFile import RecoServiceKernelFile
from pisa.reco.RecoServiceVBWKDE import RecoServiceVBWKDE
from pisa.reco.Reco import get_reco_maps

from pisa.pid import PID

from pisa.background.BackgroundServiceICC import BackgroundServiceICC 
from pisa.background.ICCBackground import add_icc_background
from pisa.sys.HoleIce import HoleIce
from pisa.sys.DomEfficiency import DomEfficiency
from pisa.sys.Resolution import Resolution

import itertools

class TemplateMaker:
    '''
    This class handles all steps needed to produce a template with a
    constant binning.

    The strategy employed will be to define all 'services' in the
    initialization process, make them members of the class, then use
    them later when needed.
    '''
    def __init__(self, template_settings, ebins, czbins, anlys_ebins,
                 oversample_e=None, oversample_cz=None, actual_oversample_e=None,
                 actual_oversample_cz=None, no_sys_maps=False, **kwargs):
        '''
        TemplateMaker class handles all of the setup and calculation of the
        templates for a given binning.

        Parameters:
        * template_settings - dictionary of all template-making settings
        * ebins - energy bin edges
        * czbins - coszen bin edges
        '''

        self.cache_params = None
        self.fluxes = None
        self.flux_maps = None
        self.osc_probs = None
        self.event_rate_maps = None
        self.event_rate_reco_maps = None
        self.event_rate_pid_maps = None
        self.wgt2_pid_map_cscd = None
        self.wgt2_pid_map_trck = None
        self.hole_ice_maps = None
        self.domeff_maps = None
        self.reco_prec_maps_e_up = None
        self.reco_prec_maps_e_down = None
        self.reco_prec_maps_cz_up = None
        self.sys_maps = None
        self.final_event_rate = None

        self.reco_mc_wt_file = template_settings['reco_mc_wt_file']
        self.aeff_weight_file = template_settings['aeff_weight_file'] 
        self.params = template_settings
        self.sim_ver = self.params['sim_ver']
        
        self.ebins = ebins
        self.czbins = czbins
        self.anlys_ebins = anlys_ebins
        self.oversample_e = oversample_e
        self.oversample_cz = oversample_cz
        self.actual_oversample_e = actual_oversample_e
        self.actual_oversample_cz = actual_oversample_cz
        self.oversample_ebins = oversample_binning(ebins, self.actual_oversample_e)
        self.oversample_czbins = oversample_binning(czbins, self.actual_oversample_cz)
        physics.debug("Using %u bins in energy from %.2f to %.2f GeV"%
                      (len(self.ebins)-1, self.ebins[0], self.ebins[-1]))
        physics.debug("Using %u bins in cos(zenith) from %.2f to %.2f"%
                      (len(self.czbins)-1, self.czbins[0], self.czbins[-1]))

        # Instantiate a flux model service
        flux_mode = template_settings['flux_mode']
        if flux_mode.lower() == 'bisplrep':
            from pisa.flux.myHondaFluxService import myHondaFluxService as HondaFluxService
            self.flux_service = HondaFluxService(oversample_e = self.oversample_e, oversample_cz = self.oversample_cz,**template_settings)
        elif flux_mode.lower() == 'integral-preserving':
            from pisa.flux.IPHondaFluxService_MC_merge import IPHondaFluxService
            self.flux_service = IPHondaFluxService(**template_settings)
        else:
            error_msg = "flux_mode: %s is not implemented! "%flux_mode
            error_msg+=" Please choose among: ['bisplrep', 'integral-preserving']"
            raise NotImplementedError(error_msg)

        # Oscillated Flux Service:
        osc_code = template_settings['osc_code']
        if osc_code == 'prob3':
            from pisa.oscillations.Prob3OscillationServiceMC_merge import Prob3OscillationServiceMC
            self.osc_service = Prob3OscillationServiceMC(
                self.oversample_ebins, self.oversample_czbins, **template_settings)
        elif osc_code == 'ocelot':
            from pisa.oscillations.OcelotOscillationServiceMC import OcelotOscillationServiceMC
            self.osc_service = OcelotOscillationServiceMC(self.oversample_ebins, self.oversample_czbins, atmos_model= 'simple', prob_model='Probabilities')
        else:
            error_msg = 'OscillationService NOT implemented for ' + \
                    'osc_code = %s' % osc_code
            raise NotImplementedError(error_msg)

        # Instantiate a PID service
        self.pid_service = PID.pid_service_factory(
            ebins= self.anlys_ebins, czbins=self.czbins, **template_settings
        )
        # set up pid ( remove pid < pid_remove and separate cscd and trck by pid_bound)
        self.pid_remove = self.params['pid_remove']
        self.pid_bound = self.params['pid_bound'] 
        print "self.pid_bound = ", self.pid_bound
        print "self.pid_remove = ", self.pid_remove

        # background service
        self.background_service = BackgroundServiceICC(self.anlys_ebins, self.czbins,
                                                     **template_settings)

        # hole ice sys
        if not no_sys_maps:
            # when we are generating fits (creating the json files)
            # for the first time ( no_sys_maps = True), this can't run 
            if self.sim_ver == 'dima':
                # only when using dima sets, do we have holeice_fwd_slope file
                self.HoleIce = HoleIce(template_settings['holeice_slope_file'], template_settings['holeice_fwd_slope_file'], sim_ver=self.sim_ver)
            else:
                self.HoleIce = HoleIce(template_settings['holeice_slope_file'], None, sim_ver=self.sim_ver)
            self.DomEfficiency = DomEfficiency(template_settings['domeff_slope_file'], sim_ver=self.sim_ver)
            self.Resolution_e_up = Resolution(template_settings['reco_prcs_coeff_file'],'e','up')
            self.Resolution_e_down = Resolution(template_settings['reco_prcs_coeff_file'],'e','down')
            self.Resolution_cz_up = Resolution(template_settings['reco_prcs_coeff_file'],'cz','up')
            self.Resolution_cz_down = Resolution(template_settings['reco_prcs_coeff_file'],'cz','down')

        #self.calc_mc_errors()

    def downsample_binning(self,maps):
        # the ugliest way to handle this.....sorry
        new_maps = {}
        new_maps['params'] = maps['params']
        for flavour in maps.keys():
            if flavour == 'params': continue
            new_maps[flavour] = {'ebins': self.ebins,
                                  'czbins': self.czbins}
            new_maps[flavour]['map'] = np.zeros((len(self.ebins)-1,len(self.czbins)-1))
            for e in xrange(len(self.ebins)-1):
                for cz in xrange(len(self.czbins)-1):
                    for e_o in xrange(self.actual_oversample_e):
                        for cz_o in xrange(self.actual_oversample_cz):
                            new_maps[flavour]['map'][e][cz] += maps[flavour]['map'][e*self.actual_oversample_e+e_o][cz*self.actual_oversample_cz+cz_o]
        return new_maps


    def calc_mc_errors(self):
        logging.info('Opening file: %s'%(self.aeff_weight_file))
        try:
            evts = events.Events(self.aeff_weight_file)
        except IOError,e:
            logging.error("Unable to open event data file %s"%simfile)
            logging.error(e)
            sys.exit(1)
        print "self.aeff_weight_file = " , self.aeff_weight_file
        
        # only keep events using bdt_score > bdt_cut
        print "self.params['further_bdt_cut'] = ", self.params['further_bdt_cut']
        for prim in ['nue', 'numu','nutau', 'nue_bar', 'numu_bar', 'nutau_bar']:
            for int_type in ['cc','nc']:
                l5_bdt_score = evts[prim][int_type]['dunkman_L5'].astype(np.float64)
                cut = l5_bdt_score >= self.params['further_bdt_cut']
                #print "np.shape of cut = ", np.shape(cut)
                #print "before bdt cut , len evts[prim][int_type][true_energy] = ", len(evts[prim][int_type]['true_energy'])
                #print "evts[prim][int_type].keys() = ", evts[prim][int_type].keys()
                for var in evts[prim][int_type].keys():
                    if var=='BARR_splines' or var=='GENSYS_splines':
                        for sys in evts[prim][int_type][var].keys():
                            evts[prim][int_type][var][sys] = evts[prim][int_type][var][sys][cut]
                        continue
                    try:
                        evts[prim][int_type][var] = evts[prim][int_type][var][cut]
                    except KeyError:
                        evts[prim][int_type][var] = np.ones_like(evts[prim][int_type]['true_energy'])
                #print "after bdt cut , len evts[prim][int_type][true_energy] = ", len(evts[prim][int_type]['true_energy'])
        
        osc_probs = get_osc_probs(evts, self.params, self.osc_service, ebins=self.ebins)
        all_reco_e = np.array([])
        all_reco_cz = np.array([])
        all_weight = np.array([])
        all_pid = np.array([])
        for prim in ['nue', 'numu','nutau', 'nue_bar', 'numu_bar', 'nutau_bar']:
            for int_type in ['cc','nc']:
                true_e = evts[prim][int_type]['true_energy']
                reco_e = evts[prim][int_type]['reco_energy']
                reco_cz = evts[prim][int_type]['reco_coszen']
                aeff_weights = evts[prim][int_type]['weighted_aeff']
                pid = evts[prim][int_type]['pid']
                nue_flux = evts[prim][int_type]['neutrino_nue_flux']
                numu_flux = evts[prim][int_type]['neutrino_numu_flux']
                if self.params['use_cut_on_trueE']:
                    cut = np.logical_and(true_e<self.ebins[-1], true_e>= self.ebins[0])
                    true_e = true_e[cut]
                    reco_e = reco_e[cut]
                    reco_cz = reco_cz[cut]
                    aeff_weights = aeff_weights[cut]
                    pid = pid[cut]
                    nue_flux = nue_flux[cut]
                    numu_flux = numu_flux[cut]
                osc_flux = nue_flux*osc_probs[prim][int_type]['nue_maps']+ numu_flux*osc_probs[prim][int_type]['numu_maps']
                final_weight = osc_flux * aeff_weights
                # get all reco_e, reco_cz, pid, weight:
                all_reco_e = np.append(all_reco_e, reco_e)
                all_reco_cz = np.append(all_reco_cz, reco_cz)
                all_pid = np.append(all_pid, pid)
                all_weight = np.append(all_weight, final_weight)

        bins = (self.anlys_ebins,self.czbins)
        self.rel_error = {}
        #self.mc_error = {}
        for channel in ['cscd', 'trck']:
            #TODO
            if channel == 'cscd':
                pid_cut =  np.logical_and(all_pid < self.pid_bound, all_pid>=self.pid_remove)
            else:
                pid_cut =  all_pid >= self.pid_bound
            hist_w,_,_ = np.histogram2d(all_reco_e[pid_cut], all_reco_cz[pid_cut], bins=bins, weights=all_weight[pid_cut])
            hist_w2,_,_ = np.histogram2d(all_reco_e[pid_cut], all_reco_cz[pid_cut], bins=bins, weights=all_weight[pid_cut]**2)
            hist_sqrt_w2 = np.sqrt(hist_w2)
            self.rel_error[channel]= hist_sqrt_w2/hist_w
            #self.mc_error[channel]= hist_sqrt_w2

            # if encounters zero count bins use 0 as error, so convert inf to zero
            self.rel_error[channel][np.isinf(self.rel_error[channel])] = 0


    def get_template(self, params, num_data_events, return_stages=False, no_osc_maps=False, only_tau_maps=False, no_sys_maps = False, return_aeff_maps = False, apply_reco_prcs=False):
        '''
        Runs entire template-making chain, using parameters found in
        'params' dict. If 'return_stages' is set to True, returns
        output from each stage as a simple tuple. 
        '''
        # apply_reco_prcs should always be false except when generating fits for reco prcs; when apply_reco_prcs=True, no_sys_maps = True 
        if apply_reco_prcs:
            assert(no_sys_maps==True)

        # just assume all steps changed
        step_changed = [True]*5

        # now see what really changed, if we have a cached map to decide from which step on we have to recalculate
        if self.cache_params:
            step_changed = [False]*5
            for p,v in params.items():
                if self.cache_params[p] != v:
                    if p in ['nue_numu_ratio','nu_nubar_ratio']: step_changed[0] = True
                    elif p in ['deltam21','deltam31','theta12','theta13','theta23','deltacp','energy_scale','YeI','YeO','YeM']: step_changed[1] = True
                    elif p in ['livetime','nutau_norm','aeff_scale', 'PID_scale', 'PID_offset', 'Barr_nu_nubar_ratio', 'Barr_uphor_ratio', 'axm_qe', 'axm_res', 'GENSYS_AhtBY', 'GENSYS_BhtBY','GENSYS_CV1uBY','GENSYS_CV2uBY', 'GENSYS_MaCCQE','GENSYS_MaRES', 'flux_hadronic_A','flux_hadronic_B', 'flux_hadronic_C', 'flux_hadronic_D','flux_hadronic_E', 'flux_hadronic_F', 'flux_hadronic_G', 'flux_hadronic_H','flux_hadronic_I', 'flux_hadronic_W', 'flux_hadronic_X', 'flux_hadronic_Y','flux_hadronic_Z', 'flux_pion_chargeratio_Chg', 'flux_prim_norm_a','flux_prim_exp_norm_b', 'flux_prim_exp_factor_c', 'flux_spectral_index_d']: step_changed[2] = True
                    #elif (apply_reco_prcs and p in ['e_reco_precision_up', 'cz_reco_precision_up', 'e_reco_precision_down','cz_reco_precision_down']): step_changed[3] = True 
                    elif (no_sys_maps==False and p in ['e_reco_precision_up', 'cz_reco_precision_up', 'up_down_e_reco_prcs', 'up_down_cz_reco_prcs','hole_ice','hole_ice_fwd', 'dom_eff']): step_changed[3] = True
                    elif p in ['atmos_mu_scale', 'atmmu_f']: step_changed[4] = True
                    # if this last statement is true, something changed that is unclear what it was....in that case just redo all steps
                    else: step_changed = [True]*5

        # update the cached.debugrmation
        self.cache_params = params

        logging.info('Extracting events from file: %s' % (self.aeff_weight_file))
        evts = events.Events(self.aeff_weight_file)
        bins = (self.ebins, self.czbins)
        anlys_bins = (self.anlys_ebins, self.czbins)

        # only keep events using bdt_score > bdt_cut
        print "self.params['further_bdt_cut'] = ", self.params['further_bdt_cut']
        for prim in ['nue', 'numu','nutau', 'nue_bar', 'numu_bar', 'nutau_bar']:
            for int_type in ['cc','nc']:
                l5_bdt_score = evts[prim][int_type]['dunkman_L5'].astype(np.float64)
                cut = l5_bdt_score >= self.params['further_bdt_cut']
                #print "np.shape of cut = ", np.shape(cut)
                #print "before bdt cut , len evts[prim][int_type][true_energy] = ", len(evts[prim][int_type]['true_energy'])
                #print "evts[prim][int_type].keys() = ", evts[prim][int_type].keys()
                for var in evts[prim][int_type].keys():
                    if var=='BARR_splines' or var=='GENSYS_splines':
                        for sys in evts[prim][int_type][var].keys():
                            evts[prim][int_type][var][sys] = evts[prim][int_type][var][sys][cut]
                        continue
                    try:
                        evts[prim][int_type][var] = evts[prim][int_type][var][cut]
                    except KeyError:
                        evts[prim][int_type][var] = np.ones_like(evts[prim][int_type]['true_energy'])
                #print "after bdt cut , len evts[prim][int_type][true_energy] = ", len(evts[prim][int_type]['true_energy'])

        # set up flux
        if any(step_changed[:1]):
            with Timer() as t:
                self.fluxes = {}
                for prim in ['nue', 'nue_bar', 'numu', 'numu_bar', 'nutau', 'nutau_bar']:
                    self.fluxes[prim] = {}
                    isbar = '_bar' if 'bar' in prim else ''
                    oppo_isbar = '' if 'bar' in prim else '_bar'
                    for int_type in ['cc', 'nc']:
                        self.fluxes[prim][int_type] = {}
                        nue_flux = evts[prim][int_type]['neutrino_nue_flux']
                        numu_flux = evts[prim][int_type]['neutrino_numu_flux']
                        oppo_nue_flux = evts[prim][int_type]['neutrino_oppo_nue_flux']
                        oppo_numu_flux = evts[prim][int_type]['neutrino_oppo_numu_flux']
                        true_e = evts[prim][int_type]['true_energy']
                        # apply flux systematics (nue_numu_ratio, nu_nubar_ratio)
                        nue_flux, numu_flux = apply_flux_ratio(prim, nue_flux, numu_flux, oppo_nue_flux, oppo_numu_flux, true_e, params)
                        self.fluxes[prim][int_type]['nue'] = nue_flux
                        self.fluxes[prim][int_type]['numu'] = numu_flux
            profile.debug("==> elapsed time to set up flux : %s sec"%t.secs)
        else:
            profile.debug("STAGE 1: Reused from step before...")

        # Get osc probability maps
        if any(step_changed[:2]):
            with Timer(verbose=False) as t:
                physics.debug("STAGE 2: Getting osc prob maps...")
                self.osc_probs = get_osc_probs(evts, params, self.osc_service, ebins=self.ebins)
            profile.debug("==> elapsed time to get osc_prob : %s sec"%t.secs)
        else:
            profile.info("STAGE 2: Reused from step before...")

        if any(step_changed[:3]):
            self.flux_maps = {}
            self.event_rate_maps = {'params':params}
            tmp_event_rate_reco_maps = {}
            tmp_event_rate_cscd = {}
            tmp_wgt2_cscd = {}
            tmp_wgt2_trck = {}
            true_tmp_event_rate_cscd = {}
            tmp_event_rate_trck = {}
            true_tmp_event_rate_trck = {}
            for prim in ['nue', 'nue_bar', 'numu', 'numu_bar', 'nutau', 'nutau_bar']:
                self.flux_maps[prim] = {}
                tmp_event_rate_reco_maps[prim] = {}
                self.event_rate_maps[prim] = {}
                tmp_event_rate_cscd[prim] = {}
                tmp_wgt2_cscd[prim] = {}
                tmp_wgt2_trck[prim] = {}
                tmp_event_rate_trck[prim] = {}
                true_tmp_event_rate_cscd[prim] = {}
                true_tmp_event_rate_trck[prim] = {}
                for int_type in ['cc', 'nc']:
                    isbar = '_bar' if 'bar' in prim else ''
                    nc_scale = 1.0    # for nutau CC, nc_scale = nc_norm; otherwise nc_scale = 1
                    if int_type == 'nc':
                        nc_scale = params['nc_norm']
                    nutau_scale = 1.0    # for nutau CC, nutau_scale = nutau_norm; otherwise nutau_scale = 1
                    if prim == 'nutau' or prim == 'nutau_bar':
                        if int_type == 'cc':
                            nutau_scale = params['nutau_norm']
                    true_e = evts[prim][int_type]['true_energy']
                    true_cz = evts[prim][int_type]['true_coszen']
                    reco_e = evts[prim][int_type]['reco_energy']
                    if params['use_oscFit_genie_barr_sys']:
                        linear_coeff_MaCCQE = evts[prim][int_type]['linear_fit_MaCCQE']
                        linear_coeff_MaCCRES = evts[prim][int_type]['linear_fit_MaCCRES']
                        linear_coeff_MaNCRES = evts[prim][int_type]['linear_fit_MaNCRES']
                        quad_coeff_MaCCQE = evts[prim][int_type]['quad_fit_MaCCQE']
                        quad_coeff_MaCCRES = evts[prim][int_type]['quad_fit_MaCCRES']
                        quad_coeff_MaNCRES = evts[prim][int_type]['quad_fit_MaNCRES']
                    reco_cz = evts[prim][int_type]['reco_coszen']
                    aeff_weights = evts[prim][int_type]['weighted_aeff']
                    if params['use_pisa_genie_barr_sys']:
                        gensys_splines = evts[prim][int_type]['GENSYS_splines']
                        barr_splines = evts[prim][int_type]['BARR_splines']
                    pid = evts[prim][int_type]['pid']
                    # get flux from self.fluxes
                    nue_flux = self.fluxes[prim][int_type]['nue']
                    numu_flux = self.fluxes[prim][int_type]['numu']

                    # apply spectral index, use one pivot energy for all flavors
                    egy_pivot =  24.0900951261  # the value that JP's using
                    nue_flux, numu_flux = apply_spectral_index(nue_flux, numu_flux, true_e, egy_pivot, aeff_weights, params) 

                    # apply Barr systematics (oscFit way, Barr_nu_nubar_ratio and Barr_uphor_ratio)
                    if params['use_oscFit_genie_barr_sys']:
                        nue_flux, numu_flux = apply_Barr_flux_ratio(prim, nue_flux, numu_flux, true_e, true_cz, **params)

                    # apply Barr systematics (pisa way)
                    if params['use_pisa_genie_barr_sys']:
                        nue_flux, numu_flux = apply_Barr_mod(prim, self.ebins, nue_flux, numu_flux, true_e, true_cz, barr_splines, **params)

                    # use cut on trueE ( b/c PISA has a cut on true E)
                    if params['use_cut_on_trueE']:
                        cut = np.logical_and(true_e<self.ebins[-1], true_e>= self.ebins[0])
                        true_e = true_e[cut]
                        true_cz = true_cz[cut]
                        reco_e = reco_e[cut]
                        reco_cz = reco_cz[cut]
                        if params['use_oscFit_genie_barr_sys']:
                            linear_coeff_MaCCQE = linear_coeff_MaCCQE[cut]
                            linear_coeff_MaCCRES = linear_coeff_MaCCRES[cut]
                            linear_coeff_MaNCRES = linear_coeff_MaNCRES[cut]
                            quad_coeff_MaCCQE = quad_coeff_MaCCQE[cut]
                            quad_coeff_MaCCRES = quad_coeff_MaCCRES[cut]
                            quad_coeff_MaNCRES = quad_coeff_MaNCRES[cut]
                        aeff_weights = aeff_weights[cut]
                        if params['use_pisa_genie_barr_sys']:
                            gensys_splines = gensys_splines[cut]
                        pid = pid[cut]
                        nue_flux = nue_flux[cut]
                        numu_flux = numu_flux[cut]

                    # apply axm_qe and axm_res (oscFit-way)
                    if params['use_oscFit_genie_barr_sys']:
                        aeff_weights = apply_GENIE_mod_oscFit(aeff_weights, linear_coeff_MaCCQE, quad_coeff_MaCCQE, params['axm_qe'])
                        aeff_weights = apply_GENIE_mod_oscFit(aeff_weights, linear_coeff_MaCCRES, quad_coeff_MaCCRES, params['axm_res'])
                    if params['use_pisa_genie_barr_sys']:
                        # apply GENIE systematics (on aeff weight, PISA-way)
                        aeff_weights = apply_GENIE_mod(prim, int_type, self.ebins, true_e, true_cz, aeff_weights, gensys_splines, **params)

                    # when generating fits for reco prcs, change reco_e and reco_cz:
                    if apply_reco_prcs and (params['e_reco_precision_up'] != 1 or params['cz_reco_precision_up'] != 1 or params['e_reco_precision_down'] != 1 or params['cz_reco_precision_down'] !=1):
                        reco_e, reco_cz = apply_reco_sys(true_e, true_cz, reco_e, reco_cz, params['e_reco_precision_up'], params['e_reco_precision_down'], params['cz_reco_precision_up'], params['cz_reco_precision_down'])

                    # Get flux maps
                    self.flux_maps[prim][int_type] = {} 
                    weighted_hist_nue_flux, _, _ = np.histogram2d(true_e, true_cz, weights= nue_flux * aeff_weights, bins=bins)
                    weighted_hist_numu_flux, _, _ = np.histogram2d(true_e, true_cz, weights= numu_flux * aeff_weights, bins=bins)
                    self.flux_maps[prim][int_type] = {}
                    self.flux_maps[prim][int_type]['nue'+isbar] = weighted_hist_nue_flux 
                    self.flux_maps[prim][int_type]['numu'+isbar] = weighted_hist_numu_flux 
                    self.flux_maps[prim][int_type]['ebins']=self.ebins
                    self.flux_maps[prim][int_type]['czbins']=self.czbins

                    # Get osc_flux 
                    if no_osc_maps:
                        #use no oscillation, osc_flux is just flux
                        if 'nue' in prim:
                            osc_flux = nue_flux
                        elif 'numu' in prim:
                            osc_flux = numu_flux
                        else:
                            osc_flux = np.zeros(len(nue_flux))
                    else:
                        osc_flux = nue_flux*self.osc_probs[prim][int_type]['nue_maps']+ numu_flux*self.osc_probs[prim][int_type]['numu_maps']

                    # Get event_rate(true) maps
                    final_weights = osc_flux * aeff_weights
                    weighted_hist_true, _, _ = np.histogram2d(true_e, true_cz, weights= final_weights, bins=bins)
                    self.event_rate_maps[prim][int_type] = {}
                    self.event_rate_maps[prim][int_type]['map'] = weighted_hist_true * params['livetime'] * year * params['aeff_scale'] * nutau_scale * nc_scale
                    self.event_rate_maps[prim][int_type]['ebins']=self.ebins
                    self.event_rate_maps[prim][int_type]['czbins']=self.czbins

                    # Get event_rate_reco maps (step1, tmp maps in 12 flavs)
                    weighted_hist_reco, _, _ = np.histogram2d(reco_e, reco_cz, weights= final_weights, bins=anlys_bins)
                    tmp_event_rate_reco_maps[prim][int_type] = weighted_hist_reco * params['livetime'] * year * params['aeff_scale'] * nutau_scale * nc_scale

                    # Get event_rate_pid maps (step1, tmp maps in 12 flavs)
                    pid_cscd =  np.logical_and(pid < self.pid_bound, pid>=self.pid_remove)
                    #pid_cscd =  pid < self.pid_bound
                    pid_trck =  pid >= self.pid_bound
                    weighted_hist_cscd,_, _ = np.histogram2d(reco_e[pid_cscd], reco_cz[pid_cscd], weights= final_weights[pid_cscd], bins=anlys_bins)
                    weighted_hist_trck,_, _ = np.histogram2d(reco_e[pid_trck], reco_cz[pid_trck], weights= final_weights[pid_trck], bins=anlys_bins)
                    wgt2_hist_cscd,_, _ = np.histogram2d(reco_e[pid_cscd], reco_cz[pid_cscd], weights= (final_weights[pid_cscd])**2, bins=anlys_bins)
                    wgt2_hist_trck,_, _ = np.histogram2d(reco_e[pid_trck], reco_cz[pid_trck], weights= (final_weights[pid_trck])**2, bins=anlys_bins)
                    tmp_wgt2_cscd[prim][int_type] = wgt2_hist_cscd * (params['livetime'] * year * params['aeff_scale'] * nutau_scale * nc_scale)**2
                    tmp_wgt2_trck[prim][int_type] = wgt2_hist_trck * (params['livetime'] * year * params['aeff_scale'] * nutau_scale * nc_scale)**2
                    tmp_event_rate_cscd[prim][int_type] = weighted_hist_cscd * params['livetime'] * year * params['aeff_scale'] * nutau_scale * nc_scale
                    tmp_event_rate_trck[prim][int_type] = weighted_hist_trck * params['livetime'] * year * params['aeff_scale'] * nutau_scale * nc_scale

                    # Get event_rate_pid maps in true e vs true cz (step1, tmp maps in 12 flavs)
                    true_weighted_hist_cscd,_, _ = np.histogram2d(true_e[pid_cscd], true_cz[pid_cscd], weights= final_weights[pid_cscd], bins=anlys_bins)
                    true_weighted_hist_trck,_, _ = np.histogram2d(true_e[pid_trck], true_cz[pid_trck], weights= final_weights[pid_trck], bins=anlys_bins)
                    true_tmp_event_rate_cscd[prim][int_type] = true_weighted_hist_cscd * params['livetime'] * year * params['aeff_scale'] * nutau_scale * nc_scale
                    true_tmp_event_rate_trck[prim][int_type] = true_weighted_hist_trck * params['livetime'] * year * params['aeff_scale'] * nutau_scale * nc_scale

            # Get event_rate_reco maps (step2, combine nu and nubar for cc, and all flavs for nc)
            self.event_rate_reco_maps = {'params': params}
            for prim in ['nue', 'numu', 'nutau']:
                self.event_rate_reco_maps[prim+'_cc'] = {'map': tmp_event_rate_reco_maps[prim]['cc'] + tmp_event_rate_reco_maps[prim+'_bar']['cc'],
                                                         'ebins': self.anlys_ebins, 'czbins': self.czbins}
            event_rate_reco_map_nuall_nc = np.zeros(np.shape(tmp_event_rate_reco_maps['nue']['nc']))
            for prim in ['nue','numu','nutau','nue_bar','numu_bar','nutau_bar']:
                event_rate_reco_map_nuall_nc += tmp_event_rate_reco_maps[prim]['nc']
            self.event_rate_reco_maps['nuall_nc']= {'map': event_rate_reco_map_nuall_nc, 'ebins': self.anlys_ebins, 'czbins': self.czbins}

            # Get event_rate_pid maps (step2, combine all flavs)
            self.event_rate_pid_maps = {'params':params}
            event_rate_pid_map_cscd = np.zeros(np.shape(tmp_event_rate_cscd['nue']['nc']))
            event_rate_pid_map_trck = np.zeros(np.shape(tmp_event_rate_trck['nue']['nc']))
            for prim in ['nue','numu','nutau','nue_bar','numu_bar','nutau_bar']:
                for int_type in ['cc', 'nc']:
                    event_rate_pid_map_cscd += tmp_event_rate_cscd[prim][int_type]
                    event_rate_pid_map_trck += tmp_event_rate_trck[prim][int_type]
            self.event_rate_pid_maps['cscd'] = {'map': event_rate_pid_map_cscd, 'ebins': self.anlys_ebins, 'czbins': self.czbins}
            self.event_rate_pid_maps['trck'] = {'map': event_rate_pid_map_trck, 'ebins': self.anlys_ebins, 'czbins': self.czbins}

            # getting wgt2_hist
            self.wgt2_pid_map_cscd = np.zeros(np.shape(tmp_wgt2_cscd['nue']['nc']))
            self.wgt2_pid_map_trck = np.zeros(np.shape(tmp_wgt2_trck['nue']['nc']))
            for prim in ['nue','numu','nutau','nue_bar','numu_bar','nutau_bar']:
                for int_type in ['cc', 'nc']:
                    self.wgt2_pid_map_cscd += tmp_wgt2_cscd[prim][int_type]
                    self.wgt2_pid_map_trck += tmp_wgt2_trck[prim][int_type]
        else:
            profile.debug("STAGE 3: Reused from step before...")

        if any(step_changed[:4]):
            physics.debug("STAGE 4: Applying systematics...")
            with Timer(verbose=False) as t:
                if no_sys_maps:
                    self.sys_maps = self.event_rate_pid_maps
                else: 
                    if self.sim_ver != 'dima':
                        self.hole_ice_maps = self.HoleIce.apply_sys(self.event_rate_pid_maps, params['hole_ice'])
                    else:
                        self.hole_ice_maps = self.HoleIce.apply_hi_hifwd(self.event_rate_pid_maps, params['hole_ice'], params['hole_ice_fwd'])
                    self.domeff_maps = self.DomEfficiency.apply_sys(self.hole_ice_maps, params['dom_eff'])
                    #self.domeff_maps = self.event_rate_pid_maps
                    if params['e_reco_precision_up']==1 and params['e_reco_precision_down']==1 and params['cz_reco_precision_up']==1 and params['cz_reco_precision_down']==1:
                        self.sys_maps = self.domeff_maps
                    else:
                        self.reco_prec_maps_e_up = self.Resolution_e_up.apply_sys(self.domeff_maps, params['e_reco_precision_up'])
                        e_param_down = 1. + params['up_down_e_reco_prcs']*(params['e_reco_precision_up']-1.)
                        # for generating plots, to show its effect, otherwise, as long as e_reco_precision_up =1, e_param_down returns 1 
                        #e_param_down = params['e_reco_precision_down']         
                        self.reco_prec_maps_e_down = self.Resolution_e_down.apply_sys(self.reco_prec_maps_e_up, e_param_down)
                        self.reco_prec_maps_cz_up = self.Resolution_cz_up.apply_sys(self.reco_prec_maps_e_down, params['cz_reco_precision_up'])
                        # in a fit; only allow e_reco_precision_up and _down to go in the same direction
                        cz_param_down = 1. + params['up_down_cz_reco_prcs']*(params['cz_reco_precision_up']-1.)    
                        # for generating plots to show its effect
                        #cz_param_down = params['cz_reco_precision_down']       
                        self.sys_maps = self.Resolution_cz_down.apply_sys(self.reco_prec_maps_cz_up, cz_param_down)
            profile.debug("==> elapsed time for sys stage: %s sec"%t.secs)
        else:
            profile.debug("STAGE 4: Reused from step before...")
        if any(step_changed[:5]):
            physics.debug("STAGE 5: Getting bkgd maps...")
            with Timer(verbose=False) as t:
                self.final_event_rate = add_icc_background(self.sys_maps, self.background_service, num_data_events, **params)
            profile.debug("==> elapsed time for bkgd stage: %s sec"%t.secs)
        else:
            profile.debug("STAGE 5: Reused from step before...")

        # Calculate the sum_w2, method 1: sum_w2 is calculated at baseline histogram
        #self.final_event_rate['cscd']['sumw2_nu'] = (self.final_event_rate['cscd']['map_nu']* self.rel_error['cscd'])**2    
        #self.final_event_rate['trck']['sumw2_nu'] = (self.final_event_rate['trck']['map_nu']* self.rel_error['trck'])**2
        # Calculate the sum_w2, method 2: sum_w2 get updated every time get_template() is called
        self.final_event_rate['cscd']['sumw2_nu'] = self.wgt2_pid_map_cscd     
        #print "self.wgt2_pid_map_cscd = ", self.wgt2_pid_map_cscd
        self.final_event_rate['trck']['sumw2_nu'] = self.wgt2_pid_map_trck
        self.final_event_rate['cscd']['sumw2'] = self.final_event_rate['cscd']['sumw2_nu'] + self.final_event_rate['cscd']['sumw2_mu']
        self.final_event_rate['trck']['sumw2'] = self.final_event_rate['trck']['sumw2_nu'] + self.final_event_rate['trck']['sumw2_mu']

        if not return_stages: return self.final_event_rate

        # Otherwise, return all stages as a simple tuple
        return (self.flux_maps, self.event_rate_maps, self.event_rate_reco_maps, self.sys_maps, self.final_event_rate)


if __name__ == '__main__':

    # parser
    parser = ArgumentParser(
        description='''Runs the template making process.''',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-t', '--template_settings', type=str,
                        metavar='JSONFILE', required=True,
                        help='''settings for the template generation''')
    hselect = parser.add_mutually_exclusive_group(required=False)
    hselect.add_argument('--normal', dest='normal', default=True,
                        action='store_true', help="select the normal hierarchy")
    hselect.add_argument('--inverted', dest='normal', default = False,
                        action='store_false',
                         help="select the inverted hierarchy")
    parser.add_argument('-v', '--verbose', action='count', default=None,
                        help='set verbosity level.')
    parser.add_argument('--no_osc_maps', action='store_true', default=False,
                        help="Apply no osc.")
    parser.add_argument('-s', '--save_all', action='store_true', default=False,
                        help="Save all stages.")
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE',
                        type=str, action='store',default="template.json",
                        help='file to store the output')
    args = parser.parse_args()

    set_verbosity(args.verbose)

    with Timer() as t:
        #Load all the settings
        model_settings = from_json(args.template_settings)
        no_osc_maps = args.no_osc_maps

        #Select a hierarchy
        physics.debug('Selected %s hierarchy'%
                     ('normal' if args.normal else 'inverted'))
        params =  select_hierarchy(model_settings['params'],
                                   normal_hierarchy=args.normal)

        #Intialize template maker
        template_maker = TemplateMaker(get_values(params),
                                       **model_settings['binning'])
    profile.debug("  ==> elapsed time to initialize templates: %s sec"%t.secs)

    #Now get the actual template
    with Timer(verbose=False) as t:
        template_maps = template_maker.get_template(get_values(params),
                                                    return_stages=args.save_all,no_osc_maps=args.no_osc_maps)
    profile.debug("==> elapsed time to get template: %s sec"%t.secs)

    physics.debug("Saving file to %s"%args.outfile)
    to_json(template_maps, args.outfile)

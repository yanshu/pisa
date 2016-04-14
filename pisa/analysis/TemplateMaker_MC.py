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
from scipy.constants import Julian_year

import h5py

from pisa.utils.log import physics, profile, set_verbosity, logging
from pisa.resources.resources import find_resource
from pisa.utils.params import get_fixed_params, get_free_params, get_values, select_hierarchy
from pisa.utils.jsons import from_json, to_json, json_string
from pisa.utils.utils import Timer, oversample_binning
import pisa.utils.flavInt as flavInt
import pisa.utils.events as events

from pisa.flux.myHondaFluxService import myHondaFluxService as HondaFluxService
from pisa.flux.IPHondaFluxService_MC import IPHondaFluxService
from pisa.flux.Flux import get_flux_maps

from pisa.oscillations.Oscillation import get_osc_flux

from pisa.aeff.AeffServiceMC import AeffServiceMC
from pisa.aeff.AeffServicePar import AeffServicePar
from pisa.aeff.Aeff import get_event_rates

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
                 oversample_e=None, oversample_cz=None, actual_oversample_e=None, actual_oversample_cz=None,sim_ver=None,**kwargs):
        '''
        TemplateMaker class handles all of the setup and calculation of the
        templates for a given binning.

        Parameters:
        * template_settings - dictionary of all template-making settings
        * ebins - energy bin edges
        * czbins - coszen bin edges
        '''

        self.cache_params = None
        self.flux_maps = None
        self.weights = None
        self.osc_flux_maps = None
        self.event_rate_maps = None
        self.event_rate_reco_maps = None
        self.event_rate_pid_maps = None
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
            self.flux_service = HondaFluxService(oversample_e = self.oversample_e, oversample_cz = self.oversample_cz,**template_settings)
        elif flux_mode.lower() == 'integral-preserving':
            self.flux_service = IPHondaFluxService(**template_settings)
        else:
            error_msg = "flux_mode: %s is not implemented! "%flux_mode
            error_msg+=" Please choose among: ['bisplrep', 'integral-preserving']"
            raise NotImplementedError(error_msg)

        # Oscillated Flux Service:
        osc_code = template_settings['osc_code']
        if osc_code == 'prob3':
            from pisa.oscillations.Prob3OscillationServiceMC import Prob3OscillationServiceMC
            self.osc_service = Prob3OscillationServiceMC(
                self.oversample_ebins, self.oversample_czbins, **template_settings)
        elif osc_code == 'ocelot':
            from pisa.oscillations.OcelotOscillationServiceMC import OcelotOscillationServiceMC
            self.osc_service = OcelotOscillationServiceMC(self.oversample_ebins, self.oversample_czbins, atmos_model= 'simple', prob_model='Probabilities')
        else:
            error_msg = 'OscillationService NOT implemented for ' + \
                    'osc_code = %s' % osc_code
            raise NotImplementedError(error_msg)

        # Aeff/True Event Rate Service:
        aeff_mode = template_settings['aeff_mode']
        if aeff_mode == 'param':
            physics.debug(" Using effective area from PARAMETRIZATION...")
            self.aeff_service = AeffServicePar(self.ebins, self.czbins,
                                               **template_settings)
        elif aeff_mode == 'MC':
            physics.debug(" Using effective area from MC EVENT DATA...")
            self.aeff_service = AeffServiceMC(self.ebins, self.czbins,
                                              **template_settings)
        else:
            error_msg = "aeff_mode: '%s' is not implemented! "%aeff_mode
            error_msg += " Please choose among: ['MC', 'param']"
            raise NotImplementedError(error_msg)

        # Reco Event Rate Service:
        reco_mode = template_settings['reco_mode']
        if reco_mode == 'MC':
            self.reco_service = RecoServiceMC(self.ebins, self.czbins,
                                                **template_settings)
        elif reco_mode == 'param':
            self.reco_service = RecoServiceParam(self.ebins, self.czbins,
                                                **template_settings)
        elif reco_mode == 'stored':
            self.reco_service = RecoServiceKernelFile(self.ebins, self.czbins,
                                                **template_settings)
        elif reco_mode == 'vbwkde':
            self.reco_service = RecoServiceVBWKDE(self.ebins, self.czbins,
                                                **template_settings)
        else:
            error_msg = "reco_mode: %s is not implemented! "%reco_mode
            error_msg+=" Please choose among: ['MC', 'param', 'stored','vbwkde']"
            raise NotImplementedError(error_msg)

        # Instantiate a PID service
        self.pid_service = PID.pid_service_factory(
            ebins= self.anlys_ebins, czbins=self.czbins, **template_settings
        )

        # background service
        self.background_service = BackgroundServiceICC(self.anlys_ebins, self.czbins,
                                                     **template_settings)

        # hole ice sys
        self.HoleIce = HoleIce(template_settings['holeice_slope_file'], sim_ver=sim_ver)
        self.DomEfficiency = DomEfficiency(template_settings['domeff_slope_file'], sim_ver=sim_ver)
        self.Resolution_e_up = Resolution(template_settings['reco_prcs_coeff_file'],'e','up')
        self.Resolution_e_down = Resolution(template_settings['reco_prcs_coeff_file'],'e','down')
        self.Resolution_cz_up = Resolution(template_settings['reco_prcs_coeff_file'],'cz','up')
        self.Resolution_cz_down = Resolution(template_settings['reco_prcs_coeff_file'],'cz','down')

        self.calc_mc_errors()

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
        logging.info('Opening file: %s'%(self.reco_mc_wt_file))
        try:
            evts = events.Events(self.reco_mc_wt_file)
        except IOError,e:
            logging.error("Unable to open event data file %s"%simfile)
            logging.error(e)
            sys.exit(1)
        all_flavors_dict = {}
        for flavor in ['nue', 'numu','nutau']:
            flavor_dict = {}
            logging.debug("Working on %s "%flavor)
            for int_type in ['cc','nc']:
                bins = (self.anlys_ebins,self.czbins)
                hist_2d,_,_ = np.histogram2d(evts[flavor][int_type]['reco_energy']+evts[flavor+'_bar'][int_type]['reco_energy'],evts[flavor][int_type]['reco_coszen']+evts[flavor +'_bar_'][int_type]['reco_coszen'],bins=bins)
                flavor_dict[int_type] = hist_2d
            all_flavors_dict[flavor] = flavor_dict
        numu_cc_map = all_flavors_dict['numu']['cc']
        nue_cc_map = all_flavors_dict['nue']['cc']
        nutau_cc_map = all_flavors_dict['nutau']['cc']
        nuall_nc_map = all_flavors_dict['numu']['nc']

        #print " before PID, total no. of MC events = ", sum(sum(numu_cc_map))+sum(sum(nue_cc_map))+sum(sum(nutau_cc_map))+sum(sum(nuall_nc_map))
        mc_event_maps = {'params':self.params}
        mc_event_maps['numu_cc'] = {u'czbins':self.czbins,u'ebins':self.ebins,u'map':numu_cc_map}
        mc_event_maps['nue_cc'] =  {u'czbins':self.czbins,u'ebins':self.ebins,u'map':nue_cc_map}
        mc_event_maps['nutau_cc'] = {u'czbins':self.czbins,u'ebins':self.ebins,u'map':nutau_cc_map}
        mc_event_maps['nuall_nc'] = {u'czbins':self.czbins,u'ebins':self.ebins,u'map':nuall_nc_map}

        final_MC_event_rate = self.pid_service.get_pid_maps(mc_event_maps)
        self.rel_error = {}
        self.rel_error['cscd']=1./(final_MC_event_rate['cscd']['map'])      
        self.rel_error['trck']=1./(final_MC_event_rate['trck']['map'])      


    def get_template(self, params, return_stages=False, no_osc_maps=False, only_tau_maps=False, no_sys_applied = False, return_aeff_maps = False, read_osc_json=False, read_osc_flux_json=False, read_flux_json=False, read_wt_json=False, use_cut_on_trueE=True):
        '''
        Runs entire template-making chain, using parameters found in
        'params' dict. If 'return_stages' is set to True, returns
        output from each stage as a simple tuple. 
        '''
        # just assume all steps changed
        step_changed = [True]*7

        # now see what really changed, if we have a cached map to decide from which step on we have to recalculate
        if self.cache_params:
            step_changed = [False]*7
            for p,v in params.items():
                if self.cache_params[p] != v:
                    if p in ['nue_numu_ratio','nu_nubar_ratio','energy_scale','atm_delta_index']: step_changed[0] = True
                    elif p in ['deltam21','deltam31','theta12','theta13','theta23','deltacp','energy_scale','YeI','YeO','YeM']: step_changed[1] = True
                    elif p in ['livetime','nutau_norm','aeff_scale']: step_changed[2] = True
                    elif (no_sys_applied and p in ['e_reco_precision_up', 'cz_reco_precision_up', 'up_down_e_reco_prcs','up_down_cz_reco_prcs']): step_changed[3] = True
                    elif p in ['PID_scale', 'PID_offset']: step_changed[4] = True
                    elif p in ['e_reco_precision_up', 'cz_reco_precision_up', 'up_down_e_reco_prcs', 'up_down_cz_reco_prcs','hole_ice','dom_eff']: step_changed[5] = True
                    elif p in ['atmos_mu_scale']: step_changed[6] = True
                    # if this last statement is true, something changed that is unclear what it was....in that case just redo all steps
                    else: steps_changed = [True]*7

        # update the cached.debugrmation
        self.cache_params = params

        logging.info('Extracting events from file: %s' % (self.aeff_weight_file))
        evts = events.Events(self.aeff_weight_file)
        bins = (self.ebins, self.czbins)
        anlys_bins = (self.anlys_ebins, self.czbins)

        # Get osc probability maps
        if read_osc_json:
            osc_probs = from_json('osc_probs.json')
        else:
            osc_probs = {}
            params =  select_hierarchy(self.params, normal_hierarchy=True)
            for prim in ['nue', 'nue_bar', 'numu', 'numu_bar', 'nutau', 'nutau_bar']:
                osc_probs[prim]= {}
                for int_type in ['cc', 'nc']:
                    osc_probs[prim][int_type] = {}
                    for nu in ['nue','numu','nue_bar','numu_bar']:
                        isbar = '_bar' if 'bar' in nu else ''
                        osc_probs[prim][int_type][nu+'_maps'] = {'nue'+isbar: [],
                                                                 'numu'+isbar: [],
                                                                 'nutau'+isbar: []}

            for prim in ['nue', 'nue_bar', 'numu', 'numu_bar', 'nutau', 'nutau_bar']:
                for int_type in ['cc', 'nc']:
                    true_e = evts[prim][int_type]['true_energy']
                    true_cz = evts[prim][int_type]['true_coszen']
                    if use_cut_on_trueE:
                        cut = np.logical_and(true_e < self.ebins[-1], true_e>= self.ebins[0])
                        true_e = true_e[cut]
                        true_cz = true_cz[cut]
                    osc_probs[prim][int_type] = self.osc_service.fill_osc_prob(true_e, true_cz, event_by_event=True, **params)

        self.flux_maps = {}
        self.osc_flux_maps = {}
        self.weights = {} 
        if read_flux_json:
            self.flux_maps = from_json('flux_maps.json')
        if read_osc_flux_json:
            self.osc_flux_maps = from_json('osc_flux_maps.json')
        if read_wt_json:
            self.weights = from_json('weights.json')

        self.event_rate_maps = {'params':params}
        sum_events_event_rate = 0
        sum_event_rate_cscd = 0
        sum_event_rate_trck = 0
        tmp_event_rate_reco_maps = {}
        tmp_event_rate_cscd = {}
        tmp_event_rate_trck = {}
        for prim in ['nue', 'nue_bar', 'numu', 'numu_bar', 'nutau', 'nutau_bar']:
            if not read_flux_json:
                self.flux_maps[prim] = {}
            if not read_osc_flux_json:
                self.osc_flux_maps[prim] = {}
            if not read_wt_json:
                self.weights[prim] = {} 
            tmp_event_rate_reco_maps[prim] = {}
            self.event_rate_maps[prim] = {}
            tmp_event_rate_cscd[prim] = {}
            tmp_event_rate_trck[prim] = {}
            for int_type in ['cc', 'nc']:
                isbar = '_bar' if 'bar' in prim else ''
                orig_flux_nue = 'nue'+ isbar
                orig_flux_numu = 'numu' + isbar
                scale = 1.0
                if prim == 'nutau' or prim == 'nutau_bar':
                    if int_type == 'cc':
                        scale = params['nutau_norm']
                true_e = evts[prim][int_type]['true_energy']
                true_cz = evts[prim][int_type]['true_coszen']
                reco_e = evts[prim][int_type]['reco_energy']
                reco_cz = evts[prim][int_type]['reco_coszen']
                aeff_weights = evts[prim][int_type]['weighted_aeff']
                pid = evts[prim][int_type]['pid']
                #event = evts[prim][int_type]['event']
                if use_cut_on_trueE:
                    cut = np.logical_and(true_e<self.ebins[-1], true_e>= self.ebins[0])
                    true_e = true_e[cut]
                    true_cz = true_cz[cut]
                    reco_e = reco_e[cut]
                    reco_cz = reco_cz[cut]
                    aeff_weights = aeff_weights[cut]
                    pid = pid[cut]
                    #event = event[cut]

                # Get flux maps
                if not read_flux_json:
                    self.flux_maps[prim][int_type] = {} 
                    self.flux_maps[prim][int_type]['nue'+isbar] = self.flux_service.get_flux(true_e, true_cz, orig_flux_nue, event_by_event=True)
                    self.flux_maps[prim][int_type]['numu'+isbar] = self.flux_service.get_flux(true_e, true_cz, orig_flux_numu, event_by_event=True)
                # Get event_rate maps
                #print "Getting event_rate map for ", prim , " "
                nue_flux = self.flux_maps[prim][int_type]['nue'+isbar]
                numu_flux = self.flux_maps[prim][int_type]['numu'+isbar]

                # get osc_flux maps
                if read_osc_flux_json:
                    osc_flux = self.osc_flux_maps[prim][int_type]
                else:
                    isbar = '_bar' if 'bar' in prim else ''
                    osc_flux = nue_flux*osc_probs[prim][int_type]['nue'+isbar+'_maps'][prim]+ numu_flux*osc_probs[prim][int_type]['numu'+isbar+'_maps'][prim]
                    self.osc_flux_maps[prim][int_type] = osc_flux
                if read_wt_json:
                    weights = self.weights[prim][int_type]['weight']
                else:
                    weights = osc_flux * aeff_weights
                    self.weights[prim][int_type] = {} 
                    self.weights[prim][int_type]['weight'] = weights 
                    self.weights[prim][int_type]['true_energy'] = true_e 
                    self.weights[prim][int_type]['true_coszen'] = true_cz
                    #self.weights[prim][int_type]['event'] = event 
                weighted_hist_true, _, _ = np.histogram2d(true_e, true_cz, weights= osc_flux * aeff_weights, bins=bins)
                self.event_rate_maps[prim][int_type] = {}
                self.event_rate_maps[prim][int_type]['map'] = weighted_hist_true * params['livetime'] * Julian_year * params['aeff_scale'] * scale
                self.event_rate_maps[prim][int_type]['ebins']=self.ebins
                self.event_rate_maps[prim][int_type]['czbins']=self.czbins
                sum_events_event_rate += np.sum(self.event_rate_maps[prim][int_type]['map'])

                # Get event_rate_reco maps (step1, tmp maps in 12 flavs)
                weighted_hist_reco, _, _ = np.histogram2d(reco_e, reco_cz, weights= osc_flux * aeff_weights, bins=anlys_bins)
                tmp_event_rate_reco_maps[prim][int_type] = weighted_hist_reco * params['livetime'] * Julian_year * params['aeff_scale'] * scale

                # Get event_rate_pid maps (step1, tmp maps in 12 flavs)
                pid_cscd =  np.logical_and(pid < 3, pid>=-3)
                pid_trck =  pid >=3 
                weighted_hist_cscd,_, _ = np.histogram2d(reco_e[pid_cscd], reco_cz[pid_cscd], weights= (osc_flux[pid_cscd]* aeff_weights[pid_cscd]), bins=anlys_bins)
                weighted_hist_trck,_, _ = np.histogram2d(reco_e[pid_trck], reco_cz[pid_trck], weights= (osc_flux[pid_trck]* aeff_weights[pid_trck]), bins=anlys_bins)
                tmp_event_rate_cscd[prim][int_type] = weighted_hist_cscd * params['livetime'] * Julian_year * params['aeff_scale'] * scale
                tmp_event_rate_trck[prim][int_type] = weighted_hist_trck * params['livetime'] * Julian_year * params['aeff_scale'] * scale
                sum_event_rate_cscd += np.sum(tmp_event_rate_cscd[prim][int_type])
                sum_event_rate_trck += np.sum(tmp_event_rate_trck[prim][int_type])

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
        print "Getting PID map"
        self.event_rate_pid_maps = {'params':self.params}
        event_rate_pid_map_cscd = np.zeros(np.shape(tmp_event_rate_cscd['nue']['nc']))
        event_rate_pid_map_trck = np.zeros(np.shape(tmp_event_rate_trck['nue']['nc']))
        for prim in ['nue','numu','nutau','nue_bar','numu_bar','nutau_bar']:
            for int_type in ['cc', 'nc']:
                event_rate_pid_map_cscd += tmp_event_rate_cscd[prim][int_type]
                event_rate_pid_map_trck += tmp_event_rate_trck[prim][int_type]
        self.event_rate_pid_maps['cscd'] = {'map': event_rate_pid_map_cscd, 'ebins': self.anlys_ebins, 'czbins': self.czbins}
        self.event_rate_pid_maps['trck'] = {'map': event_rate_pid_map_trck, 'ebins': self.anlys_ebins, 'czbins': self.czbins}

        if not read_osc_json:
        	to_json(osc_probs, 'osc_probs.json')
        if not read_flux_json:
            to_json(self.flux_maps, 'flux_maps.json')
        if not read_osc_flux_json:
            to_json(self.osc_flux_maps, 'osc_flux_maps.json')
        if not read_wt_json:
            to_json(self.weights, 'weights.json')
        #to_json(self.event_rate_maps, 'event_rate_maps.json')
        #to_json(self.event_rate_reco_maps, 'event_rate_reco_maps.json')
        #to_json(self.event_rate_pid_maps, 'event_rate_pid_maps.json')


        if any(step_changed[:6]):
            physics.debug("STAGE 6: Applying systematics...")
            with Timer(verbose=False) as t:
                if no_sys_applied:
                    self.sys_maps = self.event_rate_pid_maps
                else: 
                    self.hole_ice_maps = self.HoleIce.apply_sys(self.event_rate_pid_maps, params['hole_ice'])
                    self.domeff_maps = self.DomEfficiency.apply_sys(self.hole_ice_maps, params['dom_eff'])
                    self.reco_prec_maps_e_up = self.Resolution_e_up.apply_sys(self.domeff_maps, params['e_reco_precision_up'])
                    e_param_down = 1. + params['up_down_e_reco_prcs']*(params['e_reco_precision_up']-1.)
                    self.reco_prec_maps_e_down = self.Resolution_e_down.apply_sys(self.reco_prec_maps_e_up, e_param_down)
                    self.reco_prec_maps_cz_up = self.Resolution_cz_up.apply_sys(self.reco_prec_maps_e_down, params['cz_reco_precision_up'])
                    cz_param_down = 1. + params['up_down_cz_reco_prcs']*(params['cz_reco_precision_up']-1.)
                    self.sys_maps = self.Resolution_cz_down.apply_sys(self.reco_prec_maps_cz_up, cz_param_down)
            profile.debug("==> elapsed time for sys stage: %s sec"%t.secs)
        else:
            profile.debug("STAGE 6: Reused from step before...")

        if any(step_changed[:7]):
            physics.debug("STAGE 7: Getting bkgd maps...")
            with Timer(verbose=False) as t:
                self.final_event_rate = add_icc_background(self.sys_maps, self.background_service,**params)
            profile.debug("==> elapsed time for bkgd stage: %s sec"%t.secs)
        else:
            profile.debug("STAGE 7: Reused from step before...")

        self.final_event_rate['cscd']['sumw2_nu'] = self.final_event_rate['cscd']['map_nu']**2 * self.rel_error['cscd']
        self.final_event_rate['trck']['sumw2_nu'] = self.final_event_rate['trck']['map_nu']**2 * self.rel_error['trck']
        self.final_event_rate['cscd']['sumw2'] = self.final_event_rate['cscd']['sumw2_nu'] + self.final_event_rate['cscd']['sumw2_mu']
        self.final_event_rate['trck']['sumw2'] = self.final_event_rate['trck']['sumw2_nu'] + self.final_event_rate['trck']['sumw2_mu']

        if not return_stages: return self.final_event_rate

        # Otherwise, return all stages as a simple tuple
        return (self.flux_maps, self.osc_flux_maps, self.event_rate_maps,
                self.event_rate_reco_maps, self.sys_maps, self.final_event_rate)


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
                                                    return_stages=args.save_all)
    profile.debug("==> elapsed time to get template: %s sec"%t.secs)

    physics.debug("Saving file to %s"%args.outfile)
    to_json(template_maps, args.outfile)

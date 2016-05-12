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
from pisa.flux.IPHondaFluxService import IPHondaFluxService
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

class TemplateMaker:
    '''
    This class handles all steps needed to produce a template with a
    constant binning.

    The strategy employed will be to define all 'services' in the
    initialization process, make them members of the class, then use
    them later when needed.
    '''
    def __init__(self, template_settings, ebins, czbins, anlys_ebins,
                 oversample_e=None, oversample_cz=None, actual_oversample_e=None, actual_oversample_cz=None, sim_ver=None,**kwargs):
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
            from pisa.oscillations.Prob3OscillationService import Prob3OscillationService
            self.osc_service = Prob3OscillationService(
                self.oversample_ebins, self.oversample_czbins, **template_settings)
        elif osc_code == 'gpu':
            from pisa.oscillations.Prob3GPUOscillationService import Prob3GPUOscillationService
            self.osc_service = Prob3GPUOscillationService(
                self.oversample_ebins, self.oversample_czbins, oversample_e=self.oversample_e,
                oversample_cz=self.oversample_cz, **template_settings
            )
        elif osc_code == 'nucraft':
            from pisa.oscillations.NucraftOscillationService import NucraftOscillationService
            self.osc_service = NucraftOscillationService(
                self.oversample_ebins, self.oversample_czbins, **template_settings
            )
        elif osc_code == 'ocelot':
            from pisa.oscillations.OcelotOscillationService import OcelotOscillationService
            self.osc_service = OcelotOscillationService(self.oversample_ebins, self.oversample_czbins, atmos_model= 'simple', prob_model='Probabilities')
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
            #self.aeff_service = AeffServiceMC(self.ebins, self.czbins,
            self.aeff_service = AeffServiceMC(self.oversample_ebins, self.oversample_czbins,
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
        if not no_sys_maps:
            # when we are generating fits (creating the json files)
            # for the first time ( no_sys_maps = True), this can't run 
            self.HoleIce = HoleIce(template_settings['holeice_slope_file'], sim_ver = sim_ver)
            self.DomEfficiency = DomEfficiency(template_settings['domeff_slope_file'], sim_ver = sim_ver)
            self.Resolution_e_up = Resolution(template_settings['reco_prcs_coeff_file'],'e','up')
            self.Resolution_e_down = Resolution(template_settings['reco_prcs_coeff_file'],'e','down')
            self.Resolution_cz_up = Resolution(template_settings['reco_prcs_coeff_file'],'cz','up')
            self.Resolution_cz_down = Resolution(template_settings['reco_prcs_coeff_file'],'cz','down')

        self.calc_mc_errors()

    def downsample_binning(self, maps, map_type):
        # the ugliest way to handle this.....sorry
        new_maps = {}
        new_maps['params'] = maps['params']
        if map_type == 'osc_flux':
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
        elif map_type == 'aeff':
            for flavour in maps.keys():
                new_maps[flavour] = {}
                for int_type in ['cc', 'nc']:
                    if flavour == 'params': continue
                    new_maps[flavour][int_type] = {'ebins': self.ebins,
                                          'czbins': self.czbins}
                    new_maps[flavour][int_type]['map'] = np.zeros((len(self.ebins)-1,len(self.czbins)-1))
                    for e in xrange(len(self.ebins)-1):
                        for cz in xrange(len(self.czbins)-1):
                            for e_o in xrange(self.actual_oversample_e):
                                for cz_o in xrange(self.actual_oversample_cz):
                                    new_maps[flavour][int_type]['map'][e][cz] += maps[flavour][int_type]['map'][e*self.actual_oversample_e+e_o][cz*self.actual_oversample_cz+cz_o]
        else:
            raise ValueError('Only implemented downsampling for two types of maps: osc_flux and aeff')
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


    def get_template(self, params, return_stages=False, no_osc_maps=False, only_tau_maps=False, no_sys_maps = False, return_aeff_maps = False, apply_reco_prcs=False, only_upto_stage_2=False):
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
                    elif (apply_reco_prcs and p in ['e_reco_precision_up', 'cz_reco_precision_up', 'up_down_e_reco_prcs','up_down_cz_reco_prcs']): step_changed[3] = True 
                    elif p in ['PID_scale', 'PID_offset']: step_changed[4] = True
                    elif (no_sys_maps==False and p in ['e_reco_precision_up', 'cz_reco_precision_up', 'up_down_e_reco_prcs', 'up_down_cz_reco_prcs','hole_ice','dom_eff']): step_changed[5] = True
                    elif p in ['atmos_mu_scale']: step_changed[6] = True
                    # if this last statement is true, something changed that is unclear what it was....in that case just redo all steps
                    else: steps_changed = [True]*7

        # update the cached.debugrmation
        self.cache_params = params

        if any(step_changed[:1]):
            physics.debug("STAGE 1: Getting Atm Flux maps...")
            with Timer() as t:
                self.flux_maps = get_flux_maps(self.flux_service, self.oversample_ebins,self.oversample_czbins, **params)
            profile.debug("==> elapsed time for flux stage: %s sec"%t.secs)
        else:
            profile.debug("STAGE 1: Reused from step before...")
        
        if not no_osc_maps:
            if any(step_changed[:2]):
                physics.debug("STAGE 2: Getting osc prob maps...")
                with Timer() as t:
                    #osc_flux_maps = get_osc_flux(self.flux_maps, self.osc_service,oversample_e=self.oversample_e,oversample_cz=self.oversample_cz,**params)
                    self.osc_flux_maps = get_osc_flux(self.flux_maps, self.osc_service,oversample_e=self.oversample_e,oversample_cz=self.oversample_cz,**params)
                profile.debug("==> elapsed time for oscillations stage: %s sec"%t.secs)
                #self.osc_flux_maps = self.downsample_binning(osc_flux_maps, map_type = 'osc_flux')
            else:
                profile.debug("STAGE 2: Reused from step before...")

            # set e and mu flavours to zero, if requested
            if only_tau_maps:
                physics.debug("  >>Setting e and mu flavours to zero...")
                flavours = ['numu', 'numu_bar','nue','nue_bar']
                test_map = flux_maps['nue']
                for flav in flavours:
                    self.osc_flux_maps[flav] = {'map': np.zeros_like(test_map['map']),
                                       'ebins': np.zeros_like(test_map['ebins']),
                                       'czbins': np.zeros_like(test_map['czbins'])}

                #self.osc_flux_maps = self.downsample_binning(self.osc_flux_maps, map_type = 'osc_flux')
                self.osc_flux_maps = osc_flux_maps 

        else:
            # Skipping oscillation stage...
            physics.debug("  >>Skipping Stage 2 in no oscillations case...")
            flavours = ['nutau', 'nutau_bar']
            self.osc_flux_maps = self.flux_maps
            # Create the empty nutau maps:
            test_map = self.flux_maps['nue']
            for flav in flavours:
                self.osc_flux_maps[flav] = {'map': np.zeros_like(test_map['map']),
                                            'ebins': np.zeros_like(test_map['ebins']),
                                            'czbins': np.zeros_like(test_map['czbins'])}
        
        if only_upto_stage_2: return (self.flux_maps, self.osc_flux_maps)
        
        if not return_aeff_maps:
            if any(step_changed[:3]):
                physics.debug("STAGE 3: Getting event rate true maps...")
                with Timer() as t:
                    event_rate_maps = get_event_rates(self.osc_flux_maps,self.aeff_service, **params)
                    self.event_rate_maps = self.downsample_binning(event_rate_maps, map_type = 'aeff')
                profile.debug("==> elapsed time for aeff stage: %s sec"%t.secs)
            else:
                profile.debug("STAGE 3: Reused from step before...")
        else:
            osc_flux_maps_ones = self.osc_flux_maps
            flavours = ['numu', 'numu_bar','nue','nue_bar', 'nutau', 'nutau_bar']
            for flav in flavours:
                osc_flux_maps_ones[flav]['map']= np.ones_like(osc_flux_maps_ones[flav]['map'])
            return get_event_rates(osc_flux_maps_ones,self.aeff_service, **params)

        if any(step_changed[:4]):
            physics.debug("STAGE 4: Getting event rate reco maps...")
            with Timer() as t:
                self.event_rate_reco_maps = get_reco_maps(self.event_rate_maps, self.anlys_ebins, apply_reco_prcs, self.reco_service,**params)
                # apply_reco_prcs should always be false except when generating fits for reco prcs
            profile.debug("==> elapsed time for reco stage: %s sec"%t.secs)
        else:
            profile.debug("STAGE 4: Reused from step before...")

        if any(step_changed[:5]):
            physics.debug("STAGE 5: Getting pid maps...")
            with Timer(verbose=False) as t:
                self.event_rate_pid_maps = self.pid_service.get_pid_maps(self.event_rate_reco_maps)
            profile.debug("==> elapsed time for pid stage: %s sec"%t.secs)
        else:
            profile.debug("STAGE 5: Reused from step before...")

        if any(step_changed[:6]):
            physics.debug("STAGE 6: Applying systematics...")
            if no_sys_maps:
                # apply no dom_eff, hole_ice or reco_prcs
                self.sys_maps = self.event_rate_pid_maps
            else: 
                with Timer(verbose=False) as t:
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
    hselect.add_argument('--apply_reco_prcs', dest='apply_reco_prcs', default=False,
                        action='store_true', help='''Apply reco precision in RecoMCService.py,
                        set it True only when generating fits for reco precision parameters (at
                        the same time, keep get_reco_prcs=True), otherwise always keep it false.''')
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
                                                    return_stages=args.save_all, only_upto_stage_2=False)
    profile.debug("==> elapsed time to get template: %s sec"%t.secs)

    physics.debug("Saving file to %s"%args.outfile)
    to_json(template_maps, args.outfile)

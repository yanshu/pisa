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

from pisa.utils.log import logging, profile, set_verbosity
from pisa.resources.resources import find_resource
from pisa.utils.params import get_fixed_params, get_free_params, get_values, select_hierarchy
from pisa.utils.jsons import from_json, to_json, json_string
from pisa.utils.utils import Timer

from pisa.flux.myHondaFluxService import myHondaFluxService as HondaFluxService
#from pisa.flux.HondaFluxService import HondaFluxService
from pisa.flux.Flux import get_flux_maps

from pisa.oscillations.Prob3OscillationService import Prob3OscillationService
from pisa.oscillations.NucraftOscillationService import NucraftOscillationService
try:
    #print "Trying to import Prob3GPUOscillationService..."
    from pisa.oscillations.Prob3GPUOscillationService import Prob3GPUOscillationService
except:
    pass
    #print "CAN NOT import Prob3GPUOscillationService..."
from pisa.oscillations.Oscillation import get_osc_flux

from pisa.aeff.AeffServiceMC import AeffServiceMC
from pisa.aeff.AeffServicePar import AeffServicePar
from pisa.aeff.Aeff import get_event_rates

from pisa.reco.RecoServiceMC import RecoServiceMC
from pisa.reco.RecoServiceParam import RecoServiceParam
from pisa.reco.RecoServiceKernelFile import RecoServiceKernelFile
from pisa.reco.RecoServiceVBWKDE import RecoServiceVBWKDE
from pisa.reco.Reco import get_reco_maps

from pisa.pid.PIDServiceParam import PIDServiceParam
from pisa.pid.PIDServiceKernelFile import PIDServiceKernelFile
from pisa.pid.PID import get_pid_maps

from pisa.background.BackgroundServiceICC_nutau import BackgroundServiceICC 
from pisa.background.ICCBackground_nutau import add_icc_background
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
                 oversample_e=None, oversample_cz=None, **kwargs):
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
        self.dom_eff_maps = None
        self.final_event_rate = None
        
        self.ebins = ebins
        self.czbins = czbins
        self.anlys_ebins = anlys_ebins
        self.oversample_e = oversample_e
        self.oversample_cz = oversample_cz
        logging.debug("Using %u bins in energy from %.2f to %.2f GeV"%
                      (len(self.ebins)-1, self.ebins[0], self.ebins[-1]))
        logging.debug("Using %u bins in cos(zenith) from %.2f to %.2f"%
                      (len(self.czbins)-1, self.czbins[0], self.czbins[-1]))

        # Instantiate a flux model service
        self.flux_service = HondaFluxService(**template_settings)

        # Oscillated Flux Service:
        osc_code = template_settings['osc_code']
        if osc_code == 'prob3':
            self.osc_service = Prob3OscillationService(
                self.ebins, self.czbins, **template_settings)
        elif osc_code == 'gpu':
            self.osc_service = Prob3GPUOscillationService(
                self.ebins, self.czbins, oversample_e=self.oversample_e,
                oversample_cz=self.oversample_cz, **template_settings
            )
        elif osc_code == 'nucraft':
            self.osc_service = NucraftOscillationService(
                self.ebins, self.czbins, **template_settings
            )
        else:
            error_msg = 'OscillationService NOT implemented for ' + \
                    'osc_code = %s' % osc_code
            raise NotImplementedError(error_msg)

        # Aeff/True Event Rate Service:
        aeff_mode = template_settings['aeff_mode']
        if aeff_mode == 'param':
            logging.info(" Using effective area from PARAMETRIZATION...")
            self.aeff_service = AeffServicePar(self.ebins, self.czbins,
                                               **template_settings)
        elif aeff_mode == 'MC':
            logging.info(" Using effective area from MC EVENT DATA...")
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

        # PID Service:
        pid_mode = template_settings['pid_mode']
        if pid_mode == 'param':
            self.pid_service = PIDServiceParam( self.anlys_ebins, self.czbins,
                                                 **template_settings)
        elif pid_mode == 'stored':
            self.pid_service = PIDServiceKernelFile(self.anlys_ebins, self.czbins,
                                                    **template_settings)
        else:
            error_msg = "pid_mode: %s is not implemented! "%pid_mode
            error_msg+=" Please choose among: ['stored', 'param']"
            raise NotImplementedError(error_msg)

        # background service
        self.background_service = BackgroundServiceICC(self.anlys_ebins, self.czbins,
                                                     **template_settings)

        # hole ice sys
        self.HoleIce = HoleIce(template_settings['domeff_holeice_slope_file'])
        self.DomEfficiency = DomEfficiency(template_settings['domeff_holeice_slope_file'])
        self.Resolution = Resolution(template_settings['reco_prcs_coeff_file'])

    def get_template(self, params, return_stages=False):
        '''
        Runs entire template-making chain, using parameters found in
        'params' dict. If 'return_stages' is set to True, returns
        output from each stage as a simple tuple. 
        '''
        # just assume all steps changed
        step_changed = [True]*6

        # now see what really changed, if we have a cached map to decide from which step on we have to recalculate
        if self.cache_params:
            step_changed = [False]*6
            for p,v in params.items():
                if self.cache_params[p] != v:
                    if p in ['nue_numu_ratio','nu_nubar_ratio','energy_scale','atm_delta_index']: step_changed[0] = True
                    elif p in ['deltam21','deltam31','theta12','theta13','theta23','deltacp','energy_scale','YeI','YeO','YeM']: step_changed[1] = True
                    elif p in ['livetime','nutau_norm','aeff_scale']: step_changed[2] = True
                    elif p in ['']: step_changed[4] = True
                    elif p in ['atmos_mu_scale']: step_changed[5] = True
                    # if this last statement is true, something changed that is unclear what it was....in that case just redo all steps
                    else: steps_changed = [True]*6

        # update the cached information
        self.cache_params = params

        if any(step_changed[:1]):
            logging.info("STAGE 1: Getting Atm Flux maps...")
            with Timer() as t:
                self.flux_maps = get_flux_maps(self.flux_service, self.ebins,self.czbins, **params)
            profile.debug("==> elapsed time for flux stage: %s sec"%t.secs)
        else:
            profile.info("STAGE 1: Reused from step before...")
        
        if any(step_changed[:2]):
            logging.info("STAGE 2: Getting osc prob maps...")
            with Timer() as t:
                self.osc_flux_maps = get_osc_flux(self.flux_maps, self.osc_service,oversample_e=self.oversample_e,oversample_cz=self.oversample_cz,**params)
            profile.debug("==> elapsed time for oscillations stage: %s sec"%t.secs)
        else:
            profile.info("STAGE 2: Reused from step before...")

        if any(step_changed[:3]):
            logging.info("STAGE 3: Getting event rate true maps...")
            with Timer() as t:
                self.event_rate_maps = get_event_rates(self.osc_flux_maps,self.aeff_service, **params)
            profile.debug("==> elapsed time for aeff stage: %s sec"%t.secs)
        else:
            profile.info("STAGE 3: Reused from step before...")

        if any(step_changed[:4]):
            logging.info("STAGE 4: Getting event rate reco maps...")
            with Timer() as t:
                self.event_rate_reco_maps = get_reco_maps(self.event_rate_maps, self.anlys_ebins,self.reco_service,**params)
            profile.debug("==> elapsed time for reco stage: %s sec"%t.secs)
        else:
            profile.info("STAGE 4: Reused from step before...")

        if any(step_changed[:5]):
            logging.info("STAGE 5: Getting pid maps...")
            with Timer(verbose=False) as t:
                self.event_rate_pid_maps = get_pid_maps(self.event_rate_reco_maps,self.pid_service)
            profile.debug("==> elapsed time for pid stage: %s sec"%t.secs)
        else:
            profile.info("STAGE 5: Reused from step before...")

        #self.hole_ice_maps = self.HoleIce.apply_sys(self.event_rate_pid_maps, params['hole_ice'])
        # test
        self.hole_ice_maps = self.event_rate_pid_maps 

        if any(step_changed[:6]):
            logging.info("STAGE 6: Getting bkgd maps...")
            with Timer(verbose=False) as t:
                self.final_event_rate = add_icc_background(self.hole_ice_maps, self.background_service,**params)
            profile.debug("==> elapsed time for bkgd stage: %s sec"%t.secs)
        else:
            profile.info("STAGE 6: Reused from step before...")

        if not return_stages:
            
            # right now this is after the bakgd stage, just for tests, these will move between stages 5 and 6
            sys_maps = self.HoleIce.apply_sys(self.final_event_rate, params['hole_ice'])
            sys_maps = self.DomEfficiency.apply_sys(sys_maps, params['dom_eff'])
            sys_maps = self.Resolution.apply_sys(sys_maps, params['e_reco_precision_up'], params['e_reco_precision_down'], params['cz_reco_precision_up'], params['cz_reco_precision_down'])

            return sys_maps

        # Otherwise, return all stages as a simple tuple
        return (self.flux_maps, self.osc_flux_maps, self.event_rate_maps,
                self.event_rate_reco_maps, self.final_event_rate)

    def get_tau_template(self, params, return_stages=False):
        '''
        Runs template making chain, only return tau neutrinos map.
        '''

        logging.info("STAGE 1: Getting Atm Flux maps...")
        with Timer() as t:
            flux_maps = get_flux_maps(self.flux_service, self.ebins,
                                      self.czbins, **params)
        profile.debug("==> elapsed time for flux stage: %s sec"%t.secs)

        logging.info("STAGE 2: Getting osc prob maps...")
        with Timer() as t:
            osc_flux_maps = get_osc_flux(flux_maps, self.osc_service,
                                         oversample_e=self.oversample_e,
                                         oversample_cz=self.oversample_cz,
                                         **params)
        flavours = ['numu', 'numu_bar','nue','nue_bar']
        test_map = flux_maps['nue']
        for flav in flavours:
            osc_flux_maps[flav] = {'map': np.zeros_like(test_map['map']),
                               'ebins': np.zeros_like(test_map['ebins']),
                               'czbins': np.zeros_like(test_map['czbins'])}
        profile.debug("==> elapsed time for oscillations stage: %s sec"%t.secs)

        logging.info("STAGE 3: Getting event rate true maps...")
        with Timer() as t:
            event_rate_maps = get_event_rates(osc_flux_maps,
                                              self.aeff_service, **params)
        profile.debug("==> elapsed time for aeff stage: %s sec"%t.secs)

        logging.info("STAGE 4: Getting event rate reco maps...")
        with Timer() as t:
            event_rate_reco_maps = get_reco_maps(event_rate_maps,
                                                 self.reco_service,
                                                 **params)
        profile.debug("==> elapsed time for reco stage: %s sec"%t.secs)

        logging.info("STAGE 5: Getting pid maps...")
        with Timer(verbose=False) as t:
            final_event_rate = get_pid_maps(event_rate_reco_maps,
                                            self.pid_service)
        profile.debug("==> elapsed time for pid stage: %s sec"%t.secs)

        # if "residual_up_down" is true, then the final event rate map is
        # upgoing - reflected downgoing map.
        if params["residual_up_down"]:
            czbin_edges = len(final_event_rate['cscd']['czbins'])
            czbin_mid_idx = (czbin_edges-1)/2
            residual_event_rate = {flav:{
                'map': np.nan_to_num((final_event_rate[flav]['map'][:,0:czbin_mid_idx]   
                          - np.fliplr(final_event_rate[flav]['map'][:,czbin_mid_idx:]))),
                'ebins':final_event_rate[flav]['ebins'],
                'czbins': final_event_rate[flav]['czbins'][0:czbin_mid_idx+1] }
                           for flav in ['cscd','trck']}
            final_event_rate = residual_event_rate

        # if "ratio_up_down" is true, then the final event rate map is
        # an array: [upgoing map,reflected downgoing map].
        if params["ratio_up_down"]:
            czbin_edges = len(final_event_rate['cscd']['czbins'])
            czbin_mid_idx = (czbin_edges-1)/2
            ratio_event_rate = {flav:{
                'map': np.array([final_event_rate[flav]['map'][:,0:czbin_mid_idx],
                          np.fliplr(final_event_rate[flav]['map'][:,czbin_mid_idx:])]),
                'ebins':final_event_rate[flav]['ebins'],
                'czbins': final_event_rate[flav]['czbins'][0:czbin_mid_idx+1] }
                           for flav in ['cscd','trck']}
            final_event_rate = ratio_event_rate

        if not return_stages:
            return final_event_rate

        # Otherwise, return all stages as a simple tuple
        return (flux_maps, osc_flux_maps, event_rate_maps,
                event_rate_reco_maps, final_event_rate)

    def get_template_no_osc(self, params):
        '''
        Runs template making chain, but without oscillations
        '''

        logging.info("STAGE 1: Getting Atm Flux maps...")
        with Timer() as t:
            flux_maps = get_flux_maps(self.flux_service, self.ebins,
                                      self.czbins, **params)
        profile.debug("==> elapsed time for flux stage: %s sec"%t.secs)

        # Skipping oscillation stage...
        logging.info("  >>Skipping Stage 2 in no oscillations case...")
        flavours = ['nutau', 'nutau_bar']
        # Create the empty nutau maps:
        test_map = flux_maps['nue']
        for flav in flavours:
            flux_maps[flav] = {'map': np.zeros_like(test_map['map']),
                               'ebins': np.zeros_like(test_map['ebins']),
                               'czbins': np.zeros_like(test_map['czbins'])}

        logging.info("STAGE 3: Getting event rate true maps...")
        with Timer() as t:
            event_rate_maps = get_event_rates(flux_maps, self.aeff_service,
                                              **params)
        profile.debug("==> elapsed time for aeff stage: %s sec"%t.secs)

        logging.info("STAGE 4: Getting event rate reco maps...")
        with Timer() as t:
            event_rate_reco_maps = get_reco_maps(event_rate_maps,
                                                 self.reco_service, **params)
        profile.debug("==> elapsed time for reco stage: %s sec"%t.secs)

        logging.info("STAGE 5: Getting pid maps...")
        with Timer(verbose=False) as t:
            final_event_rate = get_pid_maps(event_rate_reco_maps,
                                            self.pid_service)
        profile.debug("==> elapsed time for pid stage: %s sec"%t.secs)

        return final_event_rate


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
        logging.info('Selected %s hierarchy'%
                     ('normal' if args.normal else 'inverted'))
        params =  select_hierarchy(model_settings['params'],
                                   normal_hierarchy=args.normal)

        #Intialize template maker
        template_maker = TemplateMaker(get_values(params),
                                       **model_settings['binning'])
    profile.info("  ==> elapsed time to initialize templates: %s sec"%t.secs)

    #Now get the actual template
    with Timer(verbose=False) as t:
        template_maps = template_maker.get_template(get_values(params),
                                                    return_stages=args.save_all)
    profile.info("==> elapsed time to get template: %s sec"%t.secs)

    logging.info("Saving file to %s"%args.outfile)
    to_json(template_maps, args.outfile)

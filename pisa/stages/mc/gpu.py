import sys, os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time

from pisa.core.stage import Stage
from pisa.utils.resources import find_resource
from pisa.utils.log import logging
from pisa.stages.osc.prob3gpu import Prob3GPU
from pisa.utils.GPUhist import GPUhist
from pisa.stages.mc.GPUweight import GPUweight
from pisa.utils.const import FTYPE
from pisa.core.events import Events
from pisa import ureg, Q_
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.map import Map, MapSet
from pisa.utils.log import logging
from pisa.utils.comparisons import normQuant
from pisa.utils.hash import hash_obj


def copy_dict_to_d(events):
    d_events = {}
    for key, val in events.items():
        d_events[key] = cuda.mem_alloc(val.nbytes)
        cuda.memcpy_htod(d_events[key], val)
    return d_events

class gpu(Stage):

    def __init__(self, params, output_binning, disk_cache=None,
                memcaching_enabled=True, error_method=None,
                outputs_cache_depth=20, debug_mode=None):

        self.osc_params = (
            'detector_depth',
            'earth_model',
            'prop_height', 
            'YeI',
            'YeO',
            'YeM',
            'theta12',
            'theta13',
            'theta23',
            'deltam21',
            'deltam31',
            'deltacp',
            'no_nc_osc',
            )


        self.weight_params = (
            'nu_nubar_ratio',
            'nue_numu_ratio',
            'livetime',
            'aeff_scale',
            'pid_bound',
            'pid_remove',
            'delta_index',
        )

        self.other_params = (
            'events_file',
            'nu_nc_norm',
            'nutau_cc_norm',
            'reco_e_res_raw',
            'reco_e_scale_raw',
            'reco_cz_res_raw'
        )

        expected_params = self.osc_params + self.weight_params + self.other_params

        output_names = ('trck','cscd')

        super(self.__class__, self).__init__(
            use_transforms=False,
            stage_name='mc',
            service_name='gpu',
            params=params,
            expected_params=expected_params,
            output_names=output_names,
            error_method=error_method,
            disk_cache=disk_cache,
            memcaching_enabled=memcaching_enabled,
            outputs_cache_depth=outputs_cache_depth,
            output_binning=output_binning,
            debug_mode=debug_mode
        )


    def _compute_nominal_outputs(self):

        self.osc_hash = None
        self.weight_hash = None

        # initialize classes
        earth_model = find_resource(self.params.earth_model.value)
        YeI = self.params.YeI.value.m_as('dimensionless')
        YeO = self.params.YeO.value.m_as('dimensionless')
        YeM = self.params.YeM.value.m_as('dimensionless')
        prop_height = self.params.prop_height.value.m_as('km')
        detector_depth = self.params.detector_depth.value.m_as('km')

        self.osc = Prob3GPU(detector_depth,
                            earth_model,
                            prop_height,
                            YeI,
                            YeO,
                            YeM)

        self.weight = GPUweight()

        self.e_dim_num = self.output_binning.names.index('reco_energy')
        self.cz_dim_num = self.output_binning.names.index('reco_coszen')
        self.bin_names = self.output_binning.names
        self.bin_edges = []
        for name in self.bin_names:
            if 'energy' in  name:
                bin_edges = self.output_binning[name].bin_edges.to('GeV').magnitude.astype(FTYPE)
            else:
                bin_edges = self.output_binning[name].bin_edges.magnitude.astype(FTYPE)
            self.bin_edges.append(bin_edges)
            

        self.histogrammer = GPUhist(*self.bin_edges)

        # --- Load events
        # open Events file
        evts = Events(self.params.events_file.value)

        # Load and copy events
        variables = ['true_energy', 'true_coszen', 'reco_energy', 'reco_coszen',
                    'neutrino_nue_flux', 'neutrino_numu_flux', 'neutrino_oppo_nue_flux',
                    'neutrino_oppo_numu_flux', 'weighted_aeff', 'pid']
        empty = ['prob_e', 'prob_mu', 'weight_trck', 'weight_cscd']
        if self.error_method == 'sumw2':
            empty += ['sumw2_trck', 'sumw2_cscd']
        self.flavs = ['nue_cc', 'numu_cc', 'nutau_cc', 'nue_nc', 'numu_nc', 'nutau_nc',
                'nuebar_cc', 'numubar_cc', 'nutaubar_cc', 'nuebar_nc', 'numubar_nc', 'nutaubar_nc']
        kFlavs = [0, 1, 2] * 4
        kNuBars = [1] *6 + [-1] * 6

        logging.info('read in events and copy to GPU')
        start_t = time.time()
        # setup all arrays that need to be put on GPU
        self.events_dict = {}
        for flav, kFlav, kNuBar in zip(self.flavs, kFlavs, kNuBars):
            self.events_dict[flav] = {}
            # neutrinos: 1, anti-neutrinos: -1 
            self.events_dict[flav]['kNuBar'] = kNuBar
            # electron: 0, muon: 1, tau: 2
            self.events_dict[flav]['kFlav'] = kFlav
            # host arrays
            self.events_dict[flav]['host'] = {}
            for var in variables:
                if var == 'reco_energy':
                    # apply energy reco sys
                    f = self.params.reco_e_res_raw.value.m_as('dimensionless')
                    g = self.params.reco_e_scale_raw.value.m_as('dimensionless')
                    self.events_dict[flav]['host']['reco_energy'] = (g * ((1.-f) * evts[flav]['true_energy'] + f * evts[flav]['reco_energy'])).astype(FTYPE)
                elif var == 'reco_coszen':
                    # apply coszen reco sys
                    f = self.params.reco_cz_res_raw.value.m_as('dimensionless')
                    reco_coszen = ((1.-f) * evts[flav]['true_coszen'] + f * evts[flav]['reco_coszen']).astype(FTYPE)
                    while np.any(reco_coszen<-1) or np.any(reco_coszen>1):
                        reco_coszen[reco_coszen>1] = 2-reco_coszen[reco_coszen>1]
                        reco_coszen[reco_coszen<-1] = -2-reco_coszen[reco_coszen<-1]
                    self.events_dict[flav]['host']['reco_coszen'] = reco_coszen 
                else:
                    self.events_dict[flav]['host'][var] = evts[flav][var].astype(FTYPE)
            self.events_dict[flav]['n_evts'] = np.uint32(len(self.events_dict[flav]['host'][variables[0]]))
            for var in empty:
                if self.params.no_nc_osc and ( (flav in ['nue_nc', 'nuebar_nc'] and var == 'prob_e') or (flav in ['numu_nc', 'numubar_nc'] and var == 'prob_mu') ):
                    self.events_dict[flav]['host'][var] = np.ones(self.events_dict[flav]['n_evts'], dtype=FTYPE)
                else:
                    self.events_dict[flav]['host'][var] = np.zeros(self.events_dict[flav]['n_evts'], dtype=FTYPE)
            # calulate layers
            self.events_dict[flav]['host']['numLayers'], self.events_dict[flav]['host']['densityInLayer'], self.events_dict[flav]['host']['distanceInLayer'] = self.osc.calc_Layers(self.events_dict[flav]['host']['true_coszen'])
        end_t = time.time()
        logging.debug('layers done in %.4f ms'%((end_t - start_t) * 1000))

        # copy arrays to GPU
        start_t = time.time()
        for flav in self.flavs:
            self.events_dict[flav]['device'] = copy_dict_to_d(self.events_dict[flav]['host'])
        end_t = time.time()
        logging.debug('copy done in %.4f ms'%((end_t - start_t) * 1000))

    def _compute_outputs(self, inputs=None):
        logging.info('retreive weighted histo')
        # get hash to decide wether weight and/or osc needs to be racalculated
        osc_hash = hash_obj(normQuant([self.params[name].value for name in self.osc_params]))
        weight_hash = hash_obj(normQuant([self.params[name].value for name in self.weight_params]))
        recalc_osc = not (osc_hash == self.osc_hash)
        recalc_weight = not (weight_hash == self.weight_hash)
        recalc_weight = True

        if recalc_weight:
            livetime = self.params.livetime.value.m_as('seconds')
            pid_bound = self.params.pid_bound.value.m_as('dimensionless')
            pid_remove = self.params.pid_remove.value.m_as('dimensionless')
            aeff_scale = self.params.aeff_scale.value.m_as('dimensionless')
            nue_numu_ratio = self.params.nue_numu_ratio.value.m_as('dimensionless')
            nu_nubar_ratio = self.params.nu_nubar_ratio.value.m_as('dimensionless')
            delta_index = self.params.delta_index.value.m_as('dimensionless')

        if recalc_osc:
            theta12 = self.params.theta12.value.m_as('rad')
            theta13 = self.params.theta13.value.m_as('rad')
            theta23 = self.params.theta23.value.m_as('rad')
            deltam21 = self.params.deltam21.value.m_as('eV**2')
            deltam31 = self.params.deltam31.value.m_as('eV**2')
            deltacp = self.params.deltacp.value.m_as('rad')
            self.osc.update_MNS(theta12, theta13, theta23, deltam21, deltam31, deltacp)

        tot = 0
        start_t = time.time()
        for flav in self.flavs:
            # calculate osc probs
            if recalc_osc and not (self.params.no_nc_osc and flav.endswith('_nc')):
                self.osc.calc_probs(self.events_dict[flav]['kNuBar'], self.events_dict[flav]['kFlav'],
                                self.events_dict[flav]['n_evts'], **self.events_dict[flav]['device'])

            # calculate weights
            if recalc_weight:
                self.weight.calc_weight(self.events_dict[flav]['n_evts'], livetime=livetime,
                                    pid_bound=pid_bound, pid_remove=pid_remove,
                                    aeff_scale=aeff_scale, nue_numu_ratio=nue_numu_ratio, 
                                    nu_nubar_ratio=nu_nubar_ratio, kNuBar=self.events_dict[flav]['kNuBar'],
                                    delta_index=delta_index,
                                    **self.events_dict[flav]['device'])

                if self.error_method == 'sumw2':
                    self.weight.calc_sumw2(self.events_dict[flav]['n_evts'], **self.events_dict[flav]['device'])

            tot += self.events_dict[flav]['n_evts']
        end_t = time.time()
        logging.debug('GPU calc done in %.4f ms for %s events'%(((end_t - start_t) * 1000),tot))

        if recalc_osc or recalc_weight:
            start_t = time.time()
            # histogram events and download fromm GPU, if either weights or osc changed
            for flav in self.flavs:
                self.events_dict[flav]['hist_cscd'] = self.histogrammer.get_hist(self.events_dict[flav]['n_evts'],
                                                                        self.events_dict[flav]['device'][self.bin_names[0]],
                                                                        self.events_dict[flav]['device'][self.bin_names[1]],
                                                                        self.events_dict[flav]['device']['weight_cscd'])

                self.events_dict[flav]['hist_trck'] = self.histogrammer.get_hist(self.events_dict[flav]['n_evts'],
                                                                        self.events_dict[flav]['device'][self.bin_names[0]],
                                                                        self.events_dict[flav]['device'][self.bin_names[1]],
                                                                        self.events_dict[flav]['device']['weight_trck'])

                if self.error_method == 'sumw2':
                    self.events_dict[flav]['sumw2_cscd'] = self.histogrammer.get_hist(self.events_dict[flav]['n_evts'],
                                                                            self.events_dict[flav]['device'][self.bin_names[0]],
                                                                            self.events_dict[flav]['device'][self.bin_names[1]],
                                                                            self.events_dict[flav]['device']['sumw2_cscd'])

                    self.events_dict[flav]['sumw2_trck'] = self.histogrammer.get_hist(self.events_dict[flav]['n_evts'],
                                                                            self.events_dict[flav]['device'][self.bin_names[0]],
                                                                            self.events_dict[flav]['device'][self.bin_names[1]],
                                                                            self.events_dict[flav]['device']['sumw2_trck'])
            end_t = time.time()
            logging.debug('GPU hist done in %.4f ms for %s events'%(((end_t - start_t) * 1000),tot))

        
        # set new hash
        self.osc_hash = osc_hash
        self.weight_hash = weight_hash

        maps = []
        # apply scales, add up all cscds and tracks, and pack them up in final PISA MapSet
        for i,flav in enumerate(self.flavs):
            if flav in ['nutau_cc','nutaubar_cc']:
                f = self.params.nutau_cc_norm.value.m_as('dimensionless')
            elif flav.endswith('_nc'):
                f = self.params.nu_nc_norm.value.m_as('dimensionless')
            else:
                f = 1.0
            # add up
            if i == 0:
                hist_cscd = np.copy(self.events_dict[flav]['hist_cscd']) * f
                hist_trck = np.copy(self.events_dict[flav]['hist_trck']) * f
                if self.error_method == 'sumw2':
                    sumw2_cscd = np.copy(self.events_dict[flav]['sumw2_cscd']) * f * f
                    sumw2_trck = np.copy(self.events_dict[flav]['sumw2_trck']) * f * f
            else:
                hist_cscd += self.events_dict[flav]['hist_cscd'] * f
                hist_trck += self.events_dict[flav]['hist_trck'] * f
                if self.error_method == 'sumw2':
                    sumw2_cscd += self.events_dict[flav]['sumw2_cscd'] * f * f
                    sumw2_trck += self.events_dict[flav]['sumw2_trck'] * f * f

        if self.error_method == 'sumw2':
            maps.append(Map(name='cscd', hist=hist_cscd, error_hist=np.sqrt(sumw2_cscd), binning=self.output_binning))
            maps.append(Map(name='trck', hist=hist_trck, error_hist=np.sqrt(sumw2_trck), binning=self.output_binning))
        else:
            maps.append(Map(name='cscd', hist=hist_cscd, binning=self.output_binning))
            maps.append(Map(name='trck', hist=hist_trck, binning=self.output_binning))

        return MapSet(maps,name='gpu_mc')
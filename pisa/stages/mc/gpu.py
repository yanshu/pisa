# authors: P.Eller (pde3@psu.edu)
# date:   September 2016

import sys, os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time

from pisa import ureg, Q_
from pisa.core.stage import Stage
from pisa.core.param import ParamSet
from pisa.core.events import Events
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.map import Map, MapSet
from pisa.stages.osc.prob3gpu import prob3gpu
from pisa.stages.mc.GPUweight import GPUweight
from pisa.utils.resources import find_resource
from pisa.utils.log import logging
from pisa.utils.gpu_hist import GPUhist
from pisa.utils.const import FTYPE
from pisa.utils.log import logging
from pisa.utils.comparisons import normQuant
from pisa.utils.hash import hash_obj

class gpu(Stage):
    '''
    GPU accelerated (pyCUDA) service to compute histograms based on reweighted MC events.

    Parameters
    ----------
    params : ParamSet or sequence with which to instantiate a ParamSet.
        Expected params are:

            detector_depth : float
            earth_model : PREM file path
            prop_height : quantity (dimensionless)
            YeI : quantity (dimensionless)
            YeO : quantity (dimensionless)
            YeM : quantity (dimensionless)
            theta12 : quantity (angle)
            theta13 : quantity (angle)
            theta23 : quantity (angle)
            deltam21 : quantity (mass^2)
            deltam31 : quantity (mass^2)
            deltacp : quantity (angle)
            no_nc_osc : bool
                don't oscillate NC events, but rather assign probabilities 1 for nue->nue and numu->numu, and 0 for nutau->nutau
            nu_nubar_ratio : quantity (dimensionless)
            nue_numu_ratio : quantity (dimensionless)
            livetime : quantity (time)
            aeff_scale : quantity (dimensionless)
            delta_index : quantity (dimensionless)
            Barr_uphor_ratio : quantity (dimensionless)
            Barr_nu_nubar_ratio : quantity (dimensionless)
            Genie_Ma_QE : quantity (dimensionless)
            Genie_Ma_RES : quantity (dimensionless)
            events_file : hdf5 file path (output from make_events), including flux weights and Genie systematics coefficients
            nutau_cc_norm : quantity (dimensionless)
            nutau_norm : quantity (dimensionless)
            reco_e_res_raw : quantity (dimensionless)
            reco_e_scale_raw : quantity (dimensionless)
            reco_cz_res_raw :quantity (dimensionless)
            bdt_cut : quantity (dimensionless)

    Notes
    -----

    This stage takes as inputs hdf5 events files that already iclude the nominal flux weights and genie systematic coefficients. The files are expected to contain
        true_energy : true energy of the event in GeV
        true_coszen : true coszen of the event
        reco_energy : reco energy of the event in GeV
        reco_coszen : reco coszen of the event
        neutrino_nue_flux : flux weight for nu_e in case of neutrino, or anit-nu_e in case of anti-neutrino event
        neutrino_numu_flux : flux weight for nu_mu in case of neutrino, or anit-nu_mu in case of anti-neutrino event
        neutrino_oppo_nue_flux:flux weight for anti-nu_e in case of neutrino, or nu_e in case of anti-neutrino event
        neutrino_oppo_numu_flux :flux weight for anti-nu_mu in case of neutrino, or nu_mu in case of anti-neutrino event
        weighted_aeff : effective are weight for event
        pid : pid value

    and more optional:
        dunkman_l5 : BDT value
        linear_fit_maccqe : Genie CC quasi elastic linear coefficient
        quad_fit_maccqe : Genie CC quasi elastic quadratic coefficient
        linear_fit_maccres : Genie CC resonance linear coefficient
        quad_fit_maccres : Genie CC resonance quadratic coefficient

    the dictionary self.events_dict is the central object here:
    it contains two dictionaries for every flavour:
        * host : all event arrays on CPU
        * device : pointers to the same arrays, but on GPU
    and some additional keys like:
        * n_evts : number of events
        * hist : retreived histogram
        * ...

    All floats (arrays) on GPU are of type FTYPE, currently double precision
        
    '''

    def __init__(self, params, output_binning, disk_cache=None,
                 memcache_deepcopy=True, error_method=None,
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
            'no_nc_osc')

        self.weight_params = (
            'nu_nubar_ratio',
            'nue_numu_ratio',
            'livetime',
            'aeff_scale',
            'delta_index',
            'Barr_uphor_ratio',
            'Barr_nu_nubar_ratio',
            'Genie_Ma_QE',
            'Genie_Ma_RES')

        self.other_params = (
            'events_file',
            'nutau_cc_norm',
            'nutau_norm',
            'reco_e_res_raw',
            'reco_e_scale_raw',
            'reco_cz_res_raw',
            'bdt_cut')

        expected_params = (self.osc_params + self.weight_params +
                           self.other_params)

        output_names = ('nue_cc+nuebar_cc', 'numu_cc+numubar_cc', 'nutau_cc+nutaubar_cc', 'nuall_nc+nuallbar_nc')

        super(self.__class__, self).__init__(
            use_transforms=False,
            params=params,
            expected_params=expected_params,
            output_names=output_names,
            error_method=error_method,
            disk_cache=disk_cache,
            memcache_deepcopy=memcache_deepcopy,
            outputs_cache_depth=outputs_cache_depth,
            output_binning=output_binning,
            debug_mode=debug_mode
        )

    def validate_params(self, params):
        # not a good idea to scale nutau norm, without the NC events being oscillated
        if params.nutau_norm.value != 1.0 or params.nutau_norm.is_fixed == False:
            assert (params.no_nc_osc.value == False), 'if you want NC tau events scaled, you should osciallte them -> set no_nc_osc to False!!!'

    def _compute_nominal_outputs(self):
        # these are for storing hashes for caching that is done inside the stage
        self.osc_hash = None
        self.weight_hash = None

        # reset fixed errors
        self.fixed_error = None

        # initialize classes
        earth_model = find_resource(self.params.earth_model.value)
        YeI = self.params.YeI.value.m_as('dimensionless')
        YeO = self.params.YeO.value.m_as('dimensionless')
        YeM = self.params.YeM.value.m_as('dimensionless')
        prop_height = self.params.prop_height.value.m_as('km')
        detector_depth = self.params.detector_depth.value.m_as('km')

        # Prob3 GPU oscialltions
	osc_params_subset = []
        for param in self.params:
            if param.name in self.osc_params:
                osc_params_subset.append(param)
        osc_params_subset = ParamSet(osc_params_subset)

        self.osc = prob3gpu(params=osc_params_subset, 
                            input_binning=None, 
                            output_binning=None,
                            error_method=None,
                            memcache_deepcopy=False,
                            transforms_cache_depth=0,
                            outputs_cache_depth=0,
                            )

        # Weight calculator
        self.weight = GPUweight()

        self.bin_names = self.output_binning.names
        self.bin_edges = []

        for name in self.bin_names:
            if 'energy' in  name:
                bin_edges = self.output_binning[name].bin_edges.to('GeV').magnitude.astype(FTYPE)
            else:
                bin_edges = self.output_binning[name].bin_edges.magnitude.astype(FTYPE)
            self.bin_edges.append(bin_edges)
            
        # GPU histogramer
        self.histogrammer = GPUhist(*self.bin_edges)

        # load events
        self.load_events()

    def load_events(self):
        # --- Load events
        # open Events file
        evts = Events(self.params.events_file.value)
        bdt_cut = self.params.bdt_cut.value.m_as('dimensionless')

        # Load and copy events
        variables = ['true_energy',
                        'true_coszen',
                        'reco_energy',
                        'reco_coszen',
                        'neutrino_nue_flux',
                        'neutrino_numu_flux',
                        'neutrino_oppo_nue_flux',
                        'neutrino_oppo_numu_flux',
                        'weighted_aeff',
                        'pid',
                        'dunkman_L5',
                        'linear_fit_MaCCQE',
                        'quad_fit_MaCCQE',
                        'linear_fit_MaCCRES',
                        'quad_fit_MaCCRES',
                    ]
        # allocate empty arrays (filled with 1s) on GPU
        empty = ['prob_e',
                    'prob_mu',
                    'weight',
                    'scaled_nue_flux',
                    'scaled_numu_flux',
                    'scaled_nue_flux_shape',
                    'scaled_numu_flux_shape'
                ]

        if self.error_method in ['sumw2', 'fixed_sumw2']:
            empty += ['sumw2']

        # list of flav_ints to use and corresponding number used in several parts of the code
        self.flavs = ['nue_cc',
                        'numu_cc',
                        'nutau_cc',
                        'nue_nc',
                        'numu_nc',
                        'nutau_nc',
                        'nuebar_cc',
                        'numubar_cc',
                        'nutaubar_cc',
                        'nuebar_nc',
                        'numubar_nc',
                        'nutaubar_nc'
                    ]

        # corrwsponding numbers for the flavours defined above, needed bi Prob3
        kFlavs = [0, 1, 2] * 4
        kNuBars = [1] *6 + [-1] * 6

        # only keep events using bdt_score > bdt_cut
        for flav, kFlav, kNuBar in zip(self.flavs, kFlavs, kNuBars):
            if evts[flav].has_key('dunkman_L5'):
                l5_bdt_score = evts[flav]['dunkman_L5'].astype(FTYPE)
                cut = l5_bdt_score >= bdt_cut
            else:
                cut = None
            for var in variables:
                try:
                    if cut is not None:
                        evts[flav][var] = evts[flav][var][cut]
                except KeyError:
                    pass

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
                try:
                    self.events_dict[flav]['host'][var] = evts[flav][var].astype(FTYPE)
                except KeyError:
                    # if variable doesn't exist (e.g. axial mass coeffs, just fill in ones)
                    logging.warning('replacing variable %s by ones'%var)
                    self.events_dict[flav]['host'][var] = np.ones_like(evts[flav]['true_energy']).astype(FTYPE)
            self.events_dict[flav]['n_evts'] = np.uint32(len(self.events_dict[flav]['host'][variables[0]]))
            #select even 50%
            #self.events_dict[flav]['host']['weighted_aeff'][::2] = 0
            #select odd 50%
            #self.events_dict[flav]['host']['weighted_aeff'][1::2] = 0
            #every 10th event only
            #cut = np.zeros_like(self.events_dict[flav]['host']['weighted_aeff'])
            #cut[9::10] = 1
            #self.events_dict[flav]['host']['weighted_aeff']*=cut
            for var in empty:
                if self.params.no_nc_osc and ( (flav in ['nue_nc', 'nuebar_nc'] and var == 'prob_e') or \
                        (flav in ['numu_nc', 'numubar_nc'] and var == 'prob_mu') ):
                    # in case of not oscillating NC events, we can set the probabilities of nue->nue and numu->numu at 1, and nutau->nutau at 0
                    self.events_dict[flav]['host'][var] = np.ones(self.events_dict[flav]['n_evts'], dtype=FTYPE)
                else:
                    self.events_dict[flav]['host'][var] = np.zeros(self.events_dict[flav]['n_evts'], dtype=FTYPE)
            # calulate layers (every particle crosses a number of layers in the earth with different densities, and for a given length
            # these depend only on the earth model (PREM) and the true coszen of an event. Therefore we can calculate these for once and are done
            self.events_dict[flav]['host']['numLayers'], \
                self.events_dict[flav]['host']['densityInLayer'], \
                self.events_dict[flav]['host']['distanceInLayer'] = \
                self.osc.calc_Layers(self.events_dict[flav]['host']['true_coszen'])
        end_t = time.time()
        logging.debug('layers done in %.4f ms'%((end_t - start_t) * 1000))

        # copy arrays to GPU
        start_t = time.time()
        for flav in self.flavs:
            self.events_dict[flav]['device'] = {}
            for key, val in self.events_dict[flav]['host'].items():
                self.events_dict[flav]['device'][key] = cuda.mem_alloc(val.nbytes)
                cuda.memcpy_htod(self.events_dict[flav]['device'][key], val)
        end_t = time.time()
        logging.debug('copy done in %.4f ms'%((end_t - start_t) * 1000))

        # apply raw reco sys
        self.apply_reco()


    def apply_reco(self):
        ''' applying raw reco systematics (to use as inputs to polyfit stage) '''
        for flav in self.flavs:
            # apply energy reco sys
            f = self.params.reco_e_res_raw.value.m_as('dimensionless')
            g = self.params.reco_e_scale_raw.value.m_as('dimensionless')
            self.events_dict[flav]['host']['reco_energy'] = \
                (g * ((1.-f) * self.events_dict[flav]['host']['true_energy'] + \
                f * self.events_dict[flav]['host']['reco_energy'])).astype(FTYPE)

            # apply coszen reco sys
            f = self.params.reco_cz_res_raw.value.m_as('dimensionless')
            self.events_dict[flav]['host']['reco_coszen'] = \
                ((1.-f) * self.events_dict[flav]['host']['true_coszen'] + \
                f * self.events_dict[flav]['host']['reco_coszen']).astype(FTYPE)
            # make sure everything is within -1 <= coszen <= 1, otherwise reflect
            while np.any(self.events_dict[flav]['host']['reco_coszen']<-1) or \
                np.any(self.events_dict[flav]['host']['reco_coszen']>1):
                self.events_dict[flav]['host']['reco_coszen'][self.events_dict[flav]['host']['reco_coszen']>1] = \
                    2-self.events_dict[flav]['host']['reco_coszen'][self.events_dict[flav]['host']['reco_coszen']>1]
                self.events_dict[flav]['host']['reco_coszen'][self.events_dict[flav]['host']['reco_coszen']<-1] = \
                    -2-self.events_dict[flav]['host']['reco_coszen'][self.events_dict[flav]['host']['reco_coszen']<-1]

            self.update_device_arrays(flav, 'reco_energy')
            self.update_device_arrays(flav, 'reco_coszen')


    def update_device_arrays(self, flav, var):
        ''' helper function to update device arrays '''
        self.events_dict[flav]['device'][var].free()
        self.events_dict[flav]['device'][var] = cuda.mem_alloc(self.events_dict[flav]['host'][var].nbytes)
        cuda.memcpy_htod(self.events_dict[flav]['device'][var], self.events_dict[flav]['host'][var])
        
    def get_device_arrays(self):
        ''' Copy back a dictionary with event by event information'''
        return_events = {}
        variables = ['true_energy', 'true_coszen', 'reco_energy', 'reco_coszen',
                        'weight']
        for flav in self.flavs:
            return_events[flav] = {}
            for var in variables:
                buff = np.ones(self.events_dict[flav]['n_evts'])
                cuda.memcpy_dtoh(buff, self.events_dict[flav]['device'][var])
                return_events[flav][var] = buff
        return return_events

    def sum_array(self, x, n_evts):
        '''
        helper function to compute the sum over a device array
        '''
	out = np.array([0.]).astype(FTYPE)
        d_out = cuda.mem_alloc(out.nbytes)
        cuda.memcpy_htod(d_out, out)
	self.weight.calc_sum(n_evts, x, d_out)
	cuda.memcpy_dtoh(out, d_out)
	return out[0]


    def _compute_outputs(self, inputs=None):
        logging.info('retreive weighted histo')
        # get hash to decide wether expensive stuff needs to be recalculated 
        osc_hash = hash_obj(normQuant([self.params[name].value for name in self.osc_params]))
        weight_hash = hash_obj(normQuant([self.params[name].value for name in self.weight_params]))
        recalc_osc = not (osc_hash == self.osc_hash)
        recalc_weight = True

        if recalc_weight:
            livetime = self.params.livetime.value.m_as('seconds')
            aeff_scale = self.params.aeff_scale.value.m_as('dimensionless')
            nue_numu_ratio = self.params.nue_numu_ratio.value.m_as('dimensionless')
            nu_nubar_ratio = self.params.nu_nubar_ratio.value.m_as('dimensionless')
            delta_index = self.params.delta_index.value.m_as('dimensionless')
            Barr_uphor_ratio = self.params.Barr_uphor_ratio.value.m_as('dimensionless')
            Barr_nu_nubar_ratio = self.params.Barr_nu_nubar_ratio.value.m_as('dimensionless')
            Genie_Ma_QE = self.params.Genie_Ma_QE.value.m_as('dimensionless')
            Genie_Ma_RES = self.params.Genie_Ma_RES.value.m_as('dimensionless')

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

            # calculate osc probs, filling the device arrays with probabilities
            if recalc_osc:
                if not (self.params.no_nc_osc.value and flav.endswith('_nc')):
                    self.osc.calc_probs(self.events_dict[flav]['kNuBar'], self.events_dict[flav]['kFlav'],
                                    self.events_dict[flav]['n_evts'], **self.events_dict[flav]['device'])

            # calculate weights
            if recalc_weight:
                # calcukate the flux weights
                self.weight.calc_flux(self.events_dict[flav]['n_evts'],
                                    nue_numu_ratio=nue_numu_ratio, 
                                    nu_nubar_ratio=nu_nubar_ratio, kNuBar=self.events_dict[flav]['kNuBar'],
                                    delta_index=delta_index,
                                    Barr_uphor_ratio=Barr_uphor_ratio, Barr_nu_nubar_ratio=Barr_nu_nubar_ratio,
                                    **self.events_dict[flav]['device'])

                # calculate global scales for flux normalizations
                #nue_flux_norm_n = self.sum_array(self.events_dict[flav]['device']['scaled_nue_flux'], self.events_dict[flav]['n_evts'])
                #nue_flux_norm_d = self.sum_array(self.events_dict[flav]['device']['scaled_nue_flux_shape'], self.events_dict[flav]['n_evts'])
                #nue_flux_norm = nue_flux_norm_n / nue_flux_norm_d
                nue_flux_norm = 1.
                #numu_flux_norm_n = self.sum_array(self.events_dict[flav]['device']['scaled_numu_flux'], self.events_dict[flav]['n_evts'])
                #numu_flux_norm_d = self.sum_array(self.events_dict[flav]['device']['scaled_numu_flux_shape'], self.events_dict[flav]['n_evts'])
                #numu_flux_norm = numu_flux_norm_n / numu_flux_norm_d
                numu_flux_norm = 1.
                
                # calculate the event weights, from osc. probs and flux weights
                # global scaling factors for the nue and numu flux can be given, for normalization purposes
                self.weight.calc_weight(self.events_dict[flav]['n_evts'], livetime=livetime,
                                    nue_flux_norm=nue_flux_norm, numu_flux_norm=numu_flux_norm,
                                    aeff_scale=aeff_scale, kNuBar=self.events_dict[flav]['kNuBar'],
                                    Genie_Ma_QE=Genie_Ma_QE, Genie_Ma_RES=Genie_Ma_RES,
                                    **self.events_dict[flav]['device'])

                
                # calculate weights squared, for error propagation
                if self.error_method in ['sumw2', 'fixed_sumw2']:
                    self.weight.calc_sumw2(self.events_dict[flav]['n_evts'], **self.events_dict[flav]['device'])

            tot += self.events_dict[flav]['n_evts']
        end_t = time.time()
        logging.debug('GPU calc done in %.4f ms for %s events'%(((end_t - start_t) * 1000),tot))

        if recalc_osc or recalc_weight:
            start_t = time.time()
            # histogram events and download fromm GPU, if either weights or osc changed
            for flav in self.flavs:
                self.events_dict[flav]['hist'] = self.histogrammer.get_hist(self.events_dict[flav]['n_evts'],
                                                                        d_x = self.events_dict[flav]['device'][self.bin_names[0]],
                                                                        d_y = self.events_dict[flav]['device'][self.bin_names[1]],
                                                                        d_z = self.events_dict[flav]['device'][self.bin_names[2]],
                                                                        d_w = self.events_dict[flav]['device']['weight'])

                if self.error_method in ['sumw2', 'fixed_sumw2']:
                    self.events_dict[flav]['sumw2'] = self.histogrammer.get_hist(self.events_dict[flav]['n_evts'],
                                                                            d_x = self.events_dict[flav]['device'][self.bin_names[0]],
                                                                            d_y = self.events_dict[flav]['device'][self.bin_names[1]],
                                                                            d_z = self.events_dict[flav]['device'][self.bin_names[2]],
                                                                            d_w = self.events_dict[flav]['device']['sumw2'])
            end_t = time.time()
            logging.debug('GPU hist done in %.4f ms for %s events'%(((end_t - start_t) * 1000),tot))

        
        # set new hash
        self.osc_hash = osc_hash
        self.weight_hash = weight_hash

        # add histos together into output names, and apply nutau normalizations
        # errors (sumw2) are also added, while scales are applied in quadrature of course
        out_hists = {}
        out_sumw2 = {}
        for name in self.output_names:
            for flav in self.flavs:
                f = 1.0
                if flav in ['nutau_cc','nutaubar_cc']:
                    f *= self.params.nutau_cc_norm.value.m_as('dimensionless')
                if 'nutau' in flav:
                    f *= self.params.nutau_norm.value.m_as('dimensionless')
                if ('bar_nc' in flav and 'allbar_nc' in name) or ('_nc' in flav and 'all_nc' in name) or (flav in name):
                    if out_hists.has_key(name):
                        out_hists[name] += self.events_dict[flav]['hist'] * f
                        if self.error_method in ['sumw2', 'fixed_sumw2']:
                            out_sumw2[name] += self.events_dict[flav]['sumw2'] * f * f
                    else:
                        out_hists[name] = np.copy(self.events_dict[flav]['hist']) * f
                        if self.error_method in ['sumw2', 'fixed_sumw2']:
                            out_sumw2[name] = np.copy(self.events_dict[flav]['sumw2']) * f * f

        maps = []
        # pack everything in a final PISA MapSet
        for name, hist in out_hists.items():
            if self.error_method == 'sumw2':
                maps.append(Map(name=name, hist=hist, error_hist=np.sqrt(out_sumw2[name]), binning=self.output_binning))
            # this is a special case where we always want the error to be the same....so for the first Mapet it is taken from
            # the calculation, and every following time it is just euqal to the first one
            elif self.error_method == 'fixed_sumw2':
                if self.fixed_error == None:
                    self.fixed_error = {}
                if not self.fixed_error.has_key(name):
                    self.fixed_error[name] = np.sqrt(out_sumw2[name])
                maps.append(Map(name=name, hist=hist, error_hist=self.fixed_error[name], binning=self.output_binning))
            else:
                maps.append(Map(name=name, tex=name, hist=hist, binning=self.output_binning))

        return MapSet(maps,name='gpu_mc')

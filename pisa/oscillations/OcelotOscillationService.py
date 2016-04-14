#
# This is a service which will return an oscillation probability map
# corresponding to the desired binning. Code is based on the prob3 oscillation
# package.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#

import sys
import numpy as np
from pisa.utils.log import logging, tprofile
from pisa.utils.proc import get_params, report_params
from pisa.oscillations.OscillationServiceBase import OscillationServiceBase
from ocelot.NeutrinoMixing import NeutrinoMixing
from ocelot.PREM import PREM
from ocelot.NeutrinoMixing import MixingParameters 
from ocelot.Atmosphere import SimpleAtmosphere
from ocelot.Atmosphere import NormalAtmosphere
from ocelot.Probabilities import Probabilities
from ocelot.BargerProbabilities import BargerProbabilities
from ocelot.pisa_interface import pisa_interface

class OcelotOscillationService(OscillationServiceBase):
    """
    This class handles all tasks related to the oscillation
    probability calculations using the prob3 oscillation code
    """
    def __init__(self, ebins, czbins, atmos_model, prob_model, **kwargs):
        """
        Parameters needed to instantiate a OcelotOscillationService:
        * ebins: Energy bin edges
        * czbins: cos(zenith) bin edges
        * earth_model: Earth density model used for matter oscillations.
        * detector_depth: Detector depth in km.
        * prop_height: Height in the atmosphere to begin in km.
        """
        OscillationServiceBase.__init__(self, ebins, czbins)
        logging.info('Initializing %s...'%self.__class__.__name__)

        report_params(get_params(),['km','','km'])

        self.atmos_model = atmos_model
        self.prob_model = prob_model
        self.earth_model = PREM()

    def fill_osc_prob(self, osc_prob_dict, ecen, czcen, event_by_event=False,
                      theta12=None, theta13=None, theta23=None,
                      deltam21=None, deltam31=None, deltacp=None,
                      energy_scale=None,**kwargs):
        '''
        Loops over ecen,czcen and fills the osc_prob_dict maps, with
        probabilities calculated according to prob3
        '''

        neutrinos = ['nue','numu','nutau']
        anti_neutrinos = ['nue_bar','numu_bar','nutau_bar']
        mID = ['','_bar']

        nu_barger = {'nue':1,'numu':2,'nutau':3,
                     'nue_bar':1,'numu_bar':2,'nutau_bar':3}

        logging.info("Defining osc_prob_dict from BargerPropagator...")

        params = MixingParameters(delta_M21_sq = deltam21,    
                                  delta_M31_sq = deltam31,
                                  theta_12 = theta12,
                                  theta_23 = theta23,
                                  theta_13 = theta13,
                                  delta_cp = deltacp)
        mixing = NeutrinoMixing(params)
        if self.atmos_model == 'simple':
            atmosphere = SimpleAtmosphere()
        elif self.atmos_model == 'normal':
            atmosphere = NormalAtmosphere()
        else:
            raise ValueError( "model allowed: ['simple', 'normal']")
        if self.prob_model == 'Probabilities':
            oscillation_calculator = Probabilities(mixing, self.earth_model, atmosphere)
        if self.prob_model == 'BargerProbabilities':
            oscillation_calculator = BargerProbabilities(mixing, self.earth_model)

        tprofile.info("start ocelot oscillation calculation")
        evals = []
        czvals = []
        print "event_by_event = ", event_by_event
        if event_by_event:
            assert(len(ecen) == len(czcen))
            total_bins = len(czcen)
        else:
            total_bins = int(len(ecen)*len(czcen))
        mod = total_bins/20
        loglevel = logging.root.getEffectiveLevel()

        flav_num = {'nue': 0, "numu" : 1, "nutau" : 2, "nue_bar" : 3, "numu_bar" : 4, "nutau_bar" : 5}
        #raw_probs = np.zeros((len(czcen), len(ecen), 6, 6))
        for icz, coszen in enumerate(czcen):
            if event_by_event:
                ecen_for_loop = np.array([ecen[icz]])   # because (e , cz),  ie = icz
            else:
                ecen_for_loop = ecen
            for ie,energy in enumerate(ecen_for_loop):
                if event_by_event:
                    ie = icz
                evals.append(energy)
                czvals.append(coszen)
                scaled_energy = energy*energy_scale
                zenith_in_rad = np.arccos(coszen)
                oscillation_prob = oscillation_calculator.matter_probabilities(zenith= zenith_in_rad, energy=scaled_energy)
                #raw_probs[icz, ie] = oscillation_prob

                for nu in ['nue','numu']:
                    for to_nu in neutrinos:
                        osc_prob_dict[nu+'_maps'][to_nu].append(oscillation_prob[flav_num[to_nu]][flav_num[nu]])
                for nu in ['nue_bar','numu_bar']:
                    for to_nu in anti_neutrinos:
                        osc_prob_dict[nu+'_maps'][to_nu].append(oscillation_prob[flav_num[to_nu]][flav_num[nu]])

                if loglevel <= logging.INFO:
                    if( (ie+1)*(icz+1) % mod == 0):
                        sys.stdout.write(".")
                        sys.stdout.flush()

        ### allows faster dict creation
        #probs = np.transpose(raw_probs)
        ## List the array indicies
        #nue = 0; numu = 1; nutau = 2; nuebar = 3; numubar = 4; nutaubar = 5
        #osc_prob_dict['nue_maps']     = {'nue':probs[nue, nue], 'numu':probs[nue, numu], 'nutau':probs[nue, nutau]}
        #osc_prob_dict['numu_maps']    = {'nue':probs[numu, nue], 'numu':probs[numu, numu], 'nutau':probs[numu, nutau]}
        #osc_prob_dict['nue_bar_maps'] = {'nue_bar':probs[nuebar, nuebar], 'numu_bar':probs[nuebar, numubar], 'nutau_bar':probs[nuebar,nutaubar]}
        #osc_prob_dict['numu_bar_maps']= {'nue_bar':probs[numubar, nuebar], 'numu_bar':probs[numubar, numubar], 'nutau_bar':probs[numubar,nutaubar]}
        #if loglevel <= logging.INFO: sys.stdout.write("\n")

        tprofile.info("stop oscillation calculation")
        return evals, czvals

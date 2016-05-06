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
from pisa.oscillations.OscillationServiceBase import OscillationServiceBase
from pisa.oscillations.prob3.BargerPropagator import BargerPropagator
from pisa.resources.resources import find_resource
from pisa.utils.proc import get_params, report_params


class Prob3OscillationServiceMC(OscillationServiceBase):
    """
    This class handles all tasks related to the oscillation
    probability calculations using the prob3 oscillation code
    """
    def __init__(self, ebins, czbins, detector_depth=None, earth_model=None,
                 prop_height=None, **kwargs):
        """
        Parameters needed to instantiate a Prob3OscillationService:
        * ebins: Energy bin edges
        * czbins: cos(zenith) bin edges
        * earth_model: Earth density model used for matter oscillations.
        * detector_depth: Detector depth in km.
        * prop_height: Height in the atmosphere to begin in km.
        """
        OscillationServiceBase.__init__(self, ebins, czbins)
        logging.info('Initializing %s...'%self.__class__.__name__)

        report_params(get_params(),['km','','km'])

        self.prop_height = prop_height
        earth_model = find_resource(earth_model)
        self.barger_prop = BargerPropagator(earth_model, detector_depth)
        self.barger_prop.UseMassEigenstates(False)

    def fill_osc_prob(self, ecen, czcen, prim, event_by_event=True,
                      theta12=None, theta13=None, theta23=None,
                      deltam21=None, deltam31=None, deltacp=None,
                      energy_scale=None, YeI = None, YeO = None,
                      YeM = None,**kwargs):
        #NOTE: fill_osc_prob is not a good name, 'get_osc_prob' would be more appropriate
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

        osc_prob_dict = {}
        for nu in ['nue','numu']:
            # NOTE: Here only has 'nue_maps' and 'numu_maps', but each one can refer to both nu and nubar,
            # e.g. 'nue_maps' can refer either to nue or nuebar, depending on the input prim, if prim is 
            # anti-neutrino, then 'nue_maps' and 'numu_maps' actually mean 'nue_bar' and 'numu_bar', should
            # do better than this, only use this temporarily.
            # (reason to do this: for full Monte Carlo analysis, each input (e,cz) event has its flavor known, so 
            # we're only interested in two oscillation probabilities, so osc_prob_dict is a matrix of 2 by 1 
            # rather than the previous 4 by 6; saves time)
            osc_prob_dict[nu+'_maps'] = []

        if type(prim)==list:
            assert(len(prim)==len(ebins))
        else:
            assert(type(prim)==str)

        tprofile.info("start prob3 oscillation calculation (event by event)")
        # Set to true, since we are using sin^2(theta) variables
        kSquared = True
        sin2th12Sq = np.sin(theta12)**2
        sin2th13Sq = np.sin(theta13)**2
        sin2th23Sq = np.sin(theta23)**2
        if event_by_event:
            assert(len(ecen) == len(czcen))
        total_bins = len(ecen)
        mod = total_bins/20
        loglevel = logging.root.getEffectiveLevel()
        for ie,energy in enumerate(ecen):
            if type(prim)==list:
                flav = prim[idx]
            else:
                flav = prim
            if event_by_event:
                icz = ie
                czcen_for_loop = czcen[icz]
                czcen_for_loop = np.array([czcen_for_loop])
            else:
                czcen_for_loop = czcen
            for icz, coszen in enumerate(czcen_for_loop):
                if event_by_event:
                    icz = ie
                scaled_energy = energy*energy_scale
                if loglevel <= logging.INFO:
                    if( (ie+1)*(icz+1) % mod == 0):
                        sys.stdout.write(".")
                        sys.stdout.flush()

                # In BargerPropagator code, it takes the "atmospheric
                # mass difference"-the nearest two mass differences, so
                # that it takes as input deltam31 for IMH and deltam32
                # for NMH
                mAtm = deltam31 if deltam31 < 0.0 else (deltam31 - deltam21)

                ########### FIRST FOR NEUTRINOS ##########
                if 'bar' not in flav:
                    kNuBar = 1 # +1 for nu -1 for nubar
                    self.barger_prop.SetMNS(sin2th12Sq,sin2th13Sq,sin2th23Sq,deltam21,mAtm,
                                            deltacp,scaled_energy,kSquared,kNuBar)

                    self.barger_prop.DefinePath(coszen, self.prop_height, YeI, YeO, YeM)
                    self.barger_prop.propagate(kNuBar)
 
                    for nu in ['nue','numu']:
                        nu_i = nu_barger[nu]
                        to_nu = flav 
                        nu_f = nu_barger[to_nu]
                        osc_prob_dict[nu+'_maps'].append(self.barger_prop.GetProb(nu_i,nu_f))

                ########### SECOND FOR ANTINEUTRINOS ##########
                else:
                    kNuBar = -1
                    self.barger_prop.SetMNS(sin2th12Sq,sin2th13Sq,sin2th23Sq,deltam21,
                                            mAtm,deltacp,scaled_energy,kSquared,kNuBar)
                    self.barger_prop.DefinePath(coszen, self.prop_height, YeI, YeO, YeM)
                    self.barger_prop.propagate(kNuBar)

                    for nu in ['nue','numu']:
                        nu_i = nu_barger[nu+'_bar']
                        to_nu = flav 
                        nu_f = nu_barger[to_nu]
                        osc_prob_dict[nu+'_maps'].append(self.barger_prop.GetProb(nu_i,nu_f))

        if loglevel <= logging.INFO: sys.stdout.write("\n")

        tprofile.info("stop oscillation calculation")
        return osc_prob_dict

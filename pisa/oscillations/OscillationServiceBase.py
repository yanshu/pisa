#
# This is the base class all other oscillation services should be derived from
#
# author: Lukas Schulte <lschulte@physik.uni-bonn.de>
#         Timothy C. Arlen tca3@psu.edu
#
# date:   July 31, 2014
#

import sys
import numpy as np
from pisa.utils.log import logging, tprofile
from pisa.utils.utils import get_smoothed_map, get_bin_centers, is_coarser_binning, is_linear, is_logarithmic, check_fine_binning, oversample_binning, Timer


class OscillationServiceBase:
    """
    Base class for all oscillation services.
    """

    def __init__(self, ebins, czbins):
        """
        Parameters needed to instantiate any oscillation service:
        * ebins: Energy bin edges
        * czbins: cos(zenith) bin edges
        If further member variables are needed, extend this method.
        """
        logging.trace('Instantiating %s'%self.__class__.__name__)
        self.ebins = np.array(ebins)
        self.czbins = np.array(czbins)
        for ax in [self.ebins, self.czbins]:
            if (len(np.shape(ax)) != 1):
                raise IndexError('Axes must be 1d! '+str(np.shape(ax)))


    def get_osc_prob_maps(self, **kwargs):
        """
        Returns an oscillation probability map dictionary calculated
        at the values of the input parameters:
          deltam21,deltam31,theta12,theta13,theta23,deltacp
        for flavor_from to flavor_to, with the binning of ebins,czbins.
        The dictionary is formatted as:
          'nue_maps': {'nue':map,'numu':map,'nutau':map},
          'numu_maps': {...}
          'nue_bar_maps': {...}
          'numu_bar_maps': {...}
        NOTES:
          * expects all angles in [rad]
          * this method doesn't calculate the oscillation probabilities
            itself, but calls get_osc_probLT_dict internally, to get a
            high resolution map of the oscillation probs,
        """

        #Get the finely binned maps as implemented in the derived class
        logging.info('Retrieving finely binned maps')
        with Timer(verbose=False) as t:
            fine_maps = self.get_osc_probLT_dict(**kwargs)
        print "       ==> elapsed time to get all fine maps: %s sec"%t.secs

        logging.info("Smoothing fine maps...")
        smoothed_maps = {}
        smoothed_maps['ebins'] = self.ebins
        smoothed_maps['czbins'] = self.czbins

        with Timer(verbose=False) as t:
            for from_nu, tomap_dict in fine_maps.items():
                if 'vals' in from_nu: continue
                new_tomaps = {}
                for to_nu, pvals in tomap_dict.items():
                    logging.debug("Getting smoothed map %s/%s"%(from_nu,to_nu))

                    new_tomaps[to_nu] = get_smoothed_map(
                        pvals,fine_maps['evals'],fine_maps['czvals'],
                        self.ebins, self.czbins)

                smoothed_maps[from_nu] = new_tomaps

        tprofile.debug("       ==> elapsed time to smooth maps: %s sec"%t.secs)

        return smoothed_maps


    def get_osc_probLT_dict(self, ebins=None, czbins=None,
                            oversample_e=None,oversample_cz=None, **kwargs):
        """
        This will create the oscillation probability map lookup tables
        (LT) corresponding to atmospheric neutrinos oscillation
        through the earth, and will return a dictionary of maps:
        {'nue_maps':[to_nue_map, to_numu_map, to_nutau_map],
         'numu_maps: [...],
         'nue_bar_maps': [...],
         'numu_bar_maps': [...],
         'czbins':czbins,
         'ebins': ebins}
        Will call fill_osc_prob to calculate the individual
        probabilities on the fly.
        By default, the standard binning is oversampled by a factor 10.
        Alternatively, the oversampling factor can be changed or a fine
        binning specified explicitly. In the latter case, the oversampling
        factor is ignored.
        """
        #First initialize the fine binning if not explicitly given
        if not check_fine_binning(ebins, self.ebins):
            ebins = oversample_binning(self.ebins, oversample_e)
        if not check_fine_binning(czbins, self.czbins):
            czbins = oversample_binning(self.czbins, oversample_cz)
        ecen = get_bin_centers(ebins)
        czcen = get_bin_centers(czbins)

        osc_prob_dict = {}
        for nu in ['nue_maps','numu_maps','nue_bar_maps','numu_bar_maps']:
            isbar = '_bar' if 'bar' in nu else ''
            osc_prob_dict[nu] = {'nue'+isbar: [],
                                 'numu'+isbar: [],
                                 'nutau'+isbar: [],}

        evals,czvals = self.fill_osc_prob(osc_prob_dict, ecen, czcen, **kwargs)
        osc_prob_dict['evals'] = evals
        osc_prob_dict['czvals'] = czvals

        return osc_prob_dict


    def fill_osc_prob(self, osc_prob_dict, ecen, czcen,
                  theta12=None, theta13=None, theta23=None,
                  deltam21=None, deltam31=None, deltacp=None, **kwargs):
        """
        This method is called by get_osc_probLT_dict and should be
        implemented in any derived class individually as here the actual
        oscillation code should be run.
        NOTE: Expects all angles to be in [rad], and all deltam to be in [eV^2]
        """
        raise NotImplementedError('Method not implemented for %s'
                                    %self.__class__.__name__)

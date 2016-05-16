#
# Creates the pdfs of the reconstructed energy and coszen from the
# true parameters. Provides reco event rate maps using these pdfs.
# It has four reco precision parameters. Replace RecoServiceMC.py
# with this script ONLY when we need to produce different templates
# at different reco precision values ( and in order to get the cubic
# fit coefficients), otherwise RecoServiceMC.py should not have 
# reco precision parameters.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   April 9, 2014
#

import sys
import pisa.utils.events as events
import numpy as np
from itertools import product
from pisa.reco.RecoServiceBase import RecoServiceBase
from pisa.utils.log import logging
from pisa.resources.resources import find_resource
from pisa.utils.utils import Timer

class RecoServiceMC(RecoServiceBase):
    """
    From the simulation file, creates 4D histograms of
    [true_energy][true_coszen][reco_energy][reco_coszen] which act as
    2D pdfs for the probability that an event with (true_energy,
    true_coszen) will be reconstructed as (reco_energy,reco_coszen).

    From these histograms, and the true event rate maps, calculates
    the reconstructed even rate templates.
    """
    def __init__(self, ebins, czbins, reco_mc_wt_file=None, **kwargs):
        """
        Parameters needed to instantiate a MC-based reconstruction service:
        * ebins: Energy bin edges
        * czbins: cos(zenith) bin edges
        * reco_weight_file: HDF5 containing the MC events to construct the kernels
        """
        self.simfile = reco_mc_wt_file
        self.ebins = ebins
        self.czbins = czbins
        self.nominal_kernels = self.kernel_from_simfile(simfile=self.simfile, e_reco_precision_up = 1, cz_reco_precision_up= 1, e_reco_precision_down=1,cz_reco_precision_down=1)


    def kernel_from_simfile(self, simfile=None,e_reco_precision_up=None, cz_reco_precision_up= None,
                            e_reco_precision_down= None, cz_reco_precision_down=None, **kwargs):
        logging.info('Opening file: %s'%(simfile))
        try:
            evts = events.Events(find_resource(simfile))
        except IOError,e:
            logging.error("Unable to open event data file %s"%simfile)
            logging.error(e)
            sys.exit(1)

        # Create the 4D distribution kernels...
        kernels = {}
        logging.info("Creating kernel dict...")
        kernels['ebins'] = self.ebins
        kernels['czbins'] = self.czbins
        for flavor in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
            flavor_dict = {}
            logging.debug("Working on %s kernels"%flavor)
            for int_type in ['cc','nc']:
                true_energy = evts[flavor][int_type]['true_energy']
                true_coszen = evts[flavor][int_type]['true_coszen']
                reco_energy = evts[flavor][int_type]['reco_energy']
                reco_coszen = evts[flavor][int_type]['reco_coszen']
                #weight = evts[flavor][int_type]['weighted_aeff']
                #weight = evts[flavor][int_type]['mc_weight']

                if e_reco_precision_up != 1:
                    delta = reco_energy[true_coszen<=0] - true_energy[true_coszen<=0]
                    change = delta/true_energy[true_coszen<=0]
                    print 'more than 100 %% delta for %s %% of the events '%(np.count_nonzero(change[change>1.])/float(len(change))*100)
                    delta *= e_reco_precision_up
                    reco_energy[true_coszen<=0] = true_energy[true_coszen<=0] + delta

                if e_reco_precision_down != 1:
                    reco_energy[true_coszen>0] *= e_reco_precision_down
                    reco_energy[true_coszen>0] -= (e_reco_precision_down - 1) * true_energy[true_coszen>0]

                if cz_reco_precision_up != 1:
                    reco_coszen[true_coszen<=0] *= cz_reco_precision_up
                    reco_coszen[true_coszen<=0] -= (cz_reco_precision_up - 1) * true_coszen[true_coszen<=0]

                if cz_reco_precision_down != 1:
                    reco_coszen[true_coszen>0] *= cz_reco_precision_down
                    reco_coszen[true_coszen>0] -= (cz_reco_precision_down - 1) * true_coszen[true_coszen>0]

                while np.any(reco_coszen<-1) or np.any(reco_coszen>1):
                    reco_coszen[reco_coszen>1] = 2-reco_coszen[reco_coszen>1]
                    reco_coszen[reco_coszen<-1] = -2-reco_coszen[reco_coszen<-1]

                # True binning, reco binning...
                bins = (self.ebins,self.czbins,self.ebins,self.czbins)
                data = (true_energy,true_coszen,reco_energy,reco_coszen)
                #kernel,_ = np.histogramdd(data,bins=bins,weights=weight)
                kernel,_ = np.histogramdd(data,bins=bins)

                # this histo to count all true events for normalization
                count_bins = (self.ebins,self.czbins)
                count_data = (true_energy,true_coszen)
                #count_hist,_ = np.histogramdd(count_data,bins=count_bins, weights=weight)
                count_hist,_ = np.histogramdd(count_data,bins=count_bins)

                # This takes into account the correct kernel normalization:
                k_shape = np.shape(kernel)
                for i,j in product(range(k_shape[0]),range(k_shape[1])):
                    if count_hist[i,j] > 0.0:
                        kernel[i,j] /= float(count_hist[i,j])

                flavor_dict[int_type] = kernel
            kernels[flavor] = flavor_dict

        return kernels


    def _get_reco_kernels(self, apply_reco_prcs, simfile=None, **kwargs):

        for reco_scale in ['e_reco_scale', 'cz_reco_scale']:
            if reco_scale in kwargs and kwargs[reco_scale] != 1:
                raise ValueError('%s = %.2f, must be 1.0 for RecoServiceMC!'
                    %(reco_scale, kwargs[reco_scale]))

        if not simfile in [self.simfile, None]:
            logging.info('Reconstruction from non-default MC file %s!'%simfile)
            return kernel_from_simfile(simfile=simfile, e_reco_precision_up = kwargs['e_reco_precision_up'], cz_reco_precision_up= kwargs['cz_reco_precision_up'], e_reco_precision_down = kwargs['e_reco_precision_down'],cz_reco_precision_down=kwargs['cz_reco_precision_down'])

        if not hasattr(self, 'nominal_kernels'):
            logging.info('Using file %s for default reconstruction'%(simfile))
            self.nominal_kernels = self.kernel_from_simfile(simfile=self.simfile, e_reco_precision_up = 1, cz_reco_precision_up= 1, e_reco_precision_down = 1, cz_reco_precision_down= 1)

        if apply_reco_prcs == False or kwargs['e_reco_precision_up'] == 1 and kwargs['cz_reco_precision_up'] == 1 and kwargs['e_reco_precision_down'] == 1 and kwargs['cz_reco_precision_down'] == 1:
            return self.nominal_kernels
        else:
            return self.kernel_from_simfile(simfile=self.simfile, e_reco_precision_up = kwargs['e_reco_precision_up'], cz_reco_precision_up= kwargs['cz_reco_precision_up'], e_reco_precision_down = kwargs['e_reco_precision_down'],cz_reco_precision_down=kwargs['cz_reco_precision_down'])

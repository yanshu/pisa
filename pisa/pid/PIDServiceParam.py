#
# PID service for parametrized PID functions.
#
# author: Lukas Schulte
#         schulte@physik.uni-bonn.de
#
#         Timothy C. Arlen
#         tca3@psu.edu
#
# date:   Oct 21, 2014
#

import logging

import numpy as np
import scipy.stats

from pisa.pid.PIDServiceBase import PIDServiceBase
from pisa.resources.resources import find_resource
from pisa.utils.jsons import from_json
from pisa.utils.utils import get_bin_centers


class PIDServiceParam(PIDServiceBase):
    """
    Creates PID kernels from parametrization functions that are stored
    in a JSON dict. numpy is accessible as np, and scipy.stats.
    Systematic parameters 'PID_offset' and 'PID_scale' are supported.
    """

    def __init__(self, ebins, czbins, PID_offset=0., PID_scale=1.,
                 pid_paramfile=None, **kwargs):
        """
        Parameters needed to initialize a PID service with parametrizations:
        * ebins: Energy bin edges
        * czbins: cos(zenith) bin edges
        * PID_offset: energy offset of PID curves
        * PID_scale: factor to scale to-track probabilities with
                     (to-cscd prob. scaled such that sum stays the same)
        * pid_paramfile: JSON containing the parametrizations
        """
        self.offset = PID_offset
        self.scale = PID_scale
        self.paramfile = pid_paramfile
        PIDServiceBase.__init__(self, ebins, czbins, PID_offset=self.offset,
                                PID_scale=self.scale,
                                pid_paramfile=self.paramfile, **kwargs)

    def kernel_from_paramfile(self, PID_offset=None, PID_scale=None,
                              pid_paramfile=None, **kwargs):
        logging.info('Opening PID parametrization file %s'%pid_paramfile)
        try:
            param_str = from_json(find_resource(pid_paramfile))
        except IOError, e:
            logging.error("Unable to open PID parametrization file %s"
                          %pid_paramfile)
            logging.error(e)
            raise
        ecen = get_bin_centers(self.ebins)
        czcen = get_bin_centers(self.czbins)
        pid_kernels = {'binning': {'ebins': self.ebins,
                                   'czbins': self.czbins}}
        for signature in param_str.keys():
            #Generate the functions
            to_trck_func = eval(param_str[signature]['trck'])
            to_cscd_func = eval(param_str[signature]['cscd'])

            # Make maps from the functions evaluated at the bin centers
            #
            # NOTE: np.where() is to catch the low energy nutau events
            # that are undefined. Often what happens is that the nutau
            # parameterization for trck events will drop below 0.0 at
            # low energies, but there are no nutau events at these
            # energies anyway, so we just set them to zero (and cscd =
            # 1.0) if the condition arises.
            to_trck = to_trck_func(ecen-PID_offset)
            to_cscd = to_cscd_func(ecen-PID_offset)
            to_trck = np.where(to_trck < 0.0,0.0,
                               np.where(to_trck > 1.0,1.0,to_trck))
            to_cscd = np.where(to_cscd < 0.0,0.0,
                               np.where(to_cscd > 1.0,1.0,to_cscd))
            sum = to_trck + to_cscd
            to_trck_scaled = PID_scale*to_trck
            to_cscd_scaled = sum - to_trck_scaled
            _,to_trck_map = np.meshgrid(czcen, to_trck_scaled)
            _,to_cscd_map = np.meshgrid(czcen, to_cscd_scaled)

            pid_kernels[signature] = {'trck':to_trck_map,
                                      'cscd':to_cscd_map}
        # now update
        self.offset = PID_offset
        self.scale = PID_scale
        self.paramfile = pid_paramfile
        return pid_kernels

    def get_pid_kernels(self, PID_offset=None, PID_scale=None,
                        pid_paramfile=None, **kwargs):
        if all([hasattr(self, 'pid_kernels'), PID_offset==self.offset,
                PID_scale==self.scale, pid_paramfile==self.paramfile]):
            logging.info('Using existing kernels for PID')

        elif not pid_paramfile in [self.paramfile, None]:
            logging.info('PID from non-default paramfile %s!'%pid_paramfile)
            return kernel_from_paramfile(PID_offset, PID_scale, pid_paramfile,
                                         **kwargs)

        else:
            logging.info('Using paramfile %s for default PID'%pid_paramfile)
            logging.info('Scaling PID with %.3f, with offset %.3f'%(PID_scale,
                                                                    PID_offset))
            self.pid_kernels = self.kernel_from_paramfile(PID_offset, PID_scale,
                                                          pid_paramfile,
                                                          **kwargs)
        return self.pid_kernels

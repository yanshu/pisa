import numpy as np
from pisa.utils.log import logging
from pisa.utils.jsons import from_json
from pisa.resources.resources import find_resource
from pisa.sys.SysBase import SysBase

class Resolution(SysBase):

    def __init__(self, reco_prcs_coeff_file, var, direction):
        # read in cubic coeffs from file
        cubic_coeffs = from_json(find_resource(reco_prcs_coeff_file))
        self.cubic_coeffs = {} 
        for channel in ['cscd', 'trck']:
            # add up and down together
            self.cubic_coeffs[channel] = cubic_coeffs['coeffs']['data_tau']['%s_reco_precision_%s'%(var,direction)][channel]

    def get_scales(self, channel, sys_val):
        # get the sacles to be applied to a map
        scale = (self.cubic_coeffs[channel][:,:,0] * (sys_val**3 - 1.0) + self.cubic_coeffs[channel][:,:,1] * (sys_val**2 - 1.0) + self.cubic_coeffs[channel][:,:,2] * (sys_val - 1.0) +1.)
        return scale

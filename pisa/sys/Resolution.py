import numpy as np
from pisa.utils.log import logging
from pisa.utils.jsons import from_json
from pisa.resources.resources import find_resource

class Resolution():

    def __init__(self, reco_prcs_coeff_file):
        # read in cubic coeffs from file
        cubic_coeffs = from_json(find_resource(reco_prcs_coeff_file))
        self.cubic_coeffs = {} 
        for var in ['e','cz']:
            self.cubic_coeffs[var] = {}
            for channel in ['cscd', 'trck']:
                self.cubic_coeffs[var][channel] = {}
                for direction in ['up','down']:
                    self.cubic_coeffs[var][channel][direction] = cubic_coeffs['%s_reco_precision_%s'%(var,direction)][channel])

    def get_scales(self, channel, e_precision_up_val, e_precision_down_val, cz_precision_up_val, cz_precision_down_val):
        # get the sacles to be applied to a map
        scale = 1.0
        for var in ['e','cz']:
            for direction in ['up','down']:
                val = eval('%s_precision_%s_val'%(var,direction))
                scale *= (self.cubic_coeffs[var][channel][direction][:,:,0] * (val**3 - 1.0) + self.cubic_coeffs[var][channel][direction][:,:,1] * (val**2 - 1.0) + self.cubic_coeffs[var][channel][direction][:,:,2] * (val - 1.0) + 1.0)
        return scale

    def apply_sys(self, maps, e_precision_up_val, e_precision_down_val, cz_precision_up_val, cz_precision_down_val):
        for channel in ['trck', 'cscd']:
            # apply scales
            maps[channel]['map'] *= self.get_scales(channel, e_precision_up_val, e_precision_down_val, cz_precision_up_val, cz_precision_down_val)
        return maps

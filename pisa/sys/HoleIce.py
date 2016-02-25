import numpy as np
from pisa.utils.log import logging
from pisa.utils.jsons import from_json
from pisa.resources.resources import find_resource
from pisa.sys.SysBase import SysBase

class HoleIce(SysBase):

    def __init__(self, slopes_file):
        # nominal hol_ice value
        self.hole_ice_val_nominal = 0.2
        # read in slopes from file
        slopes = from_json(find_resource(slopes_file))
        self.slopes = {} 
        for channel in ['cscd', 'trck']:
            # add up and down together
            self.slopes[channel] = np.append(slopes['slopes']['data_tau']['k_HoleIce'][channel]['up'],np.fliplr(slopes['slopes']['data_tau']['k_HoleIce'][channel]['down']), axis=1)

    def get_scales(self, channel, hole_ice_val):
        # get the sacles to be applied to a map
        return self.slopes[channel]*(hole_ice_val - self.hole_ice_val_nominal) + 1.

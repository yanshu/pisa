import numpy as np
from pisa.utils.log import logging
from pisa.utils.jsons import from_json
from pisa.resources.resources import find_resource
from pisa.sys.SysBase import SysBase

class HoleIce(SysBase):

    def __init__(self, slopes_file, sim_ver):
        # read in slopes from file
        slopes = from_json(find_resource(slopes_file))
        # nominal hol_ice value
        self.hole_ice_val_nominal = slopes['nominal_value']
        self.linear = {} 
        self.quadratic = {} 
        self.fixed_ratios = {} 
        self.sim = sim_ver
        for channel in ['cscd', 'trck']:
            # add up and down together
            self.linear[channel] = slopes[channel]['linear']
            self.quadratic[channel] = slopes[channel]['quadratic']
            if self.sim == '4digit':
                self.fixed_ratios[channel] = slopes[channel]['fixed_ratios']

    def get_scales(self, channel, sys_val):
        # get the sacles to be applied to a map
        return self.linear[channel]*(sys_val - self.hole_ice_val_nominal) + self.quadratic[channel]*(sys_val - self.hole_ice_val_nominal)**2 + 1.

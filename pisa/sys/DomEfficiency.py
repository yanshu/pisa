import numpy as np
from pisa.utils.log import logging
from pisa.utils.jsons import from_json
from pisa.resources.resources import find_resource
from pisa.sys.SysBase import SysBase

class DomEfficiency(SysBase):

    def __init__(self, slopes_file, sim_ver):
        # read in slopes from file
        slopes = from_json(find_resource(slopes_file))
        # nominal hol_ice value
        self.dom_eff_val_nominal = slopes['nominal_value']
        self.slopes = {} 
        self.fixed_ratios = {}
        self.sim = sim_ver
        for channel in ['cscd', 'trck']:
            # add up and down together
            self.slopes[channel] = slopes[channel]['slopes']
            if self.sim == '4digit':
                self.fixed_ratios[channel] = slopes[channel]['fixed_ratios']

    def get_scales(self, channel, sys_val):
        # get the sacles to be applied to a map
        if self.sim == '4digit':
            return self.slopes[channel]*(sys_val - 0.91) + self.fixed_ratios[channel] 
        return self.slopes[channel]*(sys_val - 1.0) + 1.0 

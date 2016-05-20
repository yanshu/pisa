import numpy as np
from pisa.utils.log import logging
from pisa.utils.jsons import from_json
from pisa.resources.resources import find_resource
from pisa.sys.SysBase import SysBase

class HoleIce(SysBase):

    def __init__(self, hole_ice_file, hole_ice_fwd_file, sim_ver):
        # read in hi_file from file
        hi_file = from_json(find_resource(hole_ice_file))
        # nominal hol_ice value
        self.hole_ice_val_nominal = hi_file['nominal_value']
        print "in HoleIce.__init__, sim_ver = ", sim_ver
        if sim_ver=='dima':
            hifwd_file = from_json(find_resource(hole_ice_fwd_file))
            self.hole_ice_fwd_val_nominal = hifwd_file['nominal_value']
            print "self.hole_ice_fwd_val_nominal = ", self.hole_ice_fwd_val_nominal
        self.linear = {} 
        self.quadratic = {} 
        self.fixed_ratios = {} 
        self.slope = {}
        self.hifwd_slope = {}
        self.sim = sim_ver
        for channel in ['cscd', 'trck']:
            #self.linear[channel] = hi_file[channel]['linear']
            #self.quadratic[channel] = hi_file[channel]['quadratic']
            self.slope[channel] = hi_file[channel]['slopes']
            self.hifwd_slope[channel] = hi_file[channel]['slopes']
            if self.sim == '4digit':
                self.fixed_ratios[channel] = hi_file[channel]['fixed_ratios']

    def get_scales(self, channel, sys_val):
        # get the sacles to be applied to a map
        # return self.linear[channel]*(sys_val - self.hole_ice_val_nominal) + self.quadratic[channel]*(sys_val - self.hole_ice_val_nominal)**2 + 1.
        return self.slope[channel]*(sys_val - self.hole_ice_val_nominal) + 1.

    def get_hi_scales(self, channel, hole_ice_val, hole_ice_fwd_val):
        return (self.slope[channel]*(hole_ice_val - self.hole_ice_val_nominal) + 1.)*(self.hifwd_slope[channel]*(hole_ice_fwd_val - self.hole_ice_fwd_val_nominal) + 1.)

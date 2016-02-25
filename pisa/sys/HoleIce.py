import numpy as np
from pisa.utils.log import logging
from pisa.utils.jsons import from_json
from pisa.resources.resources import find_resource

class HoleIce():

    def __init__(self, slopes_file):
        # read in slopes from file
        slopes = from_json(find_resource(slopes_file))

        # nominal hol_ice value
        self.hole_ice_val_nominal = slopes['nominal_value']

        self.slopes = {} 
        self.fixed_ratios = {} 
        for channel in ['cscd', 'trck']:
            # add up and down together
            self.slopes[channel] = slopes[channel]['slopes']
            self.fixed_ratios[channel] = slopes[channel]['fixed_ratios']

    def get_scales(self, channel, hole_ice_val):
        # get the sacles to be applied to a map
        return self.slopes[channel]*(hole_ice_val - self.hole_ice_val_nominal) + 1.0

    def apply_sys(self, maps, hole_ice_val):
        for channel in ['trck', 'cscd']:
            # apply scales
            maps[channel]['map'] *= self.get_scales(channel, hole_ice_val)
        return maps

import numpy as np
from pisa.utils.log import logging
from pisa.utils.jsons import from_json
from pisa.resources.resources import find_resource

class DomEfficiency():

    def __init__(self, slopes_file):
        # nominal hol_ice value
        self.dom_eff_val_nominal = 1.0
        self.dom_eff_val_nominal = 0.91
        # read in slopes from file
        slopes = from_json(find_resource(slopes_file))
        self.slopes = {} 
        for channel in ['cscd', 'trck']:
            # add up and down together
            self.slopes[channel] = np.append(slopes['slopes']['data_tau']['k_DomEff'][channel]['up'],np.fliplr(slopes['slopes']['data_tau']['k_DomEff'][channel]['down']), axis=1)

    def get_scales(self, channel, dom_eff_val):
        # get the sacles to be applied to a map
        return self.slopes[channel]*(dom_eff_val - self.dom_eff_val_nominal) + 1.

    def apply_sys(self, maps, dom_eff_val):
        for channel in ['trck', 'cscd']:
            # apply scales
            maps[channel]['map'] *= self.get_scales(channel, dom_eff_val)
        return maps

# authors: P.Eller (pde3@psu.edu)
# date:   September 2016

import sys, os
import numpy as np
import time

from pisa import ureg, Q_
from pisa.core.stage import Stage
from pisa.core.param import ParamSet
from pisa.core.events import Events
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.core.map import Map, MapSet
from pisa.utils.resources import find_resource
from pisa.utils.log import logging
from pisa.utils.comparisons import normQuant
from pisa.utils.hash import hash_obj
from pisa.utils.config_parser import split

class nutau(Stage):
    """
    Stage combining the different maps (flav int) into right now a single map
    and apply a scale factor for nutau events

    combine_groups: dict with output map names and what maps should be contained
    for example
      {
        'evts':
          'nue_cc, nuebar_cc, numu_cc, numubar_cc, nutau_cc, nutaubar_cc, nue_nc, nuebar_nc, numu_nc, numubar_nc, nutau_nc, nutaubar_nc'
      }

    Parameters
    ----------
    params : ParamSet or sequence with which to instantiate a ParamSet.
        Expected params are:

            nu_nc_norm : quantity (dimensionless)
                global scaling factor that is applied to all *_nc maps
            nutau_norm : quantity (dimensionless)
            nutau_cc_norm : quantity (dimensionless)
    """

    def __init__(self, params, input_binning, input_names, combine_groups,
                 disk_cache=None, memcache_deepcopy=True, error_method=None,
                 outputs_cache_depth=20, debug_mode=None):

        expected_params = (
            'nu_nc_norm',
            'nutau_norm',
            'nutau_cc_norm'
        )

        input_names =  split(input_names)
        self.combine_groups = eval(combine_groups)
        for key, val in self.combine_groups.items():
            self.combine_groups[key] = split(val)
        output_names = self.combine_groups.keys()

        super(self.__class__, self).__init__(
            use_transforms=True,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            error_method=error_method,
            disk_cache=disk_cache,
            memcache_deepcopy=memcache_deepcopy,
            outputs_cache_depth=outputs_cache_depth,
            output_binning=input_binning,
            input_binning=input_binning,
            debug_mode=debug_mode
        )

    def _compute_transforms(self):
    
        dims = self.input_binning.names

        transforms = []
        for group, in_names in self.combine_groups.items():
            xform_shape = [len(in_names)] + [self.input_binning[d].num_bins for d in dims]

            xform = np.ones(xform_shape)
            input_names = self.input_names
            for i,name in enumerate(in_names):
                scale = 1
                if '_nc' in name:
                    scale *= self.params.nu_nc_norm.value.m_as('dimensionless')
                if 'nutau' in name:
                    scale *= self.params.nutau_norm.value.m_as('dimensionless')
                if 'nutau_cc' in name:
                    scale *= self.params.nutau_cc_norm.value.m_as('dimensionless')
                if scale != 1:
                    xform[i] *= scale

            transforms.append(
                BinnedTensorTransform(
                    input_names=in_names,
                    output_name=group,
                    input_binning=self.input_binning,
                    output_binning=self.output_binning,
                    xform_array=xform
                )
            )

        return TransformSet(transforms=transforms)

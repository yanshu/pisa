
import numpy as np
import pint
ureg = pint.UnitRegistry()

from pisa.core.stage import Stage
from pisa.core.map import Map, MapSet
from pisa.utils.hash import hash_obj


class dummy(Stage):
    """
    This is a Flux Service just for testing purposes, generating a random map
    m1 and a map containing ones as m2; the parameter `test` is required.
    """
    def __init__(self, params, output_binning, disk_cache=None,
                 memcaching_enabled=True, propagate_errors=True,
                 outputs_cache_depth=20):
        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = (
            'atm_delta_index', 'energy_scale', 'nu_nubar_ratio',
            'nue_numu_ratio', 'test', 'example_file', 'oversample_e',
            'oversample_cz'
        )

        # Define the names of objects that get produced by this stage
        output_names = (
            'nue', 'numu', 'nuebar', 'numubar'
        )

        # Invoke the init method from the parent class, which does a lot of
        # work for you. Note that we do not specify `input_names` here, since
        # there are no "inputs" used by this stage. (Of course there are
        # parameters, and files with info, but no maps or MC events are used
        # and transformed directly by this stage to produce its output.)
        super(self.__class__, self).__init__(
            use_transforms=False,
            stage_name='flux',
            service_name='dummy',
            params=params,
            expected_params=expected_params,
            output_names=output_names,
            disk_cache=disk_cache,
            memcaching_enabled=memcaching_enabled,
            propagate_errors=propagate_errors,
            outputs_cache_depth=outputs_cache_depth,
            output_binning=output_binning
        )

        # There might be other things to do at init time than what Stage does,
        # but typically this is not much... and it's almost always a good idea
        # to have "real work" defined in another method besides init, which can
        # then get called from init (so that if anyone else wants to do the
        # same "real work" after object instantiation, (s)he can do so easily
        # by invoking that same method).

    def _compute_outputs(self, inputs=None):
        # Following is just so that we only produce new maps when params
        # change, but produce the same maps with the same param values
        # (for a more realistic test of caching).
        seed = hash_obj(self.params.values, hash_to='int') % (2**32-1)
        np.random.seed(seed)

        # Convert a parameter that the user can specify in any (compatible)
        # units to the units used for compuation
        height = self.params['test'].value.to('meter').magnitude

        output_maps = []
        for output_name in self.output_names:
            # Generate the fake per-bin "fluxes", modified by the parameter
            hist = np.ones(self.output_binning.shape) * height

            # Put the "fluxes" into a Map object, give it the output_name
            m = Map(name=output_name, hist=hist, binning=self.output_binning)

            # Optionally turn on errors here, that will be propagated through
            # rest of pipeline (slows things down, but essential in some cases)
            #m.set_poisson_errors()
            output_maps.append(m)

        # Combine the output maps into a single MapSet object to return.
        # The MapSet contains the varous things that are necessary to make
        # caching work and also provides a nice interface for the user to all
        # of the contained maps
        return MapSet(maps=output_maps, name='flux maps')

    def validate_params(self, params):
        # do some checks on the parameters
        assert (params['test'].value.dimensionality ==
                ureg.meter.dimensionality)
        assert params['test'].value.magnitude >= 0
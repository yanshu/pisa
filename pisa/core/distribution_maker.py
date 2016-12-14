#! /usr/bin/env python
# authors: J.Lanfranchi/P.Eller
# date:   March 20, 2016
"""
DistributionMaker class definition and a simple script to generate, save, and
plot a distribution from pipeline config file(s).

"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict
import inspect
from itertools import izip, product
import os

import numpy as np

from pisa import ureg
from pisa.core.map import MapSet
from pisa.core.pipeline import Pipeline
from pisa.core.param import ParamSet
from pisa.utils.betterConfigParser import BetterConfigParser
from pisa.utils.fileio import expandPath, to_file
from pisa.utils.hash import hash_obj
from pisa.utils.log import set_verbosity, logging
from pisa.utils.random_numbers import get_random_state


__all__ = ['DistributionMaker',
           'test_DistributionMaker',
           'parse_args', 'main']


class DistributionMaker(object):
    """Container for one or more pipelines; the outputs from all contained
    pipelines are added together to create the distribution.

    Parameters
    ----------
    pipelines : Pipeline or convertible thereto, or iterable thereof
        A new pipline is instantiated with each object passed. Legal objects
        are already-instantiated Pipelines and anything interpret-able by the
        Pipeline init method.

    Notes
    -----
    Free params with the same name in two pipelines are updated at the same
    time so long as you use the `update_params`, `set_free_params`, or
    `_set_rescaled_free_params` methods. Also use `select_params` to select
    params across all pipelines (if a pipeline does not have one or more of
    the param selectors specified, those param selectors have no effect in
    that pipeline).

    `_*_rescaled_*` properties and methods are for interfacing with a
    minimizer, where values are linearly mapped onto the interval [0, 1]
    according to the parameter's allowed range. Avoid interfacing with these
    except if using a minimizer, since, e.g., units are stripped and values and
    intervals are non-physical.

    """
    def __init__(self, pipelines, label=None):
        self.label = None
        self._source_code_hash = None

        self._pipelines = []
        if isinstance(pipelines, (basestring, BetterConfigParser, OrderedDict,
                                  Pipeline)):
            pipelines = [pipelines]

        for pipeline in pipelines:
            if not isinstance(pipeline, Pipeline):
                pipeline = Pipeline(pipeline)
            self._pipelines.append(pipeline)
        #for pipeline in self:
        #    pipeline.select_params(self.param_selections,
        #                           error_on_missing=False)

    def __iter__(self):
        return iter(self._pipelines)

    def get_outputs(self, return_sum=False, sum_map_name='total',
                    sum_map_tex_name='Total', **kwargs):
        """Compute and return the outputs.

        Parameters
        ----------
        return_sum : bool
            If True, add up all Maps in all MapSets returned by all pipelines.
            The result will be a single Map contained in a MapSet.
            If False, return a list where each element is the full MapSet
            returned by each pipeline in the DistributionMaker.

        **kwargs
            Passed on to each pipeline's `get_outputs1` method.

        Returns
        -------
        MapSet if `return_sum=True` or list of MapSets if `return_sum=False`

        """
        outputs = [pipeline.get_outputs(**kwargs) for pipeline in self]
        if return_sum:
            if len(outputs) > 1:
                outputs = reduce(lambda x, y: sum(x) + sum(y), outputs)
            else:
                outputs = sum(sum(outputs))
            outputs.name = sum_map_name
            outputs.tex = sum_map_tex_name
            outputs = MapSet(outputs)
        return outputs

    def update_params(self, params):
        [pipeline.update_params(params) for pipeline in self]

    def select_params(self, selections, error_on_missing=True):
        successes = 0
        if selections is not None:
            for pipeline in self:
                try:
                    pipeline.select_params(selections, error_on_missing=True)
                except KeyError:
                    pass
                else:
                    successes += 1

            if error_on_missing and successes == 0:
                raise KeyError(
                    'None of the selections %s found in any pipeline in this'
                    ' distribution maker' %(selections,)
                )
        else:
            for pipeline in self:
                possible_selections = pipeline.param_selections
                if not len(possible_selections) == 0:
                    logging.warn("Although you didn't make a parameter "
                                 "selection, the following were available: %s."
                                 " This may cause issues."
                                 %(possible_selections))

    @property
    def pipelines(self):
        return self._pipelines

    @property
    def params(self):
        params = ParamSet()
        [params.extend(pipeline.params) for pipeline in self]
        return params

    @property
    def param_selections(self):
        selections = set()
        [selections.update(pipeline.param_selections) for pipeline in self]
        return sorted(selections)

    @property
    def source_code_hash(self):
        """Hash for the source code of this object's class.

        Not meant to be perfect, but should suffice for tracking provenance of
        an object stored to disk that were produced by a Stage.
        """
        if self._source_code_hash is None:
            self._source_code_hash = hash_obj(inspect.getsource(self.__class__))
        return self._source_code_hash

    @property
    def state_hash(self):
        return hash_obj([self.source_code_hash] + [p.state_hash for p in self])

    def set_free_params(self, values):
        """Set free parameters' values.

        Parameters
        ----------
        values : list of quantities

        """
        for name, value in izip(self.params.free.names, values):
            for pipeline in self:
                if name in pipeline.params.free.names:
                    pipeline.params[name] = value
                elif name in pipeline.params.names:
                    raise AttributeError(
                        'Trying to set value for "%s", a parameter that is'
                        ' fixed in at least one pipeline' %name
                    )

    def randomize_free_params(self, random_state=None):
        if random_state is None:
            random = np.random
        else:
            random = get_random_state(random_state)
        n = len(self.params.free)
        rand = random.rand(n)
        self._set_rescaled_free_params(rand)

    def reset_all(self):
        """Reset both free and fixed parameters to their nominal values."""
        [p.params.reset_all() for p in self]

    def reset_free(self):
        """Reset only free parameters to their nominal values."""
        [p.params.reset_free() for p in self]

    def set_nominal_by_current_values(self):
        """Define the nominal values as the parameters' current values."""
        [p.params.set_nominal_by_current_values() for p in self]

    def _set_rescaled_free_params(self, rvalues):
        """Set free param values given a simple list of [0,1]-rescaled,
        dimensionless values

        """
        names = self.params.free.names
        for pipeline in self:
            for name, rvalue in izip(names, rvalues):
                if name in pipeline.params.free.names:
                    pipeline.params[name]._rescaled_value = rvalue
                elif name in pipeline.params.names:
                    raise AttributeError(
                        'Trying to set value for "%s", a parameter that is'
                        ' fixed in at least one pipeline' %name
                    )


def test_DistributionMaker():
    #
    # Test: select_params and param_selections
    #

    hierarchies = ['nh', 'ih']
    materials = ['iron', 'pyrolite']

    t23 = dict(
        ih=49.5 * ureg.deg,
        nh=42.3 * ureg.deg
    )
    YeO = dict(
        iron=0.4656,
        pyrolite=0.4957
    )

    # Instantiate with two pipelines: first has both nh/ih and iron/pyrolite
    # param selectors, while the second only has nh/ih param selectors.
    dm = DistributionMaker(['tests/settings/test_Pipeline.cfg',
                            'tests/settings/test_Pipeline2.cfg'])

    current_mat = 'iron'
    current_hier = 'nh'

    for new_hier, new_mat in product(hierarchies, materials):
        new_YeO = YeO[new_mat]

        assert dm.param_selections == sorted([current_hier, current_mat]), \
                str(dm.params.param_selections)
        assert dm.params.theta23.value == t23[current_hier], \
                str(dm.params.theta23)
        assert dm.params.YeO.value == YeO[current_mat], str(dm.params.YeO)

        # Select just the hierarchy
        dm.select_params(new_hier)
        assert dm.param_selections == sorted([new_hier, current_mat]), \
                str(dm.param_selections)
        assert dm.params.theta23.value == t23[new_hier], \
                str(dm.params.theta23)
        assert dm.params.YeO.value == YeO[current_mat], \
                str(dm.params.YeO)

        # Select just the material
        dm.select_params(new_mat)
        assert dm.param_selections == sorted([new_hier, new_mat]), \
                str(dm.param_selections)
        assert dm.params.theta23.value == t23[new_hier], \
                str(dm.params.theta23)
        assert dm.params.YeO.value == YeO[new_mat], \
                str(dm.params.YeO)

        # Reset both to "current"
        dm.select_params([current_mat, current_hier])
        assert dm.param_selections == sorted([current_hier, current_mat]), \
                str(dm.param_selections)
        assert dm.params.theta23.value == t23[current_hier], \
                str(dm.params.theta23)
        assert dm.params.YeO.value == YeO[current_mat], \
                str(dm.params.YeO)

        # Select both hierarchy and material
        dm.select_params([new_mat, new_hier])
        assert dm.param_selections == sorted([new_hier, new_mat]), \
                str(dm.param_selections)
        assert dm.params.theta23.value == t23[new_hier], \
                str(dm.params.theta23)
        assert dm.params.YeO.value == YeO[new_mat], \
                str(dm.params.YeO)

        current_hier = new_hier
        current_mat = new_mat


def parse_args():
    parser = ArgumentParser(
        description='''Generate, store, and plot a distribution from pipeline
        configuration file(s).''',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-p', '--pipeline', type=str, required=True,
        metavar='CONFIGFILE', action='append',
        help='''Settings file for each pipeline (repeat for multiple).'''
    )
    parser.add_argument(
        '--outdir', type=str, action='store',
        help='Directory into which to store the output'
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='Set verbosity level'
    )
    args = parser.parse_args()
    return args


def main():
    from pisa.utils.plotter import Plotter
    args = parse_args()
    set_verbosity(args.v)

    distribution_maker = DistributionMaker(pipelines=args.pipeline)
    outputs = distribution_maker.get_outputs(return_sum=True)
    if args.outdir:
        # TODO: unique filename: append hash (or hash per pipeline config)
        fname = 'distribution_maker_outputs.json.bz2'
        fpath = expandPath(os.path.join(args.outdir, fname))
        to_file(outputs, fpath)
        my_plotter = Plotter(
            stamp='PISA cake test',
            outdir=args.outdir,
            fmt='pdf', log=False,
            annotate=False
        )
        my_plotter.ratio = True
        my_plotter.plot_2d_array(outputs, fname='dist_output', cmap='OrRd')

    return distribution_maker, outputs


if __name__ == '__main__':
    distribution_maker, outputs = main()

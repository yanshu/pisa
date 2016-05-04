#! /usr/bin/env python
# authors: J.Lanfranchi/P.Eller
# date:   March 20, 2016

import importlib
from collections import OrderedDict

from pisa.core.stage import Stage
from pisa.core.param import ParamSet
from pisa.utils.parse_config import parse_config
from pisa.utils.log import logging, set_verbosity

"""
Implementation of the Pipeline object, and a __main__ script to instantiate and
run a pipeline.
"""

# TODO: should we check that the output binning of a previous stage produces
# the inputs required by the current stage, or that the aggregate outputs that
# got produced by previous stages (less those that got consumed in other
# previous stages) hold what the current stage requires for inputs... or
# should we not assume either will check out, since it's possible that the
# stage requires sideband objects that are to be introduced at the top of the
# pipeline by the user (and so there's no way to verify that all inputs are
# present until we see what the user hands the pipeline as its top-level
# input)? Alternatively, the lack of apparent inputs for a stage could show
# a warning message. Or we just wait to see if it fails when the user runs the
# code.

# TODO: return an OrderedDict instead of a list if the user requests
# intermediate results? Or simply use the `outputs` attribute of each stage to
# dynamically access this?

class Pipeline(object):
    """Instantiate stages according to a parsed config object; excecute
    stages.

    Parameters
    ----------
    config : string or OrderedDict
        If string, interpret as resource location; send to the
          parse_config.parse_config() function to get a config OrderedDict.
        If OrderedDict, use directly as pipeline configuration.

    Methods
    -------
    compute_outputs
        Returns output MapSet from the (final) pipeline, or all intermediate
        outputs if `return_intermediate` is specified as True.
    update_params
        Update params of all stages using values from a passed ParamSet

    Attributes
    ----------
    params : ParamSet
        All params from all stages in the pipeline
    stages : list
        All stages in the pipeline

    """
    def __init__(self, config):
        self._stages = []
        if isinstance(config, basestring):
            config = parse_config(config=config)
        assert isinstance(config, OrderedDict)
        self.config = config
        self._init_stages()

    def __iter__(self):
        return iter(self._stages)

    def _init_stages(self):
        """Stage factory: Instantiate stages specified by self.config."""

        self._stages = []
        for stage_num, stage_name in enumerate(self.config.keys()):
            service = self.config[stage_name.lower()].pop('service').lower()
            # Import stage service
            module = importlib.import_module('pisa.stages.%s.%s'
                                             %(stage_name.lower(), service))
            # Get class
            cls = getattr(module, service)

            # Instantiate object, do basic type check
            stage = cls(**self.config[stage_name.lower()])
            assert isinstance(stage, Stage)

            # Make sure the input binning of this stage is compatible with the
            # output binning of the previous stage ("compatible binning"
            # includes if both are specified to be None)
            if len(self._stages) > 0:
                assert stage.input_binning == self._stages[-1].output_binning

            # Append stage to pipeline
            self._stages.append(stage)

        logging.debug(str(self.params))

    def compute_outputs(self, inputs=None, idx=None,
                        return_intermediate=False):
        """Run the pipeline to compute its outputs.

        Parameters
        ----------
        inputs : None or MapSet # TODO: other container(s)
            Optional inputs to send to the first stage of the pipeline.
        idx : None, int, or slice
            Specification of which stage(s) to run. If None is passed, all
            stages will be run.
        return_intermediate : bool
            If True,

        Returns
        -------
        outputs : list or MapSet
            MapSet output by final stage if `return_intermediate` is False, or
            list of MapSets output by each stage if `return_intermediate` is
            True.

        """
        idx = slice(None) if idx is None else idx
        intermediate = []
        for stage in self.stages[idx]:
            logging.debug('Working on stage %s (%s)' %(stage.stage_name,
                                                       stage.service_name))
            try:
                outputs = stage.compute_outputs(inputs)
            except:
                logging.error('Error occurred computing outputs in stage %s /'
                              ' service %s ...' %(stage.stage_name,
                                                  stage.service_name))
                raise

            logging.trace('outputs: %s' %(outputs,))

            if return_intermediate:
                intermediate.append(outputs)

            # Outputs from this stage become inputs for next stage
            inputs = outputs

        if return_intermediate:
            return intermediate

        return outputs

    def update_params(self, params):
        [stage.params.update_existing(params) for stage in self]

    @property
    def params(self):
        params = ParamSet()
        [params.extend(stage.params) for stage in self]
        return params

    @property
    def stages(self):
        return [s for s in self]


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import numpy as np
    from pisa.utils.fileio import from_file, to_file
    from pisa.utils.parse_config import parse_config

    parser = ArgumentParser()
    parser.add_argument(
        '-t', '--pipeline-settings', metavar='CONFIGFILE', type=str,
        help='File containing settings for the pipeline.'
    )
    parser.add_argument(
        '--only-stage', metavar='STAGE', type=str,
        help='''Test stage: Instantiate a single stage in the pipeline
        specification and run it in isolation (as the sole stage in a
        pipeline). If it is a stage that requires inputs, these can be
        specified with the --infile argument, or else dummy stage input maps
        (numpy.ones(...), matching the input binning specification) are
        generated for testing purposes. See also --infile and --transformfile
        arguments.'''
    )
    parser.add_argument(
        '--stop-after-stage', metavar='STAGE', type=str,
        help='''Test stage: Instantiate a pipeline up to and including
        STAGE, but stop there.'''
    )
    parser.add_argument(
        '-o', '--outputs-file', metavar='FILE', type=str,
        default='out.json',
        help='''File for storing outputs. See also --intermediate-outputs
        argument.'''
    )
    parser.add_argument(
        '-i', '--inputs-file', metavar='FILE', type=str,
        required=False,
        help='''File from which to read inputs to be fed to the pipeline.'''
    )
    parser.add_argument(
        '-T', '--transform-file', metavar='FILE', type=str,
        required=False,
        help='''File into which to store transform(s) from the pipeline.'''
    )
    parser.add_argument(
        '-I', '--intermediate', action='store_true',
        help='''Store all intermediate outputs, not just the final stage's
        outputs.'''
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='set verbosity level'
    )

    args = parser.parse_args()

    set_verbosity(args.v)

    pipeline = Pipeline(args.pipeline_settings)
    m0 = pipeline.compute_outputs()
    fp = pipeline.params.free
    fp['test'].value *= 1.2
    pipeline.update_params(fp)
    m1 = pipeline.compute_outputs()
    print (m1/m0)['nue'][0,0]
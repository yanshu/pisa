#!/usr/bin/env python

# authors: T. Ehrhardt
# date:    October 14, 2016

"""
Profile LLH/Chisquare Analysis

"""

from pisa.analysis.analysis import Analysis
from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.log import logging, set_verbosity

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    import numpy as np

    from pisa import ureg, Q_
    from pisa.utils.fileio import from_file, to_file
    from pisa.utils.config_parser import parse_pipeline_config

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-d', '--data-settings', type=str,
        metavar='configfile', default=None,
        help='''Settings for the generation of "data" distributions; repeat
        this argument to specify multiple pipelines. If omitted, the same
        settings as specified for --template-settings are used to generate data
        distributions.'''
    )
    parser.add_argument(
        '-t', '--template-settings',
        metavar='CONFIGFILE', required=True,
        action='append',
        help='''Settings for generating template distributions; repeat
        this option to define multiple pipelines.'''
    )
    parser.add_argument(
        '--param-names', type=str, nargs='+', required=True, help='''Provide a
        list of parameter names to scan.'''
    )
    parser.add_argument(
        '--steps', type=int, nargs='+', required=True, help='''Provide a number
        of steps for each parameter (in the same order as the parameter names).'''
    )
    parser.add_argument(
        '--only-points', type=int, nargs='+', required=False, help='''Provide a
        point or ranges of points to be scanned specified by one or an even
        number of integer numbers (might be useful if the analysis is to be
        split up into several smaller jobs). 0-indexing is assumed. Isn't
        applied to any single parameter, but to the whole set of points
        (with steps x steps - 1 corresponding to the last).'''
    )
    parser.add_argument(
        '--no-outer', action='store_true', help='''Do not scan points as outer
        product of inner sequences.'''
    )
    parser.add_argument(
        '--param-selection', type=str, required=False,
        help='''Selection of params to use in order to generate the hypothesised
        Asimov distributions.'''
    )
    parser.add_argument(
        '--profile', action='store_true', help='''Run profile scan, i.e. optimise
        over remaining free parameters.'''
    )
    parser.add_argument(
        '-o', '--outfile', metavar='FILE',
        type=str, action='store', default='out.json',
        help='file to store the output'
    )
    parser.add_argument(
        '-m', '--minimizer-settings', type=str,
        metavar='JSONFILE', required=True,
        help='''Settings related to the minimizer used in the LLR analysis.'''
    )
    parser.add_argument(
        '--metric', type=str,
        choices=['llh', 'chi2', 'conv_llh'], required=True,
        help='''Settings related to the minimizer used in the LLR analysis.'''
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='set verbosity level'
    )
    args = parser.parse_args()

    set_verbosity(args.v)

    if args.data_settings is None:
        data_settings = args.template_settings
    else:
        data_settings = args.data_settings

    data_maker = DistributionMaker(data_settings)
    hypo_maker = DistributionMaker(args.template_settings)

    data = data_maker.get_total_outputs()

    analysis = Analysis()

    minimizer_settings = from_file(args.minimizer_settings)

    res = analysis.scan(data_dist=data, hypo_maker=hypo_maker,
                        hypo_param_selections=args.param_selection,
                        metric=args.metric, param_names=args.param_names,
                        steps=args.steps, only_points=args.only_points,
                        outer=not args.no_outer, profile=args.profile,
                        minimizer_settings=minimizer_settings,
                        outfile=args.outfile)
    to_file(res, args.outfile)
    logging.info("Done.")

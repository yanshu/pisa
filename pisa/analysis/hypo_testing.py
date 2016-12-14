#!/usr/bin/env python

# authors: J.L. Lanfranchi, P.Eller, and S. Wren
# email:   jll1062+pisa@phys.psu.edu
# date:    March 20, 2016
"""
Hypothesis testing: How do two hypotheses compare for describing MC or data?

This script/module can run either Asimov or LLR analyses. See
`hypo_testing_postprocess.py` to derive significances, etc. from the files
logged by this script.

"""


from __future__ import division

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import Mapping, OrderedDict, Sequence
from copy import copy
import getpass
from itertools import chain, product
import os
import random
import re
import socket
import string
import sys
import time
from traceback import format_exception

import pint

from pisa import _version, __version__
from pisa.analysis.analysis import Analysis
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.map import MapSet
from pisa.utils.comparisons import normQuant
from pisa.utils.fileio import from_file, get_valid_filename, mkdir, to_file
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging, set_verbosity
from pisa.utils.random_numbers import get_random_state
from pisa.utils.resources import find_resource
from pisa.utils.stats import ALL_METRICS
from pisa.utils.timing import timediffstamp, timestamp


__all__ = ['Labels', 'HypoTesting',
           'parse_args', 'normcheckpath', 'main']


class Labels(object):
    """Derive file labels and naming scheme for data and directories produced
    by the HypoTesting class.

    """
    def __init__(self, h0_name, h1_name, data_name, data_is_data,
                 fluctuate_data, fluctuate_fid):
        self.h0_name = get_valid_filename(h0_name).lower()
        self.h1_name = get_valid_filename(h1_name).lower()
        self.data_name = get_valid_filename(data_name).lower()
        self.data_is_data = data_is_data
        self.fluctuate_data = fluctuate_data
        self.fluctuate_fid = fluctuate_fid
        self._construct_names()

    def _construct_names(self):
        self.hypo_prefix = 'hypo'

        if self.h0_name == '':
            self.h0 = self.hypo_prefix
        else:
            self.h0 = '%s_%s' %(self.hypo_prefix, self.h0_name)

        if self.h1_name == '':
            self.h1 = self.hypo_prefix
        else:
            self.h1 = '%s_%s' %(self.hypo_prefix, self.h1_name)

        if self.data_is_data:
            self.data_prefix = 'data'
            self.data_suffix = ''
        else:
            self.data_prefix = 'toy'
            if self.fluctuate_data:
                self.data_suffix = 'pseudodata'
            else:
                self.data_suffix = 'asimov'

        self.generic_data = self.data_prefix
        self.data = self.data_prefix
        self.data_disp = self.data_prefix
        if self.data_name != '':
            self.data += '_' + self.data_name
            self.generic_data += '_' + self.data_name
            self.data_disp += ' ' + self.data_name
        if self.data_suffix != '':
            self.data += '_' + self.data_suffix
            self.data_disp += ' ' + self.data_suffix

        if self.fluctuate_fid:
            self.fid_disp = 'fiducial pseudodata'
            self.fid = 'fid_pseudodata'
        else:
            self.fid_disp = 'fiducial Asimov'
            self.fid = 'fid_asimov'

        # Fits to data
        self.h0_fit_to_data = '{h0}_fit_to_{data}'.format(**self.dict)
        self.h1_fit_to_data = '{h1}_fit_to_{data}'.format(**self.dict)

        for x, y in product(*[['0', '1']]*2):
            varname = 'h{x}_fit_to_h{y}_fid_'.format(x=x, y=y)
            basestr = ('{h%s}_fit_to_{h%s}_{fid}'%(x, y)).format(**self.dict)
            self.dict[varname+'base'] = basestr
            if self.fluctuate_fid:
                self.dict[varname+'re'] = re.compile(basestr + '_' +
                                                     r'(?P<fid_ind>[0-9]+)')
            else:
                self.dict[varname+'re'] = re.compile(basestr)

            # There're *always* fits performed to fid asimov
            #self.

        # Directory naming pattern
        if self.fluctuate_data:
            self.subdir_re = re.compile(self.data + '_(?P<data_ind>[0-9]+)')
        else:
            self.subdir_re = re.compile(self.data)

    def derive_fid_fits_names(self, fid_ind=None):
        # Define file name labels
        if self.fluctuate_fid:
            ind_sfx = '_%d' %fid_ind
        else:
            ind_sfx = ''

        for x, y in product(*[['0', '1']]*2):
            dst_varname = 'h{x}_fit_to_h{y}_fid'.format(x=x, y=y)
            src_varname = dst_varname + '_base'
            self.dict[dst_varname] = self.dict[src_varname]  + ind_sfx

    @property
    def dict(self):
        return self.__dict__


class HypoTesting(Analysis):
    """Tools for testing two hypotheses against one another.

    Note that duplicated `*_maker` specifications are _not_ instantiated
    separately, but instead are re-used for all duplicate definitions.
    `*_param_selections` allows for this reuse, whereby sets of parameters
    infixed with the corresponding param_selectors can be switched among to
    simulate different physics using the same DistributionMaker (e.g.,
    switching between h0 and h1 hypotheses).


    Parameters
    ----------
    logdir : string
        Base directory in which to create the directory that stores all
        results. Note that `logdir` will be (recursively) generated if it does
        not exist.

    minimizer_settings : string
        Minimizer settings file or resource path.

    data_maker : None, DistributionMaker or instantiable thereto
        Data maker specification, or None (specify an already-generated data
        distribution with `data_dist`).

    data_param_selections : None, string, or sequence of strings
        Param selections to use for data, or None to accept any param
        selections already made in `data_maker`.

    data_name : None or string
        Name for data distribution. If None, a name is auto-generated.

    data_dist : None, MapSet or instantiable thereto
        Specify an existing distribution as the data distribution instead of a
        generating a new one. Use this instead of `data_maker`.

    h0_name : None or string
        Name for hypothesis 0. If None, a name is auto-generated.

    h0_maker : DistributionMaker or instantiable thereto
        Hypothesis-0-maker specification.

    h0_param_selections : None, string, or sequence of strings
        Param selections to use for hypothesis 0, or None to accept any param
        selections already made in `h0_maker`.

    h0_fid_asimov_dist : None, MapSet or instantiable thereto
        TODO: this parameter is NOT currently used, but is intended to remove
        requirement to re-generate this distribution if it's already been
        generated in a previous run

    h1_name : None or string
        Name for hypothesis 1. If None, a name is auto-generated.

    h1_maker : None, DistributionMaker or instantiable thereto
        Hypothesis-1-maker specification. If None, `h0_maker` is used also for
        hypothesis 1 (but in this case, be sure to specify
        `h1_param_selections` so that h0 and h1 come out to be different).

    h1_param_selections : None, string, or sequence of strings
        Param selections to use for hypothesis 1, or None to accept any param
        selections already made in `h1_maker`.

    h1_fid_asimov_dist : None, MapSet or instantiable thereto
        TODO: this parameter is NOT currently used, but is intended to remove
        requirement to re-generate this distribution if it's already been
        generated in a previous run

    num_data_trials : int >= 1
        Number of (pseudo)data trials to run. For each trial, a new pseudodata
        distribution is generated, and then all subsequent fits are preformed.
        Note that data trials recorded to disk are not duplicated in subsequent
        runs (assuming the same `logdir` is specified for each run).

    num_fid_trials : int >= 1
        Number of fiducial-fit trials to run. For each trial, a new fluctuated
        fiducial distribution is generated, and then fits are preformed to
        that. Note that fiducial trials recorded to disk are not duplicated in
        subsequent runs (assuming the same `logdir` is specified for each run).

    data_start_ind : int >= 0 but < 2**(?)
        Start data trials at this index. Valid indexes begin with 0. The final
        data trial index is (data_start_ind + num_data_trials - 1). Any data
        trials already recorded to disk will be skipped.

    fid_start_ind : int >= 0 but < 2**(?)
        Start fiducial trials at this index. Valid indexes begin with 0. The
        final fiducial trial index is (fid_start_ind + num_fid_trials - 1). Any
        fiducial trials already recorded to disk will be skipped.

    check_octant : bool
        If True and theta23 is a free parameter, minimization is performed once
        starting wtih theta23 = theta23.nominal_value and then a second time
        starting with theta23 = (180 deg - theta23.nominal_value). The best fit
        found from each of these two fits is taken to be the best overall fit.

    metric : string
        Metric for minimizer to use for comparing distributions. Valid metrics
        are defined by `pisa.utils.stats.ALL_METRICS`.

    other_metrics : None, string, or sequence of strings
        Other metric to record to compare distributions. These are not used by
        the minimizer. Valid metrics are defined by
        `pisa.utils.stats.ALL_METRICS`.

    blind : bool
        Set to True to run a blind analysis, whereby free parameter values are
        hidden from terminal display and are removed before storing to log
        files.

    allow_dirty : bool
        !USE WITH CAUTION! Allow for running code despite a "dirty" git
        repository (i.e., files in the repository have been changed but not
        committed). Setting to True is dangerous since this might result in
        irreproducible results.

    allow_no_git_info : bool
        !USE WITH CAUTION! Allow for running without knowing git version
        information. Setting to True is dangerous since this might result in
        irreproducible results.

    pprint : bool
        If True, display fit information as a single line on the terminal that
        updates in-place as the fit proceeds. If False, this information is
        output as a separate line for each iteration.


    Notes
    -----
    LLR analysis is a very thorough (and computationally expensive) method to
    compare discrete hypotheses. In general, a total of

        num_data_trials * (2 + 4*num_fid_trials)

    fits must be performed (and note that for each fit, many distributions
    (typically dozens or even hundreds) must be generated).

    If the "data" used in the analysis is pseudodata (i.e., `data_maker` uses
    Monte Carlo to produce its distributions, and these are then
    Poisson-fluctuated--`fluctuate_data` is True), then `num_data_trials`
    should be as large as is computationally feasible.

    Likewise, if the fiducial-fit data is to be pseudodata (i.e.,
    `fluctuate_fid` is True and regardless if `data_maker` uses Monte
    Carlo), `num_fid_trials` should be as large as computationally
    feasible.

    Typical analyses include the following:
        * Asimov analysis of data: `data_maker` uses (actual, measured) data
          and both `fluctuate_data` and `fluctuate_fid` are False.
        * Pseudodata analysis of data: `data_maker` uses (actual, measured)
          data, `fluctuate_data` is False, and `fluctuate_fid` is True.
        * Asimov analysis of Monte Carlo: `data_maker` uses Monte Carlo to
          produce its distributions and both `fluctuate_data` and
          `fluctuate_fid` are False.
        * Pseudodata analysis of Monte Carlo: `data_maker` uses Monte Carlo to
          produce its distributions and both `fluctuate_data` and
          `fluctuate_fid` are False.


    References
    ----------
    TODO


    Examples
    --------
    TODO

    """
    def __init__(self, logdir, minimizer_settings,
                 data_is_data,
                 fluctuate_data, fluctuate_fid, metric, other_metrics=None,
                 h0_name=None, h0_maker=None,
                 h0_param_selections=None, h0_fid_asimov_dist=None,
                 h1_name=None, h1_maker=None,
                 h1_param_selections=None, h1_fid_asimov_dist=None,
                 data_name=None, data_maker=None,
                 data_param_selections=None, data_dist=None,
                 num_data_trials=1, num_fid_trials=1,
                 data_start_ind=0, fid_start_ind=0,
                 check_octant=True,
                 allow_dirty=False, allow_no_git_info=False,
                 blind=False, store_minimizer_history=True, pprint=False):
        assert num_data_trials >= 1
        assert num_fid_trials >= 1
        assert data_start_ind >= 0
        assert fid_start_ind >= 0
        assert metric in ALL_METRICS

        # Instantiate h0 distribution maker to ensure it is a valid spec
        if h0_maker is None:
            raise ValueError('`h0_maker` must be specified (and not None)')
        if not isinstance(h0_maker, DistributionMaker):
            h0_maker = DistributionMaker(h0_maker)

        if isinstance(h0_param_selections, basestring):
            h0_param_selections = h0_param_selections.strip().lower()
            if h0_param_selections == '':
                h0_param_selections = None
            else:
                h0_param_selections = [h0_param_selections]
        if isinstance(h1_param_selections, basestring):
            h1_param_selections = h1_param_selections.strip().lower()
            if h1_param_selections == '':
                h1_param_selections = None
            else:
                h1_param_selections = [h1_param_selections]
        if isinstance(data_param_selections, basestring):
            data_param_selections = data_param_selections.strip().lower()
            if data_param_selections == '':
                data_param_selections = None
            else:
                data_param_selections = [h0_param_selections]

        if (isinstance(h0_param_selections, Sequence)
                and len(h0_param_selections) == 0):
            h0_param_selections = None
        if (isinstance(h1_param_selections, Sequence)
                and len(h1_param_selections) == 0):
            h1_param_selections = None
        if (isinstance(data_param_selections, Sequence)
                and len(data_param_selections) == 0):
            data_param_selections = None

        # Cannot specify either of `data_maker` or `data_param_selections` if
        # `data_dist` is supplied.
        if data_dist is not None:
            assert data_maker is None
            assert data_param_selections is None
            assert num_data_trials == 1
            if isinstance(data_dist, basestring):
                data_dist = from_file(data_dist)
            if not isinstance(data_dist, MapSet):
                data_dist = MapSet(data_dist)

        # Ensure num_{fid_}data_trials is one if fluctuate_{fid_}data is False
        if not fluctuate_data and num_data_trials != 1:
            logging.warn(
                'More than one data trial is unnecessary because'
                ' `fluctuate_data` is False (i.e., all `num_data_trials` data'
                ' distributions will be identical). Forcing `num_data_trials`'
                ' to 1.'
            )
            num_data_trials = 1

        if not fluctuate_fid and num_fid_trials != 1:
            logging.warn(
                'More than one fid trial is unnecessary because'
                ' `fluctuate_fid` is False (i.e., all'
                ' `num_fid_trials` data distributions will be identical).'
                ' Forcing `num_fid_trials` to 1.'
            )
            num_fid_trials = 1

        # Identify duplicate `*_maker` specifications
        self.h1_maker_is_h0_maker = False
        if h1_maker is None or h1_maker == h0_maker:
            self.h1_maker_is_h0_maker = True

        self.data_maker_is_h0_maker = False
        if data_maker is None or data_maker == h0_maker:
            self.data_maker_is_h0_maker = True

        self.data_maker_is_h1_maker = False
        if data_maker == h1_maker:
            self.data_maker_is_h1_maker = True

        # If no data maker settings AND no data param selections are provided,
        # assume that data param selections are to come from hypo h0
        if data_maker is None and data_param_selections is None:
            data_param_selections = h0_param_selections

        # If no h1 maker settings AND no h1 param selections are
        # provided, then we really can't proceed since h0 and h1 will be
        # identical in every way and there's nothing of substance to be done.
        if h1_maker is None and h1_param_selections is None:
            raise ValueError(
                'Hypotheses h0 and h1 to be generated will use the same'
                ' distribution maker configured the same way, leading to'
                ' trivial behavior. If you wish for this behavior, you'
                ' must explicitly specify `h1_maker` and/or'
                ' `h1_param_selections`.'
            )

        # If analyzing actual data, fluctuations should not be applied to the
        # data_dist (fluctuating fiducial-fits Asimov dist is still fine,
        # though).
        if data_is_data and fluctuate_data:
            raise ValueError('Adding fluctuations to actual data distribution'
                             ' is invalid.')

        # Instantiate distribution makers only where necessary (otherwise copy)
        if not isinstance(h1_maker, DistributionMaker):
            if self.h1_maker_is_h0_maker:
                h1_maker = h0_maker
            else:
                h1_maker = DistributionMaker(h1_maker)

        # Cannot know if data came from same dist maker if we're given the data
        # distribution directly
        if data_dist is not None:
            self.data_maker_is_h0_maker = False
            self.data_maker_is_h1_maker = False
        # Otherwise instantiate or copy the data dist maker
        else:
            if not isinstance(data_maker, DistributionMaker):
                if self.data_maker_is_h0_maker:
                    data_maker = h0_maker
                elif self.data_maker_is_h1_maker:
                    data_maker = h1_maker
                else:
                    data_maker = DistributionMaker(data_maker)

        # Read in minimizer settings
        if isinstance(minimizer_settings, basestring):
            minimizer_settings = from_file(minimizer_settings)
        assert isinstance(minimizer_settings, Mapping)

        # Store variables to `self` for later access

        self.logdir = logdir
        self.minimizer_settings = minimizer_settings
        self.check_octant = check_octant

        self.h0_maker = h0_maker
        self.h0_param_selections = h0_param_selections

        self.h1_maker = h1_maker
        self.h1_param_selections = h1_param_selections

        self.data_is_data = data_is_data
        self.data_maker = data_maker
        self.data_param_selections = data_param_selections

        self.metric = metric
        self.other_metrics = other_metrics
        self.fluctuate_data = fluctuate_data
        self.fluctuate_fid = fluctuate_fid

        self.num_data_trials = num_data_trials
        self.num_fid_trials = num_fid_trials
        self.data_start_ind = data_start_ind
        self.fid_start_ind = fid_start_ind

        self.data_ind = self.data_start_ind
        self.fid_ind = self.fid_start_ind

        self.allow_dirty = allow_dirty
        self.allow_no_git_info = allow_no_git_info

        self.blind = blind
        self.store_minimizer_history = store_minimizer_history
        self.pprint = pprint

        # Storage for most recent Asimov (un-fluctuated) distributions
        self.toy_data_asimov_dist = None
        self.h0_fid_asimov_dist = None
        self.h1_fid_asimov_dist = None

        # Storage for most recent "data" (either un-fluctuated--if Asimov
        # analysis being run or if actual data is being used--or fluctuated--if
        # pseudodata is being generated) data
        self.data_dist = data_dist
        self.h0_fid_dist = None
        self.h1_fid_dist = None

        # Populate names with user-specified strings or try to make intelligent
        # guesses based on configuration
        if h0_name is None:
            if self.h0_param_selections is not None:
                h0_name = ','.join(self.h0_param_selections)
            else:
                h0_name = 'h0'
        h0_name = get_valid_filename(h0_name).lower()

        if h1_name is None:
            if (self.h1_maker == self.h0_maker
                    and self.h1_param_selections == self.h0_param_selections):
                h1_name = h0_name
            elif self.h1_param_selections is not None:
                h1_name = ','.join(self.h1_param_selections)
            else:
                h1_name = 'h1'

        if data_name is None:
            if (self.data_maker == self.h0_maker
                    and self.data_param_selections == self.h0_param_selections):
                data_name = h0_name
            elif (self.data_maker == self.h1_maker
                  and self.data_param_selections == self.h1_param_selections):
                data_name = h1_name
            elif self.data_param_selections is not None:
                data_name = ','.join(self.data_param_selections)
            else:
                data_name = ''

        self.labels = Labels(
            h0_name=h0_name, h1_name=h1_name,
            data_name=data_name, data_is_data=self.data_is_data,
            fluctuate_data=self.fluctuate_data,
            fluctuate_fid=self.fluctuate_fid
        )

    def run_analysis(self):
        """Run the defined analysis.

        Progress and estimated time remaining is written to stdout/stderr, and
        results are logged to an appropriate directory within `self.logdir`.

        """
        logging.info('Running LLR analysis.')
        self.analysis_start_time = time.time()

        self.setup_logging()
        self.write_config_summary()
        self.write_minimizer_settings()
        self.write_run_info()

        t0 = time.time()
        try:
            # Loop for multiple (if fluctuated) data distributions
            for self.data_ind in xrange(self.data_start_ind,
                                        self.data_start_ind
                                        + self.num_data_trials):
                data_trials_complete = self.data_ind-self.data_start_ind
                pct_data_complete = (
                    100.*(data_trials_complete)/self.num_data_trials
                )
                logging.info(
                    'Working on %s set ID %d (will stop after ID %d).'
                    ' %0.2f%s of %s sets completed.'
                    %(self.labels.data_disp,
                      self.data_ind,
                      self.data_start_ind+self.num_data_trials-1,
                      pct_data_complete,
                      '%',
                      self.labels.data_disp)
                )

                self.generate_data()
                self.fit_hypos_to_data()

                # Loop for multiple (if fluctuated) fiducial data distributions
                for self.fid_ind in xrange(self.fid_start_ind,
                                           self.fid_start_ind
                                           + self.num_fid_trials):
                    fid_trials_complete = self.fid_ind-self.fid_start_ind
                    pct_fid_dist_complete = (
                        100*(fid_trials_complete)/self.num_fid_trials
                    )

                    dt = time.time() - t0
                    total_complete = (self.num_fid_trials*data_trials_complete
                                      + fid_trials_complete)
                    trials_to_go = (self.num_data_trials*self.num_fid_trials
                                    - total_complete)

                    ts_remaining = '???'
                    if total_complete > 0:
                        sec_per_fid = dt / total_complete
                        time_to_go = sec_per_fid * trials_to_go
                        ts_remaining = timediffstamp(time_to_go, sec_decimals=0,
                                                     hms_always=True)

                    logging.info(
                        ('Working on {data_disp} set ID %d / {fid_disp} set ID'
                         ' %d. %d trials to go, est time remaining: %s'
                         %(self.data_ind, self.fid_ind, trials_to_go,
                           ts_remaining)).format(**self.labels.dict)
                    )

                    self.produce_fid_data()
                    self.fit_hypos_to_fid()
        except:
            exc = sys.exc_info()
        else:
            exc = (None, None, None)
        finally:
            if exc[0] is not None:
                logging.error('`run_analysis` body failed with exception:')
                for line in format_exception(*exc):
                    [logging.error(' '*4 + sl) for sl in line.splitlines()]

            try:
                self.write_run_stop_info(exc=exc)
            except:
                exc_l = sys.exc_info()
            else:
                exc_l = (None, None, None)

            if exc_l[0] is not None:
                logging.error('`write_run_stop_info` failed with exception:')
                for line in format_exception(*exc_l):
                    [logging.error(' '*4 + sl) for sl in line.splitlines()]
                raise exc_l

            if exc[0] is not None:
                raise exc

    def generate_data(self):
        logging.info('Generating %s distributions.' %self.labels.data_disp)
        # Ambiguous whether we're dealing with Asimov or regular data if the
        # data set is provided for us, so just return it.
        if self.num_data_trials == 1 and self.data_dist is not None:
            return self.data_dist

        # Dealing with data: No such thing as Asimov
        if self.data_is_data:
            if self.data_dist is None:
                self.data_maker.select_params(self.data_param_selections)
                self.data_dist = self.data_maker.get_outputs(return_sum=True)
                self.h0_fit_to_data = None
                self.h1_fit_to_data = None
            return self.data_dist

        # Otherwise: Toy data (MC)...

        # Produce Asimov dist if we don't already have it
        if self.toy_data_asimov_dist is None:
            self.data_maker.select_params(self.data_param_selections)
            self.toy_data_asimov_dist = (
                self.data_maker.get_outputs(return_sum=True)
            )
            self.h0_fit_to_data = None
            self.h1_fit_to_data = None

        if self.fluctuate_data:
            assert self.data_ind is not None
            # Random state for data trials is defined by:
            #   * data vs fid-dist = 0  : data part (outer loop)
            #   * data trial = data_ind : data trial number (use same for data
            #                             and and fid data trials, since on the
            #                             same data trial)
            #   * fid trial = 0         : always 0 since data stays the same
            #                             for all fid trials in this data trial
            data_random_state = get_random_state([0, self.data_ind, 0])

            self.data_dist = self.toy_data_asimov_dist.fluctuate(
                method='poisson', random_state=data_random_state
            )

        else:
            self.data_dist = self.toy_data_asimov_dist

        return self.data_dist

    # TODO: use hashes to ensure fits aren't repeated that don't have to be?
    def fit_hypos_to_data(self):
        """Fit both hypotheses to "data" to produce fiducial Asimov
        distributions from *each* of the hypotheses. (i.e., two fits are
        performed unless redundancies are detected).

        """
        # Setup directory for logging results
        self.thisdata_dirpath = self.data_dirpath
        if self.fluctuate_data:
            self.thisdata_dirpath += '_' + format(self.data_ind, 'd')
        mkdir(self.thisdata_dirpath)

        # If h0 maker is same as data maker, we know the fit will end up with
        # the data maker's params. Set these param values and record them.
        if (not self.data_is_data and self.data_maker_is_h0_maker
                and self.h0_param_selections == self.data_param_selections
                and not self.fluctuate_data):
            logging.info('Hypo %s will reproduce exactly %s distributions; not'
                         ' running corresponding fit.'
                         %(self.labels.h0_name, self.labels.data_disp))

            self.data_maker.select_params(self.data_param_selections)
            self.data_maker.reset_free()

            self.h0_fit_to_data = self.nofit_hypo(
                data_dist=self.data_dist,
                hypo_maker=self.data_maker,
                hypo_param_selections=self.data_param_selections,
                hypo_asimov_dist=self.toy_data_asimov_dist,
                metric=self.metric, other_metrics=self.other_metrics,
                blind=self.blind
            )

        # Otherwise, we do have to do the fit.
        else:
            logging.info('Fitting hypo %s to %s distributions.'
                         %(self.labels.h0_name, self.labels.data_disp))
            self.h0_fit_to_data, alternate_fits = self.fit_hypo(
                data_dist=self.data_dist,
                hypo_maker=self.h0_maker,
                hypo_param_selections=self.h0_param_selections,
                metric=self.metric,
                other_metrics=self.other_metrics,
                minimizer_settings=self.minimizer_settings,
                check_octant=self.check_octant,
                pprint=self.pprint,
                blind=self.blind
            )
        self.h0_fid_asimov_dist = self.h0_fit_to_data['hypo_asimov_dist']

        self.log_fit(fit_info=self.h0_fit_to_data,
                     dirpath=self.thisdata_dirpath,
                     label=self.labels.h0_fit_to_data)

        if (not self.data_is_data and self.data_maker_is_h1_maker
                and self.h1_param_selections == self.data_param_selections
                and not self.fluctuate_data):
            logging.info('Hypo %s will reproduce exactly %s distributions; not'
                         ' running corresponding fit.'
                         %(self.labels.h1_name, self.labels.data_disp))

            self.data_maker.select_params(self.data_param_selections)
            self.data_maker.reset_free()

            self.h1_fit_to_data = self.nofit_hypo(
                data_dist=self.data_dist,
                hypo_maker=self.h1_maker,
                hypo_param_selections=self.h1_param_selections,
                hypo_asimov_dist=self.toy_data_asimov_dist,
                metric=self.metric, other_metrics=self.other_metrics,
                blind=self.blind
            )
        elif (self.h1_maker_is_h0_maker
              and self.h1_param_selections == self.h0_param_selections):
            self.h1_fit_to_data = copy(self.h0_fit_to_data)
        else:
            logging.info('Fitting hypo %s to %s distributions.'
                         %(self.labels.h1_name, self.labels.data_disp))
            self.h1_fit_to_data, alternate_fits = self.fit_hypo(
                data_dist=self.data_dist,
                hypo_maker=self.h1_maker,
                hypo_param_selections=self.h1_param_selections,
                metric=self.metric,
                other_metrics=self.other_metrics,
                minimizer_settings=self.minimizer_settings,
                check_octant=self.check_octant,
                pprint=self.pprint,
                blind=self.blind
            )
        self.h1_fid_asimov_dist = self.h1_fit_to_data['hypo_asimov_dist']

        self.log_fit(fit_info=self.h1_fit_to_data,
                     dirpath=self.thisdata_dirpath,
                     label=self.labels.h1_fit_to_data)

    def produce_fid_data(self):
        logging.info('Generating %s distributions.' %self.labels.fid_disp)
        # Retrieve event-rate maps for best fit to data with each hypo

        if self.fluctuate_fid:
            # Random state for data trials is defined by:
            #   * data vs fid-dist = 1     : fid data part (inner loop)
            #   * data trial = data_ind    : data trial number (use same for
            #                                data and and fid data trials,
            #                                since on the same data trial)
            #   * fid trial = fid_ind      : always 0 since data stays the same
            #                                for all fid trials in this data
            #                                trial
            fid_random_state = get_random_state([1, self.data_ind,
                                                 self.fid_ind])

            # Fluctuate h0 fid Asimov
            self.h0_fid_dist = self.h0_fid_asimov_dist.fluctuate(
                method='poisson',
                random_state=fid_random_state
            )
            # The state of `random_state` will be moved forward now as compared
            # to what it was upon definition above. This is the desired
            # behavior, so the *exact* same random state isn't used to
            # fluctuate h1 as was used to fluctuate h0.
            self.h1_fid_dist = self.h1_fid_asimov_dist.fluctuate(
                method='poisson',
                random_state=fid_random_state
            )
        else:
            self.h0_fid_dist = self.h0_fid_asimov_dist
            self.h1_fid_dist = self.h1_fid_asimov_dist

        return self.h1_fid_dist, self.h0_fid_dist

    def fit_hypos_to_fid(self):
        self.labels.derive_fid_fits_names(fid_ind=self.fid_ind)

        fpath = os.path.join(self.thisdata_dirpath,
                             self.labels.h0_fit_to_h0_fid + '.json')
        if not os.path.isfile(fpath):
            # If fid isn't fluctuated, it's redundant to fit a hypo to a dist
            # it generated
            self.h0_maker.select_params(self.h0_param_selections)
            self.h0_maker.reset_free()
            if not self.fluctuate_fid:
                logging.info(
                    'Hypo %s %s is not fluctuated; fitting this hypo to its'
                    ' own %s distributions is unnecessary.'
                    %(self.labels.h0_name, self.labels.fid_disp,
                      self.labels.fid_disp)
                )
                self.h0_fit_to_h0_fid = self.nofit_hypo(
                    data_dist=self.h0_fid_dist,
                    hypo_maker=self.h0_maker,
                    hypo_param_selections=self.h0_param_selections,
                    hypo_asimov_dist=self.h0_fid_asimov_dist,
                    metric=self.metric, other_metrics=self.other_metrics,
                    blind=self.blind
                )
            else:
                logging.info('Fitting hypo %s to its own %s distributions.'
                             %(self.labels.h0_name, self.labels.fid_disp))
                self.h0_fit_to_h0_fid, alternate_fits = self.fit_hypo(
                    data_dist=self.h0_fid_dist,
                    hypo_maker=self.h0_maker,
                    hypo_param_selections=self.h0_param_selections,
                    metric=self.metric,
                    other_metrics=self.other_metrics,
                    minimizer_settings=self.minimizer_settings,
                    check_octant=self.check_octant,
                    pprint=self.pprint,
                    blind=self.blind
                )

            self.log_fit(fit_info=self.h0_fit_to_h0_fid,
                         dirpath=self.thisdata_dirpath,
                         label=self.labels.h0_fit_to_h0_fid)

        fpath = os.path.join(self.thisdata_dirpath,
                             self.labels.h1_fit_to_h1_fid + '.json')
        if not os.path.isfile(fpath):
            self.h1_maker.select_params(self.h1_param_selections)
            self.h1_maker.reset_free()
            if not self.fluctuate_fid:
                logging.info(
                    'Hypo %s %s is not fluctuated; fitting this hypo to its'
                    ' own %s distributions is unnecessary.'
                    %(self.labels.h1_name, self.labels.fid_disp,
                      self.labels.fid_disp)
                )
                self.h1_fit_to_h1_fid = self.nofit_hypo(
                    data_dist=self.h1_fid_dist,
                    hypo_maker=self.h1_maker,
                    hypo_param_selections=self.h1_param_selections,
                    hypo_asimov_dist=self.h1_fid_asimov_dist,
                    metric=self.metric, other_metrics=self.other_metrics,
                    blind=self.blind
                )
            else:
                logging.info('Fitting hypo %s to its own %s distributions.'
                             %(self.labels.h1_name, self.labels.fid_disp))
                self.h1_fit_to_h1_fid, alternate_fits = self.fit_hypo(
                    data_dist=self.h1_fid_dist,
                    hypo_maker=self.h1_maker,
                    hypo_param_selections=self.h1_param_selections,
                    metric=self.metric,
                    other_metrics=self.other_metrics,
                    minimizer_settings=self.minimizer_settings,
                    check_octant=self.check_octant,
                    pprint=self.pprint,
                    blind=self.blind
                )

            self.log_fit(fit_info=self.h1_fit_to_h1_fid,
                         dirpath=self.thisdata_dirpath,
                         label=self.labels.h1_fit_to_h1_fid)

        # TODO: remove redundancy if h0 and h1 are identical
        #if (self.h1_maker_is_h0_maker
        #    and self.h1_param_selections == self.h0_param_selections):
        #    self.h0_fit_to_h1_fid =

        # Perform fits of one hypo to fid dist produced by other hypo

        fpath = os.path.join(self.thisdata_dirpath,
                             self.labels.h1_fit_to_h0_fid + '.json')
        if not os.path.isfile(fpath):
            if ((not self.fluctuate_data) and (not self.fluctuate_fid)
                    and self.data_maker_is_h0_maker
                    and self.h0_param_selections == self.data_param_selections):
                logging.info(
                    'Fitting hypo %s to hypo %s %s distributions is'
                    ' unnecessary since former was already fit to %s'
                    ' distributions, which are identical distributions.'
                    %(self.labels.h1_name, self.labels.h0_name,
                      self.labels.fid_disp, self.labels.data_disp)
                )
                self.h1_fit_to_h0_fid = copy(self.h1_fit_to_data)
            else:
                logging.info('Fitting hypo %s to hypo %s %s distributions.'
                             %(self.labels.h1_name, self.labels.h0_name,
                               self.labels.fid_disp))
                self.h1_maker.select_params(self.h1_param_selections)
                self.h1_maker.reset_free()
                self.h1_fit_to_h0_fid, alternate_fits = self.fit_hypo(
                    data_dist=self.h0_fid_dist,
                    hypo_maker=self.h1_maker,
                    hypo_param_selections=self.h1_param_selections,
                    metric=self.metric,
                    other_metrics=self.other_metrics,
                    minimizer_settings=self.minimizer_settings,
                    check_octant=self.check_octant,
                    pprint=self.pprint,
                    blind=self.blind
                )

            self.log_fit(fit_info=self.h1_fit_to_h0_fid,
                         dirpath=self.thisdata_dirpath,
                         label=self.labels.h1_fit_to_h0_fid)

        fpath = os.path.join(self.thisdata_dirpath,
                             self.labels.h0_fit_to_h1_fid + '.json')
        if not os.path.isfile(fpath):
            if ((not self.fluctuate_data) and (not self.fluctuate_fid)
                    and self.data_maker_is_h1_maker
                    and self.h1_param_selections == self.data_param_selections):
                logging.info(
                    'Fitting hypo %s to hypo %s %s distributions is'
                    ' unnecessary since former was already fit to %s'
                    ' distributions, which are identical distributions.'
                    %(self.labels.h0_name, self.labels.h1_name,
                      self.labels.fid_disp, self.labels.data_disp)
                )
                self.h0_fit_to_h1_fid = copy(self.h0_fit_to_data)
            else:
                logging.info('Fitting hypo %s to hypo %s %s distributions.'
                             %(self.labels.h0_name, self.labels.h1_name,
                               self.labels.fid_disp))
                self.h0_maker.select_params(self.h0_param_selections)
                self.h0_maker.reset_free()
                self.h0_fit_to_h1_fid, alternate_fits = self.fit_hypo(
                    data_dist=self.h1_fid_dist,
                    hypo_maker=self.h0_maker,
                    hypo_param_selections=self.h0_param_selections,
                    metric=self.metric,
                    other_metrics=self.other_metrics,
                    minimizer_settings=self.minimizer_settings,
                    check_octant=self.check_octant,
                    pprint=self.pprint,
                    blind=self.blind
                )

            self.log_fit(fit_info=self.h0_fit_to_h1_fid,
                         dirpath=self.thisdata_dirpath,
                         label=self.labels.h0_fit_to_h1_fid)

    def setup_logging(self):
        """
        Should store enough information for the following two purposes:
            1. Be able to completely reproduce the results, assuming access to
               the same git repository.
            2. Be able to easily identify (as a human) the important / salient
               features of this config that might make it different from
               another.

        `config_hash` is generated by creating a list of the following and
        hashing that list:
            * git sha256 for latest commit (will not run if this info isn't
              present or cannot be ascertained or if code is updated since last
              commit and `unsafe_run` is True)
            * hash of instantiated `minimizer_settings` object (sent through
              normQuant)
            * `check_octant`
            * pipelines info for each used for each hypo:
                - stage name, service name, service source code hash
            * name of metric used for minimization

        config_summary.info : Human-readable metadata used to construct hash:
            * config_hash : str
            * source_provenance : dict
                - git_commit_sha256 : str
                - git_repo (?) : str
                - git_branch : str
                - git_remote_url : str
                - git_tag : str
            * minimizer_info : dict
                - minimizer_config_hash : str
                - minimizer_name : str
                - metric_minimized : str
                - check_octant : bool
            * data_is_data : bool
            * data_pipelines : list
                - p0 : list
                    - s0 : dict
                        - stage name : str
                        - service name : str
                        - service source code hash : str
                    - s1 : dict
                    ...
                - p1 : list
                ...
            * data_param_selections
            * h0_pipelines (similarly to data pipelines)
            * h0_param_selections
            * h1_pipelines (list containing list per pipeline)
            * h1_param_selections

        mininimzer_settings.ini : copy of the minimzer settings used

        run_info_<datetime in microseconds, UTC>_<hostname>.info
            * fluctuate_data : bool
            * fluctuate_fid : bool
            * data_start_ind (if toy pseudodata)
            * num_data_trials (if toy pseudodata)
            * fid_start_ind (if fid fits to pseudodata)
            * num_fid_trials (if fid fits to pseudodata)

        h0_pipeline0.ini
        h0_pipeline1.ini
        ...
        h1_pipeline0.ini
        h1_pipeline1.ini
        ...

        Directory Structure
        -------------------
        The base directory for storing data unique to this configuration is

            basedir = logdir/hypo_<h0_name>__hypo_<h1_name>_<config_hash>

        where `config_hash` is derived from the full configuration and is
        independent of `h0_name` and `h1_name` since the latter two entities
        are user-provided and can vary while yielding the same configuration.

        Within the base directory, if we're actually working with data
        (`data_is_data` is True), the following directory is created:

            <basedir>/data_fits

        If "data" actually comes from MC (i.e., `data_is_data` is False), a
        directory

            <basedir>/toy_pseudodata_fits<data_ind>

        is created for each `data_ind` if fluctuations are applied to produce
        pseudodata for fitting to. Otherwise if no fluctuations are applied to
        the toy data distribtuion for fitting to, the directory

            <basedir>/toy_asimov_fits

        is created.

        Files
        -----
        In order to record the full configuration

            /fid<fid_ind>/
            {toy_}data_fits{<data_ind>}/fid<fid_ind>/

        Create or update the files:
            logdir/h0_<h0_name>__h1_<h1_name>/reservations.sqlite
            logdir/h0_<h0_name>__h1_<h1_name>/run_info_<datetime>_<hostname>.info

        run_id comes from (??? hostname and microsecond timestamp??? settings???)

        """
        self.h0_maker.select_params(self.h0_param_selections)
        self.h0_maker.reset_free()
        self.h0_hash = self.h0_maker.state_hash

        self.h1_maker.select_params(self.h1_param_selections)
        self.h1_maker.reset_free()
        self.h1_hash = self.h1_maker.state_hash

        self.data_maker.select_params(self.data_param_selections)
        self.data_maker.reset_free()
        self.data_hash = self.data_maker.state_hash

        # Single unique hash for hypotheses and data configurations
        self.config_hash = hash_obj([self.h0_hash, self.h1_hash,
                                     self.data_hash], hash_to='x')

        # Unique id string for settings related to minimization
        self.minimizer_settings_hash = hash_obj(
            normQuant(self.minimizer_settings), hash_to='x'
        )
        co = 'co1' if self.check_octant else 'co0'
        self.minsettings_flabel = (
            'min_' + '_'.join([self.minimizer_settings_hash, co, self.metric])
        )

        # Code versioning
        self.__version__ = __version__
        self.version_info = _version.get_versions()

        no_git_info = self.version_info['error'] is not None
        if no_git_info:
            msg = 'No info about git repo. Version info: %s' %self.version_info
            if self.allow_no_git_info:
                logging.warn(msg)
            else:
                raise Exception(msg)

        dirty_git_repo = self.version_info['dirty']
        if dirty_git_repo:
            msg = 'Dirty git repo. Version info: %s' %self.version_info
            if self.allow_dirty:
                logging.warn(msg)
            else:
                raise Exception(msg)

        logging.debug('Code version: %s' %self.__version__)

        # Construct root dir name and create dir if necessary
        dirname = '__'.join([self.labels.h0, self.labels.h1,
                             self.labels.generic_data,
                             self.config_hash, self.minsettings_flabel,
                             'pisa' + self.__version__])
        dirpath = os.path.join(self.logdir, dirname)
        mkdir(dirpath)
        normpath = find_resource(dirpath)
        self.logroot = normpath
        logging.info('Output will be saved to dir "%s"' %self.logroot)

        self.data_dirpath = os.path.join(self.logroot, self.labels.data)

        # Filenames and paths
        self.config_summary_fname = 'config_summary.json'
        self.config_summary_fpath = os.path.join(self.logroot,
                                                 self.config_summary_fname)
        self.invocation_datetime = timestamp(utc=True, winsafe=True)
        self.hostname = socket.gethostname()
        chars = string.ascii_lowercase + string.digits
        self.random_suffix = ''.join([random.choice(chars) for i in range(8)])
        self.pid = os.getpid()
        self.user = getpass.getuser()
        self.minimizer_settings_fpath = os.path.join(
            self.logroot, 'minimizer_settings.json'
        )
        self.run_info_fname = (
            'run_%s_%s_%s.info'
            %(self.invocation_datetime, self.hostname, self.random_suffix)
        )
        self.run_info_fpath = os.path.join(self.logroot, self.run_info_fname)

    def write_config_summary(self):
        if os.path.isfile(self.config_summary_fpath):
            return
        summary = OrderedDict()
        d = OrderedDict()
        d['version'] = self.version_info['version']
        d['git_revision_sha256'] = self.version_info['full-revisionid']
        d['git_dirty'] = self.version_info['dirty']
        d['git_error'] = self.version_info['error']
        summary['source_provenance'] = d

        d = OrderedDict()
        d['minimizer_name'] = self.minimizer_settings['method']['value']
        d['minimizer_settings_hash'] = self.minimizer_settings_hash
        d['check_octant'] = self.check_octant
        d['metric_optimized'] = self.metric
        summary['minimizer_info'] = d

        summary['data_is_data'] = self.data_is_data

        self.data_maker.select_params(self.data_param_selections)
        self.data_maker.reset_free()
        summary['data_name'] = self.labels.data_name
        summary['data_is_data'] = self.data_is_data
        summary['data_hash'] = self.data_hash
        summary['data_param_selections'] = ','.join(self.data_param_selections)
        summary['data_params_state_hash'] = self.data_maker.params.state_hash
        summary['data_params'] = [str(p) for p in self.data_maker.params]
        summary['data_pipelines'] = self.summarize_dist_maker(self.data_maker)

        self.h0_maker.select_params(self.h0_param_selections)
        self.h0_maker.reset_free()
        summary['h0_name'] = self.labels.h0_name
        summary['h0_hash'] = self.h0_hash
        summary['h0_param_selections'] = ','.join(self.h0_param_selections)
        summary['h0_params_state_hash'] = self.h0_maker.params.state_hash
        summary['h0_params'] = [str(p) for p in self.h0_maker.params]
        summary['h0_pipelines'] = self.summarize_dist_maker(self.h0_maker)

        self.h1_maker.select_params(self.h1_param_selections)
        self.h1_maker.reset_free()
        summary['h1_name'] = self.labels.h1_name
        summary['h1_hash'] = self.h1_hash
        summary['h1_param_selections'] = ','.join(self.h1_param_selections)
        summary['h1_params_state_hash'] = self.h1_maker.params.state_hash
        summary['h1_params'] = [str(p) for p in self.h1_maker.params]
        summary['h1_pipelines'] = self.summarize_dist_maker(self.h1_maker)

        # Reverse the order so it serializes to a file as intended
        # (want top-to-bottom file convention vs. fifo streaming data
        # convention)
        od = OrderedDict()
        for ok, ov in summary.items():
            if isinstance(ov, OrderedDict):
                od1 = OrderedDict()
                for ik, iv in ov.items():
                    od1[ik] = iv
                ov = od1
            od[ok] = ov

        to_file(od, self.config_summary_fpath, sort_keys=False)

    @staticmethod
    def summarize_dist_maker(dist_maker):
        pipeline_info = []
        for pipeline in dist_maker:
            stage_info = OrderedDict()
            for stage in pipeline:
                k = ':'.join([stage.stage_name, stage.service_name,
                              str(stage.state_hash)])
                d = OrderedDict()
                for attr in ['input_binning', 'output_binning']:
                    if (hasattr(stage, attr)
                            and getattr(stage, attr) is not None):
                        d[attr] = str(getattr(stage, attr))
                stage_info[k] = d
            pipeline_info.append(stage_info)
        return pipeline_info

    def write_run_info(self):
        run_info = []
        run_info.append('invocation_datetime = %s' %self.invocation_datetime)
        run_info.append('hostname = %s' %self.hostname)
        run_info.append('random_suffix = %s' %self.random_suffix)
        run_info.append('pid = %s' %self.pid)
        run_info.append('user = %s' %self.user)

        run_info.append('logdir = %s' %self.logdir)

        run_info.append('fluctuate_data = %s' %self.fluctuate_data)
        run_info.append('fluctuate_fid = %s' %self.fluctuate_fid)
        if self.fluctuate_data:
            run_info.append('data_start_ind = %d' %self.data_start_ind)
            run_info.append('num_data_trials = %d' %self.num_data_trials)
        if self.fluctuate_fid:
            run_info.append('fid_start_ind = %d' %self.fid_start_ind)
            run_info.append('num_fid_trials = %d' %self.num_fid_trials)
        run_info.append('metric = %s' %self.metric)
        run_info.append('other_metrics = %s' %self.other_metrics)
        run_info.append('blind = %s' %self.blind)
        run_info.append('allow_dirty = %s' %self.allow_dirty)
        run_info.append('allow_no_git_info = %s' %self.allow_no_git_info)
        run_info.append('store_minimizer_history = %s'
                        %self.store_minimizer_history)
        run_info.append('pprint = %s' %self.pprint)
        for env_var in ['PISA_FTYPE', 'PISA_RESOURCES',
                        'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
                        'CUDA_VISIBLE_DEVICES',
                        'PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH']:
            if env_var in os.environ:
                val = os.environ[env_var]
            else:
                val = ''
            run_info.append('%s = %s' %(env_var, val))

        for prefix in ['PBS_']:
            [run_info.append('%s = %s' %(env_var, val))
             for env_var, val in os.environ.iteritems()
             if env_var.startswith(prefix)]

        with file(self.run_info_fpath, 'w') as f:
            f.write('\n'.join(run_info) + '\n')
        logging.info('Run info written to: ' + self.run_info_fpath)

    def write_minimizer_settings(self):
        if os.path.isfile(self.minimizer_settings_fpath):
            return
        to_file(self.minimizer_settings, self.minimizer_settings_fpath)

    def write_run_stop_info(self, exc=None):
        if isinstance(exc, Sequence):
            if exc[0] is None:
                exc = None

        self.stop_datetime = timestamp(utc=True, winsafe=True)
        self.stop_time = time.time()
        self.analysis_runtime = self.stop_time - self.analysis_start_time
        dt_stamp = timediffstamp(self.analysis_runtime, hms_always=True,
                                 sec_decimals=0)

        run_info = []
        run_info.append('stop_datetime = %s' %self.stop_datetime)
        run_info.append('analysis_runtime = %s' %dt_stamp)
        if self.fluctuate_data:
            run_info.append('data_stop_ind = %d' %self.data_ind)
        if self.fluctuate_fid:
            run_info.append('fid_stop_ind = %d' %self.fid_ind)
        if exc is None:
            run_info.append('completed = True')
            run_info.append('exception = None')
            run_info.append('traceback = None')
        else:
            run_info.append('completed = False')
            run_info.append('exception = %s: %s' % (exc[0], exc[1]))
            tb = format_exception(*exc)
            formatted_tb = ('\n' + ' '*2).join(
                chain.from_iterable((l.splitlines() for l in tb))
            )
            run_info.append('traceback = %s' %formatted_tb)

        with file(self.run_info_fpath, 'a') as f:
            f.write('\n'.join(run_info) + '\n')

        logging.info('Run stop info written to: ' + self.run_info_fpath)
        logging.info('Total analysis run time: ' + dt_stamp)

    def log_fit(self, fit_info, dirpath, label):
        serialize = ['metric', 'metric_val', 'params', 'minimizer_time',
                     'detailed_metric_info', 'minimizer_metadata']
        if self.store_minimizer_history:
            serialize.append('fit_history')

        info = OrderedDict()
        for k, v in fit_info.iteritems():
            if k not in serialize:
                continue
            if k == 'params':
                d = OrderedDict()
                for param in v.free:
                    d[param.name] = str(param.value)
                v = d
            if k == 'minimizer_metadata':
                if 'hess_inv' in v:
                    try:
                        v['hess_inv'] = v['hess_inv'].todense()
                    except AttributeError:
                        v['hess_inv'] = v['hess_inv']
            if isinstance(v, pint.quantity._Quantity):
                v = str(v)
            info[k] = v
        to_file(info, os.path.join(dirpath, label + '.json'), sort_keys=False)


def parse_args():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-d', '--logdir', required=True,
        metavar='DIR', type=str,
        help='Directory into which to store results and metadata.'
    )
    parser.add_argument(
        '-m', '--minimizer-settings',
        type=str, metavar='MINIMIZER_CFG', required=True,
        help='''Settings related to the optimizer used in the LLR analysis.'''
    )
    parser.add_argument(
        '--no-octant-check',
        action='store_true',
        help='''Disable fitting hypotheses in theta23 octant opposite initial
        octant.'''
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--data-is-data', action='store_true',
        help='''Data pipeline is based upon actual, measured data. The naming
        scheme for stored results is chosen accordingly.'''
    )
    group.add_argument(
        '--data-is-mc', action='store_true',
        help='''Data pipeline is based upon Monte Carlo simulation, and not
        actual data. The naming scheme for stored results is chosen
        accordingly. If this is selected, --fluctuate-data is forced off.'''
    )
    parser.add_argument(
        '--h0-pipeline', required=True,
        type=str, action='append', metavar='PIPELINE_CFG',
        help='''Settings for the generation of hypothesis h0
        distributions; repeat this argument to specify multiple pipelines.'''
    )
    parser.add_argument(
        '--h0-param-selections',
        type=str, default=None, metavar='PARAM_SELECTOR_LIST',
        help='''Comma-separated (no spaces) list of param selectors to apply to
        hypothesis h0's distribution maker's pipelines.'''
    )
    parser.add_argument(
        '--h0-name',
        type=str, metavar='NAME', default=None,
        help='''Name for hypothesis h0. E.g., "NO" for normal
        ordering in the neutrino mass ordering analysis. Note that the name
        here has no bearing on the actual process, so it's important that you
        be careful to use a name that appropriately identifies the
        hypothesis.'''
    )
    parser.add_argument(
        '--h1-pipeline',
        type=str, action='append', default=None, metavar='PIPELINE_CFG',
        help='''Settings for the generation of hypothesis h1 distributions;
        repeat this argument to specify multiple pipelines. If omitted, the
        same settings as specified for --h0-pipeline are used to generate
        hypothesis h1 distributions (and so you have to use the
        --h1-param-selections argument to generate a hypotheses distinct
        from hypothesis h0 but still use h0's distribution maker).'''
    )
    parser.add_argument(
        '--h1-param-selections',
        type=str, default=None, metavar='PARAM_SELECTOR_LIST',
        help='''Comma-separated (no spaces) list of param selectors to apply to
        hypothesis h1 distribution maker's pipelines.'''
    )
    parser.add_argument(
        '--h1-name',
        type=str, metavar='NAME', default=None,
        help='''Name for hypothesis h1. E.g., "IO" for inverted
        ordering in the neutrino mass ordering analysis. Note that the name
        here has no bearing on the actual process, so it's important that you
        be careful to use a name that appropriately identifies the
        hypothesis.'''
    )
    parser.add_argument(
        '--data-pipeline',
        type=str, action='append', default=None, metavar='PIPELINE_CFG',
        help='''Settings for the generation of "data" distributions; repeat
        this argument to specify multiple pipelines. If omitted, the same
        settings as specified for --h0-pipeline are used to generate data
        distributions (i.e., data is assumed to come from hypothesis h0.'''
    )
    parser.add_argument(
        '--data-param-selections',
        type=str, default=None, metavar='PARAM_SELECTOR_LIST',
        help='''Comma-separated list of param selectors to apply to the data
        distribution maker's pipelines. If neither --data-pipeline nor
        --data-param-selections are specified, *both* are copied from
        --h0-pipeline and --h0-param-selections, respectively. However,
        if --data-pipeline is specified while --data-param-selections is not,
        then the param selections in the pipeline config file(s) specified are
        used to produce data distributions.'''
    )
    parser.add_argument(
        '--data-name',
        type=str, metavar='NAME', default=None,
        help='''Name for the data. E.g., "NO" for normal ordering in the
        neutrino mass ordering analysis. Note that the name here has no bearing
        on the actual process, so it's important that you be careful to use a
        name that appropriately identifies the hypothesis.'''
    )
    parser.add_argument(
        '--fluctuate-data',
        action='store_true',
        help='''Apply fluctuations to the data distribution. This should *not*
        be set for analyzing "real" (measured) data, and it is common to not
        use this feature even for Monte Carlo analysis. Note that if this is
        not set, --num-data-trials and --data-start-ind are forced to 1 and 0,
        respectively.'''
    )
    parser.add_argument(
        '--fluctuate-fid',
        action='store_true',
        help='''Apply fluctuations to the fiducaial distributions. If this flag
        is not set, --num-fid-trials and --fid-start-ind are forced to 1 and 0,
        respectively.'''
    )
    parser.add_argument(
        '--metric',
        type=str, default=None, metavar='METRIC',
        help='''Name of metric to use for optimizing the fit. Must be one of
        %s.''' % (ALL_METRICS,)
    )
    parser.add_argument(
        '--other-metric',
        type=str, default=None, metavar='METRIC', action='append',
        choices=['all'] + sorted(ALL_METRICS),
        help='''Name of another metric to evaluate at the best-fit point. Must
        be either 'all' or one of %s. Repeat this argument (or use 'all') to
        specify multiple metrics.''' % (ALL_METRICS,)
    )
    parser.add_argument(
        '--num-data-trials',
        type=int, default=1,
        help='''When performing Monte Carlo analysis, set to > 1 to produce
        multiple pseudodata distributions from the data distribution maker's
        Asimov distribution. This is overridden if --fluctuate-data is not
        set (since each data distribution will be identical if it is not
        fluctuated). This is typically left at 1 (i.e., the Asimov distribution
        is assumed to be representative.'''
    )
    parser.add_argument(
        '--data-start-ind',
        type=int, default=0,
        help='''Fluctated data set index.'''
    )
    parser.add_argument(
        '--num-fid-trials',
        type=int, default=1,
        help='''Number of fiducial pseudodata trials to run. In our experience,
        it takes ~10^3-10^5 fiducial psuedodata trials to achieve low
        uncertainties on the resulting significance, though that exact number
        will vary based upon the details of an analysis.'''
    )
    parser.add_argument(
        '--fid-start-ind',
        type=int, default=0,
        help='''Fluctated fiducial data index.'''
    )
    parser.add_argument(
        '--blind',
        action='store_true',
        help='''Blinded analysis. Do not show parameter values or store to
        logfiles.'''
    )
    parser.add_argument(
        '--allow-dirty',
        action='store_true',
        help='''Warning: Use with caution. (Allow for run despite dirty
        repository.)'''
    )
    parser.add_argument(
        '--allow-no-git-info',
        action='store_true',
        help='''*** DANGER! Use with extreme caution! (Allow for run despite
        complete inability to track provenance of code.)'''
    )
    parser.add_argument(
        '--no-minimizer-history',
        action='store_true',
        help='''Do not store minimizer history (steps). This behavior is also
        enforced if --blind is specified.'''
    )
    parser.add_argument(
        '--pprint',
        action='store_true',
        help='''Live-updating one-line vew of metric and parameter values. (The
        latter are not displayed if --blind is specified.)'''
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='set verbosity level'
    )
    return parser.parse_args()


# TODO: make this work with Python package resources, not merely absolute
# paths! ... e.g. hash on the file or somesuch?
# TODO: move to a central loc prob. in utils
def normcheckpath(path, checkdir=False):
    normpath = find_resource(path)
    if checkdir:
        kind = 'dir'
        check = os.path.isdir
    else:
        kind = 'file'
        check = os.path.isfile

    if not check(normpath):
        raise IOError('Path "%s" which resolves to "%s" is not a %s.'
                      %(path, normpath, kind))
    return normpath


def main():
    args = parse_args()
    init_args_d = vars(args)

    # NOTE: Removing extraneous args that won't get passed to instantiate the
    # HypoTesting object via dictionary's `pop()` method.

    set_verbosity(init_args_d.pop('v'))
    init_args_d['check_octant'] = not init_args_d.pop('no_octant_check')

    init_args_d['data_is_data'] = not init_args_d.pop('data_is_mc')

    init_args_d['store_minimizer_history'] = (
        not init_args_d.pop('no_minimizer_history')
    )

    other_metrics = init_args_d.pop('other_metric')
    if other_metrics is not None:
        other_metrics = [s.strip().lower() for s in other_metrics]
        if 'all' in other_metrics:
            other_metrics = sorted(ALL_METRICS)
        if init_args_d['metric'] in other_metrics:
            other_metrics.remove(init_args_d['metric'])
        if len(other_metrics) == 0:
            other_metrics = None
        else:
            logging.info('Will evaluate other metrics %s' %other_metrics)
        init_args_d['other_metrics'] = other_metrics

    # Normalize and convert `*_pipeline` filenames; store to `*_maker`
    # (which is argument naming convention that HypoTesting init accepts).
    for maker in ['h0', 'h1', 'data']:
        filenames = init_args_d.pop(maker + '_pipeline')
        if filenames is not None:
            filenames = sorted(
                [normcheckpath(fname) for fname in filenames]
            )
        init_args_d[maker + '_maker'] = filenames

        ps_name = maker + '_param_selections'
        ps_str = init_args_d[ps_name]
        if ps_str is None:
            ps_list = None
        else:
            ps_list = [x.strip().lower() for x in ps_str.split(',')]
        init_args_d[ps_name] = ps_list

    # Instantiate the analysis object
    hypo_testing = HypoTesting(**init_args_d)

    # Run the analysis
    hypo_testing.run_analysis()


if __name__ == '__main__':
    main()

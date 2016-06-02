#
# PISA authors: Lukas Schulte
#               schulte@physik.uni-bonn.de
#               Justin L. Lanfranchi
#               jll1062+pisa@phys.psu.edu
#
# CAKE author: Shivesh Mandalia
#              s.p.mandalia@qmul.ac.uk
#
# date:    2016-05-13
"""
The purpose of this stage is to simulate the event classification of PINGU,
sorting the reconstructed nue CC, numu CC, nutau CC, and NC events into the
track and cascade channels.

This service in particular takes in events from a PISA HDF5 file to transform
a set of input map into a set of track and cascade maps.

For each particle "signature", a histogram in the input binning dimensions is
created, which gives the PID probabilities in each bin. The input maps are
transformed according to these probabilities to provide an output containing a
map for track-like events ('trck') and shower-like events ('cscd'), which is
then returned.

"""


from itertools import product

import numpy as np

from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils.dataProcParams import DataProcParams
from pisa.utils.events import Events
from pisa.utils.flavInt import NuFlavIntGroup, ALL_NUFLAVINTS
from pisa.utils.log import logging
from pisa.utils.PIDSpec import PIDSpec
from pisa.utils.profiler import profile


class hist(Stage):
    """Parameterised MC PID based on an input PISA events HDF5 file.

    Transforms an input map of the specified particle "signature" (aka ID) into
    a map of the track-like events ('trck') and a map of the shower-like events
    ('cscd').

    Parameters
    ----------
    params : ParamSet or sequence with which to instantiate a ParamSet

        Parameters which set everything besides the binning.

        If str, interpret as resource location and load params from resource.
        If dict, set contained params. Format expected is
            {'<param_name>': <Param object or passable to Param()>}

        Parameters required by this service are
            * pid_events : Events or filepath
                Events object or file path to HDF5 file containing events

            * pid_ver : string
                Version of PID to use (as defined for this
                detector/geometry/processing)

            * pid_remove_true_downgoing : bool
                Remove MC-true-downgoing events

            TODO(shivesh): Either `pid_spec` or `pid_spec_source` can be used
            to define the PID specifications. Implement this behaviour and for
            the case when `pid_spec` is used, do a check to confirm that the
            pid_events object has the matching PID spec metadata
            * pid_spec : PIDSpec
                PIDSpec object which specifies the PID specifications

            * pid_spec_source : filepath
                Resource for loading PID specifications

            * compute_error : bool
                Compute histogram errors

    input_binning : MultiDimBinning
        Arbitrary number of dimensions accepted. Contents of the input
        `pid_events` parameter defines the possible binning dimensions. Name(s)
        of given binning(s) must match to a reco variable in `pid_events`.

    output_binning : MultiDimBinning

    transforms_cache_depth : int >= 0
    outputs_cache_depth : int >= 0


    Input Names
    ----------
    The `inputs` container must include objects with `name` attributes:
        * 'nue_cc'
        * 'nue_nc'
        * 'nuebar_cc'
        * 'nuebar_nc'
        * 'numu_cc'
        * 'numu_nc'
        * 'numubar_cc'
        * 'numubar_nc'
        * 'nutau_cc'
        * 'nutau_nc'
        * 'nutaubar_cc'
        * 'nutaubar_nc'

    Output Names
    ----------
    The `outputs` container generated by this service will be objects with the
    following `name` attribute:
        * 'nue_cc_trck'
        * 'numu_cc_trck'
        * 'nutau_cc_trck'
        * 'nuall_nc_trck'
        * 'nue_cc_cscd'
        * 'numu_cc_cscd'
        * 'nutau_cc_cscd'
        * 'nuall_nc_cscd'

    Notes
    ----------
    This service takes in events from a **joined** PISA HDF5 file. The current
    implementation of this service requires that the nodes on these file match
    a certain flavour/interaction combination or "particle signature", which is
    `nue_cc, numu_cc, nutau_cc, nuall_nc`. Thus, only the HDF5 files with the
    naming convention
    ```
    events__*__joined_G_nue_cc+nuebar_cc_G_numu_cc+numubar_cc_G_nutau_cc+nutaubar_cc_G_nuall_nc+nuallbar_nc.hdf5
    ```
    should be used as input. The structure of the datafile is
    ```
    flavour / int_type / value
    ```
    where
        *  `flavour` is one of `nue, nue_bar, numu, numu_bar, nutau, nutau_bar`
        *  `int_type` is one of `cc` or `nc`
        *  `values` is one of
            * `pid` : the pid score per event
            * `reco_*param*` : the reco *param* of the event
            * `weighted_aeff`: the effective area weight per event
              (see Stage 3, Effective Area)

    For the 'joined' event files, the charged current components for the
    particle and antiparticle of a specific neutrino flavour are summed so
    that, for example, the data in the nodes `nue/cc` and `nue_bar/cc` both
    contain their own and each others events. The combined neutral current
    interaction for all neutrino flavours is also summed in the same way, so
    that any `nc` node contains the data of all neutrino flavours.

    Once the file has been read in, for each particle signature, the input PID
    specification is used to set a limit on the pid score above which the
    events are distinguished as being track-like `trck` and below as
    shower-like `cscd`.

    Next, a histogram in the input binning dimensions is created for all
    combinations of particle signature and pid channel and then normalised to
    one with respect to the particle signature to give the probability of an
    event lying in a particular bin. The input maps for each signature can then
    be transformed according to these probabilities to provide an output which
    will separate each signature into `trck` and `cscd` maps, and this is then
    returned.

    """
    def __init__(self, params, input_binning, output_binning, disk_cache=None,
                 transforms_cache_depth=20, outputs_cache_depth=20):
        # All of the following params (and no more) must be passed via
        # the `params` argument.
        expected_params = (
            'pid_events', 'pid_ver', 'pid_remove_true_downgoing', 'pid_spec',
            'pid_spec_source', 'compute_error'
        )

        # Define the names of objects that are required by this stage (objects
        # will have the attribute "name": i.e., obj.name)
        input_names = [str(fi) for fi in ALL_NUFLAVINTS]

        # Define the names of objects that get produced by this stage

        self.output_channels = ('trck', 'cscd')
        output_names = [self.suffix_channel(in_name, out_chan) for in_name,
                        out_chan in product(input_names, self.output_channels)]

        self.transforms_combined_flavints = tuple([
            NuFlavIntGroup(s)
            for s in ('nue_cc+nuebar_cc', 'numu_cc+numubar_cc',
                      'nutau_cc+nutaubar_cc', 'nuall_nc+nuallbar_nc')
        ])

        super(self.__class__, self).__init__(
            use_transforms=True,
            stage_name='pid',
            service_name='hist',
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            disk_cache=disk_cache,
            outputs_cache_depth=outputs_cache_depth,
            transforms_cache_depth=transforms_cache_depth,
            input_binning=input_binning,
            output_binning=output_binning
        )

    def suffix_channel(self, flavint, channel):
        return '%s_%s' % (flavint, channel)

    @profile
    def _compute_transforms(self):
        """Compute new PID transforms."""
        logging.info('Updating pid.hist PID histograms...')

        # TODO(shivesh): As of now, events do not have units as far as PISA
        # is concerned

        # Works only if either energy, coszen or azimuth is in input_binning
        bin_names = ('energy', 'coszen', 'azimuth')
        if set(self.input_binning.names).isdisjoint(bin_names):
            raise ValueError('Input binning must contain either one or a '
                             'combination of "energy", "coszen" or "azimuth" '
                             'dimensions.')

        # TODO: not handling rebinning in this stage or within Transform
        # objects; implement this! (and then this assert statement can go away)
        assert self.input_binning == self.output_binning

        events = Events(self.params['pid_events'].value)

        # TODO: in future, the events file will not have these combined
        # already, and it should be done here (or in a nominal transform,
        # etc.). See below about taking this step when we move to directly
        # using the I3-HDF5 files.
        events_file_combined_flavints = tuple([
            NuFlavIntGroup(s) for s in events.metadata['flavints_joined']
        ])

        # This check is still necessary to verify the assumptions below, which
        # require these flavints be combined in the events file
        assert set(self.transforms_combined_flavints) == \
                set(events_file_combined_flavints)

        # TODO: take events object as an input instead of as a param that
        # specifies a file? Or handle both cases?

        # TODO: include here the logic from the make_events_file.py script so
        # we can go directly from a (reasonably populated) icetray-converted
        # HDF5 file (or files) to a nominal transform, rather than having to
        # rely on the intermediate step of converting that HDF5 file (or files)
        # to a PISA HDF5 file that has additional column(s) in it to account
        # for the combinations of flavors, interaction types, and/or simulation
        # runs. Parameters can include which groupings to use to formulate an
        # output.

        data_proc_params = DataProcParams(
            detector=events.metadata['detector'],
            proc_ver=events.metadata['proc_ver']
        )

        if self.params['pid_remove_true_downgoing'].value:
            # TODO(shivesh): more options for cuts?
            cut_events = data_proc_params.applyCuts(
                events, cuts='true_upgoing_coszen'
            )
        else:
            cut_events = events

        pid_spec = PIDSpec(
            detector=events.metadata['detector'],
            geom=events.metadata['geom'],
            proc_ver=events.metadata['proc_ver'],
            pid_spec_ver=self.params['pid_ver'].value,
            pid_specs=self.params['pid_spec_source'].value
        )
        u_out_names = map(unicode, self.output_channels)
        if set(u_out_names) != set(pid_spec.get_signatures()):
            msg = 'PID criteria from `pid_spec` {0} does not match {1}'
            raise ValueError(msg.format(pid_spec.get_signatures(),
                                        u_out_names))

        # TODO: add importance weights, error computation

        logging.info("Separating events by PID...")
        var_names = ['reco_%s' % bin_name
                     for bin_name in self.output_binning.names]
        var_names += ['weighted_aeff']
        separated_events = pid_spec.applyPID(
            events=cut_events,
            return_fields=var_names
        )

        # These get used in innermost loop, so produce it just once here
        all_bin_edges = [edges.magnitude
                         for edges in self.output_binning.bin_edges]

        # Derive transforms by combining flavints that behave similarly, but
        # apply the derived transforms to the input flavints separately
        # (leaving combining these together to later)

        transforms = []
        for fi_group_str in self.transforms_combined_flavints:
            fi_group = NuFlavIntGroup(fi_group_str)
            rep_flavint = fi_group[0]

            # TODO(shivesh): errors
            # TODO(shivesh): total histo check?
            raw_histo = {}
            total_histo = np.zeros(self.output_binning.shape)
            for sig in self.output_channels:
                raw_histo[sig] = {}
                flav_sigdata = separated_events[rep_flavint][sig]
                reco_params = [flav_sigdata[vn] for vn in var_names]
                raw_histo[sig], _ = np.histogramdd(
                    sample=reco_params[:-1],
                    #weights=reco_params[-1],
                    bins=all_bin_edges
                )
                total_histo += raw_histo[sig]

            for sig in self.output_channels:
                with np.errstate(divide='ignore', invalid='ignore'):
                    xform_array = raw_histo[sig] / total_histo

                num_invalid = np.sum(~np.isfinite(xform_array))
                if num_invalid > 0:
                    logging.warn(
                        'Group "%s", PID signature "%s" has %d bins with no'
                        ' events (and hence the ability to separate events'
                        ' by PID cannot be ascertained). These are being'
                        ' masked off from any further computations.'
                        % (fi_group, sig, num_invalid)
                    )
                    xform_array = np.ma.masked_invalid(xform_array)

                # Double check that no NaN remain
                assert not np.any(np.isnan(xform_array))

                # Copy this transform to use for each flavint in the group
                for flavint in fi_group:
                    xform = BinnedTensorTransform(
                        input_names=str(flavint),
                        output_name=self.suffix_channel(flavint, sig),
                        input_binning=self.input_binning,
                        output_binning=self.output_binning,
                        xform_array=xform_array
                    )
                    transforms.append(xform)

        return TransformSet(transforms=transforms)

    def validate_params(self, params):
        # do some checks on the parameters

        # Check type of pid_events
        assert isinstance(params['pid_events'].value, (basestring, Events))

        # Check type of compute_error, pid_remove_true_downgoing
        assert isinstance(params['compute_error'].value, bool)
        assert isinstance(params['pid_remove_true_downgoing'].value, bool)

        # Check type of pid_ver, pid_spec_source
        assert isinstance(params['pid_ver'].value, basestring)
        assert isinstance(params['pid_spec_source'].value, basestring)

        # Check the groupings of the pid_events file
        events = Events(params['pid_events'].value)
        should_be_joined = sorted([
            NuFlavIntGroup('nue_cc + nuebar_cc'),
            NuFlavIntGroup('numu_cc + numubar_cc'),
            NuFlavIntGroup('nutau_cc + nutaubar_cc'),
            NuFlavIntGroup('nuall_nc + nuallbar_nc'),
        ])
        are_joined = sorted([
            NuFlavIntGroup(s)
            for s in events.metadata['flavints_joined']
        ])
        if are_joined != should_be_joined:
            raise ValueError('Events passed have %s joined groupings but'
                             ' it is required to have %s joined groupings.'
                             % (are_joined, should_be_joined))

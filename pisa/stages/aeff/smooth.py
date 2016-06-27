# authors: J.Lanfranchi/P.Eller
# date:   March 20, 2016


import copy
from itertools import product

import numpy as np
from scipy.interpolate import interp2d, splrep, splev

from pisa import ureg, Q_
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils.events import Events
from pisa.utils.flavInt import flavintGroupsFromString, NuFlavInt, NuFlavIntGroup, ALL_NUFLAVINTS, FlavIntData
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging, set_verbosity
from pisa.utils.profiler import profile

# TODO: the below logic does not generalize to muons, but probably should
# (rather than requiring an almost-identical version just for muons). For
# example, an input arg can dictate neutrino or muon, which then sets the
# input_names and output_names.

class smooth(Stage):
    """Example stage with maps as inputs and outputs, and no disk cache. E.g.,
    histogrammed oscillations stages will work like this.

    Parameters
    ----------
    params : ParamSet
        Must exclusively have parameters:

        aeff_weight_file
        livetime
        aeff_scale

    particles : string
        Must be one of 'neutrinos' or 'muons' (though only neutrinos are
        supported at this time).

    transform_groups : string
        Specifies which particles/interaction types to combine together in
        computing the transforms. (See Notes.)

    combine_grouped_flavints : bool
        Whether to combine the event-rate maps for the flavint groupings
        specified by `transform_groups`.

    input_binning : MultiDimBinning or convertible thereto
        Input binning is in true variables, with names prefixed by "true_".
        Each must match a corresponding dimension in `output_binning`.

    output_binning : MultiDimBinning or convertible thereto
        Output binning is in reconstructed variables, with names (traditionally
        in PISA but not necessarily) prefixed by "reco_". Each must match a
        corresponding dimension in `input_binning`.

    disk_cache
    transforms_cache_depth
    outputs_cache_depth

    Notes
    -----
    Example input names would be:
    See Conventions section in the documentation for more informaton on
    particle naming scheme in PISA. As an example

    """
    def __init__(self, params, particles, transform_groups,
                 combine_grouped_flavints, input_binning, output_binning, disk_cache=None,
                 transforms_cache_depth=20, outputs_cache_depth=20):
        self.events_hash = None
        """Hash of events file or Events object used"""

        assert particles in ['neutrinos', 'muons']
        self.particles = particles
        """Whether stage is instantiated to process neutrinos or muons"""

        self.transform_groups = flavintGroupsFromString(transform_groups)
        """Particle/interaction types to group for computing transforms"""

        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = (
            'aeff_weight_file', 'livetime', 'aeff_scale',
            'e_smooth_factor', 'cz_smooth_factor',
            'interp_kind'
        )

        # Define the names of objects expected in inputs and produced as
        # outputs
        if self.particles == 'neutrinos':
            input_names = (
                'nue', 'numu', 'nutau', 'nuebar', 'numubar', 'nutaubar'
            )
            if combine_grouped_flavints:
                output_names = tuple([str(g) for g in self.transform_groups])

            else:
                output_names = (
                    'nue_cc', 'numu_cc', 'nutau_cc', 'nuebar_cc', 'numubar_cc',
                    'nutaubar_cc',
                    'nue_nc', 'numu_nc', 'nutau_nc', 'nuebar_nc', 'numubar_nc',
                    'nutaubar_nc'
                )

        # Invoke the init method from the parent class, which does a lot of
        # work for you.
        super(self.__class__, self).__init__(
            use_transforms=True,
            stage_name='aeff',
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

        # Can do these now that binning has been set up in call to Stage's init
        self.include_attrs_for_hashes('particles')
        self.include_attrs_for_hashes('transform_groups')

    def load_events(self):
        evts = self.params.aeff_weight_file.value
        this_hash = hash_obj(evts)
        if this_hash == self.events_hash:
            return
        logging.debug('Extracting events from Events obj or file: %s' %evts)
        self.events = Events(evts)
        self.events_hash = this_hash


    def rearrange_dimensions(func):
        '''Possible decorator/wrapper to swap axes and undo swap.'''
        pass


    def slice_smooth(self, hist, hist_binning):
        '''Generate splines. Based on pisa.utils.slice_smooth_aeff'''
        # TODO add support for azimuth
        # TODO add support for error
        # TODO access hist dimensions by name

        # Swap hist dimensions to be [true_coszen, true_energy, true_azimuth]
        names = hist_binning.names
        cz_i = names.index('true_coszen')
        e_i = names.index('true_energy')
#        az_i = names.index('true_azimuth')

        if cz_i != 0:
            hist = np.swapaxes(hist, 0, cz_i)
#        if e_i != 1:
#            hist = np.swapaxes(hist, 1, e_i)

        # Load events file
        # NOTE This is also done in _compute_nominal_transforms
        # Is it possible to get the wrong metadata in any case?
        events = self.events

        # Copy metadata from events (to become metadata for smoothed transform)
        metadata = copy.deepcopy(events.metadata)
            # NOTE How do we know this is the metadata that corresponds to the
            # transform? Can events be assumed to have generated transform?

        # Get smooth factors from stage parameters
        e_smooth_factor = self.params.e_smooth_factor.value
        cz_smooth_factor = self.params.cz_smooth_factor.value

        # Adding smooth factors to metadata
        metadata.update(dict(e_smooth_factor=e_smooth_factor,
                             cz_smooth_factor=cz_smooth_factor))

        # Separate binning dimensions
        czbins = hist_binning.true_coszen
        ebins = hist_binning.true_energy

        czbin_midpoints = czbins.midpoints
        ebin_midpoints = ebins.midpoints

        # Add binning info to metadata
        metadata.update(dict(emin=ebins[0],
                             emax=ebins[-1],
                             n_ebins=len(ebins),
                             czmin=czbins[0],
                             czmax=czbins[-1],
                             n_czbins=len(ebins)))

        # Smooth cz-slices of hist
        smoothed_cz_slices = []
        #for index, czbin in enumerate(czbins):
            # NOTE AssertionError 400
            # NOTE __iter__ for OneDimBinning is commented out
            # and uncommenting it leads to issues with pickle (in hashing)
        for index in xrange(len(czbins)): # czbins.size instead?
            cz_slice = hist[index,:]
            # NOTE Go through cz slices without enumeration. There's probably a better way.
            # NOTE transform.xform_array is d-dimensional hist
            # but aeff_data[flavint] is 2-dimensional.
            # Add support for transforms with azimuth.

            # Remove extra dimensions
            s_cz_slice = np.squeeze(cz_slice)
            # TODO check if does anything --> It appears to

            # Fit spline to cz-slices
            cz_slice_spline = splrep(
                ebins.midpoints, s_cz_slice,
                k=3, s=e_smooth_factor
            )

            # Sample cz-spline over ebin midpoints
            smoothed_cz_slice = splev(ebins.midpoints, cz_slice_spline)

            # Assert that there are no nan or infinite values
            assert not np.any(np.isnan(smoothed_cz_slice) +
                              np.isinf(np.abs(smoothed_cz_slice)))

            smoothed_cz_slices.append(smoothed_cz_slice)

        # Convert list of slices to array
        smoothed_cz_slices = np.array(smoothed_cz_slices)

        # Iterate through e-slices
        smoothed_e_slices = []
        for e_slice_num in xrange(smoothed_cz_slices.shape[1]):
            e_slice = smoothed_cz_slices[:,e_slice_num]

            # Fit spline to e-slice
            e_slice_spline = splrep(
                czbin_midpoints, e_slice, w=None,
                k=3, s=cz_smooth_factor
            )

            # Evaluate spline at bin midpoints
            smoothed_aeff = splev(czbin_midpoints, e_slice_spline)

            smoothed_e_slices.append(smoothed_aeff)

        # Convert list of slices to array with cz as first index
        smoothed_hist = np.array(smoothed_e_slices).T

        # Clip out negative values
        smoothed_hist = smoothed_hist.clip(0)

        # Un-swap axes
        if cz_i != 0:
            smoothed_hist = np.swapaxes(smoothed_hist, 0, cz_i)
#        if e_i != 1:
#            smoothed_hist = np.swapaxes(smoothed_hist, 1, e_i)

        return smoothed_hist, metadata


    def smooth_transforms(self, transforms):
        new_transforms = []
        # TODO avoid redundant calculations in flav int groups
        # TODO put metadata somewhere

        v = [None]*len(self.transform_groups)
        done = dict(zip(self.transform_groups, v))

        for transform in transforms:
            t_group = NuFlavIntGroup(transform.output_name)
            if done[t_group] is not None:
                new_transform = BinnedTensorTransform(
                    input_names=transform.input_names,
                    output_name=transform.output_name,
                    input_binning=done[t_group].input_binning,
                    output_binning=done[t_group].output_binning,
                    xform_array=done[t_group].xform_array
                )
                new_transforms.append(new_transform)
            else:
                smooth_xform, metadata = self.slice_smooth(
                    hist=transform.xform_array,
                    hist_binning=transform.output_binning
                )

                new_transform = BinnedTensorTransform(
                    input_names=transform.input_names,
                    output_name=transform.output_name,
                    input_binning=transform.input_binning,
                    output_binning=transform.output_binning,
                    xform_array=smooth_xform
                )
                new_transforms.append(new_transform)
                assert done[t_group] is None
                done[t_group] = new_transform
        return TransformSet(transforms=new_transforms)


    def interpolate_hist(self, hist, hist_binning, new_binning):

        interp_kind = self.params.interp_kind.value

        interpolant = interp2d(
            x=hist_binning.true_energy.midpoints,
            y=hist_binning.true_coszen.midpoints,
            z=hist,
            kind=interp_kind, copy=True, fill_value=None)

        # NOTE Is this still "true" dimensions
        return interpolant(new_binning.true_energy.midpoints,
                           new_binning.true_coszen.midpoints)


    def interpolate_transforms(self, transforms, new_binning):
        new_transforms = []
        v = [None]*len(self.transform_groups)
        done = dict(zip(self.transform_groups, v))

        for transform in transforms:
            t_group = NuFlavIntGroup(transform.output_name)
            if done[t_group] is not None:
                new_transform = BinnedTensorTransform(
                    input_names=transform.input_names,
                    output_name=transform.output_name,
                    input_binning=done[t_group].input_binning,
                    output_binning=done[t_group].output_binning,
                    xform_array=done[t_group].xform_array
                )
                new_transforms.append(new_transform)
            else:
                interp_xform = self.interpolate_hist(
                    hist=transform.xform_array,
                    hist_binning=transform.output_binning,
                    new_binning=new_binning
                )

                new_transform = BinnedTensorTransform(
                    input_names=transform.input_names,
                    output_name=transform.output_name,
                    input_binning=new_binning,
                    output_binning=new_binning,
                    xform_array=interp_xform
                )
                new_transforms.append(new_transform)
                assert done[t_group] is None
                done[t_group] = new_transform

        return TransformSet(transforms=new_transforms)


    @profile
    def _compute_nominal_transforms(self):
        self.load_events()
        # Units must be the following for correctly converting a sum-of-
        # OneWeights-in-bin to an average effective area across the bin.
        comp_units = dict(true_energy='GeV', true_coszen=None,
                          true_azimuth='rad')

        # Only works if energy is in input_binning
        if 'true_energy' not in self.input_binning:
            raise ValueError('Input binning must contain "true_energy"'
                             ' dimension, but does not.')

        # coszen and azimuth are both optional, but no further dimensions are
        excess_dims = set(self.input_binning.names).difference(
            comp_units.keys()
        )
        if len(excess_dims) > 0:
            raise ValueError('Input binning has extra dimension(s): %s'
                             %sorted(excess_dims))

        # TODO: not handling rebinning in this stage or within Transform
        # objects; implement this! (and then this assert statement can go away)
        assert self.input_binning == self.output_binning

        # Select only the units in the input/output binning for conversion
        # (can't pass more than what's actually there)
        in_units = {dim: unit for dim, unit in comp_units.items()
                    if dim in self.input_binning}
        out_units = {dim: unit for dim, unit in comp_units.items()
                   if dim in self.output_binning}

        # These will be in the computational units
        input_binning = self.input_binning.to(**in_units)
        output_binning = self.output_binning.to(**out_units)

        # Account for "missing" dimension(s) (dimensions OneWeight expects for
        # computation of bin volume), and accommodate with a factor equal to
        # the full range. See IceCube wiki/documentation for OneWeight for
        # more info.
        missing_dims_vol = 1
        if 'true_azimuth' not in input_binning:
            missing_dims_vol *= 2*np.pi
        if 'true_coszen' not in input_binning:
            missing_dims_vol *= 2

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

        # Make binning for smoothing
        # TODO make this binning look like input? remove assertion
        assert input_binning.names == ['true_coszen', 'true_energy']
        s_ebins = OneDimBinning(name='true_energy', tex=r'E_\nu',
                                is_log=True, num_bins=39, domain=[1,80]*ureg.GeV)
        s_czbins = OneDimBinning(name='true_coszen',
                                 tex=r'\cos\theta_\nu', is_lin=True, num_bins=40,
                                 domain=[-1,1]*ureg(None))
        smoothing_binning = MultiDimBinning([s_czbins, s_ebins])

        # This gets used in innermost loop, so produce it just once here
        all_bin_edges = [dim.bin_edges.magnitude
                         for dim in smoothing_binning.dimensions]

        raw_transforms = []
        for flav_int_group in self.transform_groups:
            logging.debug("Working on %s effective areas" %flav_int_group)

            # Since events (as of now) are combined before placing in the file,
            # just need to extract the data for one "representative" flav/int.
            repr_flav_int = flav_int_group[0]

            # Extract the columns' data into a list for histogramming
            sample = [self.events[repr_flav_int][name]
                      for name in smoothing_binning.names]

            # Extract the weights
            weights = self.events[repr_flav_int]['weighted_aeff']

            aeff_transform, _ = np.histogramdd(
                sample=sample,
                weights=weights,
                bins=all_bin_edges
            )

            # Divide histogram by
            #   (energy bin width x coszen bin width x azimuth bin width)
            # volumes to convert from sums-of-OneWeights-in-bins to
            # effective areas. Note that volume correction factor for
            # missing dimensions is applied here.
            bin_volumes = smoothing_binning.bin_volumes(attach_units=False)
            aeff_transform /= (bin_volumes * missing_dims_vol)

            # Store copy of transform for each member of the group
            flav_names = [str(flav) for flav in flav_int_group.flavs()]
            for input_name in self.input_names:
                if input_name not in flav_names:
                    continue
                for output_name in self.output_names:
                    if output_name not in flav_int_group:
                        continue
                    xform = BinnedTensorTransform(
                        input_names=input_name,
                        output_name=output_name,
                        input_binning=smoothing_binning,
                        output_binning=smoothing_binning,
                        xform_array=aeff_transform,
                    )
                    raw_transforms.append(xform)

        nominal_transforms = TransformSet(transforms=raw_transforms)
        smooth_transforms = self.smooth_transforms(raw_transforms)
        interp_transforms = self.interpolate_transforms(smooth_transforms,
                new_binning=input_binning)

        debug = True
        if debug:
            import matplotlib.pyplot as plt
            from pisa.utils.plotter import plotter




        # PLOT MC, SMOOTHED, AND INTERP AEFF TRANSFORMS
        # TODO handle plotting and saving options properly
        #      nom, smt, and interp transforms will be passed out of stage
        make_plots = True
        if make_plots:
            from matplotlib.cm import Paired
            from pisa.utils.plotter import plotter
            plots = plotter()

            plots.init_fig()
            plots.plot_2d_array(nominal_transforms, n_rows=2, n_cols=6,
                                cmap=Paired)
            plots.dump('aeff_transforms')

            plots.init_fig()
            plots.plot_2d_array(smooth_transforms, n_rows=2, n_cols=6,
                                cmap=Paired)
            plots.dump('smoothed_aeff_transforms')

#            plots.init_fig()
#            plots.plot_2d_array(interp_transforms, n_rows=2, n_cols=6,
#                                cmap=Paired)
#            plots.dump('interp_aeff_transforms')


        # SMOOTHING COMPARISONS
        # Compare smoothed and original
        compare_results = True
        values = []
        if compare_results:
            smooth_vs_orig = []
            perc_diff_values = np.array([])
            for smooth, orig in zip(smooth_transforms, nominal_transforms):
                smooth_arr = smooth.xform_array
                orig_arr = orig.xform_array
                assert (smooth.input_names == orig.input_names), (
                    smooth.input_names, orig.input_names)
                assert smooth.output_name == smooth.output_name, (
                    smooth.output_name, orig.output_name)

                perc_diff = (smooth_arr - orig_arr) / orig_arr

                perc_diff_finite = perc_diff[np.isfinite(perc_diff)]
                # Some numbers
                mean = np.mean(perc_diff_finite)
                stddev = np.std(perc_diff_finite)
                mad = np.median(np.abs(perc_diff_finite - np.median(perc_diff_finite)))
                med = np.median(perc_diff_finite)
                min_val = min(perc_diff_finite)
                max_val = max(perc_diff_finite)

#                print np.mean(perc_diff[:,15:][np.isfinite(perc_diff[:,15:])])

                values.append([mean, stddev, mad, med, min_val, max_val])
                perc_diff_values = np.concatenate((perc_diff_values, perc_diff_finite.flatten()))

                # Deal with 0s in orig_arr. Set nans in perc_diff to 0
                perc_diff[perc_diff != perc_diff] = 0

                perc_diff = BinnedTensorTransform(
                    input_names=smooth.input_names,
                    output_name=smooth.output_name,
                    input_binning=smooth.input_binning,
                    output_binning=smooth.output_binning,
                    xform_array=perc_diff
                )
                smooth_vs_orig.append(perc_diff)

#                import matplotlib.pyplot as plt
#                plt.close('all')
#                plt.hist(perc_diff_finite, bins=300, histtype='step')
#                plt.xlabel('Fractional Difference Between Smoothed and MC Aeff Transforms')
#                plt.title('Distribution of Fractional Difference Values in\nSmoothed-vs-MC Aeff Tranform Comparison (%s to %s transform)'%(perc_diff.input_names, perc_diff.output_name))
#                plt.savefig('frac_diff_histogram.png')
#
#                raw_input()

            smooth_vs_orig = TransformSet(transforms=smooth_vs_orig)

#            import matplotlib.pyplot as plt
#            plt.close('all')
#            #perc_diff_values = perc_diff_values[(perc_diff_values>=-1) & (perc_diff_values<3)]
#            plt.hist(perc_diff_finite, bins=300, histtype='step')
#            plt.xlabel('Fractional Difference Between Smoothed and MC Aeff Transforms')
#            plt.title('Distribution of Fractional Difference Values in\nSmoothed-vs-MC Aeff Tranform Comparison')
#            plt.savefig('frac_diff_histogram.png')


            # Plotting
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            from matplotlib.cm import Paired
            from matplotlib.offsetbox import AnchoredText

            from pisa.utils.plotter import plotter

            plots = plotter(stamp='Comparison Between'+'\n'
                    'Smoothed and Original Aeff'+'\n'
                    r'Plotted value: $\frac{smoothed - orig}{orig}$')

            plots.init_fig()
            plots.log = False
            plots.plot_2d_array(smooth_vs_orig, n_rows=2, n_cols=6,
                    cmap=plt.get_cmap('bwr'), vmin=-1, vmax=1)
            # TODO add text boxes to correct axes
            for i, arr in enumerate(smooth_vs_orig):
                plt.subplot(2, 6, i+1)
                textstr = (#r'Values for $\frac{smooth - orig}{orig}$'+'\n'
                           'mean\t%f\n'
                           'std\t%f\n'
                           'mad\t%f\n'
                           'med\t%f\n'
                           'min\t%f\n'
                           'max\t%f') % tuple(values[i])
                textstr = textstr.expandtabs()
                a_text = AnchoredText(textstr, loc=1, frameon=False)
                plt.gca().add_artist(a_text)
            plots.dump('smooth_vs_orig')


            # Plot smoothed-vs-original cz slice comparison
            # Make cz slices into Maps
            # Plot with plot1darray
            from pisa.core.map import Map
            # assert smooth binning == nom binning
            # assert binning.names == [true_cz, true_e]
            i_cz = 30
            ebins = nominal_transforms.transforms[0].input_binning.true_energy
            czbin = nominal_transforms.transforms[0].input_binning.true_coszen[i_cz]
            binning = MultiDimBinning([czbin, ebins])
            nom_cz_slice = nominal_transforms.transforms[0].xform_array[i_cz]
            nom_cz_slice = nom_cz_slice.reshape((1,-1))
            nom_cz_slice = Map(name='nominal transform cz slice', hist=nom_cz_slice, binning=binning)
            smth_cz_slice = smooth_transforms.transforms[0].xform_array[i_cz]
            smth_cz_slice = smth_cz_slice.reshape((1,-1))
            smth_cz_slice = Map(name='smooth transform cz slice', hist=smth_cz_slice, binning=binning)
            plots = plotter(stamp='Aeff transform smoothing comparison')
            plots.init_fig()
            #plots.label = 'un-smoothed'
            plots.plot_1d_projection(nom_cz_slice, 'true_energy')
            #plots.label = 'smoothed'
            plots.plot_1d_projection(smth_cz_slice, 'true_energy')
            #plots.add_leg()
            plots.add_stamp('Smoothed-vs-original\n'
                            + str(czbin) + '\n'
                            + 'input_names: '+str(nominal_transforms.transforms[0].input_names)+'\n'
                            + 'output_name: '+str(nominal_transforms.transforms[0].output_name))
            plots.dump('cz_slice_comparison')


#            plt.close('all')
#            ax = plt.subplot(111)
#            plt.plot(nominal_transforms.transforms[0].input_binning.true_energy.bin_edges[:-1], 
#                    nominal_transforms.transforms[0].xform_array[20], 'bo-',
#                    label='original')
#            plt.plot(smooth_transforms.transforms[0].input_binning.true_energy.bin_edges[:-1], 
#                    smooth_transforms.transforms[0].xform_array[20], 'go-',
#                    label='smoothed')
#            ax.set_xscale('log', nonposx='clip')
#            ax.set_yscale('log', nonposy='clip')
#            plt.xlabel(repr(nominal_transforms.transforms[0].input_binning.true_energy))
#            plt.ylabel('Effective area transform values')
#            plt.title('Comparison of Original and Smoothed Coszen Slices (nuall_nc, cz_index=20)')
#            plt.legend(loc='best')
#            plt.savefig('cz_slice_comparison.png')


        # TODO Should there be one transform per input or per group?
        return interp_transforms


    @profile
    def _compute_transforms(self):
        """Compute new oscillation transforms"""
        # Read parameters in in the units used for computation
        aeff_scale = self.params.aeff_scale.value.m_as('dimensionless')
        livetime_s = self.params.livetime.value.m_as('sec')
        logging.trace('livetime = %s --> %s sec'
                      %(self.params.livetime.value, livetime_s))

        new_transforms = []
        for flav_int_group in self.transform_groups:
            repr_flav_int = flav_int_group[0]
            flav_names = [str(flav) for flav in flav_int_group.flavs()]
            aeff_transform = None
            for transform in self.nominal_transforms:
                if transform.input_names[0] in flav_names \
                        and transform.output_name in flav_int_group:
                    if aeff_transform is None:
                        aeff_transform = transform.xform_array * (aeff_scale *
                                                                  livetime_s)
                    new_xform = BinnedTensorTransform(
                        input_names=transform.input_names,
                        output_name=transform.output_name,
                        input_binning=transform.input_binning,
                        output_binning=transform.output_binning,
                        xform_array=aeff_transform
                    )
                    new_transforms.append(new_xform)

        return TransformSet(new_transforms)

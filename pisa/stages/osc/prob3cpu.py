# authors: T.Arlen, J.Lanfranchi, P.Eller
# date:   March 20, 2016

from itertools import izip, product

import numpy as np
import pint; ureg = pint.UnitRegistry()

from pisa.core.binning import MultiDimBinning
from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.resources.resources import find_resource
from pisa.stages.osc.prob3.BargerPropagator import BargerPropagator
from pisa.utils.log import logging
from pisa.utils.numerical import normQuant
from pisa.utils.profiler import profile


SIGFIGS = 12
"""Significant figures for determining if numbers and quantities normalised
(using normQuant) are equal. Make sure this is less than the numerical
precision that calculations are being performed in to have the desired effect
that "essentially equal" things evaluate to be equal."""

# Codes defined in Barger code
NUE_CODE, NUMU_CODE, NUTAU_CODE = 1, 2, 3

# Indices that are used for transform datastructs created here
NUE_IDX, NUMU_IDX, NUTAU_IDX = 0, 1, 2

INPUTS = ((NUE_IDX, NUE_CODE), (NUMU_IDX, NUMU_CODE))
OUTPUTS = ((NUE_IDX, NUE_CODE), (NUMU_IDX, NUMU_CODE), (NUTAU_IDX, NUTAU_CODE))

# More Barger definitions
K_NEUTRINOS = 1
K_ANTINEUTRINOS = -1


class prob3cpu(Stage):
    """Neutrino oscillations calculation via Prob3.

    Parameters
    ----------
    params : ParamSet
        All of the following param names (and no more) must be in `params`.
        Oversampling parameters:
            * osc_oversample_energy : int >= 1
            * osc_oversample_coszen : int >= 1
        Earth parameters:
            * earth_model : str (resource location with earth model file)
            * YeI : float (electron fraction, inner core)
            * YeM : float (electron fraction, mantle)
            * YeO : float (electron fraction, outer core)
        Detector parameters:
            * detector_depth : float >= 0
            * prop_height
        Oscillation parameters:
            * deltacp
            * deltam21
            * deltam31
            * theta12
            * theta13
            * theta23

    input_binning : MultiDimBinning
    output_binning : MultiDimBinning
    transforms_cache_depth : int >= 0
    outputs_cache_depth : int >= 0


    Input Names
    -----------
    The `inputs` container must include objects with `name` attributes:
      * 'nue'
      * 'numu'
      * 'nuebar'
      * 'numubar'


    Output Names
    ------------
    The `outputs` container generated by this service will be objects with the
    following `name` attribute:
      * 'nue'
      * 'numu'
      * 'nutau'
      * 'nuebar'
      * 'numubar'
      * 'nutaubar'

    """
    def __init__(self, params, input_binning, output_binning,
                 transforms_cache_depth=20, outputs_cache_depth=20):
        expected_params = (
            'earth_model', 'YeI', 'YeM', 'YeO',
            'detector_depth', 'prop_height',
            'deltacp', 'deltam21', 'deltam31',
            'theta12', 'theta13', 'theta23'
        )

        # Define the names of objects that are required by this stage (objects
        # will have the attribute "name": i.e., obj.name)
        input_names = (
            'nue', 'numu', 'nuebar', 'numubar'
        )

        # Define the names of objects that get produced by this stage
        output_names = (
            'nue', 'numu', 'nutau', 'nuebar', 'numubar', 'nutaubar'
        )

        assert input_binning == output_binning

        # Invoke the init method from the parent class (Stage), which does a
        # lot of work (caching, providing public interfaces, etc.)
        super(self.__class__, self).__init__(
            use_transforms=True,
            stage_name='osc',
            service_name='prob3cpu',
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            disk_cache=None,
            outputs_cache_depth=outputs_cache_depth,
            transforms_cache_depth=transforms_cache_depth,
            input_binning=input_binning,
            output_binning=output_binning
        )

        self.compute_binning_constants()

    def compute_binning_constants(self):
        # Only works if energy and coszen are in input_binning
        if 'energy' not in self.input_binning \
                or 'coszen' not in self.input_binning:
            raise ValueError('Input binning must contain both "energy" and'
                             ' "coszen" dimensions.')

        # Not handling rebinning (or oversampling)
        assert self.input_binning == self.output_binning

        # Get the energy/coszen (ONLY) weighted centers here, since these
        # are actually used in the oscillations computation. All other
        # dimensions are ignored. Since these won't change so long as the
        # binning doesn't change, attache these to self.
        self.ecz_binning = MultiDimBinning(
            self.input_binning.energy.to('GeV'),
            self.input_binning.coszen.to('dimensionless')
        )
        e_centers, cz_centers = self.ecz_binning.weighted_centers
        self.e_centers = e_centers.magnitude
        self.cz_centers = cz_centers.magnitude

        self.num_czbins = self.input_binning.coszen.num_bins

        self.e_dim_num = self.input_binning.names.index('energy')
        self.cz_dim_num = self.input_binning.names.index('coszen')

        self.extra_dim_nums = range(self.input_binning.num_dims)
        [self.extra_dim_nums.remove(d) for d in (self.e_dim_num,
                                                 self.cz_dim_num)]

    def create_transforms_datastructs(self):
        xform_shape = [3, 2] + list(self.input_binning.shape)
        part_xform = np.empty(xform_shape)
        antipart_xform = np.empty(xform_shape)
        return part_xform, antipart_xform

    def setup_barger_propagator(self):
        params = self.params
        # If already instantiated with same parameters, don't instantiate again
        if ( hasattr(self, 'barger_propagator')
            and hasattr(self, '_barger_earth_model')
            and hasattr(self, '_barger_detector_depth')
            and normQuant(self._barger_detector_depth, sigfigs=SIGFIGS)
                == normQuant(params.detector_depth.value, sigfigs=SIGFIGS)
            and params.earth_model.value == self._barger_earth_model):
            return

        # Some private variables to keep track of the state of the barger
        # propagator that has been instantiated, so if it is requested to be
        # instantiated again with equivalent parameters, this step can be
        # skipped (see checks above).
        self._barger_detector_depth = params.detector_depth.value.to('km')
        self._barger_earth_model = params.earth_model.value

        # TODO: can we pass kwargs to swig-ed C++ code?
        self.barger_propagator = BargerPropagator(
            find_resource(self._barger_earth_model),
            self._barger_detector_depth.magnitude
        )
        self.barger_propagator.UseMassEigenstates(False)

    @profile
    def _compute_transforms(self):
        """Compute oscillation transforms to apply to maps."""
        params = self.params

        # Run this every time, but it will only do something if relevant
        # parameters have changed
        self.setup_barger_propagator()

        # Read parameters in, convert to the units used internally for
        # computation, and then strip the units off. Note that this also
        # enforces compatible units (but does not sanity-check the numbers).
        theta12 = self.params.theta12.value.m_as('rad')
        theta13 = self.params.theta13.value.m_as('rad')
        theta23 = self.params.theta23.value.m_as('rad')
        deltam21 = self.params.deltam21.value.m_as('eV**2')
        deltam31 = self.params.deltam31.value.m_as('eV**2')
        deltacp = self.params.deltacp.value.m_as('rad')
        #energy_scale = self.params.energy_scale.value.m_as('')
        YeI = self.params.YeI.value.m_as('')
        YeO = self.params.YeO.value.m_as('')
        YeM = self.params.YeM.value.m_as('')
        prop_height = self.params.prop_height.value.m_as('km')

        logging.info("Defining osc_prob_dict from BargerPropagator...")
        #tprofile.info("start oscillation calculation")

        # Set to true, since we are using sin^2(theta) variables
        kSquared = True
        sin2th12Sq = np.sin(theta12)**2
        sin2th13Sq = np.sin(theta13)**2
        sin2th23Sq = np.sin(theta23)**2

        # In BargerPropagator code, it takes the "atmospheric
        # mass difference"-the nearest two mass differences, so
        # that it takes as input deltam31 for IMH and deltam32
        # for NMH
        m_atm = deltam31 if deltam31 < 0.0 else (deltam31 - deltam21)

        # `:` slices for all binning dimensions (except energy and coszen,
        # which get populated with their integer indices inside the for loop).
        # Used to duplicate the oscillation E,CZ result to all other dimensions
        # present in the binning
        indexer = [slice(None)]*self.input_binning.num_dims

        part_xform, antipart_xform = self.create_transforms_datastructs()
        for i, (energy, coszen) in enumerate(product(self.e_centers,
                                                     self.cz_centers)):
            # Construct indices in energy and coszen, and populate to bin
            # indexer
            e_idx = i // self.num_czbins
            cz_idx = i - e_idx*self.num_czbins
            indexer[self.e_dim_num] = e_idx
            indexer[self.cz_dim_num] = cz_idx

            scaled_energy = energy #* energy_scale

            # Neutrinos
            mns_args = [
                sin2th12Sq, sin2th13Sq, sin2th23Sq,
                deltam21, m_atm, deltacp, scaled_energy,
                kSquared
            ]
            self.barger_propagator.SetMNS(*(mns_args + [K_NEUTRINOS]))
            self.barger_propagator.DefinePath(coszen, prop_height, YeI,YeO,YeM)
            self.barger_propagator.propagate(K_NEUTRINOS)

            for (in_idx, in_code), (out_idx, out_code) in product(INPUTS,
                                                                  OUTPUTS):
                full_indexer = tuple([out_idx, in_idx] + indexer)
                part_xform[full_indexer] = \
                        self.barger_propagator.GetProb(in_code, out_code)

            # Antineutrinos
            self.barger_propagator.SetMNS(*(mns_args + [K_ANTINEUTRINOS]))
            self.barger_propagator.DefinePath(coszen, prop_height,
                                              YeI, YeO, YeM)
            self.barger_propagator.propagate(K_ANTINEUTRINOS)

            for (in_idx, in_code), (out_idx, out_code) in product(INPUTS,
                                                                  OUTPUTS):
                full_indexer = tuple([out_idx, in_idx] + indexer)
                antipart_xform[full_indexer] = \
                        self.barger_propagator.GetProb(in_code, out_code)

        # Slice up the transform arrays into views to populate each transform
        transforms = []
        for out_idx, output_name in enumerate(self.output_names):
            out_idx = out_idx % 3
            if 'bar' not in output_name:
                xform = part_xform[out_idx, :, ...]
                input_names = self.input_names[0:2]
            else:
                xform = antipart_xform[out_idx, :, ...]
                input_names = self.input_names[2:4]

            transforms.append(
                BinnedTensorTransform(
                    input_names=input_names,
                    output_name=output_name,
                    input_binning=self.input_binning,
                    output_binning=self.output_binning,
                    xform_array=xform
                )
            )

        return TransformSet(transforms=transforms)

    def validate_params(self, params):
        pass


from collections import Sequence
import os # Needed to get the absolute path to mosc3.cu and mosc.cu or else nvcc fails

import numpy as np
import pycuda.compiler
import pycuda.driver as cuda
import pycuda.autoinit

from pisa import ureg, Q_
from pisa.core.binning import MultiDimBinning
from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.stages.osc.grid_propagator.GridPropagator import GridPropagator
from pisa.utils.log import logging
from pisa.utils.comparisons import normQuant
from pisa.utils.profiler import line_profile, profile
from pisa.utils.resources import find_resource


class prob3gpu(Stage):
    """Neutrino oscillations calculation via Prob3 with a GPU.

    Parameters
    ----------
    params : ParamSet
        All of the following param names (and no more) must be in `params`.
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
    debug_mode : bool
    gpu_id: If running on a system with multiple GPUs, it will choose
            the one with gpu_id. Otherwise, defaults to 0

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
    # Define CUDA kernel
    KERNEL_TEMPLATE = '''
    #include "mosc.cu"
    #include "mosc3.cu"
    #include "utils.h"
    #include "constants.h"
    #include <stdio.h>

    /* If we use some kind of oversampling then we need the original
     * binning with nebins and nczbins. In the current version we use a
     * fine binning for the first stages and do not need any
     * oversampling.
     */
    __global__ void propagateGrid(fType* d_smooth_maps,
                                  fType d_dm[3][3], fType d_mix[3][3][2],
                                  const fType* const d_ecen_fine,
                                  const fType* const d_czcen_fine,
                                  const int nebins_fine, const int nczbins_fine,
                                  const int nebins, const int nczbins,
                                  const int maxLayers,
                                  const int* const d_numberOfLayers,
                                  const fType* const d_densityInLayer,
                                  const fType* const d_distanceInLayer) {
      const int2 thread_2D_pos = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
                                           blockIdx.y*blockDim.y + threadIdx.y);

      // ensure we don't access memory outside of bounds!
      if (thread_2D_pos.x >= nczbins_fine || thread_2D_pos.y >= nebins_fine)
        return;
      const int thread_1D_pos = thread_2D_pos.y*nczbins_fine + thread_2D_pos.x;

      int eidx = thread_2D_pos.y;
      int czidx = thread_2D_pos.x;

      int kNuBar;
      //if (threadIdx.z == 0)
      //  kNuBar = 1;
      if (blockIdx.z == 0)
        kNuBar = 1;
      else
        kNuBar=-1;

      bool kUseMassEstates = false;

      fType TransitionMatrix[3][3][2];
      fType TransitionProduct[3][3][2];
      fType TransitionTemp[3][3][2];
      fType RawInputPsi[3][2];
      fType OutputPsi[3][2];
      fType Probability[3][3];

      clear_complex_matrix(TransitionMatrix);
      clear_complex_matrix(TransitionProduct);
      clear_complex_matrix(TransitionTemp);
      clear_probabilities(Probability);

      int layers = *(d_numberOfLayers + czidx);

      fType energy = d_ecen_fine[eidx];
      //fType coszen = d_czcen_fine[czidx];
      for (int i=0; i<layers; i++) {
        fType density = *(d_densityInLayer + czidx*maxLayers + i);
        fType distance = *(d_distanceInLayer + czidx*maxLayers + i);

        get_transition_matrix(kNuBar,
                              energy,
                              density,
                              distance,
                              TransitionMatrix,
                              0.0,
                              d_mix,
                              d_dm);

        if (i==0) {
          copy_complex_matrix(TransitionMatrix, TransitionProduct);
        } else {
          clear_complex_matrix(TransitionTemp);
          multiply_complex_matrix(TransitionMatrix, TransitionProduct, TransitionTemp);
          copy_complex_matrix(TransitionTemp, TransitionProduct);
        }
      } // end layer loop

      // loop on neutrino types, and compute probability for neutrino i:
      // We actually don't care about nutau -> anything since the flux there is zero!
      for (unsigned i=0; i<2; i++) {
        for (unsigned j = 0; j < 3; j++) {
          RawInputPsi[j][0] = 0.0;
          RawInputPsi[j][1] = 0.0;
        }

        if (kUseMassEstates)
          convert_from_mass_eigenstate(i+1, kNuBar, RawInputPsi, d_mix);
        else
          RawInputPsi[i][0] = 1.0;

        multiply_complex_matvec(TransitionProduct, RawInputPsi, OutputPsi);
        Probability[i][0] += OutputPsi[0][0]*OutputPsi[0][0] + OutputPsi[0][1]*OutputPsi[0][1];
        Probability[i][1] += OutputPsi[1][0]*OutputPsi[1][0] + OutputPsi[1][1]*OutputPsi[1][1];
        Probability[i][2] += OutputPsi[2][0]*OutputPsi[2][0] + OutputPsi[2][1]*OutputPsi[2][1];
      } // end of neutrino loop

      int efctr = nebins_fine/nebins;
      int czfctr = nczbins_fine/nczbins;
      int eidx_smooth = eidx/efctr;
      int czidx_smooth = czidx/czfctr;
      fType scale = fType(efctr*czfctr);
      for (int i=0;i<2;i++) {
        int iMap = 0;
        if (kNuBar == 1)
          iMap = i*3;
        else
          iMap = 6 + i*3;

        for (unsigned to_nu=0; to_nu<3; to_nu++) {
          int k = (iMap+to_nu);
          fType prob = Probability[i][to_nu];
          atomicAdd_custom((d_smooth_maps + k*nczbins*nebins +
              eidx_smooth*nczbins + czidx_smooth), prob/scale);
        }
      }
    }
    '''

    def __init__(self, params, input_binning, output_binning,
                 error_method=None, transforms_cache_depth=20,
                 outputs_cache_depth=20, debug_mode=None, gpu_id=None):
        expected_params = (
            'earth_model', 'YeI', 'YeM', 'YeO',
            'detector_depth', 'prop_height',
            'deltacp', 'deltam21', 'deltam31',
            'theta12', 'theta13', 'theta23'
        )

        # Define the names of objects that are required by this stage (objects
        # will have the attribute `name`: i.e., obj.name)
        input_names = (
            'nue', 'numu', 'nuebar', 'numubar'
        )

        # Define the names of objects that get produced by this stage
        output_names = (
            'nue', 'numu', 'nutau', 'nuebar', 'numubar', 'nutaubar'
        )

        self.gpu_id = gpu_id

        #if self.gpu_id is None:
        #    self.gpu_id = cuda.Context.get_device()
        #self.context = cuda.Device(self.gpu_id).make_context()

        # Invoke the init method from the parent class (Stage), which does a
        # lot of work (caching, providing public interfaces, etc.)
        super(self.__class__, self).__init__(
            use_transforms=True,
            stage_name='osc',
            service_name='prob3gpu',
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            error_method=error_method,
            disk_cache=None,
            outputs_cache_depth=outputs_cache_depth,
            transforms_cache_depth=transforms_cache_depth,
            input_binning=input_binning,
            output_binning=output_binning,
            debug_mode=debug_mode
        )
        self.compute_binning_constants()
        self.initialize_kernel()

    def compute_binning_constants(self):
        # Only works if energy and coszen are in input_binning
        if 'true_energy' not in self.input_binning \
                or 'true_coszen' not in self.input_binning:
            raise ValueError('Input binning must contain both "true_energy"'
                             ' and "true_coszen" dimensions.')

        # TODO: Check for single precision
        self.FTYPE = np.float64

        self.fine_energy = self.input_binning.true_energy
        self.fine_cz = self.input_binning.true_coszen
        # Get the energy/coszen (ONLY) weighted centers here, since these
        # are actually used in the oscillations computation. All other
        # dimensions are ignored. Since these won't change so long as the
        # binning doesn't change, attache these to self.
        self.ecz_binning = MultiDimBinning([
            self.fine_energy.to('GeV'),
            self.fine_cz.to('dimensionless')
        ])

        ecen_fine, czcen_fine = self.ecz_binning.weighted_centers
        self.ecen_fine = ecen_fine.magnitude
        self.czcen_fine = czcen_fine.magnitude

        self.e_dim_num = self.input_binning.names.index('true_energy')
        self.cz_dim_num = self.input_binning.names.index('true_coszen')

        self.extra_dim_nums = range(self.input_binning.num_dims)
        [self.extra_dim_nums.remove(d) for d in (self.e_dim_num,
                                                 self.cz_dim_num)]

    def create_transforms_datastructs(self):
        xform_shape = [3, 2] + list(self.output_binning.shape)
        nu_xform = np.empty(xform_shape)
        antinu_xform = np.empty(xform_shape)
        return nu_xform, antinu_xform

    def _derive_nominal_transforms_hash(self):
        """No nominal transforms implemented for this service."""
        return None

    @profile
    def _compute_transforms(self):
        """Compute oscillation transforms using grid_propagator GPU code."""

        # Read parameters in, convert to the units used internally for
        # computation, and then strip the units off. Note that this also
        # enforces compatible units (but does not sanity-check the numbers).
        theta12 = self.params.theta12.m_as('rad')
        theta13 = self.params.theta13.m_as('rad')
        theta23 = self.params.theta23.m_as('rad')
        deltam21 = self.params.deltam21.m_as('eV**2')
        deltam31 = self.params.deltam31.m_as('eV**2')
        deltacp = self.params.deltacp.m_as('rad')
        YeI = self.params.YeI.m_as('dimensionless')
        YeO = self.params.YeO.m_as('dimensionless')
        YeM = self.params.YeM.m_as('dimensionless')
        prop_height = self.params.prop_height.m_as('km')

        sin2th12Sq = np.sin(theta12)**2
        sin2th13Sq = np.sin(theta13)**2
        sin2th23Sq = np.sin(theta23)**2

        mAtm = deltam31 if deltam31 < 0.0 else (deltam31 - deltam21)

        # TODO: check this, and possibly remove comment if unnecessary (should
        # be no # copy-paste even in comments unless absolutely necessary)

        # Comment GridPropagator.cpp::SetMNS()
        # "For the inverted Hierarchy, adjust the input
        # by the solar mixing (should be positive)
        # to feed the core libraries the correct value of m32."
        #if mAtm < 0.0: mAtm -= deltam21;

        self.grid_prop.SetMNS(deltam21, mAtm, sin2th12Sq, sin2th13Sq,
                              sin2th23Sq, deltacp)
        self.grid_prop.SetEarthDensityParams(prop_height, YeI, YeO, YeM)
        self.prepare_device_arrays()

        dm_mat = np.zeros((3,3), dtype=self.FTYPE)
        self.grid_prop.Get_dm_mat(dm_mat)
        mix_mat = np.zeros((3,3,2), dtype=self.FTYPE)
        self.grid_prop.Get_mix_mat(mix_mat)

        logging.trace('dm_mat: \n %s' %str(dm_mat))
        logging.trace('mix[re]: \n %s' %str(mix_mat[:,:,0]))

        d_dm_mat = cuda.mem_alloc(dm_mat.nbytes)
        d_mix_mat = cuda.mem_alloc(mix_mat.nbytes)
        cuda.memcpy_htod(d_dm_mat, dm_mat)
        cuda.memcpy_htod(d_mix_mat, mix_mat)

        nebins_fine = np.uint32(len(self.ecen_fine))
        nczbins_fine = np.uint32(len(self.czcen_fine))

        # Earlier versions had self.ecen_fine*energy_scale but energy_scale is
        # not used anymore
        cuda.memcpy_htod(self.d_ecen_fine, self.ecen_fine)

        smooth_maps = np.zeros((nczbins_fine*nebins_fine*12), dtype=self.FTYPE)
        d_smooth_maps = cuda.mem_alloc(smooth_maps.nbytes)
        cuda.memcpy_htod(d_smooth_maps, smooth_maps)

        block_size = (16,16,1)
        grid_size = (nczbins_fine/block_size[0] + 1, nebins_fine/block_size[1] + 1, 2)
        self.propGrid(d_smooth_maps,
                      d_dm_mat, d_mix_mat,
                      self.d_ecen_fine, self.d_czcen_fine,
                      nebins_fine, nczbins_fine,
                      nebins_fine, nczbins_fine,
                      np.uint32(self.maxLayers),
                      self.d_numLayers, self.d_densityInLayer,
                      self.d_distanceInLayer,
                      block=block_size, grid=grid_size)
                      #shared=16384)

        cuda.memcpy_dtoh(smooth_maps, d_smooth_maps)
        #self.free_device_memory()
        #d_smooth_maps.free()
        #d_dm_mat.free()
        #d_mix_mat.free()
        ## Finish this context cleanly since we don't need it anymore
        #self.context.pop()

        # Return TransformSet
        smooth_maps = np.reshape(smooth_maps, (12, nebins_fine, nczbins_fine))
        # Slice up the transform arrays into views to populate each transform
        transforms = []
        x_shape = [2] + list(self.input_binning.shape)
        for out_idx, output_name in enumerate(self.output_names):
            xform = np.empty(x_shape)
            if out_idx < 3:
                # Neutrinos
                xform[0] = smooth_maps[out_idx]
                xform[1] = smooth_maps[out_idx+3]
                input_names = self.input_names[0:2]
            else:
                # Antineutrinos
                xform[0] = smooth_maps[out_idx+3]
                xform[1] = smooth_maps[out_idx+6]
                input_names = self.input_names[2:4]
            transforms.append(
                BinnedTensorTransform(input_names=input_names,
                output_name=output_name,
                input_binning=self.input_binning,
                output_binning=self.output_binning,
                xform_array=xform)
            )
        return TransformSet(transforms=transforms)

    def validate_params(self, params):
        pass

    def initialize_kernel(self):
        """Initialize 1) the grid_propagator class, 2) the device arrays that
        will be passed to the `propagateGrid()` kernel, and 3) the kernel
        module.

        """
        self.grid_prop = GridPropagator(
            find_resource(self.params.earth_model.value),
            self.FTYPE(self.czcen_fine),
            self.FTYPE(self.params.detector_depth.m_as('km'))
        )

        # Path relative to `resources` directory
        file_path = find_resource('../stages/osc/grid_propagator/mosc3.cu')
        dir_path = os.path.dirname(file_path)
        include_path = os.path.abspath(dir_path)
        logging.debug('  pycuda INC PATH: %s' %include_path)
        logging.debug('  pycuda FLAGS: %s' %pycuda.compiler.DEFAULT_NVCC_FLAGS)
        self.module = pycuda.compiler.SourceModule(
            self.KERNEL_TEMPLATE, include_dirs=[include_path], keep=True
        )
        self.propGrid = self.module.get_function('propagateGrid')

    def prepare_device_arrays(self):
        self.maxLayers = self.grid_prop.GetMaxLayers()
        nczbins_fine = len(self.czcen_fine)
        numLayers = np.zeros(nczbins_fine, dtype=np.int32)
        densityInLayer = np.zeros((nczbins_fine*self.maxLayers),
                                  dtype=self.FTYPE)
        distanceInLayer = np.zeros((nczbins_fine*self.maxLayers),
                                   dtype=self.FTYPE)

        self.grid_prop.GetNumberOfLayers(numLayers)
        self.grid_prop.GetDensityInLayer(densityInLayer)
        self.grid_prop.GetDistanceInLayer(distanceInLayer)

        # Copy all these earth info arrays to device:
        self.d_numLayers = cuda.mem_alloc(numLayers.nbytes)
        self.d_densityInLayer = cuda.mem_alloc(densityInLayer.nbytes)
        self.d_distanceInLayer = cuda.mem_alloc(distanceInLayer.nbytes)
        cuda.memcpy_htod(self.d_numLayers, numLayers)
        cuda.memcpy_htod(self.d_densityInLayer, densityInLayer)
        cuda.memcpy_htod(self.d_distanceInLayer, distanceInLayer)

        self.d_ecen_fine = cuda.mem_alloc(self.ecen_fine.nbytes)
        self.d_czcen_fine = cuda.mem_alloc(self.czcen_fine.nbytes)
        cuda.memcpy_htod(self.d_ecen_fine, self.ecen_fine)
        cuda.memcpy_htod(self.d_czcen_fine, self.czcen_fine)

    def free_device_memory(self):
        self.d_numLayers.free()
        self.d_densityInLayer.free()
        self.d_distanceInLayer.free()

        self.d_ecen_fine.free()
        self.d_czcen_fine.free()

    def validate_params(self, params):
        pass

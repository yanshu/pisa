"""
Calculation of Earth layers and electron densities.
"""


from __future__ import division

import numpy as np
try:
    import numba
except ImportError:
    numba = None

from pisa import FTYPE
from pisa.utils.fileio import from_file
from pisa.utils.log import logging, set_verbosity
from pisa.utils.profiler import profile


__all__ = ['extCalcLayers', 'Layers']


if numba is None:
    class jit(object):
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args):
            return args[0]
    signature = None
else:
    jit = numba.jit
    ftype = numba.typeof(FTYPE(1))
    signature = (
        numba.types.Tuple((numba.int32[:], ftype[:], ftype[:]))(
            ftype[:],
            ftype,
            ftype,
            ftype,
            numba.int32,
            ftype,
            ftype[:],
            ftype[:],
            ftype[:],
            ftype,
            ftype[:],
            ftype[:]
        )
    )


@jit(signature, nopython=True, nogil=True, cache=True)
def extCalcLayers(
        cz,
        r_detector,
        prop_height,
        detector_depth,
        max_layers,
        min_detector_depth,
        rhos,
        YeFrac,
        YeOuterRadius,
        default_elec_frac,
        coszen_limit,
        radii):
    """Layer density/distance calculator for each coszen specified.

    Accelerated with Numba if present.

    Parameters
    ----------
    cz
    r_detector
    prop_height
    detector_depth
    max_layers
    min_detector_depth
    rhos
    YeFrac
    YeOuterRadius
    default_elec_frac
    coszen_limit

    Returns
    -------
    n_layers : int number of layers
    density : array of densities, flattened from (cz, max_layers)
    distance : array of distances per layer, flattened from (cz, max_layers)

    """
    # Something to store the final results in
    shape = (np.int64(len(cz)), np.int64(max_layers))
    n_layers = np.zeros(shape[0], dtype=np.int32)
    distance = np.zeros(shape=shape, dtype=FTYPE)
    density = np.zeros(shape=shape, dtype=FTYPE)

    # Loop over all CZ values
    for k, coszen in enumerate(cz):
        tot_earth_len = -2 * coszen * r_detector

        # To store results
        traverse_rhos = np.zeros(max_layers, dtype=FTYPE)
        traverse_dist = np.zeros(max_layers, dtype=FTYPE)
        traverse_electron_frac = np.zeros(max_layers, dtype=FTYPE)

        # Above horizon
        if coszen >= 0:
            kappa = (detector_depth + prop_height)/r_detector
            path_len = (
                r_detector * np.sqrt(coszen**2 - 1 + (1 + kappa)**2)
                - r_detector * coszen
            )

            # Path through the air:
            kappa = detector_depth / r_detector
            lam = (
                coszen + np.sqrt(coszen**2 - 1 + (1 + kappa) * (1 + kappa))
            )
            lam *= r_detector
            path_thru_atm = (
                prop_height * (prop_height + 2*detector_depth + 2*r_detector)
                / (path_len + lam)
            )
            path_thru_outerlayer = path_len - path_thru_atm
            traverse_rhos[0] = 0.0
            traverse_dist[0] = path_thru_atm
            traverse_electron_frac[0] = default_elec_frac

            # In that case the neutrino passes through some earth (?)
            layers = 1
            if detector_depth > min_detector_depth:
                traverse_rhos[1] = rhos[0]
                traverse_dist[1] = path_thru_outerlayer
                traverse_electron_frac[1] = YeFrac[-1]
                layers += 1

        # Below horizon
        else:
            path_len = (
                np.sqrt((r_detector + prop_height + detector_depth)**2
                        - r_detector**2 * (1 - coszen**2))
                - r_detector * coszen
            )

            # Path through air (that's down from production height in the
            # atmosphere?)
            traverse_rhos[0] = 0
            traverse_dist[0] = (
                prop_height * (prop_height + detector_depth + 2*r_detector)
                / path_len
            )

            # TODO: Why default here?
            traverse_electron_frac[0] = default_elec_frac
            i_trav = 1

            # Path through the final layer above the detector (if necessary)
            # NOTE: outer top layer is assumed to be the same as the next layer
            # inward.
            if (detector_depth > min_detector_depth):
                traverse_rhos[1] = rhos[0]
                traverse_dist[1] = path_len - tot_earth_len - traverse_dist[0]
                traverse_electron_frac[1] = YeFrac[-1]
                i_trav += 1

            # See how many layers we will pass
            layers = 0
            for val in coszen_limit:
                if coszen < val:
                    layers += 1

            # The zeroth layer is the air!
            # ... and the first layer is the top layer (if detector is not on
            # surface)
            for i in range(layers):
                # this is the density
                traverse_rhos[i+i_trav] = rhos[i]
                # TODO: Why default? is this air with density 0 and electron
                # fraction just doesn't matter?
                traverse_electron_frac[i+i_trav] = default_elec_frac
                for rad_i in range(len(YeOuterRadius)):
                    # TODO: why 1.001 here?
                    if radii[i] < (YeOuterRadius[rad_i] * 1.001):
                        traverse_electron_frac[i+i_trav] = YeFrac[rad_i]
                        break

                # now calculate the distance travele in layer
                c2 = coszen**2
                R2 = r_detector**2
                s1 = radii[i]**2 - R2*(1 -c2)
                s2 = radii[i+1]**2 - R2*(1 -c2)
                cross_this = 2. * np.sqrt(s1)
                if i < layers - 1:
                    cross_next = 2. * np.sqrt(s2)
                    traverse_dist[i+i_trav]  =  0.5 * (cross_this - cross_next)
                else:
                    traverse_dist[i+i_trav]  =  cross_this

                #assumes azimuthal symmetry
                if 0 < i and i < layers:
                    index = 2 * layers - i + i_trav - 1
                    traverse_rhos[index] = traverse_rhos[i+i_trav-1]
                    traverse_dist[index] = traverse_dist[i+i_trav-1]
                    traverse_electron_frac[index] = traverse_electron_frac[i+i_trav-1]

            # That is now the total
            layers = 2 * layers + i_trav - 1

        n_layers[k] = np.int32(layers)
        density[k] = traverse_rhos * traverse_electron_frac
        distance[k] = traverse_dist

    return n_layers, density.ravel(), distance.ravel()


class Layers(object):
    """
    Calculate the path through earth for a given layer model with densities
    (PREM), the electron fractions (Ye) and an array of coszen values

    Parameters
    ----------
    prem_file : str
        path to PREM file containing layer radii and densities as white space
        separated txt

    detector_depth : float
        depth of detector underground in km

    prop_height : float
        the production height of the neutrinos in the atmosphere in km (?)


    Attributes:
    ----------
    max_layers : int
            maximum number of layers (this is important for the shape of the
            output! if less than maximumm number of layers are crossed, it's
            filled up with 0s

    n_layers : 1d int array of length len(cz)
            number of layers crossed for every CZ value

    density : 1d float array of length (max_layers * len(cz))
            containing density values and filled up with 0s otherwise

    distance : 1d float array of length (max_layers * len(cz))
            containing distance values and filled up with 0s otherwise

    """
    def __init__(self, prem_file, detector_depth=1., prop_height=2.):
        # Load earth model
        prem = from_file(prem_file, as_array=True)
        self.rhos = prem[...,1][::-1].astype(FTYPE)
        self.radii = prem[...,0][::-1].astype(FTYPE)
        r_earth = prem[-1][0]
        self.r_detector = r_earth - detector_depth
        self.default_elec_frac = 0.5
        self.prop_height = prop_height
        self.detector_depth = detector_depth
        self.min_detector_depth = 1.0e-3 # <-- Why? // [km] so min is ~ 1 m
        n_prem = len(self.radii) - 1
        self.max_layers = 2 * n_prem + 1

        # Change outermost radius to a bit underground, where the detector
        if self.detector_depth >= self.min_detector_depth:
            self.radii[0] -= detector_depth
            self.max_layers += 1

        self.computeMinLengthToLayers()

    def setElecFrac(self, YeI, YeO, YeM):
        """
        Parameters
        ----------
        YeI, YeO, YeM : scalars
            Three electron fractions (Ye), where I=inner core, O=outer core,
            and M=mantle

        """
        self.YeFrac = np.array([YeI, YeO, YeM], dtype=FTYPE)

        # TODO: these numbers are just hard coded for some reason...?
        self.YeOuterRadius = np.array([1121.5, 3480.0, self.r_detector],
                                      dtype=FTYPE)

    def computeMinLengthToLayers(self):
        # Compute which layer is tangeted at which angle
        coszen_limit = []
        # First element of self.radii is largest radius!
        for i,rad in enumerate(self.radii):
            # Using a cosine threshold instead!
            if i == 0:
                x = 0
            else:
                x = - np.sqrt(1 - (rad**2 / self.r_detector**2))
            coszen_limit.append(x)
        self.coszen_limit = np.array(coszen_limit, dtype=FTYPE)

    def calcLayers(self, cz):
        """

        Parameters
        ----------
        cz : 1d float array
            Array of coszen values

        """
        # run external function
        self._n_layers, self._density, self._distance = extCalcLayers(
            cz=cz,
            r_detector=self.r_detector,
            prop_height=self.prop_height,
            detector_depth=self.detector_depth,
            max_layers=self.max_layers,
            min_detector_depth=self.min_detector_depth,
            rhos=self.rhos,
            YeFrac=self.YeFrac,
            YeOuterRadius=self.YeOuterRadius,
            default_elec_frac=self.default_elec_frac,
            coszen_limit=self.coszen_limit,
            radii=self.radii
        )

    @property
    def n_layers(self):
        return self._n_layers

    @property
    def density(self):
        return self._density

    @property
    def distance(self):
        return self._distance


def test_Layers():
    layer = Layers('osc/PREM_4layer.dat')
    layer.setElecFrac( 0.4656, 0.4656, 0.4957)
    cz = np.linspace(-1, 1, 1e5, dtype=FTYPE)
    layer.calcLayers(cz);
    logging.info('n_layers = %s' %layer.n_layers)
    logging.info('density  = %s' %layer.density)
    logging.info('distance = %s' %layer.distance)
    logging.info('<< PASS : test_Layers >>')


if __name__ == '__main__':
    set_verbosity(3)
    test_Layers()

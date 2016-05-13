
import numpy as np

from pisa.core.binning import MultiDimBinning
from pisa.core.map import Map, MapSet
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging, set_verbosity

# TODO: Include option for propagating/not propagating errors, so that while
# e.g. a minimizer runs to match templates to "data," the overhead is not
# incurred. But this then requires -- if the user does want errors -- for a
# final iteration after a match has been found where all outputs are
# re-computed but with the propagate_errors option set to True. The output
# caches must all then "miss" so that actual output including error is
# computed.

# TODO: use a generic container, *not* a MapSet to store sets of maps for
# inputs and outputs (in fact I think there should be no MapSet object at all,
# so we can trnsparently handle e.g. events alongside maps where one is a
# sideband object for the other in a given stage, but which is which should be
# irrelevant).

# TODO: Add Sequence capabilities to TransformSet (e.g. it'd be nice to have at
# least append, extend, ...)
class TransformSet(object):
    """
    Parameters
    ----------
    transforms
    name

    Properties
    ----------
    hash
    input_names
    name
    nonvolatile_hash
    num_inputs
    num_outputs
    output_names
    transforms

    Methods
    -------
    apply
    check_predecessor_compat
    check_successor_compat

    """
    def __init__(self, transforms, name=None):
        self.transforms = transforms
        self.name = name
        self.hash = None

    def __iter__(self):
        return iter(self.transforms)

    #@property
    #def hash(self):
    #    xform_hashes = [x.hash for x in transforms]
    #    if all([(h != None) for h in xform_hashes]):
    #        if all([(h == xform_hashes[0]) for h in xform_hashes]):
    #            return xform_hashes[0]
    #        return hash_obj(tuple(xform_hashes))
    #    return None

    # TODO: implement a non-volatile hash that includes source code hash in
    # addition to self.hash from the contained transforms
    @property
    def nonvolatile_hash(self):
        return None

    @property
    def input_names(self):
        input_names = set()
        [input_names.update(x.input_names) for x in self]
        return tuple(sorted(input_names))

    @property
    def num_inputs(self):
        return len(self.input_names)

    @property
    def num_outputs(self):
        return len(self)

    @property
    def output_names(self):
        output_names = []
        [output_names.append(x.output_name) for x in self]
        return tuple(output_names)

    def apply(self, inputs):
        """Apply each transform to `inputs`; return computed outputs.

        Parameters
        -----------
        inputs : sequence of objects

        Returns
        -------
        outputs : container with computed outputs (no sideband objects)

        """
        outputs = [xform.apply(inputs) for xform in self]

        # TODO: what to set for name, tex, ... ?
        return MapSet(maps=outputs)


class Transform(object):
    """
    Transform.

    Parameters
    ----------

    Properties
    ----------
    hash
    input_names
    num_inputs
    output_name
    tex

    """
    # Attributes that __setattr__ will allow setting
    _slots = ('_input_names', '_output_name', '_tex', '_hash', '_hash')
    # Attributes that should be retrieved to fully describe state
    _state_attrs = ('input_names', 'output_name', 'tex', 'hash')

    def __init__(self, input_names, output_name, input_binning=None,
                 output_binning=None, tex=None, hash=None):
        if isinstance(input_names, basestring):
            input_names = [input_names]
        assert isinstance(output_name, basestring)
        self._input_names = input_names
        self._output_name = output_name
        if input_binning is not None:
            self._input_binning = MultiDimBinning(input_binning)
        else:
            self._input_binning = None
        if output_binning is not None:
            self._output_binning = MultiDimBinning(output_binning)
        else:
            self._output_binning = None
        self._tex = tex if tex is not None else output_name
        self._hash = hash

    @property
    def hash(self):
        return self._hash

    @property
    def input_names(self):
        return self._input_names

    @property
    def num_inputs(self):
        return len(self.input_names)

    @property
    def output_name(self):
        return self._output_name

    @property
    def input_binning(self):
        return self._input_binning

    @property
    def output_binning(self):
        return self._output_binning

    def apply(self, inputs):
        output = self._apply(inputs)
        # TODO: tex, etc.?
        output.name = self.output_name
        return output

    def _apply(self, inputs):
        """Override this method in subclasses"""
        raise NotImplementedError('Override this method in subclasses')

    def validate_transform(xform):
        """Override this method in subclasses"""
        raise NotImplementedError('Override this method in subclasses')

    def validate_input(self, inputs):
        """Override this method in subclasses"""
        raise NotImplementedError('Override this method in subclasses')


# TODO: integrate uncertainties module in with this so that a transform can
#       introduce (augment) error of an input Map for producing a more accurate
#       estimate of the error in the output map.
class BinnedTensorTransform(Transform):
    """

    Parameters
    ----------
    input_names : string or sequence thereof
        Names of maps expected in the input MapSet. See Notes for how multiple
        inputs are ito be indexed in the `xform_array`.

    output_name : string
        Name of Map that will be generated.

    input_binning : MultiDimBinning
        Binning required for inputs maps.

    output_binning : MultiDimBinning
        Binning used for generated output maps.

    xform_array : numpy ndarray
        The actual transform's numerical values. Shape must be in accordance
        with `input_binning` and `output_binning` to accommodate the type
        of transform being implemented. See Notes for more detail on allowed
        shapes.

    tex : string
        TeX label for e.g. automatic plot labelling.

    hash : immutable object (usually integer)
        A hash value the user can attach


    Properties
    ----------
    hash
    source_hash


    Notes
    -----
    For an input map that is M_ebins x N_czbins, the transform must either be
    2-dimensional of shape (M x N) or 4-dimensional of shape (M x N x M x N).
    The latter case can be thought of as a 2-dimensional (M x N) array, each
    element of which is a 2-dimensional (M x N) array, and is currently used
    for the reconstruction stage's convolution kernels where there is one
    (M_ebins x N_czbins)-size kernel for each (energy, coszen) bin.

    There can be extra objects in `inputs` that are not used by this transform
    ("sideband" objects, which are simply ignored here). If multiple input maps
    are used by the transform, they are combined via
    numpy.stack((map0, map1, ... ), axis=0) I.e., the first dimension of the
    input sent to the transform has a length the same number of input maps
    requested by the transform.

    """
    _slots = tuple(list(Transform._slots) +
                   ['_input_binning', '_output_binning', '_xform_array'])

    _state_attrs = tuple(list(Transform._state_attrs) +
                         ['input_binning', 'output_binning', 'xform_array'])

    def __init__(self, input_names, output_name, input_binning, output_binning,
                 xform_array, tex=None, hash=None):
        super(self.__class__, self).__init__(input_names=input_names,
                                             output_name=output_name,
                                             input_binning=input_binning,
                                             output_binning=output_binning,
                                             tex=tex,
                                             hash=hash)
        self.xform_array = xform_array

    @property
    def xform_array(self):
        return self._xform_array

    @xform_array.setter
    def xform_array(self, x):
        self.validate_transform(self.input_binning, self.output_binning, x)
        self._xform_array = x

    # TODO: validate transform...
    def validate_transform(self, input_binning, output_binning, xform_array):
        """Superficial validation that the transform being set is reasonable.

        As of now, only checks shape.

        Expected transform shape is:
            (
             <input binning n_ebins>,
             <input binning n_czbins>,
             {if num_inputs > 1: <num_inputs>,}
             <output binning n_ebins>,
             <output binning n_czbins>,
             {if num_outputs > 1: <num_outputs>}
            )

        """
        #in_dim = [] if self.num_inputs == 1 else [self.num_inputs]
        #out_dim = [] if self.num_outputs == 1 else [self.num_outputs]
        #assert xform_array.shape == tuple(list(input_binning.shape) + in_dim +
        #                                  list(output_binning.shape) + out_dim)
        pass

    def validate_input(self, inputs):
        for input_name in self.input_names:
            assert input_name in inputs, \
                    'Input "%s" expected but not present.' % input_name
            assert inputs[input_name].binning == self.input_binning

    # TODO: make _apply work with multiple inputs (i.e., concatenate
    # these into a higher-dimensional array) and make logic for applying
    # element-by-element multiply and tensordot generalize to any dimension
    # given the (concatenated) input dimension and the dimension of the
    # transform kernel

    def _apply(self, inputs):
        """Apply transforms to input maps to compute output maps.

        Parameters
        ----------
        inputs : Mapping
            Container class that must contain (at least) the maps to be
            transformed.

        Returns
        -------
        output : Map
            Result of applying the transform to the input map(s).

        """
        self.validate_input(inputs)
        if self.num_inputs > 1:
            input_array = np.stack([inputs[name].hist
                                    for name in self.input_names], axis=-1)
        else:
            input_array = inputs[self.input_names[0]].hist

        if self.xform_array.shape == input_array.shape:
            output = input_array * self.xform_array

        # TODO: Check that
        #   len(xform.shape) == 2*len(input_array.shape)
        # and then check that
        #   xform.shape == (input_array.shape, input_array.shape) (roughly)
        # and then apply tensordot appropriately for this generic case...
        elif len(self.xform_array.shape) == 2*len(input_array.shape):
            output = np.tensordot(input_array, self.xform_array,
                                  axes=([0,1], [0,1]))
        else:
            raise NotImplementedError(
                'Unhandled shapes for input(s) "%s": %s and'
                ' transform "%s": %s.'
                %(', '.join(self.input_names), input_array.shape, self.name,
                  self.xform_array.shape)
            )

        if self.num_inputs > 1:
            output = np.sum(output, axis=-1)

        # TODO: do rebinning here? (aggregate, truncate, and/or
        # concatenate 0's?)

        output = Map(name=self.output_name,
                     hist=output,
                     binning=self.output_binning)

        return output


def test_BinnedTensorTransform():
    import pint; ureg = pint.UnitRegistry()

    from pisa.core.map import Map, MapSet
    from pisa.core.binning import MultiDimBinning

    binning = MultiDimBinning([
        dict(name='energy', is_log=True, domain=(1,80)*ureg.GeV, num_bins=10),
        dict(name='coszen', is_lin=True, domain=(-1,0), num_bins=5)
    ])

    nue_map = Map(name='nue',
                  binning=binning,
                  hist=np.random.random(binning.shape))
    nue_map.set_poisson_errors()
    numu_map = Map(name='numu',
                  binning=binning,
                  hist=np.random.random(binning.shape))
    numu_map.set_poisson_errors()
    inputs = MapSet(
        name='inputs',
        maps=[nue_map, numu_map],
    )

    xform0 = BinnedTensorTransform(
        input_names='nue',
        output_name='nue',
        input_binning=binning,
        output_binning=binning,
        xform_array=2*np.ones(binning.shape)
    )

    xform1 = BinnedTensorTransform(
        input_names=['numu'],
        output_name='numu',
        input_binning=binning,
        output_binning=binning,
        xform_array=3*np.ones(binning.shape)
    )

    xform2 = BinnedTensorTransform(
        input_names=['nue', 'numu'],
        output_name='nue_numu',
        input_binning=binning,
        output_binning=binning,
        xform_array=np.stack([2*np.ones(binning.shape),
                              3*np.ones(binning.shape)], axis=-1)
    )

    xforms = TransformSet(
        name='scaling',
        transforms=[xform0, xform1, xform2],
    )

    outputs = xforms.apply(inputs)


if __name__ == "__main__":
    test_BinnedTensorTransform()

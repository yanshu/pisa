# Authors: J.L.Lanfranchi/P.Eller
# Date   : 2016-05-13

from collections import OrderedDict, Sequence
from copy import copy, deepcopy
from functools import wraps
import importlib
import inspect
import sys

import numpy as np
from uncertainties import unumpy as unp

from pisa.core.binning import MultiDimBinning
from pisa.core.map import Map, MapSet, rebin
from pisa.utils.comparisons import normQuant, recursiveEquality
from pisa.utils.hash import hash_obj
from pisa.utils import jsons
from pisa.utils.log import logging, set_verbosity
from pisa.utils.profiler import line_profile, profile


HASH_SIGFIGS = 12


__all__ = ['TransformSet', 'Transform', 'BinnedTensorTransform']


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

# TODO: dtype arg to BinnedTensorTransform (or a global def like HASH_SIGFIGS?)
# to allow for single precision and hence any associated speedups

## TODO: numba implementation of BinnedTensorTransform faster or not?
#import numba
#@numba.jit("float64[:,:] (float64[:,:], float64[:,:,:,:])",
#           nopython=True, nogil=True, cache=True)
#def map2d_kernel4d(x, xform):
#    I = x.shape[0]
#    J = x.shape[1]
#    out = np.empty((I,J))
#    for i1 in range(I):
#        for j1 in range(J):
#            out[i1,j1] = 0.0
#            for i0 in range(I):
#                for j0 in range(J):
#                    out[i1,j1] += x[i1,j1] * xform[i0,j0,i1,j1]
#    return out


# TODO: Add Sequence capabilities to TransformSet (e.g. it'd be nice to have at
# least append, extend, ...)
TRANS_SET_SLOTS = ('name', 'hash', 'transforms', '__iter__',
                   'nonvolatile_hash', 'input_names', 'num_inputs',
                   'output_names', 'apply', '__getattribute__',
                   'to_json', 'from_json')
class TransformSet(object):
    """
    Set of Transform objects.

    Parameters
    ----------
    transforms
    name

    """
    def __init__(self, transforms, name=None, hash=None):
        self._transforms = transforms
        self.name = name
        self.hash = hash

    @property
    def _serializable_state(self):
        state = OrderedDict()
        state['transforms'] = [
            (t.__module__, t.__class__.__name__, t._serializable_state)
            for t in self
        ]
        state['name'] = self.name
        return state

    @property
    def _hashable_state(self):
        state = OrderedDict()
        state['transforms'] = [
            (t.__module__, t.__class__.__name__, t._hashable_state)
            for t in self
        ]
        state['name'] = self.name
        return state

    def __iter__(self):
        return iter(self._transforms)

    def __len__(self):
        return len(self._transforms)

    def __eq__(self, other):
        if not isinstance(other, TransformSet):
            return False
        return recursiveEquality(self._hashable_state, other._hashable_state)

    def to_json(self, filename, **kwargs):
        """Serialize the state to a JSON file that can be instantiated as a new
        object later.

        Parameters
        ----------
        filename : str
            Filename; must be either a relative or absolute path (*not
            interpreted as a PISA resource specification*)
        **kwargs
            Further keyword args are sent to `pisa.utils.jsons.to_json()`

        See Also
        --------
        from_json : Intantiate new object from the file written by this method
        pisa.utils.jsons.to_json

        """
        jsons.to_json(self._serializable_state, filename=filename, **kwargs)

    @classmethod
    def from_json(cls, resource):
        """Instantiate a new TransformSet object from a JSON file.

        Parameters
        ----------
        resource : str
            A PISA resource specification (see pisa.utils.resources)

        See Also
        --------
        to_json
        pisa.utils.jsons.to_json

        """
        state = jsons.from_json(resource)
        transforms = []
        for module, classname, transform_state in state['transforms']:
            clsmembers = inspect.getmembers(sys.modules[__name__],
                                            inspect.isclass)
            # First try to get a class within this module/namespace
            classes = [c[1] for c in clsmembers if c[0] == classname]
            if len(classes) > 0:
                class_ = classes[0]
            else:
                # Otherwise try to import the module recorded in the JSON file
                module = importlib.import_module(module)
                # And then get the class
                class_ = getattr(module, classsname)
            transforms.append(class_(**transform_state))
        state['transforms'] = transforms
        # State is a dict, so instantiate with double-asterisk syntax
        return cls(**state)

    @property
    def hash(self):
        hashes = self.hashes
        if len(hashes) > 0:
            if all([(h is not None and h == hashes[0]) for h in hashes]):
                return hashes[0]
            if all([(h is not None) for h in hashes]):
                return hash_obj(hashes)
        return None

    @hash.setter
    def hash(self, val):
        if val is not None:
            [setattr(xform, 'hash', val) for xform in self]

    @property
    def hashes(self):
        return [t.hash for t in self]

    # TODO: implement a non-volatile hash that includes source code hash in
    # addition to self.hash from the contained transforms
    #@property
    #def nonvolatile_hash(self):
    #    return hash_obj((self.source_code_hash,

    @property
    def input_names(self):
        input_names = set()
        [input_names.update(x.input_names) for x in self]
        return sorted(input_names)

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
        return output_names

    def get(self, input_names, output_name):
        """Retrieve the transform that maps `input_names` to `output_name`.
        Note that `input_names` can be a single name.

        Parameters
        ----------
        input_names : string or sequence thereof
        output_name : string

        Returns
        -------
        transform : Transform object

        """
        if isinstance(input_names, basestring):
            input_names = [input_names]
        for transform in self:
            if set(input_names) == set(transform.input_names) \
               and output_name == transform.output_name:
                return transform

    @profile
    def apply(self, inputs):
        """Apply each transform to `inputs`; return computed outputs.

        Parameters
        -----------
        inputs : sequence of objects

        Returns
        -------
        outputs : container with computed outputs (no sideband objects)

        """
        output_names = []
        outputs = []

        # If any outputs have the same name, add them together to form a single
        # output for that name
        for xform in self:
            output = xform.apply(inputs)
            name = output.name
            try:
                idx = output_names.index(name)
                outputs[idx] = outputs[idx] + output
                outputs[idx].name = name
            except ValueError:
                outputs.append(output)
                output_names.append(name)

        # Automatically attach a sensible hash (this may be overwritten, but
        # the below should be a reasonable hash in most cases)
        if inputs.hash is None or self.hash is None:
            hash = None
        else:
            hash = hash_obj((inputs.hash, self.hash))

        # TODO: what to set for map set's name, tex, etc. ?
        return MapSet(maps=outputs, hash=hash)

    def __getattr__(self, attr):
        if attr in TRANS_SET_SLOTS:
            return super(TransformSet, self).__getattribute__(attr)
        # TODO: return maps based upon name?
        #if attr in
        return TransformSet([getattr(t, attr) for t in self], name=self.name)

    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)):
            return self._transforms[idx]
        # TODO: search for transform by output name?
        #if isinstance(idx, basestring):
        raise IndexError('`idx` %s not found in map set.' %idx)


class Transform(object):
    """
    Base class for building a transform.

    """
    # TODO: get rid of the tex attribute, or add back in the name attribute?

    # Attributes that __setattr__ will allow setting
    _slots = ('_input_names', '_output_name', '_tex', '_hash', '_hash',
              'error_method')
    # Attributes that should be retrieved to fully describe state
    _state_attrs = ('input_names', 'output_name', 'tex', 'hash', 'error_method')

    def __init__(self, input_names, output_name, input_binning=None,
                 output_binning=None, tex=None, hash=None, error_method=None):
        # Convert to sequence of single string if a single string was passed
        # for uniform interfacing
        if isinstance(input_names, basestring):
            input_names = [input_names]
        self._input_names = input_names

        assert isinstance(output_name, basestring)
        self._output_name = output_name

        if input_binning is not None:
            if not isinstance(input_binning, MultiDimBinning):
                if isinstance(input_binning, Sequence):
                    input_binning = MultiDimBinning(input_binning)
                else:
                    input_binning = MultiDimBinning(**input_binning)
            self._input_binning = input_binning
        else:
            self._input_binning = None

        if output_binning is not None:
            if not isinstance(output_binning, MultiDimBinning):
                if isinstance(output_binning, Sequence):
                    output_binning = MultiDimBinning(output_binning)
                else:
                    output_binning = MultiDimBinning(**output_binning)
            self._output_binning = output_binning
        else:
            self._output_binning = None

        self._tex = tex if tex is not None else output_name
        self._hash = hash
        if bool(error_method) == False:
            self._error_method = None
        else:
            self._error_method = error_method

    @property
    def _serializable_state(self):
        state = OrderedDict()
        state['input_names'] = self.input_names
        state['output_name'] = self.output_name
        state['input_binning'] = self.input_binning._serializable_state
        state['output_binning'] = self.output_binning._serializable_state
        state['tex'] = self.tex
        state['error_method'] = self.error_method
        state['hash'] = self.hash
        return state

    @property
    def _hashable_state(self):
        state = OrderedDict()
        state['input_names'] = self.input_names
        state['output_name'] = self.output_name
        state['input_binning'] = self.input_binning._hashable_state
        state['output_binning'] = self.output_binning._hashable_state
        state['tex'] = self.tex
        state['error_method'] = self.error_method
        return state

    def to_json(self, filename, **kwargs):
        """Serialize the state to a JSON file that can be instantiated as a new
        object later.

        Parameters
        ----------
        filename : str
            Filename; must be either a relative or absolute path (*not
            interpreted as a PISA resource specification*)
        **kwargs
            Further keyword args are sent to `pisa.utils.jsons.to_json()`

        See Also
        --------
        from_json : Intantiate new object from the file written by this method
        pisa.utils.jsons.to_json

        """
        jsons.to_json(self._serializable_state, filename=filename, **kwargs)

    @classmethod
    def from_json(cls, resource):
        """Instantiate a new Map object from a JSON file.

        The format of the JSON is generated by the `Map.to_json` method, which
        converts a Map object to basic types and then numpy arrays are
        converted in a call to `pisa.utils.jsons.to_json`.

        Parameters
        ----------
        resource : str
            A PISA resource specification (see pisa.utils.resources)

        See Also
        --------
        to_json
        pisa.utils.jsons.to_json

        """
        state = jsons.from_json(resource)
        # State is a dict for Map, so instantiate with double-asterisk syntax
        return cls(**state)

    @property
    def hash(self):
        return self._hash

    @hash.setter
    def hash(self, val):
        self._hash = val

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

    @property
    def tex(self):
        return self._tex

    @property
    def error_method(self):
        return self._error_method

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
    BinnedTensorTransform implementing common transforms used in PISA:
        1) Element-by-element multiplicaton. E.g., to transform an N x M map,
           the transform array is also M x N.

        2) Smearing kernels, e.g. to characterize resolutions. For a transform
           indended to be applied to a 2D map of dimensions (M x N), for each
           of the M x N bins of the input map, there is one M x N kernel that
           represents how that bin gets "smeared out" to the entire output map.
           The effect of each input bin is added up to yield in the end a single
           M x N output map.


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

    sum_inputs : bool
        If true, add inputs together if multiple are specified. Otherwise,
        stack input maps along the first dimension (increases dimensionality of
        the input array by one).

    tex : string
        TeX label for e.g. automatic plot labelling.

    hash : immutable object (usually integer)
        A hash value the user can attach

    error_method : None, bool, or string
        Define the method for error propaation on unumpy arrays

    output_name : string

    input_binning : MultiDimBinning

    output_binning : MultiDimBinning

    xform_array : numpy ndarray

    error_array : None or numpy ndarray

    params_hash : immutable object (usually integer)


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
    are used by the transform, if `sum_inputs` is Ture, they are summed;
    otherwise, they are combined via numpy.stack((map0, map1, ... ), axis=0)
    I.e., the first dimension of the input sent to the transform has a length
    the same number of input maps requested by the transform.

    """
    _slots = tuple(list(Transform._slots) +
                   ['_input_binning', '_output_binning', '_xform_array'])

    _state_attrs = tuple(list(Transform._state_attrs) +
                         ['input_binning', 'output_binning', 'xform_array'])

    def __init__(self, input_names, output_name, input_binning, output_binning,
                 xform_array, sum_inputs=False, error_array=None, tex=None,
                 error_method=None, hash=None):
        super(BinnedTensorTransform, self).__init__(
            input_names=input_names, output_name=output_name,
            input_binning=input_binning, output_binning=output_binning,
            tex=tex, hash=hash, error_method=error_method
        )
        self.xform_array = xform_array
        self.sum_inputs = sum_inputs
        if error_array is not None:
            self.set_errors(error_array)

    @property
    def _serializable_state(self):
        state = super(BinnedTensorTransform, self)._serializable_state
        state['xform_array'] = unp.nominal_values(self.xform_array)
        state['error_array'] = unp.std_devs(self.xform_array)
        return state

    @property
    def _hashable_state(self):
        state = super(BinnedTensorTransform, self)._hashable_state
        state['xform_array'] = normQuant(unp.nominal_values(self.xform_array),
                                         sigfigs=HASH_SIGFIGS)
        state['error_array'] = normQuant(unp.std_devs(self.xform_array),
                                         sigfigs=HASH_SIGFIGS)
        return state

    @profile
    def set_errors(self, error_array):
        """Manually define the error with an array the same shape as the
        contained histogram. Can also remove errors by passing None.

        Parameters
        ----------
        error_array : None or ndarray
            Standard deviations to apply to `self.xform_array`; shapes must
            match. If None is passed, any errors present are removed, making
            `self.xform_array` a bare numpy array.

        """
        if error_array is None:
            super(Transform, self).__setattr__(
                '_xform_array', unp.nominal_values(self._xform_array)
            )
            return
        assert error_array.shape == self.xform_array.shape
        super(BinnedTensorTransform, self).__setattr__(
            '_xform_array', unp.uarray(self._xform_array, 
                    np.ascontiguousarray(error_array))
        )

    @property
    def xform_array(self):
        return self._xform_array

    @xform_array.setter
    def xform_array(self, x):
        self.validate_transform(self.input_binning, self.output_binning, x)
        self._xform_array = np.ascontiguousarray(x)

    def _new_obj(original_function):
        """Decorator to deepcopy unaltered states into new object"""
        @wraps(original_function)
        def new_function(self, *args, **kwargs):
            new_state = OrderedDict()
            state_updates = original_function(self, *args, **kwargs)
            for slot in self._state_attrs:
                if state_updates.has_key(slot):
                    new_state[slot] = state_updates[slot]
                else:
                    new_state[slot] = deepcopy(getattr(self, slot))
            return self.__class__(**new_state)
        return new_function

    @_new_obj
    def __abs__(self):
        return dict(xform_array=np.abs(self.xform_array))

    @_new_obj
    def __add__(self, other):
        if isinstance(other, BinnedTensorTransform):
            return dict(xform_array=self.xform_array + other.xform_array)
        return dict(xform_array=self.xform_array + other)

    @_new_obj
    def __div__(self, other):
        if isinstance(other, BinnedTensorTransform):
            return dict(xform_array=self.xform_array / other.xform_array)
        return dict(xform_array=self.xform_array / other)

    def __eq__(self, other):
        if not isinstance(other, BinnedTensorTransform):
            return False
        return recursiveEquality(self._hashable_state, other._hashable_state)

    @_new_obj
    def __mul__(self, other):
        if isinstance(other, BinnedTensorTransform):
            return dict(xform_array=self.xform_array * other.xform_array)
        return dict(xform_array=self.xform_array * other)

    @_new_obj
    def __ne__(self, other):
        return not self == other

    @_new_obj
    def __neg__(self, other):
        return dict(xform_array=-self.xform_array)

    @_new_obj
    def __pow__(self, other):
        if isinstance(other, BinnedTensorTransform):
            return dict(xform_array=self.xform_array ** other.xform_array)
        return dict(xform_array=self.xform_array ** other)

    @_new_obj
    def __radd__(self, other):
        return self + other

    @_new_obj
    def __rdiv__(self, other):
        if isinstance(other, BinnedTensorTransform):
            return dict(xform_array=other.xform_array / self.xform_array)
        return dict(xform_array=other / self.xform_array)

    @_new_obj
    def __rmul__(self, other):
        if isinstance(other, BinnedTensorTransform):
            return dict(xform_array=other.xform_array * self.xform_array)
        return dict(xform_array=other * self.xform_array)

    @_new_obj
    def __rsub__(self, other):
        if isinstance(other, BinnedTensorTransform):
            return dict(xform_array=other.xform_array - self.xform_array)
        return dict(xform_array=other - self.xform_array)

    @_new_obj
    def sqrt(self):
        return dict(xform_array=np.sqrt(self.xform_array))

    @_new_obj
    def __sub__(self, other):
        if isinstance(other, BinnedTensorTransform):
            return dict(xform_array=self.xform_array - other.xform_array)
        return dict(xform_array=self.xform_array - other)

    # TODO: validate transform...
    @profile
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

    @profile
    def validate_input(self, inputs):
        for input_name in self.input_names:
            assert input_name in inputs, \
                    'Input "%s" expected; got: %s.' \
                    % (input_name, inputs.names)

    # TODO: make _apply work with multiple inputs (i.e., concatenate
    # these into a higher-dimensional array) and make logic for applying
    # element-by-element multiply and tensordot generalize to any dimension
    # given the (concatenated) input dimension and the dimension of the
    # transform kernel

    @profile
    def _apply(self, inputs):
        """Apply transforms to input maps to compute output maps.

        Parameters
        ----------
        inputs : MapSet
            Container class that must contain (at least) the maps to be
            transformed.

        Returns
        -------
        output : Map
            Result of applying the transform to the input map(s).

        """
        self.validate_input(inputs)

        # TODO: In the multiple inputs / single output case and depending upon
        # the dimensions of the transform, for efficiency purposes we should
        # make sure that an operation is not carried out like
        #
        #   (input0 [*] transform) + (input1 [*] transform) = output
        #
        # but instead is performed more efficiently as
        #
        #   (input0 + input1) [*] transform = output
        #
        # where [*] is some linear operation, like element-by-element
        # multiplication, a dot product, etc.
        #
        # E.g., for a 1D dot product (dimensionality-reducing linear operation)
        # with M_in inputs and N_el elements in each
        # vector, for the first (less efficient) formulation, there are
        #
        #   (N_el-1)*M_in + (M_in-1) = (M_in*N_el - 1) adds
        #
        # and
        #
        #   (N_el*M_in) multiplies
        #
        # while for the second (more efficient) formulation, there are
        #
        #   (M_in-1)*N_el + (N_el-1) = (M_in*N_el - 1) adds
        #
        # and
        #
        #   (N_el) multiplies
        #
        # so the benefit is a reduction of the number of multiplies necessary
        # by a factor of the number of inputs being "combined" (the number of
        # adds stays the same).

        # TODO: make sure all of these operations are compatible with
        # uncertainties module!

        names = self.input_names
        in0 = inputs[names[0]]

        if self.num_inputs == 1:
            input_array = (in0.rebin(self.input_binning)).hist

        # Stack inputs, sum inputs, *then* rebin (if necessary)
        elif self.sum_inputs:
            input_array = np.sum([inputs[n].hist for n in names], axis=0)
            input_array = rebin(input_array, orig_binning=in0.binning,
                                new_binning=self.input_binning)

        # Rebin (if necessary) then stack
        else:
            input_array = [(inputs[n].rebin(self.input_binning)).hist
                           for n in names]
            input_array = np.stack(input_array, axis=0)

        # Transform same shape: element-by-element multiplication
        if self.xform_array.shape == input_array.shape:
            if (isinstance(self.error_method, basestring) and
                    self.error_method.strip().lower() == 'fixed'):
                # don't scale errors here
                output = unp.uarray(
                    unp.nominal_values(input_array) * self.xform_array,
                    unp.std_devs(input_array)
                )
            else:
                output = input_array * self.xform_array

            # If multiple inputs were concatenated together, and we did not sum
            # these inputs together, we need to sum the results together now.

            # TODO: generalize this and the above operation (and possibly speed
            # this up) by formulating a simple inner product above and avoiding
            # an explicit sum here?
            if self.num_inputs > 1 and not self.sum_inputs:
                output = np.sum(output, axis=0)

        # TODO: Check that
        #   len(xform.shape) == 2*len(input_array.shape)
        # and then check that
        #   xform.shape == (input_array.shape, input_array.shape) (roughly)
        # and then apply tensordot appropriately for this generic case...

        # TODO: why does this fail for different input/output binning, but
        # below tensordot works?
        #elif len(self.xform_array.shape) == 4 and len(input_array.shape) == 2:
        #    output = map2d_kernel4d(input_array, self.xform_array)

        elif len(self.xform_array.shape) == 2*len(input_array.shape):
            output = np.tensordot(input_array, self.xform_array,
                                  axes=([0,1], [0,1]))

        elif (input_array.shape ==
              self.xform_array.shape[0:len(input_array.shape)]):
            axes = range(len(input_array.shape))
            output = np.tensordot(input_array, self.xform_array,
                                  axes=(axes, axes))

        else:
            raise ValueError(
                'Unhandled shapes for input(s) "%s": %s and'
                ' transform: %s.'
                %(', '.join(self.input_names), input_array.shape,
                  self.xform_array.shape)
            )

        output = Map(name=self.output_name,
                     hist=output,
                     binning=self.output_binning)

        # Rebin if necessary so output has `output_binning`
        output = output.rebin(self.output_binning)

        return output


def test_BinnedTensorTransform():
    import os
    import shutil
    import tempfile
    from pisa import ureg, Q_
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
                              3*np.ones(binning.shape)], axis=0)
    )
    assert np.all((xform2 + 2).xform_array - xform2.xform_array == 2)

    testdir = tempfile.mkdtemp()
    try:
        for i, t in enumerate([xform0, xform1, xform2]):
            t_file = os.path.join(testdir, str(i) + '.json')
            t.to_json(t_file)
            t_ = BinnedTensorTransform.from_json(t_file)
            assert t_ == t, 't=\n%s\nt_=\n%s' %(t, t_)
    finally:
        shutil.rmtree(testdir, ignore_errors=True)

    print '<< PASSED : test_BinnedTensorTransform >>'

    xforms = TransformSet(
        name='scaling',
        transforms=[xform0, xform1, xform2],
        hash=9
    )

    assert xforms.hash == 9
    xforms.hash = -20
    assert xforms.hash == -20

    outputs = xforms.apply(inputs)

    # TODO: get this working above, then test here!
    #xforms2 = xforms * 2

    testdir = tempfile.mkdtemp()
    try:
        for i, t in enumerate([xforms]):
            t_filename = os.path.join(testdir, str(i) + '.json')
            t.to_json(t_filename)
            t_ = TransformSet.from_json(t_filename)
            assert t_ == t, 't=\n%s\nt_=\n%s' %(t.transforms, t_.transforms)
    finally:
        shutil.rmtree(testdir, ignore_errors=True)

    print '<< PASSED : test_TransformSet >>'


if __name__ == "__main__":
    test_BinnedTensorTransform()

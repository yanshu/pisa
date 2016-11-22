# Authors

from collections import Iterable, Mapping, Sequence
from copy import deepcopy
import inspect
import os

from pisa.core.events import Events, Data
from pisa.core.map import MapSet
from pisa.core.param import Param, ParamSelector, ParamSet
from pisa.core.transform import TransformSet
from pisa.utils.cache import DiskCache, MemoryCache
from pisa.utils.comparisons import normQuant
from pisa.utils.fileio import mkdir
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging, set_verbosity
from pisa.utils.profiler import line_profile, profile
from pisa.utils.resources import find_resource


__all__ = ['Stage']


# TODO: mode for not propagating errors. Probably needs hooks here, but meat of
# implementation would live inside map.py and/or transform.py.

def arg_str_seq_none(inputs, name):
    """Simple input handler.

    Parameters
    ----------
    inputs : None, string, or sequence of strings
        Input value(s) provided by caller
    name : string
        Name of input, used for producing a meaningful error message

    Returns
    -------
    inputs : None, or list of strings

    Raises
    ------
    TypeError if unrecognized type

    """
    if isinstance(inputs, basestring):
        inputs = [inputs]
    elif isinstance(inputs, (Iterable, Sequence)):
        inputs = [i for i in inputs]
    elif inputs is None:
        pass
    else:
        raise TypeError('Input %s: Unhandled type %s' % (name, type(inputs)))
    return inputs


class Stage(object):
    """
    PISA stage base class. Should encompass all behaviors common to (almost)
    all stages.

    Specialization should be done via subclasses.

    Parameters
    ----------
    use_transforms : bool (required)
        Whether or not this stage takes inputs to be transformed (and hence
        implements transforms).

    params : ParamSelector, dict of ParamSelector kwargs, ParamSet, or object
             instantiable to ParamSet

    expected_params : list of strings
        List containing required `params` names.

    input_names : None or list of strings

    output_names : None or list of strings

    disk_cache : None, str, or DiskCache
        If None, no disk cache is available.
        If str, represents a path with which to instantiate a utils.DiskCache
        object. Must be concurrent-access-safe (across threads and processes).

    memcache_deepcopy : bool

    outputs_cache_depth : int >= 0

    transforms_cache_depth : int >= 0

    input_binning : None or interpretable as MultiDimBinning

    output_binning : None or interpretable as MultiDimBinning

    error_method : None, bool, or string
        If None, False, or empty string, the stage does not compute errors for
        the transforms and does not apply any (additional) error to produce its
        outputs. (If the inputs already have errors, these are propagated.)

        Otherwise, this specifies the method by which the stage should compute
        errors for the transforms to be applied in producing outputs from the
        stage.

    debug_mode : None, bool, or string
        If None, False, or empty string, the stage runs normally.

        Otherwise, the stage runs in debug mode. This disables caching (forcing
        recomputation of any nominal transforms, transforms, and outputs).
        Services that subclass from the `Stage` class can then implement
        further custom behavior when this mode is set by reading the value of
        the `self.debug_mode` attribute.

    Notes
    -----
    The following methods can be overridden in derived classes where
    applicable:
        _derive_nominal_transforms_hash
        _derive_transforms_hash
        _derive_nominal_outputs_hash
        _derive_outputs_hash
        _compute_nominal_transforms
            This is called during initialization to compute what are termed
            "nominal" transforms -- i.e, transforms with all systematic
            parameters set to their nominal values, such that they have no
            effect on the transform. It is optional to use this stage, but if
            it *is* used, then the result will be cached to memory (and
            optionally to disk cache, if one is provided) for future use. A
            nominal transform is useful when systematic parameters merely have
            the effect of modifying the nominal transform, rather than
            requiring a complete recomputation of the transform.
        _compute_nominal_outputs
            same as nominal transforms, but for outputs (e.g. used for
            non-input stages)
        _compute_transforms
            Do the actual work to produce the stage's transforms. For stages
            that specify use_transforms=False, this method is never called.
        _compute_outputs
            Do the actual work to compute the stage's output. Default
            implementation is to call self.transforms.apply(inputs); override
            if no transforms are present or if more needs to be done to
            compute outputs than this.
        validate_params
            Perform validation on any parameters.

    """
    def __init__(self, use_transforms,
                 params=None, expected_params=None, input_names=None,
                 output_names=None, error_method=None, disk_cache=None,
                 memcache_deepcopy=True, transforms_cache_depth=10,
                 outputs_cache_depth=0, input_binning=None,
                 output_binning=None, debug_mode=None):

        # Allow for string inputs, but have to populate into lists for
        # consistent interfacing to one or multiple of these things
        expected_params = arg_str_seq_none(expected_params, 'expected_params')
        input_names = arg_str_seq_none(input_names, 'input_names')
        output_names = arg_str_seq_none(output_names, 'output_names')

        self.use_transforms = use_transforms
        """Whether or not stage uses transforms"""

        module_path = self.__module__.split('.')

        self.stage_name = module_path[-2]
        """Name of the stage (e.g. flux, osc, aeff, reco, pid, etc."""

        self.service_name = module_path[-1]
        """Name of the specific service implementing the stage."""

        self.expected_params = expected_params
        """The full set of parameters (by name) that must be present in
        `params`"""

        self._events_hash = None

        self._input_names = [] if input_names is None else input_names
        self._output_names = [] if output_names is None else output_names

        self.input_binning = input_binning
        self.output_binning = output_binning
        self._source_code_hash = None

        # Storage of latest transforms and outputs; default to empty
        # TransformSet and None, respectively.
        self.transforms = TransformSet([])
        """A stage that takes to-be-transformed inputs and has had these
        transforms computed stores them here. Before computation, `transforms`
        is an empty TransformSet; a stage that does not make use of these (such
        as a no-input stage) has an empty TransformSet."""

        self.outputs = None
        """Last-computed outputs; None if no outputs have been computed yet."""

        self.memcache_deepcopy = memcache_deepcopy

        self.transforms_cache_depth = int(transforms_cache_depth)

        self.transforms_cache = None
        """Memory cache object for storing transforms"""

        self.nominal_transforms_cache = None
        """Memory cache object for storing nominal transforms"""

        self.full_hash = True
        """Whether to do full hashing if true, otherwise do fast hashing"""

        self.transforms_cache = MemoryCache(
            max_depth=self.transforms_cache_depth, is_lru=True,
            deepcopy=self.memcache_deepcopy
        )
        self.nominal_transforms_cache = MemoryCache(
            max_depth=self.transforms_cache_depth, is_lru=True,
            deepcopy=self.memcache_deepcopy
        )

        self.outputs_cache_depth = int(outputs_cache_depth)

        self.outputs_cache = None
        """Memory cache object for storing outputs (excludes sideband
        objects)."""

        self.outputs_cache = None
        if self.outputs_cache_depth > 0:
            self.outputs_cache = MemoryCache(
                max_depth=self.outputs_cache_depth, is_lru=True,
                deepcopy=self.memcache_deepcopy
            )

        self.disk_cache = disk_cache
        """Disk cache object"""

        self.disk_cache_path = None
        """Path to disk cache file for this stage/service."""

        param_selector_keys = set([
            'regular_params', 'selector_param_sets', 'selections'
        ])
        if isinstance(params, Mapping) \
                and set(params.keys()) == param_selector_keys:
            self._param_selector = ParamSelector(**params)
        elif isinstance(params, ParamSelector):
            self._param_selector = params
        else:
            self._param_selector = ParamSelector(regular_params=params)

        # Get the params from the ParamSelector, validate, and set as the
        # params object for this stage
        p = self._param_selector.params
        self._check_params(p)
        self.validate_params(p)
        self._params = p

        if bool(debug_mode) == False:
            self._debug_mode = None
        else:
            self._debug_mode = debug_mode

        if bool(error_method) == False:
            self._error_method = None
        else:
            self._error_method = error_method

        # Include each attribute here for hashing if it is defined and its
        # value is not None
        default_attrs_to_hash = ['input_names', 'output_names',
                                 'input_binning', 'output_binning']
        self._attrs_to_hash = set([])
        for attr in default_attrs_to_hash:
            if not hasattr(self, attr):
                continue
            val = getattr(self, attr)
            if val is None:
                continue
            try:
                self.include_attrs_for_hashes(attr)
            except ValueError():
                pass

        # Define useful flags and values for debugging behavior after running

        self.nominal_transforms_loaded_from_cache = None
        """Records which cache nominal transforms were loaded from, or None."""

        self.nominal_transforms_computed = False
        """Records whether nominal transforms were (re)computed."""

        self.transforms_loaded_from_cache = None
        """Records which cache transforms were loaded from, or None."""

        self.transforms_computed = False
        """Records whether transforms were (re)computed."""

        self.nominal_outputs_computed = False
        """Records whether nominal outputs were (re)computed."""

        self.outputs_loaded_from_cache = None
        """Records which cache outputs were loaded from, or None."""

        self.outputs_computed = False
        """Records whether outputs were (re)computed."""

        self.nominal_transforms_hash = None
        self.transforms_hash = None
        self.nominal_outputs_hash = None
        self.outputs_hash = None

    @profile
    def get_nominal_transforms(self, nominal_transforms_hash):
        """Load a cached transform from the nominal transform memory cache
        (which is backed by a disk cache, if one is specified) if the nominal
        transform is in the cache, or else recompute it and store to the
        cache(s).

        This method calls the `_compute_nominal_transforms` method, which by
        default does nothing.

        However, if you want to use the nominal transforms feature, override
        the `_compute_nominal_transforms` method and fill in the logic there.

        Deciding whether to invoke the `_compute_nominal_transforms` method or
        to load the nominal transforms from cache is done here, so you needn't
        think about any of this within the `_compute_nominal_transforms`
        method.

        Returns
        -------
        nominal_transforms, hash

        """
        # Reset flags
        self.nominal_transforms_loaded_from_cache = None
        self.nominal_transforms_computed = False

        if nominal_transforms_hash is None:
            nominal_transforms_hash = self._derive_nominal_transforms_hash()

        nominal_transforms = None
        # Quick way to avoid further logic is if hash value is None
        if nominal_transforms_hash is None:
            self.nominal_transforms_hash = None
            self.nominal_transforms = None
            return self.nominal_transforms, self.nominal_transforms_hash

        recompute = True
        # If hash found in memory cache, load nominal transforms from there
        if nominal_transforms_hash in self.nominal_transforms_cache \
                and self.debug_mode is None:
            nominal_transforms = \
                    self.nominal_transforms_cache[nominal_transforms_hash]
            self.nominal_transforms_loaded_from_cache = 'memory'
            recompute = False

        # Otherwise try to load from an extant disk cache
        elif self.disk_cache is not None and self.debug_mode is None:
            try:
                nominal_transforms = self.disk_cache[nominal_transforms_hash]
            except KeyError:
                pass
            else:
                self.nominal_transforms_loaded_from_cache = 'disk'
                recompute = False
                # Save to memory cache
                self.nominal_transforms_cache[nominal_transforms_hash] = \
                        nominal_transforms

        if recompute:
            self.nominal_transforms_computed = True
            nominal_transforms = self._compute_nominal_transforms()
            if nominal_transforms is None:
                # Invalidate hash value since found transforms
                nominal_transforms_hash = None
            else:
                nominal_transforms.hash = nominal_transforms_hash
                self.nominal_transforms_cache[nominal_transforms_hash] = \
                        nominal_transforms
                if self.disk_cache is not None:
                    self.disk_cache[nominal_transforms_hash] = \
                            nominal_transforms

        self.nominal_transforms = nominal_transforms
        self.nominal_transforms_hash = nominal_transforms_hash
        return nominal_transforms, nominal_transforms_hash

    @profile
    def get_transforms(self, transforms_hash=None,
                       nominal_transforms_hash=None):
        """Load a cached transform (keyed on hash of parameter values) if it
        is in the cache, or else compute a new transform from currently-set
        parameter values and store this new transform to the cache.

        This calls the private method _compute_transforms (which must be
        implemented in subclasses if the nominal transform feature is desired)
        to generate a new transform if the nominal transform is not found in
        the nominal transform cache.

        Notes
        -----
        The hash used here is only meant to be valid within the scope of a
        session; a hash on the full parameter set used to generate the
        transform *and* the version of the generating software is required for
        non-volatile storage.

        """
        # Reset flags
        self.transforms_loaded_from_cache = None
        self.transforms_computed = False

        # TODO: store nominal transforms to the transforms cache as well, but
        # derive the hash value the same way as it is done for transforms,
        # to avoid needing to apply no systematics to the nominal transforms
        # to get the (identical) transforms?
        # Problem: assumes the nominal transform is the same as the transforms
        # that will result, which *might* not be true (though it seems it will
        # usually be so)

        # Compute nominal transforms; if feature is not used, this doesn't
        # actually do much of anything. To do more than this, override the
        # `_compute_nominal_transforms` method.
        nominal_transforms, nominal_transforms_hash = \
                self.get_nominal_transforms(
                    nominal_transforms_hash=nominal_transforms_hash
                )

        # Generate hash from param values
        if transforms_hash is None:
            transforms_hash = self._derive_transforms_hash(
                nominal_transforms_hash=nominal_transforms_hash
            )
        logging.trace('transforms_hash: %s' %str(transforms_hash))

        # Load and return existing transforms if in the cache
        if self.transforms_cache is not None \
                and transforms_hash in self.transforms_cache \
                and self.debug_mode is None:
            self.transforms_loaded_from_cache = 'memory'
            logging.trace('loading transforms from cache.')
            transforms = self.transforms_cache[transforms_hash]

        # Otherwise: compute transforms, set hash, and store to cache
        else:
            self.transforms_computed = True
            logging.trace('computing transforms.')
            transforms = self._compute_transforms()
            transforms.hash = transforms_hash
            if self.transforms_cache is not None:
                self.transforms_cache[transforms_hash] = transforms

        self.check_transforms(transforms)
        self.transforms = transforms
        return transforms

    @profile
    def get_nominal_outputs(self, nominal_outputs_hash):
        """Load a cached output from the nominal outputs memory cache
        (which is backed by a disk cache, if one is specified) if the nominal
        outout is in the cache, or else recompute it and store to the
        cache(s).

        This method calls the `_compute_nominal_outputs` method, which by
        default does nothing.

        However, if you want to use the nominal outputs feature, override
        the `_compute_nominal_outputs` method and fill in the logic there.

        Deciding whether to invoke the `_compute_nominal_outputs` method or
        to load the nominal outputs from cache is done here, so you needn't
        think about any of this within the `_compute_nominal_outputs`
        method.

        Returns
        -------
        nominal_outputs, hash

        """
        if self.nominal_outputs_hash is None \
           or self.nominal_outputs_hash != self._derive_nominal_outputs_hash():
            self._compute_nominal_outputs()
            self.nominal_outputs_hash = self._derive_nominal_outputs_hash()

    @profile
    def get_outputs(self, inputs=None):
        """Top-level function for computing outputs. Use this method to get
        outputs if you live outside this stage/service.

        Caching is handled here, so if the output hash returned by
        `_derive_outputs_hash` is in `outputs_cache`, it is simply returned.
        Otherwise, the `_compute_outputs` private method is invoked to do the
        actual work of computing outputs.

        Parameters
        ----------
        inputs : None or Mapping
            Any inputs to be transformed, plus any sideband objects that are to
            be passed on (untransformed) to subsequent stages.

        See also
        --------
        Overloadable methods called directly from this:
            _derive_outputs_hash
            _compute_outputs

        """
        # Reset flags
        self.outputs_loaded_from_cache = None
        self.outputs_computed = False

        # TODO: store nominal outputs to the outputs cache as well, but
        # derive the hash value the same way as it is done for outputs,
        # to avoid needing to apply no systematics to the nominal outputs
        # to get the (identical) outputs?
        # Problem: assumes the nominal transform is the same as the outputs
        # that will result, which *might* not be true (though it seems it will
        # usually be so)

        # Keep inputs for internal use and for inspection later
        self.inputs = inputs

        outputs_hash, transforms_hash, nominal_transforms_hash = \
                self._derive_outputs_hash()

        # Compute nominal outputs; if feature is not used, this doesn't
        # actually do much of anything. To do more than this, override the
        # `_compute_nominal_outputs` method.
        self.get_nominal_outputs(nominal_outputs_hash=nominal_transforms_hash)

        logging.trace('outputs_hash: %s' %outputs_hash)

        if self.outputs_cache is not None \
                and outputs_hash is not None \
                and outputs_hash in self.outputs_cache \
                and self.debug_mode is None:
            self.outputs_loaded_from_cache = 'memory'
            logging.trace('Loading outputs from cache.')
            outputs = self.outputs_cache[outputs_hash]
        else:
            self.outputs_computed = True
            logging.trace('Need to compute outputs...')

            if self.use_transforms:
                self.get_transforms(
                    transforms_hash=transforms_hash,
                    nominal_transforms_hash=nominal_transforms_hash
                )

            logging.trace('... now computing outputs.')
            outputs = self._compute_outputs(inputs=self.inputs)
            outputs.hash = outputs_hash
            self.check_outputs(outputs)

            # Store output to cache
            if self.outputs_cache is not None and outputs_hash is not None:
                self.outputs_cache[outputs_hash] = outputs

        # Keep outputs for inspection later
        self.outputs = outputs

        # Attach sideband objects (i.e., inputs not specified in
        # `self.input_names`) to the "augmented" output object
        if self.inputs is None: names_in_inputs = set([])
        else: names_in_inputs = set(self.inputs.names)
        unused_input_names = names_in_inputs.difference(self.input_names)

        if len(unused_input_names) == 0:
            return outputs

        # Create a new output container different from `outputs` but copying
        # the contents, for purposes of attaching the sideband objects found.
        augmented_outputs = MapSet(outputs)
        [augmented_outputs.append(inputs[name]) for name in unused_input_names]

        return augmented_outputs

    @profile
    def _check_params(self, params):
        """Make sure that `expected_params` is defined and that exactly the
        params specified in self.expected_params are present.

        """
        assert self.expected_params is not None
        exp_p, got_p = set(self.expected_params), set(params.names)
        if exp_p == got_p:
            return
        excess = got_p.difference(exp_p)
        missing = exp_p.difference(got_p)
        err_strs = []
        if len(excess) > 0:
            err_strs.append('Excess params provided: %s'
                            %', '.join(sorted(excess)))
        if len(missing) > 0:
            err_strs.append('Missing params: %s'
                            %', '.join(sorted(missing)))
        raise ValueError('Expected parameters: %s;\n'
                         %', '.join(sorted(exp_p))
                         + ';\n'.join(err_strs))

    @profile
    def check_transforms(self, transforms):
        """Check that transforms' inputs and outputs match those specified
        for this service.

        Parameters
        ----------
        transforms

        Raises
        ------
        ValueError if transforms' inputs/outputs don't match stage spec

        """
        assert set(transforms.input_names) == set(self.input_names), \
                "Transforms' inputs: " + str(transforms.input_names) + \
                "\nStage inputs: " + str(self.input_names)

        assert set(transforms.output_names) == set(self.output_names), \
                "Transforms' outputs: " + str(transforms.output_names) + \
                "\nStage outputs: " + str(self.output_names)

    @profile
    def check_outputs(self, outputs):
        assert set(outputs.names) == set(self.output_names), \
                "Outputs: " + str(outputs.names) + \
                "\nStage outputs: " + str(self.output_names)

    def select_params(self, selections, error_on_missing=False):
        try:
            self._param_selector.select_params(
                selections, error_on_missing=True
            )
        except KeyError:
            msg = 'None of the selections %s found in this pipeline.' \
                    %(selections,)
            if error_on_missing:
                #logging.error(msg)
                raise
            logging.trace(msg)
        else:
            logging.trace('`selections` = %s yielded `params` = %s'
                          %(selections, self.params))

    def load_events(self, events):
        if isinstance(events, Param):
            events = events.value
        elif isinstance(events, basestring):
            events = find_resource(events)
        this_hash = hash_obj(events, full_hash=self.full_hash)
        if self._events_hash is not None and this_hash == self._events_hash:
            return
        logging.debug('Extracting events from Events obj or file: %s' %events)
        self.events = Events(events)
        self._events_hash = this_hash

    def cut_events(self, keep_criteria):
        if isinstance(keep_criteria, Param):
            keep_criteria = keep_criteria.value
        if keep_criteria is not None:
            self.remaining_events = deepcopy(self.events)
            self.remaining_events.applyCut(keep_criteria=keep_criteria)
        else:
            self.remaining_events = self.events

    def instantiate_disk_cache(self):
        if isinstance(self.disk_cache, DiskCache):
            return
        if isinstance(self.disk_cache, basestring):
            self.disk_cache_path = self.disk_cache
        elif self.disk_cache is None:
            cache_root_dir = find_resource('cache')
            dirs = [cache_root_dir, self.stage_name]
            dirpath = os.path.join(*dirs)
            if self.service_name is not None and self.service_name != '':
                filename = self.service_name + '.sqlite'
            else:
                filename = 'geeric.sqlite'
            mkdir(dirpath)
            self.disk_cache_path = os.path.join(dirpath, filename)
        else:
            raise ValueError("Don't know what to do with a %s."
                             %type(self.disk_cache))
        self.disk_cache = DiskCache(self.disk_cache_path, max_depth=10,
                                    is_lru=False)

    @property
    def params(self):
        return self._params

    @property
    def param_selections(self):
        return sorted(deepcopy(self._param_selector.param_selections))

    @property
    def input_names(self):
        return deepcopy(self._input_names)

    @property
    def output_names(self):
        return deepcopy(self._output_names)

    @property
    def source_code_hash(self):
        """Hash for the source code of this object's class.

        Not meant to be perfect, but should suffice for tracking provenance of
        an object stored to disk that were produced by a Stage.
        """
        if self._source_code_hash is None:
            self._source_code_hash = hash_obj(
                inspect.getsource(self.__class__),
                full_hash=self.full_hash
            )
        return self._source_code_hash

    @property
    def state_hash(self):
        """Combines source_code_hash and params.hash for checking/tagging
        provenance of persisted (on-disk) objects."""
        objects_to_hash = [self.source_code_hash, self.params.state_hash]
        for attr in self._attrs_to_hash:
            objects_to_hash.append(hash_obj(getattr(self, attr),
                                            full_hash=self.full_hash))
        return hash_obj(objects_to_hash, full_hash=self.full_hash)

    def include_attrs_for_hashes(self, attrs):
        """Include a class attribute or attributes to be included when
        computing hashes (for all that apply: nominal transforms, transforms,
        and/or outputs).

        This is a convenience that allows some customization of hashing (and
        hence caching) behavior without having to override the hash-computation
        methods (`_derive_nominal_transforms_hash`, `_derive_transforms_hash`,
        and `_derive_outputs_hash`).

        Parameters
        ----------
        attrs : string or sequence thereof
            Name of the attribute(s) to include for hashes. Each must be an
            existing attribute of the object at the time this method is
            invoked.

        """
        if isinstance(attrs, basestring):
            attrs = [attrs]

        # Validate that all are actually attrs before setting any
        for attr in attrs:
            assert isinstance(attr, basestring)
            if not hasattr(self, attr):
                raise ValueError('"%s" not an attribute of the class; not'
                                 ' adding *any* of the passed attributes %s to'
                                 ' attrs to hash.' %(attr, attrs))

        # Include the attribute names
        [self._attrs_to_hash.add(attr) for attr in attrs]

    @property
    def debug_mode(self):
        """Read-only attribute indicating whether or not the stage is being run
        in debug mode. None indicates non-debug mode, while non-none value
        indicates a debug mode."""
        return self._debug_mode

    @property
    def error_method(self):
        """Read-only attribute indicating whether or not the stage will compute
        errors for its transforms and outputs (whichever is applicable). Errors
        on inputs are propagated regardless of this setting."""
        return self._error_method

    def _derive_outputs_hash(self):
        """Derive a hash value that unique identifies the outputs that will be
        generated based upon the current state of the stage.

        This implementation hashes together:
        * Input and output binning objects' hash values (if either input or
          output binning is not None)
        * Current params' values hash
        * Hashes from any input objects with names in `self.input_names`

        If any of the above objects is specified but returns None for its hash
        value, the entire output hash is invalidated, and None is returned.

        """
        id_objects = []

        # If stage uses inputs, grab hash from the inputs container object
        if self.outputs_cache is not None and len(self.input_names) > 0:
            inhash = self.inputs.hash
            logging.trace('inputs.hash = %s' %inhash)
            id_objects.append(inhash)

        # If stage uses transforms, get hash from the transforms
        transforms_hash = None
        if self.use_transforms:
            transforms_hash, nominal_transforms_hash = \
                    self._derive_transforms_hash()
            id_objects.append(transforms_hash)
            logging.trace('derived transforms hash = %s' %id_objects[-1])

        # Otherwise, generate sub-hash on binning and param values here
        else:
            transforms_hash, nominal_transforms_hash = None, None

            if self.outputs_cache is not None:
                id_subobjects = []
                # Include all parameter values
                id_subobjects.append(self.params.values_hash)

                # Include additional attributes of this object
                for attr in sorted(self._attrs_to_hash):
                    val = getattr(self, attr)
                    if hasattr(val, 'hash'):
                        attr_hash = val.hash
                    elif self.full_hash:
                        norm_val = normQuant(val)
                        attr_hash = hash_obj(norm_val,
                                             full_hash=self.full_hash)
                    else:
                        attr_hash = hash_obj(val, full_hash=self.full_hash)
                    id_subobjects.append(attr_hash)

                # Generate the "sub-hash"
                if any([(h == None) for h in id_subobjects]):
                    sub_hash = None
                else:
                    sub_hash = hash_obj(id_subobjects,
                                        full_hash=self.full_hash)
                id_objects.append(sub_hash)

        # If any hashes are missing (i.e, None), invalidate the entire hash
        if self.outputs_cache is None or any([(h is None) for h in id_objects]):
            outputs_hash = None
        else:
            outputs_hash = hash_obj(id_objects, full_hash=self.full_hash)

        return outputs_hash, transforms_hash, nominal_transforms_hash

    def _derive_transforms_hash(self, nominal_transforms_hash=None):
        """Compute a hash that uniquely identifies the transforms that will be
        produced from the current configuration. Note that this hash needs only
        to be valid for this run (i.e., it is a volatile hash).

        This implementation returns a hash from the current parameters' values.

        """
        id_objects = []
        h = self.params.values_hash
        logging.trace('self.params.values_hash = %s' %h)
        id_objects.append(h)

        # Grab any provided nominal transforms hash, or derive it again
        if nominal_transforms_hash is None:
            nominal_transforms_hash = self._derive_nominal_transforms_hash()
        # If a valid hash has been gotten, include it
        if nominal_transforms_hash is not None:
            id_objects.append(nominal_transforms_hash)

        for attr in sorted(self._attrs_to_hash):
            val = getattr(self, attr)
            if hasattr(val, 'hash'):
                attr_hash = val.hash
            elif self.full_hash:
                norm_val = normQuant(val)
                attr_hash = hash_obj(norm_val, full_hash=self.full_hash)
            else:
                attr_hash = hash_obj(val, full_hash=self.full_hash)
            id_objects.append(attr_hash)

        # If any hashes are missing (i.e, None), invalidate the entire hash
        if any([(h == None) for h in id_objects]):
            transforms_hash = None
        else:
            transforms_hash = hash_obj(id_objects, full_hash=self.full_hash)

        return transforms_hash, nominal_transforms_hash

    def _derive_nominal_transforms_hash(self):
        """Derive a hash to uniquely identify the nominal transform. This
        should be unique across processes and invocations bacuase the nominal
        transforms can be non-volatile (cached to disk) and must still be
        valid given their hash value upon loading from disk in the future.

        This implementation uses the nominal parameter values' hash
        combined with the source code hash to generate the final nominal
        transforms hash.

        Notes
        -----
        The hashing scheme implemented here might be sufficiently unique for
        many cases, but override this method in services according to the
        following guidelines:

        * Stages that use a nominal transform should override this method if
          the hash is more accurately computed differently from here.

        * Stages that use transforms but do not use nominal transforms can
          override this method with a simpler version that simply returns None
          to save computation time (if this method is found to be a significant
          performance hit). (This method is called each time an output
          is computed if `self.use_transforms == True`.)

        * Stages that use no transforms (i.e., `self.use_transforms == False`)
          will not call any built-in methods related to transforms, so
          overriding this method is irrelevant to such stages.

        If this method *is* overridden (and not just to return None), since the
        nominal transform may be stored to a disk cache, make sure that
        `self.source_code_hash` is included in the objects used to compute the
        final hash value. Even if all parameters are the same, a nominal
        transform stored to disk is ***invalid if the source code changes***,
        and `_derive_nominal_transforms_hash` must reflect this.

        """
        id_objects = []
        id_objects.append(self.params.nominal_values_hash)
        for attr in sorted(self._attrs_to_hash):
            val = getattr(self, attr)
            if hasattr(val, 'hash'):
                attr_hash = val.hash
            elif self.full_hash:
                norm_val = normQuant(val)
                attr_hash = hash_obj(norm_val, full_hash=self.full_hash)
            else:
                attr_hash = hash_obj(val, full_hash=self.full_hash)
            id_objects.append(attr_hash)
        id_objects.append(self.source_code_hash)

        # If any hashes are missing (i.e, None), invalidate the entire hash
        if any([(h == None) for h in id_objects]):
            nominal_transforms_hash = None
        else:
            nominal_transforms_hash = hash_obj(id_objects,
                                               full_hash=self.full_hash)

        return nominal_transforms_hash

    def _derive_nominal_outputs_hash(self):
        return self._derive_nominal_transforms_hash()

    def _compute_nominal_transforms(self):
        """Stages that start with a nominal transform and use systematic
        parameters to modify the nominal transform in order to obtain the final
        transforms should override this method for deriving the nominal
        transform."""
        return None

    def _compute_transforms(self):
        """Stages that apply transforms to inputs should override this method
        for deriving the transform. No-input stages should leave this as-is,
        simply returning None."""
        return TransformSet([])

    def _compute_nominal_outputs(self):
        return None

    @profile
    def _compute_outputs(self, inputs):
        """Override this method for no-input stages which do not use transforms.
        Input stages that compute a TransformSet needn't override this, as the
        work for computing outputs is done by the TransfromSet below."""
        return self.transforms.apply(inputs)

    def validate_params(self, params):
        """Override this method to test if params are valid; e.g., check range
        and dimensionality."""
        return

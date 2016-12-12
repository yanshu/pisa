# author : J.L. Lanfranchi
#          jll1062+pisa@phys.psu.edu
#
# date   : April 8, 2016
"""
Utilities for comparing things.

`recursiveEquality` and `recursiveAllclose` traverse into potentially nested
datstructures and compare all elements for equality.
`normQuant` performs the same kind of traversal, but returns a normalized
version of the input object whereby "essentially-equal" things are returned as
"actually-equal" objects.

These functions are at the heart of hashing behaving as one expects, and this
in turn is essential for caching to work correctly.

.. important:: Read carefully how each function in this module defines
   "equality" for various datatypes so you understand what two things being
   "equal" based upon these functions *actually* means.

   E.g., (nan == nan) == True, uncorrelated uncertainties are equal if both
   their nominal values and standard deviations are equal regardless if they
   come from independent random variables, etc.
"""


from itertools import izip
from collections import Iterable, Iterator, Mapping, OrderedDict, Sequence
from numbers import Number
import re

import numpy as np
from uncertainties.core import AffineScalarFunc
from uncertainties import ufloat
from uncertainties import unumpy as unp
import pint

from pisa import ureg, FTYPE, HASH_SIGFIGS
from pisa.utils.log import logging, set_verbosity


__all__ = ['FTYPE_PREC', 'EQUALITY_SIGFIGS', 'EQUALITY_PREC', 'ALLCLOSE_KW',
           'LOG10_2', 'NP_TYPES', 'SEQ_TYPES', 'MAP_TYPES', 'COMPLEX_TYPES',
           'isvalidname', 'isscalar', 'isbarenumeric',
           'recursiveEquality', 'recursiveAllclose', 'normQuant']


FTYPE_PREC = np.finfo(FTYPE).eps
"""Machine precision ("eps") for PISA's FLOAT datatype"""

#EQ_SIGFIGS = int(np.ceil(np.log10(FTYPE_PREC)))
EQUALITY_SIGFIGS = HASH_SIGFIGS
"""Significant figures for performing equality comparisons"""

EQUALITY_PREC = 10**-EQUALITY_SIGFIGS
"""Precision ("rtol") for performing equality comparisons"""

ALLCLOSE_KW = dict(rtol=EQUALITY_PREC, atol=0, equal_nan=True)
"""Keyword args to pass to all calls to numpy.allclose"""

# Derive the following number via:
# >>> from sympy import log, N
# >>> str(N(log(2, 10), 40))
LOG10_2 = np.float64('0.3010299956639811952137388947244930267682')

NP_TYPES = (np.ndarray, np.matrix)
SEQ_TYPES = (Sequence, np.ndarray, np.matrix)
MAP_TYPES = (Mapping,)
COMPLEX_TYPES = tuple(list(NP_TYPES) + list(SEQ_TYPES) + list(MAP_TYPES))


def isvalidname(x):
    """Name that is valid to use for a Python variable"""
    return re.compile(r'\W|^(?=\d)').match(x) is None


def isscalar(x):
    """Check if input is a scalar object.

    Best check found for now as to scalar-ness (works for pint,
    uncertainties, lists, tuples, numpy arrays, ...) but not tested against all
    things.

    See Also
    --------
    numpy.isscalar

    """
    return (not (hasattr(x, 'shape')
                 or isinstance(x, (Iterator, Mapping, Sequence)))
            or np.isscalar(x))


def isbarenumeric(x):
    """Check if input is a numerical datatype (including arrays of numbers) but
    without units. Note that for these purposes, booleans are *not* considered
    numerical datatypes.

    """
    is_bare_numeric = False
    if isinstance(x, np.ndarray):
        if x.dtype.type not in (np.bool, np.bool_, np.bool8, np.object0,
                                np.object, np.object_):
            is_bare_numeric = True
    elif isinstance(x, Number) and not isinstance(x, bool):
        is_bare_numeric = True
    return is_bare_numeric


def recursiveEquality(x, y):
    """Recursively verify equality between two objects `x` and `y`.

    Parameters
    ----------
    x, y
        Objects to be compared

    Notes
    -----
    Possibly unintuitive behaviors:
      * Sequences of any type evaluate equal if their contents are the same.
        E.g., a list can equal a tuple.

      * Mappings of any type evaluate equal if their contents are the same.
        E.g. a dict can equal an OrderedDict.

      * nan SHOULD equal nan, +inf SHOULD equal +inf, and -inf SHOULD equal -inf
        ... but this ***only*** holds true (as of now) if those values are in
        numpy arrays! (TODO!)

      * Two pint units with same __repr__ but that were derived from different
        unit registries evaluate to be equal. (This is contrary to pint's
        implementation of equality comparisons, which is careful in case a
        unit is defined differently in different registries. We'll assume this
        isn't done here in PISA, until a use case arises where this is no
        longer a good assumption.)

      * Two pint units that are compatible but different (even just in
        magnitude prefix) evaluate to be unequal.

        This behavior is chosen for the case where numbers are given
        independently of their units, and hence the automatic conversion
        facility available for comparing pint quantities is not available.
        The only reliable way to test equality for these "less intelligent"
        objects is to ensure that both the numerical values are exactly equal
        and that the units are exactly equal; the latter includes order of
        magnitude prefixes (micro, milli, ..., giga, etc.).

    """
    # NOTE: The order in which types are compared below matters, so change
    # order carefully.

    # pint units
    if isinstance(x, pint.unit._Unit):
        if not isinstance(y, pint.unit._Unit):
            logging.trace('type(x)=%s but type(y)=%s' %(type(x), type(y)))
        if repr(x) != repr(y):
            logging.trace('x: %s' %x)
            logging.trace('y: %s' %y)
            return False

    # pint quantities
    elif isinstance(x, pint.quantity._Quantity):
        if not isinstance(y, pint.quantity._Quantity):
            logging.trace('type(x)=%s but type(y)=%s' %(type(x), type(y)))
            return False
        xunit = str(x.units)
        try:
            converted_y = y.to(xunit)
        except pint.DimensionalityError:
            logging.trace('Incompatible units: x.units=%s, y.units=%s'
                          %(x.units, y.units))
            return False
        # Check for equality to HASH_SIGFIGS significant figures
        if not np.allclose(x.magnitude, converted_y.magnitude, **ALLCLOSE_KW):
            logging.trace('x.magnitude: %s' %x.magnitude)
            logging.trace('y.magnitude: %s' %y.magnitude)
            return False

    # Simple things can be compared directly
    elif (isinstance(x, basestring) or isinstance(y, basestring) or
          not (isinstance(x, COMPLEX_TYPES)
               or isinstance(y, COMPLEX_TYPES))):
        if x != y:
            is_eq = False
            try:
                if np.allclose(x, y, **ALLCLOSE_KW):
                    is_eq = True
            except TypeError:
                pass
            if not is_eq:
                logging.trace('Simple types (type(x)=%s, type(y)=%s) not equal.'
                              %(type(x), type(y)))
                logging.trace('x: %s' %x)
                logging.trace('y: %s' %y)
                return False

    # Numpy types
    elif isinstance(x, NP_TYPES) or isinstance(y, NP_TYPES):
        if np.shape(x) != np.shape(y):
            logging.trace('shape(x): %s' %np.shape(x))
            logging.trace('shape(y): %s' %np.shape(y))
            return False
        if not np.allclose(x, y, **ALLCLOSE_KW):
            logging.trace('x: %s' %x)
            logging.trace('y: %s' %y)
            return False

    # dict
    elif isinstance(x, Mapping):
        xkeys = sorted(x.keys())
        if xkeys != sorted(y.keys()):
            logging.trace('xkeys: %s' %(xkeys,))
            logging.trace('ykeys: %s' %(sorted(y.keys()),))
            return False
        else:
            for k in xkeys:
                if not recursiveEquality(x[k], y[k]):
                    logging.trace('not equal found at key: "%s"' %k)
                    return False

    # Non-numpy sequence
    elif isinstance(x, Sequence):
        if len(x) != len(y):
            logging.trace('len(x): %s' %len(x))
            logging.trace('len(y): %s' %len(y))
            return False
        else:
            for xs, ys in izip(x, y):
                if not recursiveEquality(xs, ys):
                    logging.trace('xs: %s' %xs)
                    logging.trace('ys: %s' %ys)
                    return False

    elif hasattr(x, '_hashable_state'):
        if not hasattr(y, '_hashable_state'):
            return False
        return recursiveEquality(x._hashable_state, y._hashable_state)

    # Unhandled
    else:
        raise TypeError('Unhandled type(s): %s, x=%s, y=%s' %
                        (type(x), str(x), str(y)))

    # Returns above only occur if comparisons evaluate to False; therefore, if
    # you make it here, everything is equal.
    return True


# TODO: Get recursiveAllclose working as recursiveEquality does.

def recursiveAllclose(x, y, *args, **kwargs):
    """Recursively verify close-equality between two objects x and y. If
    structure is different between the two objects, returns False

    args and kwargs are passed into numpy.allclose() function
    """
    # TODO: until the below has been verified, refuse to run
    raise NotImplementedError('recursiveAllclose not implemented yet')
    # None
    if x is None:
        if y is not None:
            return False
    # Scalar
    elif isscalar(x):
        if not isscalar(y):
            return False
        # np.allclose doesn't handle some dtypes
        try:
            eq = np.allclose(x, y, *args, **kwargs)
        except TypeError:
            eq = x == y
        if not eq:
            return False
    # Dict
    elif isinstance(x, dict):
        if not isinstance(y, dict):
            return False
        xkeys = sorted(x.keys())
        if xkeys != sorted(y.keys()):
            return False
        for k in xkeys:
            if not recursiveAllclose(x[k], y[k], *args, **kwargs):
                return False
    # Sequence
    elif hasattr(x, '__len__'):
        if len(x) != len(y):
            return False
        if isinstance(x, list) or isinstance(x, tuple):
            # NOTE: A list is allowed to be allclose to a tuple so long
            # as the contents are allclose
            if not isinstance(y, list) or isinstance(y, tuple):
                return False
            for xs, ys in izip(x, y):
                if not recursiveAllclose(xs, ys, *args, **kwargs):
                    return False
        elif isinstance(x, np.ndarray):
            # NOTE: A numpy array only evalutes to allclose if compared to
            # another numpy array
            if not isinstance(y, np.ndarray):
                return False
            # np.allclose doesn't handle arrays of some dtypes
            # TODO: this can be rolled into the above clause, I think
            try:
                eq = np.allclose(x, y, *args, **kwargs)
            except TypeError:
                eq = np.all(x == y)
            if not eq:
                return False
        else:
            raise TypeError('Unhandled type(s): %s, x=%s, y=%s' %
                            (type(x), str(x), str(y)))
    else:
        raise TypeError('Unhandled type(s): %s, x=%s, y=%s' %
                        (type(x), str(x), str(y)))
    # If you make it to here, must be close
    return True


# TODO: add an arg and logic to round to a number of significand *bits* (as
# opposed to digits) (or even a fixed number of bits that align with special
# floating point spec values -- like half, single, double) for more precise
# control (and possibly faster comp), esp. if we decide to move to FP32 or
# (even more critical) FP16?
def normQuant(obj, sigfigs=None, full_norm=True):
    """Normalize quantities such that two things that *should* be equal are
    returned as identical objects.

    Handles floating point numbers, pint quantities, uncertainties, and
    combinations thereof as standalone objects or in sequences, dicts, or numpy
    ndarrays. Numerical precision issues and equal quantities represented in
    differently-scaled or different systems of units come out identically.

    Outputs from this function (**not** the inputs) deemed to be equal by the
    above logic will compare to be equal (via the `==` operator and via
    `pisa.utils.comparisons.recursiveEquality`) and will also hash to equal
    values (via `pisa.utils.hash.hash_obj`).

    Parameters
    ----------
    obj
        Object to be normalized.

    sigfigs : None or int > 0
        Number of digits to which to round numbers' significands; if None, do
        not round numbers.

    full_norm : bool
        If True, does full translation and normalization which is good across
        independent invocations and is careful about normalizing units, etc.
        If false, certain assumptions are made that modify the behavior
        described below in the Notes section which help speed things up in the
        case of e.g. a minimizer run, where we know certain things won't
        change:
        * Units are not normalized. They are assumed to stay the same from
          run to run.
        * sigfigs are not respected; full significant figures are returned
          (since it takes time to round all values appropriately).

    Returns
    -------
    normed_obj : object roughly of same type as input `obj`
        Simple types are returned as the same type as at the input, Numpy
        ndarrays are returned in the same shape and representation as the
        input, Mappings (dicts) are returned as OrderedDict, and all other
        sequences or iterables are returned as (possibly nested) lists.

    Notes
    -----
    Conversion logic by `obj` type or types found within `obj`:

    * **Sequences and OrderedDicts** (but not numpy arrays) are iterated
      through recursively.
    * **Mappings without ordering** (e.g. dicts) are iterated through
      recursively after sorting their keys, and are returned as
      OrderedDicts (such that the output is always consistent when
      serialized).
    * **Sequences** (not numpy arrays) are iterated through recursively.
    * **Numpy ndarrays** are treated as the below data types (according to the
      array's dtype).
    * **Simple objects** (non-floating point / non-sequence / non-numpy / etc.)
      are returned unaltered (e.g. strings).
    * **Pint quantities** (numbers with units): Convert to their base units.
    * **Floating-point numbers** (including the converted pint quantities):
      Round values to `sigfigs` significant figures.
    * **Numbers with uncertainties** (via the `uncertainties` module) have
      their nominal values rounded as above but their standard deviations are
      rounded to the same order of magnitude (*not* number of significant
      figures) as the nominal.
      Therefore passing obj=10.23+/-0.25 and sigfigs=2 returns 10+/-0.0.
      Note that **correlations are lost** in the outputs of this function, so
      equality of the output requires merely having equal nomial values and
      equal standard deviations.
      The calculations leading to these equal numbers might have used
      independent random variables to arrive at them, however, and so the
      `uncertainties` module would have evaluated them to be unequal. [1]

    To achieve rounding that masks floating point precision issues, set
    `sigfigs` to a value *less than* the number of decimal digits used for the
    significand of the calculation floating point precision.

    For reference, the IEEE 754 floating point standard [2] uses the following:

    * FP16 (half precision): **3.31** significand decimal digits (11 bits)
    * FP32 (single precision): **7.22** significand decimal digits (24 bits)
    * FP64 (double precision): **15.95** significand decimal digits (53 bits)
    * FP128 (quad precision): **34.02** significand decimal digits (113 bits)

    Logic for rounding the significand for numpy arrays was derived from [3].

    References
    ----------
    [1] https://github.com/lebigot/uncertainties/blob/master/uncertainties/test_uncertainties.py#L436

    [2] https://en.wikipedia.org/wiki/IEEE_floating_point

    [3] http://stackoverflow.com/questions/18915378, answer by user BlackGriffin.

    Examples
    --------
    Pint quantities hash to unequal values if specified in different scales or
    different systems of units (even if the underlying physical quantity is
    identical).

    >>> import pint; ureg = pint.UnitRegistry()
    >>> from pisa.utils.hash import hash_obj
    >>> q0 = 1 * ureg.m
    >>> q1 = 100 * ureg.cm
    >>> q0 == q1
    True
    >>> hash_obj(q0) == hash_obj(q1)
    False

    Even the `to_base_units()` method fails for hashing to equal values, as
    `q0` is a float and `q1` is an integer.

    >>> hash_obj(q0.to_base_units()) == hash_obj(q1.to_base_units())
    False

    Even if both quantities are floating point numbers, finite precision
    effects in the `to_base_units` conversion can still cause two things which
    we "know" are equal to evaluate to be unequal.

    >>> q2 = 0.1 * ureg.m
    >>> q3 = 1e5 * ureg.um
    >>> q2 == q3
    True
    >>> q2.to_base_units() == q3.to_base_units()
    False

    `normQuant` handles all of these issues given an appropriate `sigfigs`
    argument.

    >>> q2_normed = normQuant(q2, sigfigs=12)
    >>> q3_normed = normQuant(q3, sigfigs=12)
    >>> q2_normed == q3_normed
    True
    >>> hash_obj(q2_normed) == hash_obj(q3_normed)
    True

    """
    if not full_norm:
        return obj

    # Nothing to convert for strings, None, ...
    if isinstance(obj, basestring) or obj is None:
        return obj

    round_result = False
    if sigfigs is not None:
        if not (int(sigfigs) == float(sigfigs) and sigfigs > 0):
            raise ValueError('`sigfigs` must be an integer > 0.')
        round_result = True
        sigfigs = int(sigfigs)

    # Store kwargs for easily passing to recursive calls of this function
    kwargs = dict(sigfigs=sigfigs, full_norm=full_norm)

    # Recurse into dict by its (sorted) keys (or into OrderedDict using keys in
    # their defined order) and return an OrderedDict in either case.
    if isinstance(obj, Mapping):
        if isinstance(obj, OrderedDict):
            keys = obj.keys()
        else:
            keys = sorted(obj.keys())
        normed_obj = OrderedDict()
        for key in keys:
            normed_obj[key] = normQuant(obj[key], **kwargs)
        return normed_obj

    # Sequences, etc. but NOT numpy arrays (or pint quantities, which are
    # iterable) get their elements normalized and populated to a new list for
    # returning.
    if (isinstance(obj, (Iterable, Iterator, Sequence))
            and not (isinstance(obj, np.ndarray)
                     or isinstance(obj, pint.quantity._Quantity))):
        return [normQuant(x, **kwargs) for x in obj]

    # Must be a numpy array or scalar if we got here...

    # NOTE: the order in which units (Pint module) and uncertainties
    # (uncertainties module) are handled is crucial! Essentially, it appears
    # that Pint is aware of uncertainties, but not vice versa. Hence the
    # ordering and descriptions used below.

    # The outermost "wrapper" of a number or numpy array is its Pint units. If
    # units are present, convert to base units, record the base units, and
    # strip the units off of the quantity by replacing it with its magnitude
    # (in the base units).

    has_units = False
    if isinstance(obj, pint.quantity._Quantity):
        has_units = True
        if full_norm:
            obj = obj.to_base_units()
        units = obj.units
        obj = obj.magnitude

    # The next layer possible for a number or numpy array to have is
    # uncertainties. If uncertainties are attached to `obj`, record a
    # "snapshot" (losing correlations) of the standard deviations. Then replace
    # the number or array solely with its nominal value(s).

    # NOTE: uncertainties.core.AffineScalarFunc includes such functions *and*
    # uncertainties.core.Variable objects

    has_uncertainties = False
    if isinstance(obj, AffineScalarFunc):
        has_uncertainties = True
        std_devs = obj.std_dev
        obj = obj.nominal_value
    elif isinstance(obj, np.ndarray) and np.issubsctype(obj, AffineScalarFunc):
        has_uncertainties = True
        std_devs = unp.std_devs(obj)
        obj = unp.nominal_values(obj)

    # What is done below will convert scalars into arrays, so get this info
    # before it is lost.
    is_scalar = isscalar(obj)

    if round_result:
        # frexp returns *binary* fraction (significand) and *binary* exponent
        bin_significand, bin_exponent = np.frexp(obj)
        exponent = LOG10_2 * bin_exponent
        exponent_integ = np.floor(exponent)
        exponent_fract = exponent - exponent_integ
        significand = bin_significand * 10**(exponent_fract)
        obj = np.around(significand, sigfigs-1) * 10**exponent_integ

    # Now work our way *up* through the hierarchy: First, reintroduce
    # uncertainties

    if has_uncertainties and round_result:
        std_bin_significand, std_bin_exponent = np.frexp(std_devs)
        std_exponent = LOG10_2 * std_bin_exponent
        std_exponent_integ = np.floor(std_exponent)
        std_exponent_fract = std_exponent - std_exponent_integ
        # Don't just scale magnitude by the stddev's fractional exponent; also
        # shift to be on the same scale (power-of-10) as the nominal value
        delta_order_of_mag = std_exponent_integ - exponent_integ
        std_significand = (
            std_bin_significand * 10**(std_exponent_fract + delta_order_of_mag)
        )
        # Now rounding on the stddev's significand occurs at the same order of
        # magnitude as rounding on the nominal value (and so scaling is done
        # with `exponent_integ`, NOT `std_exponent_integ`)
        std_devs = (np.around(std_significand, sigfigs-1) * 10**exponent_integ)

    if has_uncertainties:
        obj = unp.uarray(obj, std_devs)
        # If it was a scalar, it has become a len-1 array; extract the scalar
        if is_scalar:
            obj = obj[0]

    # Finally, attach units if they were present
    if has_units:
        obj = obj * units

    return obj


def test_isscalar():
    assert isscalar(0)
    assert isscalar('xyz')
    assert isscalar('')
    assert isscalar(np.nan)
    assert isscalar(np.inf)
    assert not isscalar(iter([]))
    assert not isscalar({})
    assert not isscalar(tuple())
    assert not isscalar([np.inf])
    assert not isscalar(np.array([np.inf]))
    assert not isscalar(np.array([]))

    a = np.array([-np.inf, np.nan, -1.1, -1, 0, 1, 1.1, np.inf])
    unp_a = unp.uarray(a, np.ones_like(a))
    pint_a = a * ureg.GeV
    pint_unp_a = unp_a * ureg.GeV
    for x in [a, unp_a, pint_a, pint_unp_a]:
        assert not isscalar(x), str(x) + ' should not evalute to scalar'

    u_fl = ufloat(1, 1)
    p_fl = 1 * ureg.GeV
    p_u_fl = ufloat(1, 1) * ureg.GeV
    for x in [u_fl, p_fl, p_u_fl]:
        assert isscalar(x), str(x) + ' should evaluate to scalar'
    logging.info('<< PASSED : test_isscalar >>')


def test_recursiveEquality():
    d1 = {'one':1, 'two':2, 'three': None, 'four': 'four'}
    d2 = {'one':1.0, 'two':2.0, 'three': None, 'four': 'four'}
    d3 = {'one':np.arange(0, 100),
          'two':[{'three':{'four':np.arange(1, 2)}},
                 np.arange(3, 4)]}
    d4 = {'one':np.arange(0, 100),
          'two':[{'three':{'four':np.arange(1, 2)}},
                 np.arange(3, 4)]}
    d5 = {'one':np.arange(0, 100),
          'two':[{'three':{'four':np.arange(1, 3)}},
                 np.arange(3, 4)]}
    d6 = {'one':np.arange(0, 100),
          'two':[{'three':{'four':np.arange(1.1, 2.1)}},
                 np.arange(3, 4)]}
    d7 = OrderedDict()
    d7['d1'] = d1
    d7['f'] = 7.2
    d8 = OrderedDict()
    d8['d1'] = d1
    d8['f'] = 7.2
    assert recursiveEquality(d1, d2)
    assert not recursiveEquality(d1, d3)
    assert recursiveEquality(d3, d4)
    assert not recursiveEquality(d3, d5)
    assert not recursiveEquality(d4, d5)
    assert not recursiveEquality(d3, d6)
    assert not recursiveEquality(d4, d6)
    assert recursiveEquality(d7, d8)

    # Units and quantities (numbers with units)
    ureg0 = pint.UnitRegistry()
    ureg1 = pint.UnitRegistry()
    u0 = ureg0.GeV
    u1 = ureg1.MeV
    u2 = ureg1.GeV
    u3 = ureg1.gigaelectron_volt
    u4 = ureg1.foot
    assert not recursiveEquality(u0, u1), 'noneq. of diff. unit, diff. reg'
    assert not recursiveEquality(u1, u2), 'noneq. of diff. unit same reg'
    assert recursiveEquality(u0, u2), 'eq. of same unit across registries'
    assert recursiveEquality(u2, u3), 'eq. of same unit in same registry'
    q0 = np.ones(100) * u0
    q1 = np.ones(100) * 1000.0 * u1
    assert recursiveEquality(q0, q1)
    q2 = 1e5*np.ones(100) * ureg0.um
    q3 = 0.1 * np.ones(100) * ureg0.m
    assert recursiveEquality(q2, q3)
    q4, q5 = np.ones(10) * 1000. * ureg0.MeV, np.ones(10) * ureg0.GeV
    assert recursiveEquality(q4, q5)

    # Special numerical values
    assert recursiveEquality(np.nan, np.nan)
    assert recursiveEquality(np.inf, np.inf)
    assert recursiveEquality(-np.inf, -np.inf)
    assert not recursiveEquality(np.inf, -np.inf)
    assert not recursiveEquality(np.inf, np.nan)
    logging.info('<< PASSED : test_recursiveEquality >>')


def test_normQuant():
    # TODO: test:
    # * non-numerical
    #   * single non-numerical
    #   * sequence (tuple, list) of non-numirical
    #   * np.array of non-numerical
    # * scalar
    #   * bare
    #   * with units only
    #   * with uncertainties only
    #   * with uncertanties and units
    # * multi-dimensional array
    #   * integers, bare
    #   * float32, bare
    #   * float64, bare
    #   * float64 with units only
    #   * float64 with uncertainties only
    #   * float64 with both units and uncertainties
    # * nested objects... ?
    assert normQuant('xyz') == normQuant('xyz')
    assert normQuant('xyz', sigfigs=12) == normQuant('xyz', sigfigs=12)
    assert normQuant(None) == normQuant(None)
    assert normQuant(None, sigfigs=12) == normQuant(None, sigfigs=12)
    assert normQuant(1) == normQuant(1)
    assert normQuant(1, sigfigs=12) == normQuant(1, sigfigs=12)
    assert normQuant(1) == normQuant(1.0)
    assert normQuant(1, sigfigs=12) == normQuant(1.0, sigfigs=12)
    assert normQuant(1.001) != normQuant(1.002)
    assert normQuant(1.001, sigfigs=3) == normQuant(1.002, sigfigs=3)
    s0 = 1e5*ureg.um
    s1 = 1e5*ureg.um
    # ...

    q0 = 1e5*np.ones(10)*ureg.um
    q1 = 0.1*np.ones(10)*ureg.m
    assert not np.any(q0 == q1)
    assert not np.any(q0.to_base_units() == q1.to_base_units())
    assert not np.any(normQuant(q0, None) == normQuant(q1, None))
    assert not np.any(normQuant(q0, 18) == normQuant(q1, 18))
    assert np.all(normQuant(q0, 16) == normQuant(q1, 16))
    assert np.all(normQuant(q0, 15) == normQuant(q1, 15))
    assert np.all(normQuant(q0, 1) == normQuant(q1, 1))
    assert normQuant(np.inf, sigfigs=15) == normQuant(np.inf, sigfigs=15)
    assert normQuant(-np.inf, sigfigs=15) == normQuant(-np.inf, sigfigs=15)
    assert normQuant(np.inf, sigfigs=15) != normQuant(-np.inf, sigfigs=15)
    assert normQuant(np.nan, sigfigs=15) != normQuant(np.nan, sigfigs=15)
    logging.info('<< PASSED : test_normQuant >>')


if __name__ == '__main__':
    set_verbosity(1)
    test_isscalar()
    test_recursiveEquality()
    test_normQuant()

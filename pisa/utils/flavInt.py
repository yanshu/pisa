#! /usr/bin/env python
#
# author: Justin L. Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   October 24, 2015
"""
Classes for working with neutrino flavors (NuFlav), interactions types
(IntType), "flavints" (a flavor and an interaction type) (NuFlavInt), and
flavint groups (NuFlavIntGroup) in a consistent and convenient manner.

FlavIntData class for working with data stored by flavint (flavor &
interaction type). This should replace the PISA convention of using raw
doubly-nested dictionaries indexed as [<flavor>][<interaction type>]. For
now, FlavIntData objects can be drop-in replacements for such dictionaries
(they can be accessed and written to in the same way since FlavIntData
subclasses dict) but this should be deprecated; eventually, all direct access
of the data structure should be eliminated and disallowed by the FlavIntData
object.

Define convenience tuples ALL_{x} for easy iteration

"""


# TODO: Make strings convertible to various types less liberal. E.g., I already
# converted NuFlav to NOT accept 'numucc' such that things like 'numu nue' or
# 'nu xyz mutation' would also be rejected; this should be true also for
# interaction type and possibly others I haven't thought about yet. Note that I
# achieved this using the IGNORE regex that ignores all non-alpha characters
# but asserts one and only one match to the regex (consult NuFlav for details).

# TODO: make simpleStr() method convertible back to NuFlavIntGroup, either by
# increasing the intelligence of interpret(), by modifying what simpleStr()
# produces, or by adding another function to interpret simple strings. (I'm
# leaning towards the second option at the moment, since I don't see how to
# make the first interpret both a simplestr AND nue as nuecc+nuenc, and I
# don't think there's a way to know "this is a simple str" vs not easily.)


from collections import MutableSequence, MutableMapping, Mapping, Sequence
from copy import deepcopy
from itertools import product, combinations, izip
from operator import add
import re
import sys
import traceback

import numpy as np
import pint

from pisa import ureg, Q_
from pisa.utils import fileio
from pisa.utils.log import logging, set_verbosity
from pisa.utils.comparisons import recursiveAllclose, recursiveEquality


__all__ = ['NuFlav', 'NuFlavInt', 'NuFlavIntGroup', 'FlavIntData',
           'FlavIntDataGroup', 'CombinedFlavIntData', 'xlateGroupsStr',
           'flavintGroupsFromString', 'IntType', 'BarSep', 'set_bar_ssep',
           'get_bar_ssep', 'tex', 'ALL_NUPARTICLES', 'ALL_NUANTIPARTICLES',
           'ALL_NUFLAVS']


global __BAR_SSEP__
__BAR_SSEP__ = ''


class BarSep(object):
    def __init__(self, val):
        global __BAR_SSEP__
        self.old_val = __BAR_SSEP__
        self.new_val = val

    def __enter__(self):
        global __BAR_SSEP__
        __BAR_SSEP__ = self.new_val

    def __exit__(self, type, value, traceback):
        global __BAR_SSEP__
        __BAR_SSEP__ = self.old_val


def set_bar_ssep(val):
    global __BAR_SSEP__
    assert isinstance(val, basestring)
    __BAR_SSEP__ = val


def get_bar_ssep():
    global __BAR_SSEP__
    return __BAR_SSEP__


# TODO: move this to central loc in utils
def tex(x, d=False):
    if d:
        return '$' + x.tex() + '$'
    return x.tex()


class NuFlav(object):
    """Class for handling neutrino flavors (and anti-flavors)"""
    PART_CODE = 1
    ANTIPART_CODE = -1
    NUE_CODE = 12
    NUMU_CODE = 14
    NUTAU_CODE = 16
    NUEBAR_CODE = -12
    NUMUBAR_CODE = -14
    NUTAUBAR_CODE = -16
    IGNORE = re.compile(r'[^a-zA-Z]')
    FLAV_RE = re.compile(
        r'^(?P<fullflav>(?:nue|numu|nutau)(?P<barnobar>bar){0,1})$'
    )
    def __init__(self, val):
        self.fstr2code = {
            'nue': self.NUE_CODE,
            'numu': self.NUMU_CODE,
            'nutau': self.NUTAU_CODE,
            'nuebar': self.NUEBAR_CODE,
            'numubar': self.NUMUBAR_CODE,
            'nutaubar': self.NUTAUBAR_CODE
        }
        self.barnobar2code = {
            None: self.PART_CODE,
            '': self.PART_CODE,
            'bar': self.ANTIPART_CODE,
        }
        self.f2tex = {
            self.NUE_CODE:  r'{\nu_e}',
            self.NUMU_CODE:  r'{\nu_\mu}',
            self.NUTAU_CODE:  r'{\nu_\tau}',
            self.NUEBAR_CODE: r'{\bar\nu_e}',
            self.NUMUBAR_CODE: r'{\bar\nu_\mu}',
            self.NUTAUBAR_CODE: r'{\bar\nu_\tau}',
        }
        # Instantiate this neutrino flavor object by interpreting val
        orig_val = val
        try:
            if isinstance(val, basestring):
                # Sanitize the string
                sanitized_val = self.IGNORE.sub('', val.lower())
                matches = self.FLAV_RE.findall(sanitized_val)
                if len(matches) != 1:
                    raise ValueError('Invalid NuFlav spec: "%s"' % val)
                self.__flav = self.fstr2code[matches[0][0]]
                self.__barnobar = self.barnobar2code[matches[0][1]]
            elif hasattr(val, 'flavCode'):
                self.__flav = val.flavCode()
                self.__barnobar = np.sign(self.__flav)
            elif hasattr(val, 'flav'):
                self.__flav = val.flav().flavCode()
                self.__barnobar = np.sign(self.__flav)
            else:
                if val in self.fstr2code.values():
                    self.__flav = int(val)
                    self.__barnobar = np.sign(self.__flav)
                else:
                    raise ValueError('Invalid neutrino flavor/code: "%s"' %
                                     str(val))
            # Double check than flav and barnobar codes are valid
            assert self.__flav in self.fstr2code.values()
            assert self.__barnobar in self.barnobar2code.values()
        except (AssertionError, ValueError, AttributeError):
            exc_type, exc_value, exc_traceback = sys.exc_info()
            raise ValueError(
                'Could not interpret value "' + orig_val + '":\n' + '\n'.join(
                    traceback.format_exception(exc_type, exc_value,
                                               exc_traceback)
                )
            )

    def __str__(self):
        global __BAR_SSEP__
        fstr = [s for s, code in self.fstr2code.items() if code == self.__flav]
        fstr = fstr[0]
        fstr = fstr.replace('bar', __BAR_SSEP__+'bar')
        return fstr

    # TODO: copy, deepcopy, and JSON serialization
    #def __copy__(self):
    #    return self.__str__()

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__flav)

    def __cmp__(self, other):
        if not isinstance(other, NuFlav):
            return 1
        c0 = self.flavCode()
        c1 = other.flavCode()
        ac0 = abs(c0)
        ac1 = abs(c1)
        if ac0 < ac1:
            return -1
        if ac0 == ac1:
            return c1 - c0
        if ac0 > ac1:
            return +1

    def __neg__(self):
        return NuFlav(self.__flav*-1)

    def __add__(self, other):
        return NuFlavIntGroup(self, other)

    def tex(self):
        """TeX string"""
        return self.f2tex[self.__flav]

    def flavCode(self):
        """Integer PDG code"""
        return self.__flav

    def barNoBar(self):
        """Return +/-1 for particle/antiparticle"""
        return self.__barnobar

    def isParticle(self):
        """Is this a particle (vs. antiparticle) flavor?"""
        return self.__barnobar == self.PART_CODE

    def isAntiParticle(self):
        """Is this an antiparticle flavor?"""
        return self.__barnobar == self.ANTIPART_CODE

    def pidx(self, d, *args):
        """Extract data from a nested dictionary `d` whose format is commonly
        found in PISA

        The dictionary must have the format
            d = {"<flavor>": <data object>}
            <flavor> is one of "nue", "nue_bar", "numu", "numu_bar", "nutau",
                "nutau_bar"
        """
        with BarSep('_'):
            field = d[str(self)]
        for idx in args:
            field = field[idx]
        return field


ALL_NUPARTICLES = (NuFlav(12), NuFlav(14), NuFlav(16))
ALL_NUANTIPARTICLES = (NuFlav(-12), NuFlav(-14), NuFlav(-16))
ALL_NUFLAVS = tuple(sorted(list(ALL_NUPARTICLES) + list(ALL_NUANTIPARTICLES)))


# TODO: are the following two classes redundant now?
class AllNu(object):
    def __init__(self):
        self.__flav = [p for p in ALL_NUPARTICLES]

    def flav(self):
        return self.__flav

    def __str__(self):
        return 'nuall'

    def tex(self):
        return r'{\nu_{\rm all}}'


class AllNuBar(object):
    def __init__(self):
        self.__flav = [p for p in ALL_NUANTIPARTICLES]

    def flav(self):
        return self.__flav

    def __str__(self):
        return 'nuallbar'

    def tex(self):
        return r'{\bar\nu_{\rm all}}'


class IntType(object):
    """
    Interaction type object.

    Instantiate via
      * Numerical code: 1=CC, 2=NC
      * String (case-insensitive; all characters besides valid tokens are
        ignored)
      * Instantiated IntType object (or any method implementing intTypeCode()
        which returns a valid interaction type code)
      * Instantiated NuFlavInt object (or any object implementing intType()
        which returns a valid IntType object)

    The following, e.g., are all interpreted as charged-current IntTypes:
      IntType('cc')
      IntType('\n\t _cc \n')
      IntType('numubarcc')
      IntType(1)
      IntType(1.0)
      IntType(IntType('cc'))
      IntType(NuFlavInt('numubarcc'))
    """
    CC_CODE = 1
    NC_CODE = 2
    IGNORE = re.compile(r'[^a-zA-Z]')
    IT_RE = re.compile(r'^(cc|nc)$')
    def __init__(self, val):
        self.istr2code = {
            'cc': self.CC_CODE,
            'nc': self.NC_CODE,
        }
        self.i2tex = {
            self.CC_CODE: r'{\rm CC}',
            self.NC_CODE: r'{\rm NC}'
        }

        # Interpret `val`
        try:
            orig_val = val
            if isinstance(val, basestring):
                sanitized_val = self.IGNORE.sub('', val.lower())
                int_type = self.IT_RE.findall(sanitized_val)
                if len(int_type) != 1:
                    raise ValueError('Invalid IntType spec: "%s"' % val)
                self.__int_type = self.istr2code[int_type[0]]
            elif hasattr(val, 'intType'):
                self.__int_type = val.intType().intTypeCode()
            elif hasattr(val, 'intTypeCode'):
                self.__int_type = val.intTypeCode()
            else:
                if val in self.istr2code.values():
                    self.__int_type = int(val)
                else:
                    raise TypeError('Unhandled type: "%s"' %
                                    str(type(int_type)))
            # Double check that the interaction type code set is valid
            assert self.__int_type in self.istr2code.values()
        except (AssertionError, TypeError, ValueError, AttributeError):
            exc_type, exc_value, exc_traceback = sys.exc_info()
            raise ValueError('Could not interpret value "%s":  %s' % (
                str(orig_val),
                '\n'.join(traceback.format_exception(
                    exc_type, exc_value, exc_traceback))
            ))

    def __str__(self):
        return [s for s, code in self.istr2code.items()
                if code == self.__int_type][0]

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__int_type)

    def __cmp__(self, other):
        if not isinstance(other, IntType):
            return 1
        return self.intTypeCode() - other.intTypeCode()

    def isCC(self):
        """Is this interaction type charged current (CC)?"""
        return self.__int_type == self.CC_CODE

    def isNC(self):
        """Is this interaction type neutral current (NC)?"""
        return self.__int_type == self.NC_CODE

    def intTypeCode(self):
        """Integer code for this interaction type"""
        return self.__int_type

    def tex(self):
        """TeX representation of this interaction type"""
        return self.i2tex[self.__int_type]


ALL_NUINT_TYPES = (IntType('cc'), IntType('nc'))


class NuFlavInt(object):
    """A neutrino "flavint" encompasses both the neutrino flavor and its
    interaction type.

    Instantiate via
      * String containing a single flavor and a single interaction type
        e.g.: 'numucc', 'nu_mu_cc', 'nu mu CC', 'numu_bar CC', etc.
      * Another instantiated NuFlavInt object
      * Two separate objects that can be converted to a valid NuFlav
        and a valid IntType (in that order)
      * An iterable of length two which contains such objects
      * kwargs `flav` and `int_type` specifying such objects

    String specifications simply ignore all characters not recognized as a
    valid token.

    """
    TOKENS = re.compile('(nu|e|mu|tau|bar|nc|cc)')
    FINT_RE = re.compile(
        r'(?P<fullflav>(?:nue|numu|nutau)'
        r'(?P<barnobar>bar){0,1})'
        r'(?P<int_type>cc|nc){0,1}'
    )
    FINT_SSEP = '_'
    FINT_TEXSEP = r' \, '
    # TODO: use multiple inheritance to clean up the below?
    def __init__(self, *args, **kwargs):
        if kwargs:
            if args:
                raise TypeError('Either positional or keyword args may be'
                                ' provided, but not both')
            keys = kwargs.keys()
            if len(set(keys).difference(set(('flav', 'int_type')))) != 0:
                raise TypeError('Invalid kwarg(s) specified: %s' %
                                kwargs.keys())
            flav_int = (kwargs['flav'], kwargs['int_type'])
        elif args:
            if len(args) == 0:
                raise TypeError('No flavint specification provided')
            elif len(args) == 1:
                flav_int = args[0]
            elif len(args) == 2:
                flav_int = args
            elif len(args) > 2:
                raise TypeError('More than two args')

        if not isinstance(flav_int, basestring) \
                and hasattr(flav_int, '__len__') and len(flav_int) == 1:
            flav_int = flav_int[0]

        if isinstance(flav_int, basestring):
            orig_flav_int = flav_int
            try:
                flav_int = ''.join(self.TOKENS.findall(flav_int.lower()))
                flavint_dict = self.FINT_RE.match(flav_int).groupdict()
                self.__flav = NuFlav(flavint_dict['fullflav'])
                self.__int_type = IntType(flavint_dict['int_type'])
            except (UnboundLocalError, ValueError, AttributeError):
                exc_type, exc_value, exc_traceback = sys.exc_info()
                raise ValueError(
                    'Could not interpret value "%s" as valid flavint: %s' %
                    (str(orig_flav_int),
                     '\n'.join(traceback.format_exception(exc_type, exc_value,
                                                          exc_traceback)))
                )
        elif hasattr(flav_int, '__len__'):
            assert len(flav_int) == 2, \
                    'Need 2 components to define flavor and interaction type'
            self.__flav = NuFlav(flav_int[0])
            self.__int_type = IntType(flav_int[1])
        elif isinstance(flav_int, NuFlavInt):
            self.__flav = NuFlav(flav_int.flav())
            self.__int_type = IntType(flav_int.intTypeCode())
        else:
            raise TypeError('Unhandled type: "' + str(type(flav_int)) +
                            '"; class: "' + str(flav_int.__class__) +
                            '; value: "' + str(flav_int) + '"')

    def __str__(self):
        return self.flavStr() + self.FINT_SSEP + self.intTypeStr()

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash((self.flavCode(), self.intTypeCode()))

    def __cmp__(self, other):
        if not isinstance(other, NuFlavInt):
            return 1
        return cmp(
            (self.flav(), self.intType()), (other.flav(), other.intType())
        )

    def __neg__(self):
        return NuFlavInt(-self.__flav, self.__int_type)

    def __add__(self, other):
        return NuFlavIntGroup(self, other)

    def pidx(self, d, *args):
        """Extract data from a nested dictionary `d` whose format is commonly
        found in PISA

        The dictionary must have the format
            d = {"<flavor>": {"<interaction type>": <data object>}}
            <flavor> is one of "nue", "nue_bar", "numu", "numu_bar", "nutau",
                "nutau_bar"
            <interaction type> is one of "cc", "nc"
        """

        with BarSep('_'):
            field = d[str(self.flav())][str(self.intType())]
        for idx in args:
            field = field[idx]
        return field

    def flav(self):
        """Return just the NuFlav part of this NuFlavInt"""
        return self.__flav

    def flavCode(self):
        """Return the integer PDG code for this flavor"""
        return self.__flav.flavCode()

    def barNoBar(self):
        """Return +/-1 depending on if the flavor is a particle/antiparticle"""
        return self.__flav.barNoBar()

    def isParticle(self):
        """Is this a particle (vs. antiparticle) flavor?"""
        return self.__flav.isParticle()

    def isAntiParticle(self):
        """Is this an antiparticle flavor?"""
        return self.__flav.isAntiParticle()

    def isCC(self):
        """Is this interaction type charged current (CC)?"""
        return self.__int_type.isCC()

    def isNC(self):
        """Is this interaction type neutral current (NC)?"""
        return self.__int_type.isNC()

    def intType(self):
        """Return IntType object that composes this NuFlavInt"""
        return self.__int_type

    def intTypeCode(self):
        """Return integer code for IntType that composes this NuFlavInt"""
        return self.__int_type.intTypeCode()

    def flavStr(self):
        """String representation of flavor that comprises this NuFlavInt"""
        return str(self.__flav)

    def intTypeStr(self):
        """String representation of interaction type that comprises this
        NuFlavInt"""
        return str(self.__int_type)

    def flavTex(self):
        """TeX string representation of the flavor that comprises this
        NuFlavInt"""
        return self.__flav.tex()

    def intTypeTex(self):
        """TeX string representation of interaction type that comprises this
        NuFlavInt"""
        return self.__int_type.tex()

    def tex(self):
        """TeX string representation of this NuFlavInt"""
        return '{%s%s%s}' % (self.flavTex(),
                             self.FINT_TEXSEP,
                             self.intTypeTex())


class NuFlavIntGroup(MutableSequence):
    """Grouping of neutrino flavors+interaction types (flavints)

    Grouping of neutrino flavints. Specification can be via
      * A single `NuFlav` object; this gets promoted to include both
        interaction types
      * A single `NuFlavInt` object
      * String:
        * Ignores anything besides valid tokens
        * A flavor with no interaction type specified will include both CC
          and NC interaction types
        * Multiple flavor/interaction-type specifications can be made;
          use of delimiters is optional
        * Interprets "nuall" as nue+numu+nutau and "nuallbar" as
          nuebar+numubar+nutaubar
      * Iterable containing any of the above (i.e., objects convertible to
        `NuFlavInt` objects). Note that a valid iterable is another
        `NuFlavIntGroup` object.
    """
    TOKENS = re.compile('(nu|e|mu|tau|all|bar|nc|cc)')
    IGNORE = re.compile(r'[^a-zA-Z]')
    FLAVINT_RE = re.compile(
        r'((?:nue|numu|nutau|nuall)(?:bar){0,1}(?:cc|nc){0,2})'
    )
    FLAV_RE = re.compile(r'(?P<fullflav>(?:nue|numu|nutau|nuall)(?:bar){0,1})')
    IT_RE = re.compile(r'(cc|nc)')
    def __init__(self, *args):
        self.flavint_ssep = '+'
        self.__flavints = []
        # Possibly a special case if len(args) == 2, so send as a single entity
        # if this is the case
        if len(args) == 2:
            args = [args]
        [self.__iadd__(a) for a in args]

    def __add__(self, val):
        flavint_list = sorted(set(self.__flavints + self.interpret(val)))
        return NuFlavIntGroup(flavint_list)

    def __iadd__(self, val):
        self.__flavints = sorted(set(self.__flavints + self.interpret(val)))
        return self

    def __delitem__(self, idx):
        self.__flavints.__delitem__(idx)

    def remove(self, val):
        """
        Remove a flavint from this group.

        `val` must be valid for the interpret() method
        """
        flavint_list = sorted(set(self.interpret(val)))
        for k in flavint_list:
            try:
                idx = self.__flavints.index(k)
            except ValueError:
                pass
            else:
                del self.__flavints[idx]

    def __sub__(self, val):
        cp = deepcopy(self)
        cp.remove(val)
        return cp

    def __isub__(self, val):
        self.remove(val)
        return self

    def __setitem__(self, idx, val):
        self.__flavints[idx] = val

    def insert(self, idx, val):
        self.__flavints.insert(idx, val)

    def __cmp__(self, other):
        if not isinstance(other, NuFlavIntGroup):
            return 1
        if len(other) != len(self):
            return len(self) - len(other)
        cmps = [cmp(mine, other[n]) for n, mine in enumerate(self.__flavints)]
        if all([c == 0 for c in cmps]):
            return 0
        return [c for c in cmps if c != 0][0]

    def __contains__(self, val):
        return all([(k in self.__flavints) for k in self.interpret(val)])

    def __len__(self):
        return len(self.__flavints)

    def __getitem__(self, idx):
        return self.__flavints[idx]

    def __str__(self):
        allkg = set(self.flavints())

        # Check if nuall or nuallbar CC, NC, or both
        nuallcc, nuallbarcc, nuallnc, nuallbarnc = False, False, False, False
        ccFlavInts = NuFlavIntGroup(self.ccFlavInts())
        ncFlavInts = NuFlavIntGroup(self.ncFlavInts())
        if len(ccFlavInts.particles()) == 3:
            nuallcc = True
        if len(ccFlavInts.antiParticles()) == 3:
            nuallbarcc = True
        if len(ncFlavInts.particles()) == 3:
            nuallnc = True
        if len(ncFlavInts.antiParticles()) == 3:
            nuallbarnc = True

        # Construct nuall(bar) part(s) of string
        strs = []
        if nuallcc and nuallnc:
            strs.append('nuall')
            [allkg.remove(NuFlavInt(k, 'cc')) for k in ALL_NUPARTICLES]
            [allkg.remove(NuFlavInt(k, 'nc')) for k in ALL_NUPARTICLES]
        elif nuallcc:
            strs.append('nuall' + NuFlavInt.FINT_SSEP + str(IntType('cc')))
            [allkg.remove(NuFlavInt(k, 'cc')) for k in ALL_NUPARTICLES]
        elif nuallnc:
            strs.append('nuall' + NuFlavInt.FINT_SSEP + str(IntType('nc')))
            [allkg.remove(NuFlavInt(k, 'nc')) for k in ALL_NUPARTICLES]

        if nuallbarcc and nuallbarnc:
            strs.append('nuallbar')
            [allkg.remove(NuFlavInt(k, 'cc')) for k in ALL_NUANTIPARTICLES]
            [allkg.remove(NuFlavInt(k, 'nc')) for k in ALL_NUANTIPARTICLES]
        elif nuallbarcc:
            strs.append('nuallbar' + NuFlavInt.FINT_SSEP + str(IntType('cc')))
            [allkg.remove(NuFlavInt(k, 'cc')) for k in ALL_NUANTIPARTICLES]
        elif nuallbarnc:
            strs.append('nuallbar' + NuFlavInt.FINT_SSEP + str(IntType('nc')))
            [allkg.remove(NuFlavInt(k, 'nc')) for k in ALL_NUANTIPARTICLES]

        # Among remaining flavints, group by flavor and combine if both CC and
        # NC are present for individual flavors (i.e., eliminate the intType
        # string altogether)
        for flav in ALL_NUPARTICLES + ALL_NUANTIPARTICLES:
            if flav in [k.flav() for k in allkg]:
                cc, nc = False, False
                if NuFlavInt(flav, 'cc') in allkg:
                    cc = True
                if NuFlavInt(flav, 'nc') in allkg:
                    nc = True
                if cc and nc:
                    strs.append(str(flav))
                    allkg.remove(NuFlavInt(flav, 'cc'))
                    allkg.remove(NuFlavInt(flav, 'nc'))
                elif cc:
                    strs.append(str(NuFlavInt(flav, 'cc')))
                    allkg.remove(NuFlavInt(flav, 'cc'))
                elif nc:
                    strs.append(str(NuFlavInt(flav, 'nc')))
                    allkg.remove(NuFlavInt(flav, 'nc'))
        return self.flavint_ssep.join(strs)

    def __repr__(self):
        return self.__str__()

    # TODO:
    # Technically, since this is a mutable type, the __hash__ method shouldn't
    # be implemented as this will allow for "illegal" behavior, like using
    # a NuFlavIntGroup as a key in a dict. So this should be fixed, maybe.
    #__hash__ = None
    def __hash__(self):
        return hash(tuple(self.__flavints))

    @staticmethod
    def interpret(val):
        """Interpret a NuFlavIntGroup arg"""
        if isinstance(val, basestring):
            orig_val = val
            try:
                flavints = []
                orig_val = val

                # Eliminate anything besides valid tokens
                val = NuFlavIntGroup.IGNORE.sub('', val.lower())
                #val = ''.join(NuFlavIntGroup.TOKENS.findall(val))

                # Find all flavints specified
                allflavints_str = NuFlavIntGroup.FLAVINT_RE.findall(val)
                # Remove flavints
                val = NuFlavIntGroup.FLAVINT_RE.sub('', val)

                for flavint_str in allflavints_str:
                    match = NuFlavIntGroup.FLAV_RE.match(flavint_str)
                    flav = match.groupdict()['fullflav']

                    # A flavint found above can include 'all' which is actually
                    # three different flavors
                    if 'all' in flav:
                        flavs = [flav.replace('all', x)
                                 for x in ('e', 'mu', 'tau')]
                    else:
                        flavs = [flav]

                    ints = sorted(set(
                        NuFlavIntGroup.IT_RE.findall(flavint_str)
                    ))

                    # If flavint_str does not include 'cc' or 'nc', include both
                    if len(ints) == 0:
                        ints = ['cc', 'nc']

                    # Add all combinations of (flav, int) found in this
                    # flavint_str
                    flavints.extend([''.join(fi)
                                     for fi in product(flavs, ints)])

            except (ValueError, AttributeError):
                exc_type, exc_value, exc_traceback = sys.exc_info()
                raise ValueError('Could not interpret value "' + orig_val
                                 + '":\n' + '\n'.join(
                                     traceback.format_exception(exc_type,
                                                                exc_value,
                                                                exc_traceback)
                                 ))
        elif isinstance(val, NuFlav):
            flavints = [NuFlavInt((val, 'cc')), NuFlavInt((val, 'nc'))]
        elif isinstance(val, NuFlavInt):
            flavints = [val]
        elif isinstance(val, NuFlavIntGroup):
            flavints = list(val.flavints())
        elif np.isscalar(val):
            flavints = [val]
        elif val is None:
            flavints = []
        elif hasattr(val, '__len__'):
            flavints = []
            # Treat length-2 iterables as special case, in case the two
            # elements can form a single NuFlavInt.
            if len(val) == 2:
                try_again = True
                try:
                    # Start with counter-hypothesis: that the two elements of
                    # `val` can form two valid, independent NuFlavInts...
                    k1 = NuFlavIntGroup.interpret(val[0])
                    k2 = NuFlavIntGroup.interpret(val[1])
                    if k1 and k2:
                        # Success: Two independent NuFlavInts were created
                        try_again = False
                        flavints.extend(k1)
                        flavints.extend(k2)
                except (UnboundLocalError, ValueError, AssertionError,
                        TypeError):
                    pass
                if try_again:
                    # If the two elements of the iterable did not form two
                    # NuFlavInts, try forming a single NuFlavInt with `val`
                    flavints = [NuFlavInt(val)]
            else:
                # If 1 or >2 elements in `val`, make a flavint out of each
                [flavints.extend(NuFlavIntGroup.interpret(x)) for x in val]
        else:
            raise Exception('Unhandled val: ' + str(val) + ', class '
                            + str(val.__class__) + ' type ' + str(val))

        flavint_list = []
        for k in flavints:
            try:
                nk = NuFlavInt(k)
                flavint_list.append(nk)
            except TypeError:
                # If NuFlavInt failed, try NuFlav; if this fails, give up.
                flav = NuFlav(k)
                flavint_list.append(NuFlavInt((flav, 'cc')))
                flavint_list.append(NuFlavInt((flav, 'nc')))
        return flavint_list

    def flavints(self):
        """Return tuple of all NuFlavInts that make up this group"""
        return tuple(self.__flavints)

    def flavs(self):
        """Return tuple of unique flavors that make up this group"""
        return tuple(sorted(set([k.flav() for k in self.__flavints])))

    def ccFlavInts(self):
        """Return tuple of unique charged-current-interaction NuFlavInts that
        make up this group"""
        return tuple([k for k in self.__flavints
                      if k.intType() == IntType('cc')])

    def ncFlavInts(self):
        """Return tuple of unique neutral-current-interaction NuFlavInts that
        make up this group"""
        return tuple([k for k in self.__flavints
                      if k.intType() == IntType('nc')])

    def particles(self):
        """Return tuple of unique particle (vs antiparticle) NuFlavInts that
        make up this group"""
        return tuple([k for k in self.__flavints if k.isParticle()])

    def antiParticles(self):
        """Return tuple of unique antiparticle NuFlavInts that make up this
        group"""
        return tuple([k for k in self.__flavints if k.isAntiParticle()])

    def ccFlavs(self):
        """Return tuple of unique charged-current-interaction flavors that
        make up this group. Note that only the flavors, and not NuFlavInts, are
        returned (cf. method `ccFlavInts`"""
        return tuple(sorted(set([k.flav() for k in self.__flavints
                                 if k.intType() == IntType('cc')])))

    def ncFlavs(self):
        """Return tuple of unique neutral-current-interaction flavors that
        make up this group. Note that only the flavors, and not NuFlavInts, are
        returned (cf. method `ncFlavInts`"""
        return tuple(sorted(set([k.flav() for k in self.__flavints
                                 if k.intType() == IntType('nc')])))

    #def uniqueFlavs(self):
    #    """Return tuple of unique flavors that make up this group"""
    #    return tuple(sorted(set([k.flav() for k in self.__flavints])))

    def groupFlavsByIntType(self):
        """Return a dictionary with flavors grouped by the interaction types
        represented in this group.

        The returned dictionary has format
        {
            'all_int_type_flavs': [<NuFlav object>, <NuFlav object>, ...],
            'cc_only_flavs':      [<NuFlav object>, <NuFlav object>, ...],
            'nc_only_flavs':      [<NuFlav object>, <NuFlav object>, ...],
        }
        where the lists of NuFlav objects are mutually exclusive
        """
        uniqueF = self.flavs()
        fint_d = {f:set() for f in uniqueF}
        [fint_d[k.flav()].add(k.intType()) for k in self.flavints()]
        grouped = {
            'all_int_type_flavs': [],
            'cc_only_flavs' : [],
            'nc_only_flavs': []
        }
        for f in uniqueF:
            if len(fint_d[f]) == 2:
                grouped['all_int_type_flavs'].append(f)
            elif list(fint_d[f])[0] == IntType('cc'):
                grouped['cc_only_flavs'].append(f)
            else:
                grouped['nc_only_flavs'].append(f)
        return grouped

    def __simpleStr(self, flavsep, intsep, flavintsep, addsep, func):
        grouped = self.groupFlavsByIntType()
        all_nu = AllNu()
        all_nubar = AllNuBar()
        for k, v in grouped.items():
            if all([f in v for f in all_nubar.flav()]):
                [grouped[k].remove(f) for f in all_nubar.flav()]
                grouped[k].insert(0, all_nubar)
            if all([f in v for f in all_nu.flav()]):
                [grouped[k].remove(f) for f in all_nu.flav()]
                grouped[k].insert(0, all_nu)
        all_s = flavsep.join([func(f) for f in grouped['all_int_type_flavs']])
        cc_only_s = flavsep.join([func(f) for f in grouped['cc_only_flavs']])
        nc_only_s = flavsep.join([func(f) for f in grouped['nc_only_flavs']])
        strs = []
        if len(all_s) > 0:
            strs.append(all_s + intsep + func(IntType('cc')) + addsep +
                        func(IntType('nc')))
        if len(cc_only_s) > 0:
            strs.append(cc_only_s + intsep + func(IntType('cc')))
        if len(nc_only_s) > 0:
            strs.append(nc_only_s + intsep + func(IntType('nc')))
        return flavintsep.join(strs)

    def simpleStr(self, flavsep='+', intsep=' ', flavintsep=', ',
                  addsep='+'):
        """Simple string representation of this group"""
        return self.__simpleStr(flavsep=flavsep, intsep=intsep,
                                flavintsep=flavintsep, addsep=addsep, func=str)

    def fileStr(self, flavsep='_', intsep='_', flavintsep='__', addsep=''):
        """String representation for this group useful for file names"""
        return self.__simpleStr(flavsep=flavsep, intsep=intsep,
                                flavintsep=flavintsep, addsep=addsep, func=str)

    def simpleTex(self, flavsep=r' + ', intsep=r' \, ',
                  flavintsep=r'; \; ', addsep=r'+'):
        """Simplified TeX string reperesentation of this group"""
        return self.__simpleStr(flavsep=flavsep, intsep=intsep,
                                flavintsep=flavintsep, addsep=addsep, func=tex)

    def tex(self, *args, **kwargs):
        """TeX string representation for this group"""
        return self.simpleTex(*args, **kwargs)

    def uniqueFlavsTex(self, flavsep=r' + '):
        """TeX string representation of the unique flavors present in this
        group"""
        return flavsep.join([f.tex() for f in self.flavs()])


ALL_NUFLAVINTS = NuFlavIntGroup('nuall,nuallbar')
ALL_NUCC = NuFlavIntGroup('nuall_cc,nuallbar_cc')
ALL_NUNC = NuFlavIntGroup('nuall_nc,nuallbar_nc')


class FlavIntData(dict):
    """Container class for storing data for each NuFlavInt.

    Parameters
    ----------
    val : string, dict, or None
        Data with which to populate the hierarchy.

        If string, interpret as PISA resource and load data from it
        If dict, populate data from the dictionary
        If None, instantiate with None for all data

        The interpreted version of `val` must be a valid data structure: A
        dict with keys 'nue', 'numu', 'nutau', 'nue_bar', 'numu_bar', and
        'nutau_bar'; and each item corresponding to these keys must itself be a
        dict with keys 'cc' and 'nc'.

    Notes
    -----
    Accessing data (both for getting and setting) is fairly flexible. It uses
    dict-like square-brackets syntax, but can accept any object (or two
    objects) that are convertible to a NuFlav or NuFlavInt object. In the
    former case, the entire flavor dictionary (which includes both 'cc' and
    'nc') is returned, while in the latter case whatever lives at the node is
    returned.

    Initializing, setting and getting data in various ways:
    >>> fi_dat = FlavIntData()
    >>> fi_dat['nue', 'cc'] = 1
    >>> fi_dat['nuenc'] = 2
    >>> fi_dat['numu'] = {'cc': 'cc data...', 'nc': 'nc data...'}
    >>> fi_dat[NuFlav(16), IntType(1)] == 4

    >>> fi_dat['nuecc'] == 1
    True
    >>> fi_dat['NUE_NC'] == 2
    True
    >>> fi_dat['nu_e'] == {'cc': 1, 'nc': 2}
    True
    >>> fi_dat['nu mu cc'] == 'cc data...'
    True
    >>> fi_dat['nu mu'] == {'cc': 'cc data...', 'nc': 'nc data...'}
    True
    >>> fi_dat['nutau cc'] == 4
    True

    """
    def __init__(self, val=None):
        super(FlavIntData, self).__init__()
        if isinstance(val, basestring):
            d = self.__load(val)
        elif isinstance(val, dict):
            d = val
        elif val is None:
            # Instantiate empty FlavIntData
            with BarSep('_'):
                d = {str(f): {str(it):None for it in ALL_NUINT_TYPES}
                     for f in ALL_NUFLAVS}
        else:
            raise TypeError('Unrecognized `val` type %s' % type(val))
        self.validate(d)
        self.update(d)

    @staticmethod
    def _interpret_index(idx):
        if not isinstance(idx, basestring) and hasattr(idx, '__len__') \
                and len(idx) == 1:
            idx = idx[0]
        with BarSep('_'):
            try:
                nfi = NuFlavInt(idx)
                return [str(nfi.flav()), str(nfi.intType())]
            except (AssertionError, ValueError, TypeError):
                try:
                    return [str(NuFlav(idx))]
                except:
                    raise ValueError('Invalid index: %s' %str(idx))

    def __getitem__(self, *args):
        assert len(args) <= 2
        key_list = self._interpret_index(args)
        tgt_obj = super(FlavIntData, self).__getitem__(key_list[0])
        if len(key_list) == 2:
            tgt_obj = tgt_obj[key_list[1]]
        return tgt_obj

    def __setitem__(self, *args):
        assert len(args) > 1
        item, value = args[:-1], args[-1]
        key_list = self._interpret_index(item)
        if len(key_list) == 1:
            self.__validate_inttype_dict(value)
            value = self.__translate_inttype_dict(value)
        tgt_obj = self
        for key in key_list[:-1]:
            tgt_obj = dict.__getitem__(tgt_obj, key)
        dict.__setitem__(tgt_obj, key_list[-1], value)

    def __eq__(self, other):
        """Recursive, exact equality"""
        return recursiveEquality(self, other)

    @staticmethod
    def __basic_validate(fi_container):
        for flavint in ALL_NUFLAVINTS:
            with BarSep('_'):
                f = str(flavint.flav())
                it = str(flavint.intType())
            assert isinstance(fi_container, dict), "container must be of" \
                    " type 'dict'; instead got %s" % type(fi_container)
            assert fi_container.has_key(f), "container missing flavor '%s'" % f
            assert isinstance(fi_container[f], dict), \
                    "Child of flavor '%s': must be type 'dict' but" \
                    " got %s instead" % (f, type(fi_container[f]))
            assert fi_container[f].has_key(it), \
                    "Flavor '%s' sub-dict must contain a both interaction" \
                    " types, but missing (at least) intType '%s'" % (f, it)

    @staticmethod
    def __validate_inttype_dict(d):
        assert isinstance(d, MutableMapping), \
                "Value must be an inttype (sub-) dict if you only specify a" \
                " flavor (and not an int type) as key"
        keys = d.keys()
        assert (len(keys) == 2) and \
                ([str(k).lower() for k in sorted(keys)] == ['cc', 'nc']), \
                "inttype (sub-) dict must contain exactly 'cc' and 'nc' keys"

    @staticmethod
    def __translate_inttype_dict(d):
        for key in d.keys():
            if not isinstance(key, basestring) or not (key.lower() == key):
                val = d.pop(key)
                d[str(key).lower()] = val
        return d

    def __load(self, fname, **kwargs):
        d = fileio.from_file(fname, **kwargs)
        self.validate(d)
        return d

    def flavs(self):
        return tuple(sorted([NuFlav(k) for k in self.keys()]))

    def flavints(self):
        fis = []
        [[fis.append(NuFlavInt(flav, int_type))
          for int_type in self[flav].keys()]
         for flav in self.keys()]
        return tuple(sorted(fis))

    def allclose(self, other, rtol=1e-05, atol=1e-08):
        """Returns True if all data structures are equal and all numerical
        values contained are within relative (rtol) and/or absolute (atol)
        tolerance of one another.
        """
        return recursiveAllclose(self, other, rtol=rtol, atol=atol)

    def validate(self, fi_container):
        """Perform basic validation on the data structure"""
        self.__basic_validate(fi_container)

    def save(self, fname, **kwargs):
        """Save data structure to a file; see fileio.to_file for details"""
        fileio.to_file(self, fname, **kwargs)

    def idDupes(self, rtol=None, atol=None):
        """Identify flavints with duplicated data (exactly or within a
        specified tolerance), convert these NuFlavInt's into NuFlavIntGroup's
        and returning these along with the data associated with each.

        Parameters
        ----------
        rtol
            Set to positive value to use as rtol argument for numpy allclose
        atol
            Set to positive value to use as atol argument for numpy allclose

        If either `rtol` or `atol` is 0, exact equality is enforced.

        Returns
        -------
        dupe_flavintgroups : list of NuFlavIntGroup
            A NuFlavIntGroup object is returned for each group of NuFlavInt's
            found with duplicate data
        dupe_flavintgroups_data : list of objects
            Data associated with each NuFlavIntGroup in dupe_flavintgroups.
            Each object in `dupe_flavintgroups_data` corresponds to, and in the
            same order as, the objects in `dupe_flavintgroups`.
        """
        exact_equality = True
        kwargs = {}
        if rtol is not None and rtol > 0 and atol != 0:
            exact_equality = False
            kwargs['rtol'] = rtol
        if atol is not None and atol > 0 and rtol != 0:
            exact_equality = False
            kwargs['atol'] = atol
        if exact_equality:
            cmpfunc = recursiveEquality
        else:
            cmpfunc = lambda x, y: recursiveAllclose(x, y, **kwargs)

        dupe_flavintgroups = []
        dupe_flavintgroups_data = []
        for flavint in self.flavints():
            this_datum = self[flavint]
            match = False
            for n, group_datum in enumerate(dupe_flavintgroups_data):
                if len(this_datum) != len(group_datum):
                    continue
                if cmpfunc(this_datum, group_datum):
                    dupe_flavintgroups[n] += flavint
                    match = True
                    break
            if not match:
                dupe_flavintgroups.append(NuFlavIntGroup(flavint))
                dupe_flavintgroups_data.append(this_datum)

        sort_inds = np.argsort(dupe_flavintgroups)
        dupe_flavintgroups = [dupe_flavintgroups[i] for i in sort_inds]
        dupe_flavintgroups_data = [dupe_flavintgroups_data[i]
                                   for i in sort_inds]

        return dupe_flavintgroups, dupe_flavintgroups_data


class FlavIntDataGroup(dict):
    """Container class for storing data for some set(s) of NuFlavIntGroups
    (cf. FlavIntData, which stores one datum for each NuFlavInt separately)

    Parameters
    ----------
    val: None, str, or dict
        Data with which to populate the hierarchy
    flavint_groups: None, str, or iterable
        User-defined groupings of NuFlavIntGroups. These can be specified
        in several ways.

        None
            If val == None, flavint_groups must be specified
            If val != None, flavitn_groups are deduced from the data
        string
            If val is a string, it is expected to be a comma-separated
            list, each field of which describes a NuFlavIntGroup. The
            returned list of groups encompasses all possible flavor/int
            types, but the groups are mutually exclusive.
        iterable of strings or NuFlavIntGroup
            If val is an iterable, each member of the iterable is
            interpreted as a NuFlavIntGroup.
    """
    def __init__(self, val=None, flavint_groups=None):
        super(FlavIntDataGroup, self).__init__()
        if flavint_groups is None:
            if val is None:
                raise ValueError('Error - must input at least one of '
                                 '`flavint_groups` or `val`.')
        else:
            self.flavint_groups = flavint_groups

        if val is None:
            # Instantiate empty FlavIntDataGroup
            d = {str(group): None for group in self.flavint_groups}
        else:
            if isinstance(val, basestring):
                d = self.__load(val)
            elif isinstance(val, dict):
                d = val
            else:
                raise TypeError('Unrecognized `val` type %s' % type(val))
            d = {str(NuFlavIntGroup(key)): d[key] for key in d.iterkeys()}
            if d.keys() == ['']:
                raise AssertionError('NuFlavIntGroups not found in data keys')

            fig = [NuFlavIntGroup(fig) for fig in d.iterkeys()]
            if flavint_groups is None:
                self.flavint_groups = fig
            else:
                if set(fig) != set(self.flavint_groups):
                    raise AssertionError(
                        'Specified `flavint_groups` does not match `val` '
                        'signature.\n`flavint_groups` - {0}\n`val groups` '
                        '- {1}'.format(self.flavint_groups, fig)
                    )

        self.validate(d)
        self.update(d)

    @property
    def flavint_groups(self):
        return self._flavint_groups

    @flavint_groups.setter
    def flavint_groups(self, value):
        assert 'muons' not in value
        fig = self._parse_flavint_groups(value)
        all_flavints = reduce(add, [f.flavints() for f in fig])
        for fi in set(all_flavints):
            if all_flavints.count(fi) > 1:
                raise AssertionError(
                    'FlavInt {0} referred to multiple times in flavint_group '
                    '{1}'.format(fi, fig)
                )
        self._flavint_groups = fig

    def transform_groups(self, flavint_groups):
        """Transform FlavIntDataGroup into a structure given by the input
        flavint_groups.

        Parameters
        ----------
        flavint_groups : string, or sequence of strings or sequence of
                         NuFlavIntGroups

        Returns
        -------
        transformed_fidg : FlavIntDataGroup

        """
        flavint_groups = self._parse_flavint_groups(flavint_groups)

        original_flavints = reduce(add, [list(f.flavints()) for f in
                                         self.flavint_groups])
        inputted_flavints = reduce(add, [list(f.flavints()) for f in
                                         flavint_groups])
        if not set(inputted_flavints).issubset(set(original_flavints)):
            raise AssertionError(
                'Mismatch between underlying group of flavints given as input '
                'and original flavint_group.\nOriginal {0}\nInputted '
                '{1}'.format(set(original_flavints), set(inputted_flavints))
            )

        transformed_fidg = FlavIntDataGroup(flavint_groups=flavint_groups)
        for in_fig in flavint_groups:
            for or_fig in self.flavint_groups:
                if or_fig in in_fig:
                    if transformed_fidg[in_fig] is None:
                        transformed_fidg[in_fig] = deepcopy(self[or_fig])
                    else:
                        transformed_fidg[in_fig] = \
                                self._merge(transformed_fidg[in_fig],
                                            self[or_fig])
                elif in_fig in or_fig:
                    raise AssertionError(
                        'Cannot decouple original flavint_group {0} into input'
                        'flavint_group {1}'.format(or_fig, in_fig)
                    )
        logging.trace('Transformed from\n{0}\nto '
                      '{1}'.format(self.flavint_groups, flavint_groups))
        return transformed_fidg

    def allclose(self, other, rtol=1e-05, atol=1e-08):
        """Returns True if all data structures are equal and all numerical
        values contained are within relative (rtol) and/or absolute (atol)
        tolerance of one another.
        """
        return recursiveAllclose(self, other, rtol=rtol, atol=atol)

    def validate(self, fi_container):
        """Perform basic validation on the data structure"""
        self.__basic_validate(fi_container)

    def save(self, fname, **kwargs):
        """Save data structure to a file; see fileio.to_file for details"""
        fileio.to_file(self, fname, **kwargs)

    @staticmethod
    def _parse_flavint_groups(flavint_groups):
        if isinstance(flavint_groups, basestring):
            return flavintGroupsFromString(flavint_groups)
        elif isinstance(flavint_groups, NuFlavIntGroup):
            return [flavint_groups]
        elif isinstance(flavint_groups, Sequence):
            if all(isinstance(f, NuFlavIntGroup) for f in flavint_groups):
                return flavint_groups
            elif all(isinstance(f, NuFlavInt) for f in flavint_groups):
                return [NuFlavIntGroup(f) for f in flavint_groups]
            elif all(isinstance(f, basestring) for f in flavint_groups):
                return [NuFlavIntGroup(f) for f in flavint_groups]
            else:
                raise AssertionError(
                    'Elements in `flavint_groups` not all type '
                    'NuFlavIntGroup or string: %s' % flavint_groups
                )
        else:
            raise TypeError('Unrecognized `flavint_groups` type %s' %
                            type(flavint_groups))

    @staticmethod
    def _merge(a, b, path=None):
        """Merge dictionaries `a` and `b` by recursively iterating down
        to the lowest level of the dictionary until coincident numpy
        arrays are found, after which the appropriate sub-element is
        made equal to the concatenation of the two arrays.
        """
        if path is None:
            path = []
        for key in b:
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    FlavIntDataGroup._merge(a[key], b[key], path + [str(key)])
                elif isinstance(a[key], np.ndarray) and \
                        isinstance(b[key], np.ndarray):
                    a[key] = np.concatenate((a[key], b[key]))
                elif isinstance(a[key], pint.quantity._Quantity) and \
                        isinstance(b[key], pint.quantity._Quantity):
                    if isinstance(a[key].m, np.ndarray) and \
                       isinstance(b[key].m, np.ndarray):
                        units = a[key].units
                        a[key] = np.concatenate((a[key].m, b[key].m_as(units)))
                        a[key] = a[key] * units
                    else:
                        raise Exception(
                            'Conflict at %s' % '.'.join(path + [str(key)])
                        )
                else:
                    raise Exception(
                        'Conflict at %s' % '.'.join(path + [str(key)])
                    )
            else:
                a[key] = b[key]
        return a

    def _interpret_index(self, idx):
        try:
            nfi = NuFlavIntGroup(idx)
            return str(nfi)
        except:
            raise ValueError('Invalid index: %s' % str(idx))

    def __basic_validate(self, fi_container):
        for group in self.flavint_groups:
            f = str(group)
            assert isinstance(fi_container, dict), "container must be of" \
                    " type 'dict'; instead got %s" % type(fi_container)
            assert f in fi_container, \
                    "container missing flavint group '%s'" % f

    def __load(self, fname, **kwargs):
        d = fileio.from_file(fname, **kwargs)
        return d

    def __add__(self, other):
        d = deepcopy(self)
        d = self._merge(d, other)
        combined_flavint_groups = list(
            set(self.flavint_groups + other.flavint_groups)
        )
        return FlavIntDataGroup(val=d, flavint_groups=combined_flavint_groups)

    def __getitem__(self, arg):
        key = self._interpret_index(arg)
        tgt_obj = super(FlavIntDataGroup, self).__getitem__(key)
        return tgt_obj

    def __setitem__(self, arg, value):
        key = self._interpret_index(arg)
        if NuFlavIntGroup(key) not in self.flavint_groups:
            self.flavint_groups += [NuFlavIntGroup(key)]
        super(FlavIntDataGroup, self).__setitem__(key, value)

    def __eq__(self, other):
        """Recursive, exact equality"""
        return recursiveEquality(self, other)


def flavintGroupsFromString(groups):
    """Interpret `groups` to break into neutrino flavor/interaction type(s)
    that are to be grouped together; also form singleton groups as specified
    explicitly in `groups` or for any unspecified flavor/interaction type(s).

    The returned list of groups encompasses all possible flavor/int types, but
    the groups are mutually exclusive.

    Parameters
    ----------
    groups : None, string, or sequence of strings

    Returns
    -------
    flavint_groups : list of NuFlavIntGroup

    """
    if groups is None or groups == '':
        # None are to be grouped together
        grouped = []
        # All will be singleton groups
        ungrouped = [NuFlavIntGroup(k) for k in ALL_NUFLAVINTS]
        #groups_label = 'ungrouped'
    else:
        grouped, ungrouped = xlateGroupsStr(groups)
        #evts.metadata['flavints_joined'] = [str(g) for g in grouped]
        #groups_label = 'joined_G_' + '_G_'.join([str(g) for g in grouped])

    # Find any flavints not included in the above groupings
    flavint_groups = grouped + ungrouped
    logging.trace('flav/int in the following group(s) will be joined together:'
                  + ', '.join([str(k) for k in grouped]))
    logging.trace('flav/ints treated individually:'
                  + ', '.join([str(k) for k in ungrouped]))

    # Enforce that flavints composing groups are mutually exclusive
    for grp_n, flavintgrp0 in enumerate(flavint_groups[:-1]):
        for flavintgrp1 in flavint_groups[grp_n+1:]:
            assert len(set(flavintgrp0).intersection(set(flavintgrp1))) == 0

    #flavintgrp_names = [str(flavintgrp) for flavintgrp in flavint_groups]
    return flavint_groups


class CombinedFlavIntData(FlavIntData):
    """Container class for storing data redundant for some set(s) of NuFlavInts
    (cf. FlavIntData, which stores one datum for each NuFlavInt separately)

    val
        Data with which to populate the hierarchy
    flavint_groupings
        User-defined groupings of NuFlavInts. One set of data is shared among
        all NuFlavInts in a NuFlavInt grouping. These can be specified in
        several ways, and note the `dedup` parameter which deduces groupings
        from the data.

        None
            If val == None, no NuFlavInts are grouped together
            If val != None, flavint_groupings are deduced from the data; note
              that deduplication, if specified, will occur *after* this step
        string
            If val is a string, it is expected to be a semicolon-delimited
            list, each field of which describes a NuFlavIntGroup
        iterable
            If val is an iterable, each member of the iterable is converted to
            a string and interpreted as a NuFlavIntGroup
    dedup : bool
        If True, after populating all data according to the `val` and
        `flavint_groupings` arguments, regroups the data according to those
        flavints whose data is considered equal (with tolerance set by the
        `dedup_rtol` argument), removing duplicated data in the process.
        (Default: False)
    dedup_rtol : numeric or None
        Set relative tolerance for identifying flavints with duplicated data.
        See docstring for `idDupes` method for details on how this is handled.
        (Default: None)
    dedup_atol : numeric or None
        Set absolute tolerance for identifying flavints with duplicated data.
        See docstring for `idDupes` method for details on how this is handled.
        (Default: None)
    """
    def __init__(self, val=None, flavint_groupings=None, dedup=False,
                 dedup_rtol=None, dedup_atol=None):
        raise NotImplementedError()
        # Interpret the flavint_groupings arg
        if flavint_groupings is None:
            grouped = []
            ungrouped = list(ALL_NUFLAVINTS)
        elif isinstance(flavint_groupings, basestring):
            grouped, ungrouped = xlateGroupsStr(flavint_groupings)
        elif hasattr(flavint_groupings, '__iter__'):
            strkgs = ','.join([str(x) for x in flavint_groupings])
            grouped, ungrouped = xlateGroupsStr(strkgs)
        else:
            raise TypeError('Incomprehensible `flavint_groupings`: "%s"' %
                            str(flavint_groupings))

        # Interpret the val arg
        named_g = None
        named_ung = None
        if isinstance(val, basestring):
            val = self.__load(val)

        if isinstance(val, dict):
            out_dict = {}
            groupings_found = []
            for top_key, top_val in val.iteritems():
                # If top-level key consists of a single flavor and NO int type,
                # check for a sub-dict that specifies interaction type to
                # formulate a full NuFlavInt
                try:
                    flav = NuFlav(top_key)
                    if isinstance(top_val, Mapping):
                        int_types = []
                        for level2_key, level2_data in top_val.iteritems():
                            # If *any* keys are invalid interaction type specs,
                            # invalidate the entire sequence
                            is_valid = True
                            try:
                                int_type = IntType(level2_key)
                            except ValueError:
                                is_valid = False
                            else:
                                int_types.append((int_type, level2_data))
                        if is_valid and len(int_types) != 0:
                            nfig = NuFlavIntGroup()
                            for int_type in int_types:
                                nfig += (flav, int_type)
                    else:
                        out_dict[str(flav)] = top_val

                    for nfi in NuFlavIntGroup(flav).flavints():
                        named_g = 999999999999999999999999999999999
                except ValueError:
                    nfig = NuFlavIntGroup(top_key)
                groupings_found.append(nfig)

            named_g, named_ung = xlateGroupsStr(','.join(val.keys()))
            #print 'named_g:', named_g
            #print 'named_ung:', named_ung
            # Force keys to standard naming convention (be liberal on input,
            # strict on output)
            for key in val.keys():
                for g in named_g + named_ung:
                    if (NuFlavIntGroup(key) == g) and key != str(g):
                        d[str(g)] = val.pop(key)

        elif val is None:
            if flavint_groupings is None:
                logging.warn('CombinedFlavIntData object instantiated without'
                             ' flavint groupings specified.')
            named_g = grouped
            named_ung = ungrouped
            # Instantiate empty dict with groupings as keys
            d = {str(k):None for k in named_g+named_ung}

        else:
            raise TypeError('Unrecognized `val`: "%s", type %s' %
                            (str(val), type(val)))

        if named_g and flavint_groupings:
            assert named_g == grouped
            assert named_ung == ungrouped

        self.validate(d)
        self.grouped, self.ungrouped = named_g, named_ung
        with BarSep('_'):
            self.flavints_to_keys = [(ks, str(ks)) for ks in named_g+named_ung]
        self.update(d)

        if dedup:
            self.deduplicate(rtol=dedup_rtol, atol=dedup_atol)

    def __basic_validate(self, cfid):
        assert isinstance(cfid, dict), 'container must be of' \
                ' type `dict`; instead got %s' % str(type(cfid))
        keys = cfid.keys()
        key_grps = [NuFlavIntGroup(k) for k in keys]
        for flavint in ALL_NUFLAVINTS:
            found = 0
            for grp in key_grps:
                if flavint in grp:
                    found += 1
            assert found > 0, 'container missing flavint %s' % str(flavint)

    def __eq__(self, other):
        return recursiveEquality(self, other)

    def __getitem__(self, item):
        return super(CombinedFlavIntData, self).__getitem__(item)

    def __setitem__(self, item, value):
        return super(CombinedFlavIntData, self).__setitem__(item, value)

    def deduplicate(self, rtol=None, atol=None):
        """Identify duplicate datasets and combine the associated flavints
        together, elinimating redundancy in the data.

        This forces any flavints with identical data to be tied to one another
        on into the future (hence calling the `set` method will throw an
        exception if obsolete groupings are specified).
        """
        dupe_kgs, dupe_kgs_data = self.idDupes(rtol=rtol, atol=atol)
        d = {str(kg): dat for kg, dat in izip(dupe_kgs, dupe_kgs_data)}
        self.validate(d)
        self.grouped = [kg for kg in dupe_kgs if len(kg) > 1]
        self.ungrouped = [kg for kg in dupe_kgs if len(kg) == 1]
        with BarSep('_'):
            self.flavints_to_keys = [(kg, str(kg)) for kg in dupe_kgs]
        self.clear()
        self.update(d)

    def set(self, *args):
        """Store data for the specified flavints.

        Parameters
        ----------
        arg[0], arg[1], ... arg[N-2]
            NuFlavInts for which to store data. The specified NuFlavInts must
            match exactly an existing NuFlavIntGroup within the data structure.
            (This avoids a user inadvertently setting data for an unintended
            flavint.)
        arg[N-1] (final arg)
            Data object to be stored
        """
        # TODO: do not set *anything* until it is verified that a valid
        # NuFlavIntGroup is specified (i.e., the target NuFlavIntGroup doesn't
        # span multiple NuFlavIntGroups).
        all_keys = list(args[:-1])
        new_val = deepcopy(args[-1])
        tgt_grp = NuFlavIntGroup(all_keys)
        for (flavints, key) in self.flavints_to_keys:
            match = False
            # Identical match to existing NuFlavIntGroup
            if tgt_grp == flavints:
                all_keys[0] = key
                match = True

            if match:
                branch_keys = all_keys[:-1]
                node_key = all_keys[-1]
                lvl = self
                for k in branch_keys:
                    lvl = dict.__getitem__(lvl, k)
                old_val = dict.__getitem__(lvl, node_key)
                dict.__setitem__(lvl, node_key, new_val)
                try:
                    self.validate(self)
                except:
                    dict.__setitem__(lvl, node_key, old_val)
                    raise
                return
        # If you get this far, no match was found
        raise ValueError(
            'Could not set data for NuFlavInt(Group) %s; valid'
            ' NuFlavInt(Group)s for this object are: %s' %
            (str(tgt_grp), '. '.join([str(nfig) for nfig in
                                      self.grouped]))
        )

    def get(self, *args):
        """Get data corresponding to a NuFlavInt or NuFlavIntGroup that
        comprises a subset of a NuFlavIntGroup represented in this container.

        * If `arg` is a NuFlavInt object or a string convertible to one, the
            branch whose NuFlavIntGroup contains this NuFlavInt is returned
        * If `arg` is a NuFlavIntGroup object or a string convertible to one
            and if it is a subset of a NuFlavIntGroup within this object, the
            corresponding node is returned
        * Subsequent `arg`s are treated as integer or string indices in
            sub-structures within the NuFlavIntGroup branch

        * If the NuFlavInt or NuFlavIntGroup corresponding to `arg` is not
            a subset of a single NuFlavIntGroups in this object, an exception
            is raised
        """
        #with BarSep('_'):
        all_keys = list(args)
        #print 'all_keys:', all_keys
        tgt_grp = NuFlavIntGroup(all_keys[0])
        #print 'tgt_grp0:', tgt_grp
        #print 'flavints_to_keys:', self.flavints_to_keys
        for (flavints, key) in self.flavints_to_keys:
            #print 'flavints:', flavints, 'type:', type(flavints)
            #print 'key:', key, 'type:', type(key)
            match = False
            # Identical
            if tgt_grp == flavints:
                all_keys[0] = key
                match = True
                #print 'found exact match:', tgt_grp, '==', flavints
            # Requested flavints are strict subset
            elif len(tgt_grp - flavints) == 0:
                all_keys[0] = key
                match = True
                #print 'found subset match:', tgt_grp, 'in', flavints
                logging.debug('Requesting data for subset (%s) of'
                              ' grouping %s' % (str(tgt_grp), str(flavints)))
            # Get it
            if match:
                branch_keys = all_keys[:-1]
                node_key = all_keys[-1]
                lvl = self
                for k in branch_keys:
                    lvl = dict.__getitem__(lvl, k)
                #print 'node_key:', node_key, 'type:', type(node_key)
                return deepcopy(dict.__getitem__(lvl, node_key))
        # If you get this far, no match was found
        raise ValueError('Could not locate data for group %s' % str(tgt_grp))


def xlateGroupsStr(val):
    """Translate a ","-separated string into separate `NuFlavIntGroup`s.

    val
        ","-delimited list of valid NuFlavIntGroup strings, e.g.:
            "nuall_nc,nue,numu_cc+numubar_cc"
        Note that specifying NO interaction type results in both interaction
        types being selected, e.g. "nue" implies "nue_cc+nue_nc". For other
        details of how the substrings are interpreted, see docs for
        NuFlavIntGroup.

    returns:
        grouped, ungrouped

    grouped, ungrouped
        lists of NuFlavIntGroups; the first will have more than one flavint
        in each NuFlavIntGroup whereas the second will have just one
        flavint in each NuFlavIntGroup. Either list can be of 0-length.

    This function does not enforce mutual-exclusion on flavints in the
    various flavint groupings, but does list any flavints not grouped
    together in the `ungrouped` return arg. Mutual exclusion can be
    enforced through set operations upon return.
    """
    # What flavints to group together
    grouped = [NuFlavIntGroup(s) for s in re.split('[,;]', val)]

    # Find any flavints not included in the above groupings
    all_flavints = set(ALL_NUFLAVINTS)
    all_grouped_flavints = set(NuFlavIntGroup(grouped))
    ungrouped = [NuFlavIntGroup(k) for k in
                 sorted(all_flavints.difference(all_grouped_flavints))]

    return grouped, ungrouped


def test_IntType():
    all_f_codes = [12, -12, 14, -14, 16, -16]
    all_i_codes = [1, 2]

    #==========================================================================
    # Test IntType
    #==========================================================================
    ref = IntType('cc')
    assert IntType('\n\t _cc \n') == ref
    try:
        IntType('numubarcc')
    except:
        pass
    else:
        raise Exception()
    assert IntType(1) == ref
    assert IntType(1.0) == ref
    assert IntType(IntType('cc')) == ref
    assert IntType(NuFlavInt('numubarcc')) == ref
    for i in all_i_codes:
        IntType(i)
        IntType(float(i))
    logging.info('<< PASS : test_IntType >>')


def test_NuFlav():
    all_f_codes = [12, -12, 14, -14, 16, -16]
    all_i_codes = [1, 2]

    #==========================================================================
    # Test NuFlav
    #==========================================================================
    ref = NuFlav('numu')
    assert ref.flavCode() == 14
    assert (-ref).flavCode() == -14
    assert ref.barNoBar() == 1
    assert (-ref).barNoBar() == -1
    assert ref.isParticle()
    assert not (-ref).isParticle()
    assert not ref.isAntiParticle()
    assert (-ref).isAntiParticle()

    #assert NuFlav('\n\t _ nu_ mu_ cc\n\t\r') == ref
    #assert NuFlav('numucc') == ref
    assert NuFlav(14) == ref
    assert NuFlav(14.0) == ref
    assert NuFlav(NuFlav('numu')) == ref
    assert NuFlav(NuFlavInt('numucc')) == ref
    assert NuFlav(NuFlavInt('numunc')) == ref

    for f in all_f_codes:
        NuFlav(f)
        NuFlav(float(f))
    for (f, bnb) in product(['e', 'mu', 'tau'], ['', 'bar']):
        NuFlav('nu_' + f + '_' + bnb)

    logging.info('<< PASS : test_NuFlav >>')


def test_NuFlavInt():
    all_f_codes = [12, -12, 14, -14, 16, -16]
    all_i_codes = [1, 2]

    #==========================================================================
    # Test NuFlavInt
    #==========================================================================
    try:
        NuFlavInt('numu')
    except ValueError:
        pass

    # Equality
    fi_comb = [fic for fic in product(all_f_codes, all_i_codes)]
    for (fi0, fi1) in product(fi_comb, fi_comb):
        if fi0 == fi1:
            assert NuFlavInt(fi0) == NuFlavInt(fi1)
        else:
            assert NuFlavInt(fi0) != NuFlavInt(fi1)
    assert NuFlavInt((12, 1)) != 'xyz'
    # Sorting: this is my desired sort order
    nfl0 = [NuFlavInt(fic) for fic in fi_comb]
    nfl1 = [NuFlavInt(fic) for fic in fi_comb]
    np.random.shuffle(nfl1)
    nfl_sorted = sorted(nfl1)
    assert all([v0 == nfl_sorted[n] for n, v0 in enumerate(nfl0)])
    assert len(nfl0) == len(nfl_sorted)

    # Test NuFlavInt instantiation
    nue = NuFlav('nue')
    cc = IntType('cc')
    nc = IntType('nc')
    nuebar = NuFlav('nuebar')
    flavs = list(ALL_NUFLAVS)
    flavs.extend(['nue', 'numu', 'nutau', 'nu_e', 'nu e', 'Nu E', 'nuebar',
                  'nu e bar'])
    flavs.extend(all_f_codes)
    ints = [cc, nc, 'cc', 'nc', 'CC', 'NC', 1, 2]
    nuecc = NuFlavInt('nuecc')
    nuebarnc = NuFlavInt('nuebarnc')

    # Instantiate with combinations of flavs and int types
    for f, i in product(flavs, ints):
        ref = NuFlavInt(f, i)
        assert NuFlavInt((f, i)) == ref
        assert NuFlavInt(flav=f, int_type=i) == ref
        if isinstance(f, basestring) and isinstance(i, basestring):
            assert NuFlavInt(f+i) == ref
            assert NuFlavInt(f + '_' + i) == ref
            assert NuFlavInt(f + ' ' + i) == ref

    # Instantiate with already-instantiated `NuFlavInt`s
    assert NuFlavInt(nuecc) == NuFlavInt('nuecc')
    assert NuFlavInt(nuebarnc) == NuFlavInt('nuebarnc')

    # test negating flavint
    nk = NuFlavInt('numucc')
    assert -nk == NuFlavInt('numubarcc')

    logging.info('<< PASS : test_NuFlavInt >>')


def test_NuFlavIntGroup():
    all_f_codes = [12, -12, 14, -14, 16, -16]
    all_i_codes = [1, 2]

    #==========================================================================
    # Test NuFlavIntGroup
    #==========================================================================
    fi_comb = [fic for fic in product(all_f_codes, all_i_codes)]
    nfl0 = [NuFlavInt(fic) for fic in fi_comb]
    nfl1 = [NuFlavInt(fic) for fic in fi_comb]
    nfl_sorted = sorted(nfl1)
    nkg0 = NuFlavIntGroup(nfl0)
    nkg1 = NuFlavIntGroup(nfl_sorted)
    assert nkg0 == nkg1
    assert nkg0 != 'xyz'
    assert nkg0 != 'xyz'

    # Test inputs
    assert NuFlavIntGroup('nuall,nuallbar').flavs() == \
            tuple([NuFlav(c) for c in all_f_codes])

    #
    # Test NuFlavIntGroup instantiation
    #
    nue = NuFlav('nue')
    numu = NuFlav('numu')
    nue_cc = NuFlavInt('nue_cc')
    nue_nc = NuFlavInt('nue_nc')

    # Empty args
    NuFlavIntGroup()
    NuFlavIntGroup([])

    # String flavor promoted to CC+NC
    assert set(NuFlavIntGroup('nue').flavints()) == set((nue_cc, nue_nc))
    # NuFlav promoted to CC+NC
    assert set(NuFlavIntGroup(nue).flavints()) == set((nue_cc, nue_nc))
    # List of single flav str same as above
    assert set(NuFlavIntGroup(['nue']).flavints()) == set((nue_cc, nue_nc))
    # List of single flav same as above
    assert set(NuFlavIntGroup([nue]).flavints()) == set((nue_cc, nue_nc))

    # Single flavint spec
    assert set(NuFlavIntGroup(nue_cc).flavints()) == set((nue_cc,))
    # Str with single flavint spec
    assert set(NuFlavIntGroup('nue_cc').flavints()) == set((nue_cc,))
    # List of single str containing single flavint spec
    assert set(NuFlavIntGroup(['nue_cc']).flavints()) == set((nue_cc,))

    # Multiple flavints as *args
    assert set(NuFlavIntGroup(nue_cc, nue_nc).flavints()) == set((nue_cc, nue_nc))
    # List of flavints
    assert set(NuFlavIntGroup([nue_cc, nue_nc]).flavints()) == set((nue_cc, nue_nc))
    # List of single str containing multiple flavints spec
    assert set(NuFlavIntGroup(['nue_cc,nue_nc']).flavints()) == set((nue_cc, nue_nc))
    # List of str containing flavints spec
    assert set(NuFlavIntGroup(['nue_cc', 'nue_nc']).flavints()) == set((nue_cc, nue_nc))

    # Another NuFlavIntGroup
    assert set(NuFlavIntGroup(NuFlavIntGroup(nue_cc, nue_nc)).flavints()) == set((nue_cc, nue_nc))

    # Addition of flavints promoted to NuFlavIntGroup
    assert nue_cc + nue_nc == NuFlavIntGroup(nue)
    # Addition of flavs promoted to NuFlavIntGroup including both CC & NC
    assert nue + numu == NuFlavIntGroup(nue, numu)

    # Test remove
    nkg = NuFlavIntGroup('nue_cc+numucc')
    nkg.remove(NuFlavInt((12, 1)))
    assert nkg == NuFlavIntGroup('numucc')

    # Test del
    nkg = NuFlavIntGroup('nue_cc+numucc')
    del nkg[0]
    assert nkg == NuFlavIntGroup('numucc')

    # Equivalent object when converting to string and back to NuFlavIntGroup from
    # that string
    for n in range(1, len(ALL_NUFLAVINTS)+1):
        logging.debug('NuFlavIntGroup --> str --> NuFlavIntGroup, n = %d' % n)
        for comb in combinations(ALL_NUFLAVINTS, n):
            ref = NuFlavIntGroup(comb)
            assert ref == NuFlavIntGroup(str(ref))

    # test TeX strings
    nkg = NuFlavIntGroup('nuall,nuallbar')
    logging.info(str(nkg))
    logging.info(tex(nkg))
    logging.info(nkg.simpleStr())
    logging.info(nkg.simpleTex())
    logging.info(nkg.uniqueFlavsTex())

    logging.info('<< ???? : test_NuFlavIntGroup >> checks pass upon inspection'
                 ' of above outputs and generated file(s).')


def test_FlavIntData():
    all_f_codes = [12, -12, 14, -14, 16, -16]
    all_i_codes = [1, 2]

    #==========================================================================
    # Test FlavIntData
    #==========================================================================
    # Excercise the "standard" PISA nested-python-dict features, where this
    # dict uses an '_' to separate 'bar' in key names, and the nested dict
    # levels are [flavor][interaction type].

    # Force separator to something weird before starting, to ensure everything
    # still works and this separator is still set when we're done
    oddball_sep = 'xyz'
    set_bar_ssep(oddball_sep)
    ref_pisa_dict = {f:{it:None for it in ['cc', 'nc']} for f in
                     ['nue', 'nue_bar', 'numu', 'numu_bar', 'nutau',
                      'nutau_bar']}
    fi_cont = FlavIntData()
    for f in ['nue', 'nue_bar', 'numu', 'numu_bar', 'nutau', 'nutau_bar']:
        for it in ['cc', 'nc']:
            assert fi_cont[f][it] == ref_pisa_dict[f][it]
            flavint = NuFlavInt(f, it)
            assert flavint.pidx(ref_pisa_dict) == ref_pisa_dict[f][it]
            logging.trace('%s: %s' %('flavint', flavint))
            logging.trace('%s: %s' %('f', f))
            logging.trace('%s: %s' %('it', it))
            logging.trace('%s: %s' %('fi_cont', fi_cont))
            logging.trace('%s: %s' %('fi_cont[f]', fi_cont[f]))
            logging.trace('%s: %s' %('fi_cont[f][it]', fi_cont[f][it]))
            logging.trace('%s: %s' %('fi_cont[flavint]', fi_cont[flavint]))
            logging.trace('%s: %s' %('fi_cont[flavint]', fi_cont[flavint]))
            assert fi_cont[flavint] == fi_cont[f][it]
            assert fi_cont[flavint] == fi_cont[flavint]
    assert get_bar_ssep() == oddball_sep
    set_bar_ssep('')

    # These should fail because you're only allowed to access the flav or
    # flavint part of the data structure, no longer any sub-items (use
    # subsequent [k1][k2]... to do this instead)
    fi_cont['numu', 'cc'] = {'sub-key':{'sub-sub-key': None}}
    try:
        fi_cont['numu', 'cc', 'sub-key']
    except ValueError:
        pass
    else:
        raise Exception('Test failed, exception should have been raised')

    try:
        fi_cont['numu', 'cc', 'sub-key'] = 'new sub-val'
    except ValueError:
        pass
    else:
        raise Exception('Test failed, exception should have been raised')

    # These should fail because setting flavor-only as a string would
    # invalidate the data structure (not a nested dict)
    try:
        fi_cont[NuFlav('numu')] = 'xyz'
    except AssertionError:
        pass
    else:
        raise Exception('Test failed, exception should have been raised')

    try:
        fi_cont[NuFlav('numu')] = {'cc': 'cc_xyz'}
    except AssertionError:
        pass
    else:
        raise Exception('Test failed, exception should have been raised')

    # The previously-valid fi_cont should *still* be valid, as `set` should
    # revert to the original (valid) values rather than keep the invalid values
    # that were attempted to be set above
    fi_cont.validate(fi_cont)

    # This should be okay because datastructure is still valid if the item
    # being set on the flavor (only) is a valid int-type dict
    fi_cont[NuFlav('numu')] = {'cc': 'cc_xyz', 'nc': 'nc_xyz'}

    # Test setting, getting, and JSON serialization of FlavIntData
    fi_cont['nue', 'cc'] = 'this is a string blah blah blah'
    fi_cont[NuFlavInt('nue_cc')]
    fi_cont[NuFlavInt('nue_nc')] = np.pi
    fi_cont[NuFlavInt('nue_nc')]
    fi_cont[NuFlavInt('numu_cc')] = [0, 1, 2, 3]
    fi_cont[NuFlavInt('numu_cc')]
    fi_cont[NuFlavInt('numu_nc')] = {'new':{'nested':{'dict':'xyz'}}}
    fi_cont[NuFlavInt('numu_nc')]
    fi_cont[NuFlavInt('nutau_cc')] = 1
    fi_cont[NuFlavInt('nutau_cc')]
    fi_cont[NuFlavInt('nutaubar_cc')] = np.array([0, 1, 2, 3])
    fi_cont[NuFlavInt('nutaubar_cc')]
    fname = '/tmp/test_FlavIntData.json'
    logging.info('Writing FlavIntData to file %s; inspect.' %fname)
    fileio.to_file(fi_cont, fname, warn=False)
    fi_cont2 = fileio.from_file(fname)
    assert recursiveEquality(fi_cont2, fi_cont), \
            'fi_cont=%s\nfi_cont2=%s' %(fi_cont, fi_cont2)

    logging.info('<< ???? : test_FlavIntData >> checks pass upon inspection of'
                 ' above outputs and generated file(s).')


def test_FlavIntDataGroup():
    flavint_group = 'nue, numu_cc+numubar_cc, nutau_cc'
    FlavIntDataGroup(flavint_groups=flavint_group)
    fidg1 = FlavIntDataGroup(
        flavint_groups='nuall, nu all bar CC, nuallbarnc',
        val={'nuall': np.arange(0, 100),
             'nu all bar CC': np.arange(100, 200),
             'nuallbarnc': np.arange(200, 300)}
    )
    fidg2 = FlavIntDataGroup(
        val={'nuall': np.arange(0, 100),
             'nu all bar CC': np.arange(100, 200),
             'nuallbarnc': np.arange(200, 300)}
    )
    assert fidg1 == fidg2

    try:
        fidg1 = FlavIntDataGroup(
            flavint_groups='nuall, nu all bar, nuallbar',
            val={'nuall': np.arange(0, 100),
                 'nu all bar CC': np.arange(100, 200),
                 'nuallbarnc': np.arange(200, 300)}
        )
    except AssertionError:
        pass
    else:
        raise Exception

    try:
        fidg1 = FlavIntDataGroup(flavint_groups=['nuall', 'nue'])
    except AssertionError:
        pass
    else:
        raise Exception

    assert set(fidg1.keys()) == set(('nuall', 'nuallbar_cc', 'nuallbar_nc'))
    fidg1.save('/tmp/test_FlavIntDataGroup.json', warn=False)
    fidg1.save('/tmp/test_FlavIntDataGroup.hdf5', warn=False)
    fidg3 = FlavIntDataGroup(val='/tmp/test_FlavIntDataGroup.json')
    fidg4 = FlavIntDataGroup(val='/tmp/test_FlavIntDataGroup.hdf5')
    assert fidg3 == fidg1
    assert fidg4 == fidg1

    figroups = ('nuecc+nuebarcc,numucc+numubarcc,nutaucc+nutaubarcc,'
                'nuallnc,nuallbarnc')
    cfidat = FlavIntDataGroup(flavint_groups=figroups)

    for k in cfidat.flavint_groups:
        cfidat[k] = np.arange(10)

    cfidat[NuFlavIntGroup('nuecc+nuebarcc')] = np.arange(10)

    logging.debug(str((fidg1 + fidg2)))
    assert fidg1 == fidg2
    try:
        logging.debug(str((fidg1 + cfidat)))
    except AssertionError:
        pass
    else:
        raise Exception

    d1 = {
        'numu+numubar': {
            'energy': np.arange(0, 10)
        },
        'nutau+nutaubar': {
            'energy': np.arange(0, 10)
        }
    }
    d2 = {
        'nue+nuebar': {
            'weights': np.arange(0, 10)
        },
        'nutau+nutaubar': {
            'weights': np.arange(0, 10)
        }
    }
    d1 = FlavIntDataGroup(val=d1)
    d2 = FlavIntDataGroup(val=d2)
    d3 = d1 + d2
    logging.debug(str((d3)))

    tr_d1 = d1.transform_groups(['numu+numubar+nutau+nutaubar'])
    logging.debug(str((tr_d1)))
    tr_d3 = d3.transform_groups('nue+nuebar+numu+numubar, nutau+nutaubar')
    tr_d3_1 = d3.transform_groups(['nue+nuebar+numu+numubar', 'nutau+nutaubar'])
    tr_d3_2 = d3.transform_groups([NuFlavIntGroup('nue+nuebar+numu+numubar'),
                                   NuFlavIntGroup('nutau+nutaubar')])
    logging.debug(str((tr_d3)))
    assert tr_d3 == tr_d3_1 and tr_d3 == tr_d3_2

    try:
        tr_d3.transform_groups(['nue+nuebar'])
    except AssertionError:
        pass
    else:
        raise Exception

    try:
        tr_d3.transform_groups('nue+nuebar, numu+numubar, nutau+nutaubar')
    except AssertionError:
        pass
    else:
        raise Exception

    logging.info('<< PASS : test_FlavIntDataGroup >>')


def test_CombinedFlavIntData():
    all_f_codes = [12, -12, 14, -14, 16, -16]
    all_i_codes = [1, 2]

    #==========================================================================
    # Test xlateGroupsStr function
    #==========================================================================
    # Test string parsing for flavor groupings
    gp1, ug1 = xlateGroupsStr(
        'nuall_nc, nuallbar_nc, nue, numu_cc+numubar_cc, nutau_cc'
    )
    logging.debug(str(([kg.simpleStr() for kg in gp1], ug1)))
    gp2, ug2 = xlateGroupsStr('nue,numu')
    logging.debug(str(([kg.simpleStr() for kg in gp2], ug2)))
    gp3, ug3 = xlateGroupsStr('nuall_nc')
    logging.debug(str(([kg.simpleStr() for kg in gp3], ug3)))
    gp4, ug4 = xlateGroupsStr(
        'nuall_nc+nuallbar_nc,nuall_cc+nuallbar_cc'
    )
    logging.debug(str(([kg.simpleStr() for kg in gp4], ug4)))

    logging.info('<< PASS : test_CombinedFlavIntData/xlateGroupsStr >>')

    #==========================================================================
    # Test CombinedFlavIntData class
    #==========================================================================
    # Empty container with no groupings
    CombinedFlavIntData()
    # Empty container with groupings
    CombinedFlavIntData(flavint_groupings='nuall,nuallbar')
    # Instantiate with non-standard key names
    cfid = CombinedFlavIntData(
        val={'nuall': np.arange(0, 100),
             'nu all bar CC': np.arange(100, 200),
             'nuallbarnc': np.arange(200, 300)}
    )
    assert set(cfid.keys()) == set(('nuall', 'nuallbar_cc', 'nuallbar_nc'))
    cfid.save('/tmp/test_CombinedFlavIntData.json', warn=False)
    cfid.save('/tmp/test_CombinedFlavIntData.hdf5', warn=False)
    cfid2 = CombinedFlavIntData('/tmp/test_CombinedFlavIntData.json')
    cfid3 = CombinedFlavIntData('/tmp/test_CombinedFlavIntData.hdf5')
    assert cfid2 == cfid
    assert cfid3 == cfid

    # Test deduplication
    d1 = {
        'nuecc':        np.arange(0, 10),
        'nuebarcc':     np.arange(0, 10),
        'numucc':       np.arange(1, 11),
        'numubarcc':    np.arange(1, 11),
        'nutaucc':      np.arange(2, 12),
        'nutaubarcc':   np.arange(2, 12),
        'nuallnc':      np.arange(20, 30),
        'nuallbarnc':   np.arange(10, 20),
    }
    d2 = {
        'nuecc':        np.arange(0, 10)*(1+1e-10),
        'nuebarcc':     np.arange(0, 10),
        'numucc':       np.arange(1, 11)*(1+1e-10),
        'numubarcc':    np.arange(1, 11),
        'nutaucc':      np.arange(2, 12)*(1+1e-8),
        'nutaubarcc':   np.arange(2, 12),
        'nuallnc':      np.arange(20, 30),
        'nuallbarnc':   np.arange(10, 20),
    }

    # Require exact equality, fields are exactly equal
    cfid1 = CombinedFlavIntData(val=d1, dedup=True, dedup_rtol=0)

    # Loose rtol spec, fields are not exactly equal but all within rtol (should
    # match 1)
    rtol = 1e-7
    cfid2 = CombinedFlavIntData(val=d2, dedup=True, dedup_rtol=rtol)
    assert cfid2.allclose(cfid1, rtol=rtol)

    # Tight rtol spec, fields are not exactly equal and some CC groupings are
    # now outside rtol
    rtol = 1e-9
    cfid3 = CombinedFlavIntData(val=d2, dedup=True, dedup_rtol=rtol)
    assert not cfid1.allclose(cfid3, rtol=rtol)

    # Tight rtol spec, fields are not exactly equal and all CC groupings are
    # outside rtol
    rtol = 1e-11
    cfid4 = CombinedFlavIntData(val=d2, dedup=True, dedup_rtol=rtol)
    assert not cfid1.allclose(cfid4, rtol=rtol)

    # Loose rtol, fields are exactly equal
    cfid5 = CombinedFlavIntData(val=d1, dedup=True, dedup_rtol=1e-7)
    assert cfid1 == cfid5

    # Tighter rtol, fields are exactly equal
    cfid6 = CombinedFlavIntData(val=d1, dedup=True, dedup_rtol=1e-9)
    assert cfid1 == cfid6

    # Very tight rtol, fields are exactly equal
    cfid7 = CombinedFlavIntData(val=d1, dedup=True, dedup_rtol=1e-14)
    assert cfid1 == cfid7

    cfidat = CombinedFlavIntData(
        flavint_groupings='nuecc+nuebarcc,'
                          'numucc+numubarcc,'
                          'nutaucc+nutaubarcc,'
                          'nuallnc,'
                          'nuallbarnc'
    )

    # Try to set individual NuFlavInts (all should fail since any single
    # NuFlavInt is a strict subset of the above-specified NuFlavInt groupings)
    for k in ALL_NUFLAVINTS:
        try:
            cfidat[k, np.arange(10)]
        except ValueError:
            pass
        else:
            raise Exception('Should not be able to set to a strict subset of'
                            ' grouped NuFlavInts!')

    # Try to set to a NuFlavInt group that *spans* two of the above groupings;
    # this should fail
    try:
        cfidat[NuFlavIntGroup('nuecc+numucc')] = np.arange(10)
    except ValueError:
        pass
    else:
        raise Exception('Should not be able to set to a set of NuFlavInts that'
                        'spans specified NuFlavInt groupings')

    for nfi in cfidat.grouped:
        try:
            cfidat[nfi] = np.arange(10)
        except ValueError:
            raise Exception('Should be able to set to grouped NuFlavInts!')

    cfidat.deduplicate()
    assert len(cfidat.flavints_to_keys) == 1
    assert cfidat.flavints_to_keys[0][0] == NuFlavIntGroup('nuall+nuallbar')

    logging.info(str([NuFlavIntGroup(k) for k in cfid1.keys()]))
    logging.info('<< ???? : CombinedFlavIntData >>')


if __name__ == "__main__":
    set_verbosity(1)
    test_IntType()
    test_NuFlav()
    test_NuFlavInt()
    test_NuFlavIntGroup()
    test_FlavIntData()
    test_FlavIntDataGroup()
    # Not implemented yet:
    #test_CombinedFlavIntData()

import os
import sys

import numpy as np
from pint import UnitRegistry

from ._version import get_versions


__all__ = ['__version__',
           'ureg', 'Q_',
           'FTYPE', 'C_FTYPE', 'C_PRECISION_DEF']


__version__ = get_versions()['version']
del get_versions


ureg = UnitRegistry()
Q_ = ureg.Quantity
del UnitRegistry


# Default value for FTYPE
FTYPE = np.float64
"""Global floating-point data type. C, CUDA, and Numba datatype definitions are
derived from this"""


# Set FTYPE from environment variable PISA_FTYPE, if it is defined
float32_strings = ['single', 'float32', 'fp32', '32', 'f4']
float64_strings = ['double', 'float64', 'fp64', '64', 'f8']
if 'PISA_FTYPE' in os.environ:
    pisa_ftype = os.environ['PISA_FTYPE']
    sys.stderr.write('PISA_FTYPE env var is defined as: "%s"; ' %pisa_ftype)
    if pisa_ftype.strip().lower() in float32_strings:
        FTYPE = np.float32
    elif pisa_ftype.strip().lower() in float64_strings:
        FTYPE = np.float64
    else:
        msg = (
            'Environment var PISA_FTYPE="%s" is unrecognized.\n'
            '--> For single precision set PISA_FTYPE to one of %s\n'
            '--> For double precision set PISA_FTYPE to one of %s\n'
            %(pisa_ftype, float32_strings, float64_strings)
        )
        sys.stderr.write('\n')
        raise ValueError(msg)


"""
Derive #define consts for dynamically-compiled C (and also C++ and CUDA) code
to use.

To use these in code, put in the C/C++/CUDA the following at the TOP of your
code:

  from pisa import FTYPE, C_FTYPE, C_PRECISION_DEF

  ...

  dynamic_source_code = '''
  #define fType %(C_FTYPE)s
  #define %(C_PRECISION_DEF)s

  ...

  ''' % dict(C_FTYPE=C_FTYPE, C_PRECISION_DEF=C_PRECISION_DEF)

"""
if FTYPE == np.float32:
    C_FTYPE = 'float'
    C_PRECISION_DEF = 'SINGLE_PRECISION'
    sys.stderr.write('PISA running in single precision (FP32) mode.\n\n')
elif FTYPE == np.float64:
    C_FTYPE = 'double'
    C_PRECISION_DEF = 'DOUBLE_PRECISION'
    sys.stderr.write('PISA running in double precision (FP64) mode.\n\n')
else:
    raise ValueError('FTYPE must be one of `np.float32` or `np.float64`. Got'
                     ' %s instead.' %FTYPE)
del np

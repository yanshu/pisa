"""
Decorators for profiling code

"""


from functools import wraps
from time import time

from line_profiler import LineProfiler

# Note this relative import (might be) necessary to avoid circular imports
import log


__all__ = ['line_profile', 'profile',
           'test_profile', 'test_line_profile']


class Log(object):
    """Class to redirect output into our logging stream."""
    def write(self, string):
        string = string.rstrip('\n')
        if string:
            log.tprofile.debug(string)
    def flush():
        pass


TLOG = Log()
"""Instance of a global timing logger"""

LINE_PROFILER = LineProfiler()
"""Instance of a global LineProfiler"""


def line_profile(func):
    """Use as `@line_profile` decorator for a function or class method to log
    how long each line in the function/method takes to run.

    Note that timings can be skewed by overhead from the line_profiler module,
    which is used as the core timing mechanism for this function.

    Parameters
    ----------
    func : callable
        Function or method to be profiled

    Returns
    -------
    new_func : callable
        New version of `func` that is callable just like `func` but that logs
        the time spent in each line of code in `func`.

    """
    @wraps(func)
    def profiled_func(*args, **kwargs):
        """<< docstring will be inherited from wrapped `func` >>"""
        try:
            LINE_PROFILER.enable_by_count()
            LINE_PROFILER.add_function(func)
            return func(*args, **kwargs)
        finally:
            LINE_PROFILER.disable_by_count()
            # Only print if it is the outermost function
            if LINE_PROFILER.functions[0] == func:
                LINE_PROFILER.print_stats(stream=TLOG)
    return profiled_func


def profile(func):
    """Use as `@profile` decorator for a function or class method to log the
    time that it takes to complete.

    Parameters
    ----------
    func : callable
        Function or method to profile

    Returns
    -------
    new_func : callable
        New version of `func` that is callable just like `func` but that logs
        the total time spent in `func`.

    """
    @wraps(func)
    def profiled_func(*args, **kwargs):
        """<< docstring will be inherited from wrapped `func` >>"""
        try:
            start_t = time()
            return func(*args, **kwargs)
        finally:
            end_t = time()
            log.tprofile.debug(
                'module %s, function %s: %.4f ms'
                %(func.__module__, func.__name__, (end_t - start_t)*1000)
            )
    return profiled_func


def test_profile():
    """Unit tests for `profile` functional (decorator)"""
    @profile
    def get_number():
        log.logging.trace('hello, i am get_number')
        for x in xrange(500000):
            yield x

    @profile
    def expensive_function():
        log.logging.trace('hello, i am expensive fun')
        for x in get_number():
            _ = x ^ x ^ x
        return 'some result!'

    _ = expensive_function()
    log.logging.info('<< ??? : test_profile >> inspect above outputs')


def test_line_profile():
    """Unit tests for `line_profile` functional (decorator)"""
    @line_profile
    def get_number():
        log.logging.trace('hello, i am get_number')
        for x in xrange(500000):
            yield x

    @line_profile
    def expensive_function():
        log.logging.trace('hello, i am expensive fun')
        for x in get_number():
            _ = x ^ x ^ x
        return 'some result!'

    _ = expensive_function()
    log.logging.info('<< ??? : test_line_profile >> Inspect above outputs')


if __name__ == '__main__':
    log.set_verbosity(2)
    test_profile()
    test_line_profile()

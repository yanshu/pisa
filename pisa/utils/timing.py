#!/usr/bin/env python

import time

import numpy as np

from pisa.utils.format import engfmt


__all__ = ['Timer', 'timediffstamp', 'timestamp']


# TODO: add unit tests!

class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print 'elapsed time: %f ms' % self.msecs


def timediffstamp(dt_sec, hms_always=False, sec_decimals=3):
    """Smart string formatting for a time difference (in seconds)

    Parameters
    ----------
    dt_sec : numeric
        Time difference, in seconds
    hms_always : bool
        * True
            Always display hours, minuts, and seconds regardless of the order-
            of-magnitude of dt_sec
        * False
            Display a minimal-length string that is meaningful, by omitting
            units that are more significant than those necessary to display
            dt_sec; if...
            * dt_sec < 1 s
                Use engineering formatting for the number.
            * dt_sec is an integer in the range 0-59 (inclusive)
                `sec_decimals` is ignored and the number is formatted as an
                integer
            See Notes below for handling of units.
        (Default: False)
    sec_decimals : int
        Round seconds to this number of digits

    Notes
    -----
    If colon notation (e.g. HH:MM:SS.xxx, MM:SS.xxx, etc.) is not used, the
    number is only seconds, and is appended by a space ' ' followed by units
    of 's' (possibly with a metric prefix).
    """
    sign_str = ''
    sgn = 1
    if dt_sec < 0:
        sgn = -1
        sign_str = '-'
    dt_sec = sgn*dt_sec

    r = dt_sec % 3600
    h = int((dt_sec - r)/3600)
    s = r % 60
    m = int((r - s)/60)
    strdt = ''
    if hms_always or h != 0:
        strdt += format(h, '02d') + ':'
    if hms_always or h != 0 or m != 0:
        strdt += format(m, '02d') + ':'

    if float(s) == int(s):
        s = int(s)
        s_fmt = 'd' if len(strdt) == 0 else '02d'
    else:
        # If no hours or minutes, use engineering fmt
        if (h == 0) and (m == 0) and not hms_always:
            return engfmt(dt_sec*sgn, sigfigs=100, decimals=sec_decimals) + 's'
        # Otherwise, round seconds to sec_decimals decimal digits
        s = np.round(s, sec_decimals)
        if len(strdt) == 0:
            s_fmt = '.%df' %sec_decimals
        else:
            if sec_decimals == 0:
                s_fmt = '02.0f'
            else:
                s_fmt = '0%d.%df' %(3+sec_decimals, sec_decimals)
    if len(strdt) > 0:
        strdt += format(s, s_fmt)
    else:
        strdt += format(s, s_fmt) + ' s'

    return sign_str + strdt


def timestamp(d=True, t=True, tz=True, utc=False, winsafe=False):
    """Simple utility to print out a time, date, or time+date stamp for the
    time at which the function is called.

    Parameters
    ----------:
    d : bool
        Include date (default: True)
    t : bool
        Include time (default: True)
    tz : bool
        Include timezone offset from UTC (default: True)
    utc : bool
        Include UTC time/date (as opposed to local time/date) (default: False)
    winsafe : bool
        Omit colons between hours/minutes (default: False)
    """
    if utc:
        time_tuple = time.gmtime()
    else:
        time_tuple = time.localtime()

    dts = ''
    if d:
        dts += time.strftime('%Y-%m-%d', time_tuple)
        if t:
            dts += 'T'
    if t:
        if winsafe:
            dts += time.strftime('%H%M%S', time_tuple)
        else:
            dts += time.strftime('%H:%M:%S', time_tuple)

        if tz:
            if utc:
                if winsafe:
                    dts += time.strftime('+0000')
                else:
                    dts += time.strftime('+0000')
            else:
                offset = time.strftime('%z')
                if not winsafe:
                    offset = offset[:-2:] + '' + offset[-2::]
                dts += offset
    return dts


def test_timestamp():
    print timestamp()


def test_timediffstamp():
    print timediffstamp(1234)


def test_Timer():
    with Timer() as t:
        pass


if __name__ == '__main__':
    test_timestamp()
    test_timediffstamp()
    test_Timer()

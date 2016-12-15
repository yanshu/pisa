# author : P.Eller, T.Ehrhardt, J.L.Lanfranchi
#
# date   : March 25, 2016
"""
Statistical functions

"""

import numpy as np
from scipy.special import gammaln
from uncertainties import unumpy as unp

from pisa.utils.barlow import likelihoods
from pisa.utils.comparisons import isbarenumeric
from pisa.utils.log import logging


__all__ = ['SMALL_POS', 'CHI2_METRICS', 'LLH_METRICS', 'ALL_METRICS',
           'maperror_logmsg',
           'chi2', 'llh', 'log_poisson', 'log_smear', 'conv_poisson',
           'norm_conv_poisson', 'conv_llh', 'barlow_llh', 'mod_chi2']


SMALL_POS = 1e-10
"""A small positive number with which to replace numbers smaller than it"""

CHI2_METRICS = ['chi2', 'mod_chi2']
"""Metrics defined that result in measures of chi squared"""

LLH_METRICS = ['llh', 'conv_llh', 'barlow_llh']
"""Metrics defined that result in measures of log likelihood"""

ALL_METRICS = LLH_METRICS + CHI2_METRICS
"""All metrics defined"""

METRICS_TO_MAXIMIZE = LLH_METRICS
"""Metrics that must be maximized to obtain a better fit"""


# TODO(philippeller):
# * unit tests to ensure these don't break


def maperror_logmsg(m):
    """Create message with thorough info about a map for logging purposes"""
    with np.errstate(invalid='ignore'):
        msg = ''
        msg += '    min val : %s\n' %np.nanmin(m)
        msg += '    max val : %s\n' %np.nanmax(m)
        msg += '    mean val: %s\n' %np.nanmean(m)
        msg += '    num < 0 : %s\n' %np.sum(m < 0)
        msg += '    num == 0: %s\n' %np.sum(m == 0)
        msg += '    num > 0 : %s\n' %np.sum(m > 0)
        msg += '    num nan : %s\n' %np.sum(np.isnan(m))
    return msg


def chi2(actual_values, expected_values):
    """Compute the chi-square between each value in `actual_values` and
    `expected_values`.

    Parameters
    ----------
    actual_values, expected_values : numpy.ndarrays of same shape

    Returns
    -------
    chi2 : numpy.ndarray of same shape as inputs
        chi-squared values corresponding to each pair of elements in the inputs

    Notes
    -----
    * Uncertainties are not propagated through this calculation.
    * Values in each input are clipped to the range [SMALL_POS, inf] prior to
      the calculation to avoid infinities due to the divide function.

    """
    assert actual_values.shape == expected_values.shape

    # Convert to simple numpy arrays containing floats
    if not isbarenumeric(actual_values):
        actual_values = unp.nominal_values(actual_values)
    if not isbarenumeric(expected_values):
        expected_values = unp.nominal_values(expected_values)

    with np.errstate(invalid='ignore'):
        # Mask off any nan expected values (these are assumed to be ok)
        actual_values = np.ma.masked_invalid(actual_values)
        expected_values = np.ma.masked_invalid(expected_values)

        # TODO: this check (and the same for `actual_values`) should probably
        # be done elsewhere... maybe?
        if not np.all(actual_values >= 0.0):
            msg = '`actual_values` must all be >= 0...\n' \
                    + maperror_logmsg(actual_values)
            raise ValueError(msg)

        if not np.all(expected_values >= 0.0):
            msg = '`expected_values` must all be >= 0...\n' \
                    + maperror_logmsg(expected_values)
            raise ValueError(msg)

        # TODO: is this okay to do?
        # replace 0's with small positive numbers to avoid inf in division
        np.clip(actual_values, a_min=SMALL_POS, a_max=np.inf, out=actual_values)
        np.clip(expected_values, a_min=SMALL_POS, a_max=np.inf,
                out=expected_values)

        delta = actual_values - expected_values

    return (delta * delta) / actual_values


def llh(actual_values, expected_values):
    """Compute the log-likelihoods (llh) that each count in `actual_values`
    came from the the corresponding expected value in `expected_values`.

    Parameters
    ----------
    actual_values, expected_values : numpy.ndarrays of same shape

    Returns
    -------
    llh : numpy.ndarray of same shape as the inputs
        llh corresponding to each pair of elements in `actual_values` and
        `expected_values`.

    Notes
    -----
    * Uncertainties are not propagated through this calculation.
    * Values in `expected_values` are clipped to the range [SMALL_POS, inf]
      prior to the calculation to avoid infinities due to the log function.

    """
    assert actual_values.shape == expected_values.shape

    # Convert to simple numpy arrays containing floats
    if not isbarenumeric(actual_values):
        actual_values = unp.nominal_values(actual_values)
    if not isbarenumeric(expected_values):
        expected_values = unp.nominal_values(expected_values)

    with np.errstate(invalid='ignore'):
        # Mask off any nan expected values (these are assumed to be ok)
        actual_values = np.ma.masked_invalid(actual_values)
        expected_values = np.ma.masked_invalid(expected_values)

        # Check that new array contains all valid entries
        if not np.all(actual_values >= 0.0):
            msg = '`actual_values` must all be >= 0...\n' \
                    + maperror_logmsg(actual_values)
            raise ValueError(msg)

        # TODO: How should we handle nan / masked values in the "data"
        # (actual_values) distribution? How about negative numbers?

        # Make sure actual values (aka "data") are valid -- no infs, no nans,
        # etc.
        if np.any((actual_values < 0) | ~np.isfinite(actual_values)):
            msg = '`actual_values` must be >= 0 and neither inf nor nan...\n' \
                    + maperror_logmsg(actual_values)
            raise ValueError(msg)

        # Check that new array contains all valid entries
        if not np.all(expected_values >= 0.0):
            msg = '`expected_values` must all be >= 0...\n' \
                    + maperror_logmsg(expected_values)
            raise ValueError(msg)

        # Replace 0's with small positive numbers to avoid inf in log
        np.clip(expected_values, a_min=SMALL_POS, a_max=np.inf,
                out=expected_values)

    llh_val = actual_values*np.log(expected_values) - expected_values

    # Do following to center around 0
    llh_val -= actual_values*np.log(actual_values) - actual_values

    return llh_val


def log_poisson(k, l):
    """
    calculates the log of a poisson pdf

    - poisson pdf as p(k,l) := l^k*exp(-l)/k!

    Parameters
    ----------
    k : float
    l : float

    Returns
    -------

    log of poisson

    """
    return k*np.log(l) -l - gammaln(k+1)


def log_smear(x, sigma):
    """

    Calculates the log of a normal pdf

    - normal pdf as p(x,sigma) := (sigma*sqrt(2*pi))^-1*exp(-x^2/(2*sigma^2))

    Parameters
    ----------
    x : float
    sigma : float

    Returns
    -------
    log of gaussian

    """
    return (
        -np.log(sigma) - 0.5*np.log(2*np.pi)
        - x**2 / (2*sigma**2)
    )


def conv_poisson(k, l, s, nsigma=3, steps=50):
    """

    poisson pdf as p(k,l) := l^k*exp(-l)/k!

    Parameters
    ----------
    k : float
    l : float
    s : float
        sigma for smearing term (= the uncertainty to be accounted for)
    nsigma : int
        the range in sigmas over which to do the convolution, 3 sigmas is > 99%, so should be enough
    steps : int
        number of steps to do the intergration in (actual steps are 2*steps + 1,
                so this is the steps to each side of the gaussian smearing term)

    Returns
    -------
    
    float
        convoluted poissson likelihood

    """
    # Replace 0's with small positive numbers to avoid inf in log
    l = max(SMALL_POS, l)
    st = 2*(steps + 1)
    conv_x = np.linspace(-nsigma*s, +nsigma*s, st)[:-1]+nsigma*s/(st-1.)
    conv_y = log_smear(conv_x, s)
    f_x = conv_x + l
    #f_x = conv_x + k
    # Avoid zero values for lambda
    idx = np.argmax(f_x > 0)
    f_y = log_poisson(k, f_x[idx:])
    #f_y = log_poisson(f_x[idx:], l)
    if np.isnan(f_y).any():
        logging.error('`NaN values`:')
        logging.error('idx = ', idx)
        logging.error('s = ', s)
        logging.error('l = ', l)
        logging.error('f_x = ', f_x)
        logging.error('f_y = ', f_y)
    f_y = np.nan_to_num(f_y)
    conv = np.exp(conv_y[idx:] + f_y)
    norm = np.sum(np.exp(conv_y))
    return conv.sum()/norm


def norm_conv_poisson(k, l, s, nsigma=3, steps=50):
    """
    Parameters
    ----------
    k : float
    l : float
    s : float
        sigma for smearing term (= the uncertainty to be accounted for)
    nsigma : int
        the range in sigmas over which to do the convolution, 3 sigmas is > 99%, so should be enough
    steps : int
        number of steps to do the intergration in (actual steps are 2*steps + 1,
                so this is the steps to each side of the gaussian smearing term)

    Returns
    -------

    convoluted poisson likelihood normalized so that the value at k=l (asimov) does not change

    """
    cp = conv_poisson(k, l, s, nsigma=nsigma, steps=steps)
    n1 = np.exp(log_poisson(l, l))
    n2 = conv_poisson(l, l, s, nsigma=nsigma, steps=steps)
    return cp*n1/n2


def conv_llh(actual_values, expected_values):
    """Compute the convolution llh using the uncertainty on the expected values
    to smear out the poisson PDFs

    Parameters
    ----------
    actual_values, expected_values : numpy.ndarrays of same shape

    Returns
    -------
    total log of convoluted poisson likelihood

    """
    actual_values = unp.nominal_values(actual_values).ravel()
    sigma = unp.std_devs(expected_values).ravel()
    expected_values = unp.nominal_values(expected_values).ravel()
    triplets = np.array([actual_values, expected_values, sigma]).T
    norm_triplets = np.array([actual_values, actual_values, sigma]).T
    total = 0
    for i in xrange(len(triplets)):
        total += np.log(max(SMALL_POS, norm_conv_poisson(*triplets[i])))
        total -= np.log(max(SMALL_POS, norm_conv_poisson(*norm_triplets[i])))
    return total


def barlow_llh(actual_values, expected_values):
    """Compute the barlow LLH taking into account finite stats

    Parameters
    ----------
    actual_values, expected_values : numpy.ndarrays of same shape

    Returns
    -------
    barlow_llh

    """
    l = likelihoods()
    actual_values = unp.nominal_values(actual_values).ravel()
    sigmas = [unp.std_devs(ev.ravel()) for ev in expected_values]
    expected_values = [unp.nominal_values(ev).ravel() for ev in expected_values]
    uws = [(ev/s)**2 for ev, s in zip(expected_values, sigmas)]
    ws = [s**2/ev for ev, s in zip(expected_values, sigmas)]
    l.SetData(actual_values)
    l.SetMC(np.array(ws))
    l.SetUnweighted(np.array(uws))
    return -l.GetLLH('barlow')


def mod_chi2(actual_values, expected_values):
    """Compute the chi-square value taking into account uncertainty terms
    (incl. e.g. finite stats)

    Parameters
    ----------
    actual_values, expected_values : numpy.ndarrays of same shape

    Returns
    -------
    sum(mod_chi2)

    """
    # Replace 0's with small positive numbers to avoid inf in log
    np.clip(expected_values, a_min=SMALL_POS, a_max=np.inf,
            out=expected_values)
    actual_values = unp.nominal_values(actual_values).ravel()
    sigma = unp.std_devs(expected_values).ravel()
    expected_values = unp.nominal_values(expected_values).ravel()
    m_chi2 = (
        (actual_values - expected_values)**2 / (sigma**2 + expected_values)
    )
    return np.sum(m_chi2)

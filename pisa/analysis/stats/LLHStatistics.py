#
# LLHStatistics.py
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# Generic module that has the helper functions for performing an LLH
# calculations on a set of templates. Operates on templates, which are
# numpy arrays. No input/output operations or reading from customized
# files in this module.
#

import re
import numpy as np
from scipy.special import multigammaln
from scipy.stats import poisson

from pisa.utils.jsons import from_json
from pisa.utils.utils import get_all_channel_names
from pisa.analysis.stats.Maps import get_channel_template


def generalized_ln_poisson(data, expectation):
    """
    When the data set is not integer based, we need a different way to
    calculate the poisson likelihood, so we'll use this version, which
    is appropriate for float data types (using the continuous version
    of the poisson pmf) as well as the standard integer data type for
    the discrete Poisson pmf.

    Returns: the natural logarithm of the value of the continuous form
    of the poisson probability mass function, given detected counts,
    'data' from expected counts 'expectation'.
    """

    if not np.alltrue(data >= 0.0):
        raise ValueError(
            "Template must have all bins >= 0.0! Template generation bug?")

    if bool(re.match('^int',data.dtype.name)):
        return np.log(poisson.pmf(data,expectation))
    elif bool(re.match('^float',data.dtype.name)):
        return (data*np.log(expectation) - expectation - multigammaln(data+1.0, 1))
    else:
        raise ValueError(
            "Unknown data dtype: %s. Must be float or int!"%psuedo_data.dtype)

def get_binwise_llh(pseudo_data, template, channel='all'):
    """
    Computes the log-likelihood (llh) of the pseudo_data from the
    template, where each input is expected to be a 2d numpy array
    """
    if isinstance(pseudo_data, dict) and isinstance(template, dict):
	pd_channel_tmpl = get_channel_template(pseudo_data, channel)
	hypo_tmpl = get_channel_template(template, channel)
	totalLLH = {}
	pd_chans = \
	    sorted(set(pd_channel_tmpl.keys()) & set(get_all_channel_names()))
	tmpl_chans = \
	    sorted(set(hypo_tmpl.keys()) & set(get_all_channel_names()))
	if not pd_chans == tmpl_chans:
            raise ValueError("Templates must have the same channels.")
	for chan in pd_chans:
	    if not np.alltrue(hypo_tmpl[chan] >= 0.0):
	        raise ValueError("Template must have all bins >= 0.0!"
				 " Template generation bug?")
	    totalLLH[chan] = \
	        np.sum(generalized_ln_poisson(
                    pd_channel_tmpl[chan],hypo_tmpl[chan]))
    else:
	raise TypeError("Incompatible type(s): pseudo data=%s, template=%s" %
			(type(pseudo_data), type(template)))

    return totalLLH

def get_binwise_chisquare(pseudo_data, template, channel='all'):
    """
    Computes the chisquare between the pseudo_data
    and the template, where each input is expected to be a 2d numpy array
    """
    if isinstance(pseudo_data, dict) and isinstance(template, dict):
	pd_channel_tmpl = get_channel_template(pseudo_data, channel)
	hypo_tmpl = get_channel_template(template, channel)
	total_chisquare = {}
	pd_chans = \
	    sorted(set(pd_channel_tmpl.keys()) & set(get_all_channel_names()))
	tmpl_chans = \
	    sorted(set(hypo_tmpl.keys()) & set(get_all_channel_names()))
	if not pd_chans == tmpl_chans:
            raise ValueError("Templates must have the same channels.")
	for chan in pd_chans:
	    if not np.alltrue(hypo_tmpl[chan] >= 0.0):
	        raise ValueError("Template must have all bins >= 0.0!"
				 " Template generation bug?")
	    total_chisquare[chan] = \
	        np.sum(np.divide(np.power(
		      (pd_channel_tmpl[chan] - hypo_tmpl[chan]), 2),
		       pd_channel_tmpl[chan]))
    else:
	raise TypeError("Incompatible type(s): pseudo data=%s, template=%s" %
			(type(pseudo_data), type(template)))

    return total_chisquare


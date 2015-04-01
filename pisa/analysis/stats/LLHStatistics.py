#
# LLHStatistics.py
#
# Generic module that has the helper functions for performing an LLH
# calculations on a set of templates. Operates on templates, which are
# numpy arrays. No input/output operations or reading from customized
# files in this module.
#

from pisa.utils.jsons import from_json
import numpy as np
from scipy.stats import poisson

def get_binwise_llh(pseudo_data,template):
    '''
    computes the log-likelihood (llh) of the pseudo_data from the
    template, where each input is expected to be a 2d numpy array
    '''

    if not np.alltrue(template >= 0.0):
        raise ValueError("Template must have all bins >= 0.0! Template generation bug?")
    totalLLH = np.sum(np.log(poisson.pmf(pseudo_data,np.float64(template))))

    return totalLLH

    #totalLLH = 0
    #if not len(template)==len(pseudo_data):
    #    raise ValueError("Template and pseudo_data should have equal lengths, len(template)=%i, len(pseudo_data) = %i"%(len(template),len(pseudo_data)))
    #for i in range(0,len(pseudo_data)):
    #    value_d = pseudo_data[i]
    #    value_t = template[i]
    #    if value_t<0:
    #        return -1e10
    #    if value_d==0 and value_t==0:
    #        continue
    #    if value_d!=0 and value_t==0:
    #        return -1e10
    #    #totalLLH += value_d*np.log(value_t) - value_t - (value_d*np.log(value_d)-value_d)
    #    totalLLH += value_d*np.log(value_t) - value_t
    #return totalLLH

def get_random_map(template, seed=None):
    '''
    Gets an event map with integer entries from non-integer entries
    (in general) in the template, varied according to Poisson
    statistics.
    '''
    #Set the seed if given
    if not seed is None:
        np.random.seed(seed=seed)

    return poisson.rvs(template)



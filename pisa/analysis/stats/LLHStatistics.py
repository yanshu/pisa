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
from scipy.stats import poisson,skellam

def get_binwise_llh(pseudo_data,template,template_params):
    '''
    computes the log-likelihood (llh) of the pseudo_data from the
    template, where each input is expected to be a 2d numpy array
    '''
    if template_params['residual_up_down']:
        if len(template)!=2:
            raise ValueError("Under current template settings, template must be an array of two arrays(i.e. up-going and down-going array)!")
        if not np.alltrue(template >= 0.0):
            raise ValueError("Template must have all bins >= 0.0! Template generation bug?")
        #totalLLH = np.sum(skellam.logpmf(pseudo_data,np.float64(template[0]),np.float64(template[1])))     #causes overflow when mu1,mu2 gets too big
        mu1 = np.float64(template[0])
        mu2 = np.float64(template[1])
        totalLLH = np.sum(-mu1-mu2+0.5*pseudo_data*np.log(mu1/mu2)+np.log(iv(pseudo_data,2*np.sqrt(mu1*mu2))))
    else:
        if not np.alltrue(template >= 0.0):
            raise ValueError("Template must have all bins >= 0.0! Template generation bug?")
        totalLLH = np.sum(np.log(poisson.pmf(pseudo_data,np.float64(template))))

    return totalLLH

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



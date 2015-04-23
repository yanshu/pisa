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
from scipy.special import iv

def get_binwise_llh(pseudo_data,template,template_params):
    '''
    computes the log-likelihood (llh) of the pseudo_data from the
    template, where each input is expected to be a 2d numpy array
    '''
    if not np.alltrue(template >= 0.0):
        raise ValueError("Template must have all bins >= 0.0! Template generation bug?")
    if 'residual_up_down' in template_params and template_params['residual_up_down']:
        if len(template)!=2:
            raise ValueError("Under current template settings, template must be an array of two arrays(i.e. up-going and down-going array)!")
        mu1 = np.float64(template[0])
        mu2 = np.float64(template[1])
        #totalLLH = np.sum(skellam.logpmf(pseudo_data,np.float64(template[0]),np.float64(template[1])))    
                                              # causes overflow when mu1,mu2 gets too big, it calculates pmf first, then take the log.
        totalLLH = np.sum(-mu1-mu2+0.5*pseudo_data*np.log(mu1/mu2)+np.log(iv(pseudo_data,2*np.sqrt(mu1*mu2)))) 
                                              # better, but could also end up inf if (mu1*mu2) is too large, e.g. iv(0,2*np.sqrt(127444)) ~ 1.7961e+308
    elif 'ratio_up_down' in template_params and template_params['ratio_up_down']:
        if len(template)!=2 or len(pseudo_data)!=2:
            raise ValueError("Under current template settings, template and pseudo_data must be an array of two arrays(i.e. up-going and down-going array)!")
        t_N_up = np.float64(template[0])
        t_N_down = np.float64(template[1])
        d_N_up = np.float64(pseudo_data[0])
        d_N_down = np.float64(pseudo_data[1])
        t_R = np.nan_to_num(t_N_up/t_N_down)
        d_R = np.nan_to_num(d_N_up/d_N_down)
        #t_R_err_sqr = np.nan_to_num(t_N_up*(t_N_up+t_N_down)/(t_N_down**3))
        #d_R_err_sqr = np.nan_to_num(d_N_up*(d_N_up+d_N_down)/(d_N_down**3))
        #t_R_err_sqrt_inv = np.nan_to_num(t_N_down*np.sqrt(t_N_down)/np.sqrt(t_N_up*(t_N_up+t_N_down)))
        d_R_err_sqrt_inv = np.nan_to_num(d_N_down*np.sqrt(d_N_down)/np.sqrt(d_N_up*(d_N_up+d_N_down)))
        totalLLH = -np.nan_to_num(np.sum(np.square((t_R-d_R)*d_R_err_sqrt_inv)))        # this definition is the new test statistic for ratio_analysis, it's not a likelihood function, its value should increase when two maps get more different, that's why the negative sign
        if 0 in d_N_down:
            print "d_N_down has 0"
            print "d_N_down = ", d_N_down
            print "d_N_up = ", d_N_up
            print "d_R = ", d_R
            print "d_R_err_sqrt_inv = ", d_R_err_sqrt_inv
            print "totalLLH = ", totalLLH
        if 0 in t_N_down:
            print "t_N_down has 0"
            print "t_N_down = ", t_N_down
            print "t_R = ", t_R
            print "t_R_err_sqrt_inv = ", t_R_err_sqrt_inv
            print "totalLLH = ", totalLLH
    else:
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



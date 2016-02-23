#
# Generic module that has the helper functions for performing an LLH
# calculations on a set of templates. Operates on templates, which are
# numpy arrays. No input/output operations or reading from customized
import numpy as np
from scipy.special import gammaln

def get_binwise_llh(pseudo_data, template):
    """
    Computes the log-likelihood (llh) of the pseudo_data from the
    template, where each input is expected to be a 2d numpy array
    """
    return np.sum( pseudo_data * np.log(template)- gammaln(pseudo_data+1) - template)     # avoid LLH = inf 

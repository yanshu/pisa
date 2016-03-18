#
# Generic module that has the helper functions for performing an LLH
# calculations on a set of templates. Operates on templates, which are
# numpy arrays. No input/output operations or reading from customized
import numpy as np
from scipy.special import gammaln

def log_poisson(k,l):
    return k*np.log(l) -l - gammaln(k+1)

def log_smear(x,sigma):
    return-np.log(sigma)-0.5*np.log(2*np.pi)-np.square(x)/(2*np.square(sigma))

def conv_poisson(k,l,s,nsigma=3,steps=100.):
    st = 2*(steps+1)
    conv_x = np.linspace(-nsigma*s,+nsigma*s,st)[:-1]+nsigma*s/(st-1.)
    conv_y = log_smear(conv_x,s)
    f_x = conv_x + l
    # avoid zero values for lambda
    idx = np.argmax(f_x>0)
    f_y = log_poisson(k,f_x[idx:])
    conv = np.exp(conv_y[idx:] + f_y)
    return conv.sum()*(conv_x[1]-conv_x[0])

def get_binwise_llh(pseudo_data, template):
    """
    Computes the log-likelihood (llh) of the pseudo_data from the
    template, where each input is expected to be a 2d numpy array
    """
    # replace 0 or negative entries in template with very small numbers, to avoid inf in log
    template[template <= 0] = 10e-10
    return np.sum( pseudo_data * np.log(template)- gammaln(pseudo_data+1) - template)     # avoid LLH = inf 

def get_binwise_smeared_llh(pseudo_data, template, sumw2):
    """
    Computes the log-likelihood (llh) of the pseudo_data from the
    template, where each input is expected to be a 2d numpy array
    """
    sigma = np.sqrt(sumw2)
    #template[template <= 0] = 10e-10
    triplets = np.array([pseudo_data, template, sigma]).T
    sum = 0
    for i in xrange(len(triplets)):
        sum += np.log(max(10e-10,conv_poisson(*triplets[i])))
    return sum

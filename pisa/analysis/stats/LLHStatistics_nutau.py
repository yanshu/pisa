#
# Generic module that has the helper functions for performing an LLH
# calculations on a set of templates. Operates on templates, which are
# numpy arrays. No input/output operations or reading from customized
import numpy as np
import scipy
from scipy.special import gammaln
from pisa.analysis.stats.barlow_llh import likelihoods
from pisa.utils.params import get_param_priors

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
    if np.isnan(f_y).any():
        print "idx = ", idx
        print "s = ", s
        print "l = ", l
        print "f_x = ", f_x
        print "f_y = ", f_y
    f_y = np.nan_to_num(f_y)
    conv = np.exp(conv_y[idx:] + f_y)
    return conv.sum()*(conv_x[1]-conv_x[0])

def get_chi2(data, mc, data_sumw2, mc_sumw2, params, priors):
    """
    Computes the chi2 of the (pseudo) data and the best fit
    template, where each input is expected to be a 2d numpy array
    """
    chi2 = np.sum((data-mc)**2/(data_sumw2 + mc_sumw2))
    if len(priors)!=0:
        chi2 += sum([prior.chi2(val) for (val, prior) in zip(params, priors)])
    no_uniform_prior = 0
    for prior in priors:
        if prior.kind == 'uniform':
            no_uniform_prior += 1
    dof = len(data) - no_uniform_prior 
    chi2_p = scipy.stats.chi2.sf(chi2, df=dof)
    return chi2, chi2_p, dof

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

def get_barlow_llh(data, map_nu, sumw2_nu, map_mu, sumw2_mu):
    l = likelihoods()
    uw_nu = np.square(map_nu)/sumw2_nu
    uw_nu = np.nan_to_num(uw_nu)
    uw_mu = np.square(map_mu)/sumw2_mu
    uw_mu = np.nan_to_num(uw_mu)
    w_nu = sumw2_nu/map_nu
    w_nu = np.nan_to_num(w_nu)
    w_mu = sumw2_mu/map_mu
    w_mu = np.nan_to_num(w_mu)
    l.SetData(data)
    l.SetMC(np.array([w_nu,w_mu]))
    l.SetUnweighted(np.array([uw_nu,uw_mu]))
    llh =  l.GetLLH('barlow') 
    del l
    return llh

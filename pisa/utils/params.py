#
# params.py
#
# Functions for accessing and formatting parameters taken from .json
# template settings file for analysis.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   16 October 2014
#


import re
from scipy import interpolate, optimize
import numpy as np
from copy import deepcopy
from pisa.utils.log import logging
import pisa.utils.fileio as fileio
import pisa.resources.resources as resources
from pisa.utils.jsons import from_json

class Prior(object):
    def __init__(self, **kwargs):
        self.constructor_args = deepcopy(kwargs)
        if not kwargs.has_key('kind'):
            raise TypeError(str(self.__class__) + ' __init__ requires `kind` kwarg to be specified')
        kind = kwargs.pop('kind')
        # Dispatch the correct initialization method
        if kind.lower() in ['none', 'uniform'] or kind is None:
            self.__init_uniform(**kwargs)
        elif kind.lower() == 'gaussian':
            self.__init_gaussian(**kwargs)
        elif kind.lower() == 'linterp':
            self.__init_linterp(**kwargs)
        elif kind.lower() == 'spline':
            self.__init_spline(**kwargs)
        else:
            raise TypeError('Unknown Prior kind `' + str(kind) + '`')

    @classmethod
    def from_param(cls, param):
        """Factory method for generating a Prior object from a param dict."""

        if not isinstance(param, dict):
            raise TypeError("`from_param` factory method can only instantiate a Prior object from a param dict")

        # If param has no 'prior', do not create a Prior object
        # NOTE: This is probably a poor design decision, but maintains a more
        # "sparse" config file; probably should change in future.
        if 'prior' not in param:
            return None

        prior = param['prior']

        # Old-style prior specs that translate to a uniform prior
        if prior is None or (isinstance(prior, str) and prior.lower() == 'none'):
            return cls(kind='uniform')

        # Old-style prior spec that translates to a gaussian prior
        elif isinstance(prior, (int, float)):
            fiducial = param['value']
            sigma = prior
            return cls(kind='gaussian', fiducial=fiducial, sigma=sigma)

        # New-style prior is a dictionary
        elif isinstance(prior, dict):
            return cls(**prior)

        else:
            raise TypeError("Uninterpretable param dict 'prior': " + str(prior))

    def __str__(self):
        return self._str(self)

    def __repr__(self):
        return '<' + str(self.__class__) + ' ' + self.__str__() + '>'

    def build_dict(self, node_dict=None):
        if node_dict is None:
            node_dict = {}
        node_dict['prior'] = self.constructor_args
        return node_dict

    def __init_uniform(self):
        self.kind = 'uniform'
        self.llh = lambda x: 0.*x
        self.chi2 = lambda x: 0.*x
        self.valid_range = [-np.inf, np.inf]
        self.max_at = np.nan
        self.max_at_str = "no maximum"
        self._str = lambda s: "uniform prior"

    def __init_gaussian(self, fiducial, sigma):
        self.kind = 'gaussian'
        self.fiducial = fiducial
        self.sigma = sigma
        self.llh = lambda x: -(x-self.fiducial)**2 / (2*self.sigma**2)
        self.chi2 = lambda x: (x-self.fiducial)**2 / (self.sigma**2)
        self.valid_range = [-np.inf, np.inf]
        self.max_at = self.fiducial
        self.max_at_str = format(self.max_at, '0.4e')
        self._str = lambda s: "gaussian prior: sigma=%.4e, max at %.4e" % (self.sigma, self.fiducial)

    def __init_linterp(self, x, y):
        self.kind = 'linterp'
        self.x = np.array(x)
        self.y = np.array(y)
        self.interp = interpolate.interp1d(self.x, self.y, kind='linear',
                                           copy=True, bounds_error=True)
        self.llh = lambda x_new: self.interp(x_new)
        self.chi2 = lambda x_new: -2 * self.llh(x_new)
        self.valid_range = [min(self.x), max(self.x)]
        self.max_at = self.x[self.y == np.max(self.y)]
        self.max_at_str = ", ".join([format(v, '0.4e') for v in self.max_at])
        self._str = lambda s: "linearly-interpolated prior: valid in [%0.4e, %0.4e], max at %s" % (self.valid_range[0], self.valid_range[1], self.max_at_str)

    def __init_spline(self, knots, coeffs, deg):
        '''Spline is expected to define the log likelihood (so LLH = spline and
        chi^2 = -2*spline)
        
        knots, coeffs, and deg are given by e.g. scipy.interpolate.splrep, and
        evaluation of splines is carried out by scipy.interpolate.splev
        '''
        self.kind = 'spline'
        self.knots = knots
        self.coeffs = coeffs
        self.deg = deg
        self.llh = lambda x: interpolate.splev(x, tck=(knots, coeffs, deg), ext=2)
        self.chi2 = lambda x: -2 * self.llh(x)
        self.valid_range = [np.min(knots), np.max(knots)]
        self.max_at = optimize.fminbound(
            func=self.chi2,
            x1=self.valid_range[0],
            x2=self.valid_range[1],
        )
        self.max_at_str = format(self.max_at, '0.4e')
        self._str = lambda s: "spline prior: deg=%d, valid in [%0.4e, %0.4e], max at %s" % (self.deg, self.valid_range[0], self.valid_range[1], self.max_at_str)

    def check_range(self, x_range):
        return min(x_range) >= self.valid_range[0] and max(x_range) <= self.valid_range[1]


def plot_prior(obj, param=None, x_xform=None, ax1=None, ax2=None, **plt_kwargs):
    '''Plot prior for param from template settings, params, or prior filename or dict.
   
    Arguments
    ---------
    obj : str or dict
        if str, interpret as path from which to load a dict
        if dict, can be:
            template settings dict; must supply `param` to choose which to plot
            params dict; must supply `param` to choose which to plot
            prior dict
    param
        Param name to plot; necessary if obj is either template settings or params
    x_xform
        Transform to apply to x-values. E.g., to plot against sin^2 theta, use
        x_xform = lambda x: np.sin(x)**2
    ax1, ax2
        Axes onto which to plot LLH and chi-squared, respectively. If none are
        provided, new figures & axes will be created.
    plt_kwargs
        Keyword arguments to pass on to the plot function

    Returns
    -------
    ax1, ax2
        The axes onto which plots were drawn (ax1 = LLH, ax2 = chi^2)
    '''
    import matplotlib.pyplot as plt
    if isinstance(obj, basestring):
        obj = fileio.from_file(obj)
    if 'params' in obj:
        obj = obj['params']
    if param is not None and param in obj:
        obj = obj[param]
    if 'prior' in obj:
        obj = obj['prior']

    prior = Prior(**obj)
    logging.info('Plotting Prior: %s' % prior)
    x0 = prior.valid_range[0]
    x1 = prior.valid_range[1]
    if prior.kind == 'gaussian':
        x0 = max(x0, prior.max_at - 5*prior.sigma)
        x1 = min(x1, prior.max_at + 5*prior.sigma)
    if np.isinf(x0):
        x0 = -1
    if np.isinf(x1):
        x1 = +1
    x = np.linspace(x0, x1, 5000)
    llh = prior.llh(x)
    chi2 = prior.chi2(x)

    if x_xform is not None:
        x = x_xform(x)

    if ax1 is None:
        f = plt.figure()
        ax1 = f.add_subplot(111)
    if ax2 is None:
        f = plt.figure()
        ax2 = f.add_subplot(111)

    ax1.plot(x, llh, **plt_kwargs)
    ax2.plot(x, chi2, **plt_kwargs)
    
    ax1.set_title(str(prior))
    ax2.set_title(str(prior))
    ax1.set_xlabel(param)
    ax2.set_xlabel(param)
    ax1.set_ylabel('LLH')
    ax2.set_ylabel(r'$\Delta\chi^2$')

    return ax1, ax2


def get_values(params):
    """
    Takes the params dict which is of the form:
      {'param1': {
        'value': val,
        'fixed': bool
       },
       'param2': {...},...
      }
    and returns a dictionary of names/values as:
      {'param1': val,
       'param2': val,...
      }
    """
    return deepcopy({ key: param['value'] for key, param in sorted(params.items()) })

def construct_shape_dict(keyword, kwargs): #at the moment (2016-03-01) only called in flux modification not in the genie case
    flux_mod_dict = {}
    logging.info('constructing shape dict for: %s'%keyword)
    logging.info('keys we are getting in: %s'%sorted(kwargs.keys()))
    #logging.info("params: %s "%params)
    #logging.info( "params type: %s "%type(params))
    if keyword == 'flux':
        flux_unc_dict = from_json(resources.find_resource(kwargs['flux_uncertainty_inputs']))
        par_names = flux_unc_dict.keys()
    logging.info( "\n the par names are: %s "%par_names)
    for entry in kwargs:
        logging.info( "going for : %s"%entry)
        #logging.info( "can we access it? %s "%kwargs['params'][entry])
        if entry in par_names:
            flux_mod_dict[entry] = kwargs[entry]
            logging.info( "adding to flux_mod_dict: %s "%entry)
            #logging.info( '\n flux mod dict: %s '%flux_mod_dict)
    logging.info("flux_mod_dict, keys: %s "%sorted(flux_mod_dict.keys()))
    #flux_mod_dict['fixed'] = {'fixed':'true'}
    return flux_mod_dict

def construct_genie_dict(params):
    ### make dict of genie params ###
    GENSYS = {}
    if not "GENSYS_AhtBY" in params:
        print "setting to zero: GENSYS_AhtBY"
        GENSYS["AhtBY"] = 0.0
    else: GENSYS["AhtBY"] = params["GENSYS_AhtBY"]
        
    if not "GENSYS_BhtBY" in params:
        GENSYS["BhtBY"] = 0.0
        print "setting to zero: GENSYS_BhtBY"
    else: GENSYS["BhtBY"] = params["GENSYS_BhtBY"]
        
    if not "GENSYS_CV1uBY" in params:
        GENSYS["CV1uBY"] = 0.0
        print "setting to zero GENSYS_CV1uBY"
    else: GENSYS["CV1uBY"] = params["GENSYS_CV1uBY"]
        
    if not "GENSYS_CV2uBY" in params:
        GENSYS["CV2uBY"] = 0.0
        print "setting to zero GENSYS_CV2uBY"
    else: GENSYS["CV2uBY"] = params["GENSYS_CV2uBY"]
        
    if not "GENSYS_MaCCQE" in params:
        GENSYS["MaCCQE"] = 0.0
        print "setting to zero GENSYS_MaCCQE"
    else: GENSYS["MaCCQE"] = params["GENSYS_MaCCQE"]
        
    if not "GENSYS_MaRES" in params:
        GENSYS["MaRES"] = 0.0
        print "setting to zero GENSYS_MaRES"
    else: GENSYS["MaRES"] = params["GENSYS_MaRES"]
    return GENSYS

def select_hierarchy(params, normal_hierarchy):
    """
    Corrects for a key in params being defined for both 'nh' and 'ih',
    and returns a modified dict with one value for the given hierarchy.
    """

    if not isinstance(normal_hierarchy, bool):
        raise ValueError('Hierarchy selection must be boolean value')

    newparams = {}
    for key, value in params.items():
        if key.endswith('_nh'):
            #Don't use if doesn't match request
            if not normal_hierarchy: continue
            #Use simplified key
            key = key.rsplit('_',1)[0]
        if key.endswith('_ih'):
            if normal_hierarchy: continue
            key = key.rsplit('_',1)[0]

        newparams[key] = value

    return deepcopy(newparams)

def select_hierarchy_and_nutau_norm(params,normal_hierarchy,nutau_norm_value):
    ''' Select one hierarchy and change the value of nutau_norm (for the pseudo settings file)'''
    newparams = select_hierarchy(params,normal_hierarchy)
    newparams["nutau_norm"]["value"] = nutau_norm_value
    return newparams

def change_nutau_norm_settings(params,nutau_norm_val,nutau_norm_fix,normal_hierarchy, pos_def = True):
    ''' Change the value of nutau_norm["value"] and ["fixed"] (for the template settings file)'''
    if not isinstance(nutau_norm_fix, bool):
                raise ValueError('nutau_norm_fix must be boolean value')
    newparams = select_hierarchy(params, normal_hierarchy)
    newparams["nutau_norm"]["value"] = nutau_norm_val
    newparams["nutau_norm"]["fixed"] = nutau_norm_fix
    if not(pos_def):
        newparams["nutau_norm"]["range"] = np.array([-2.5, 2.5])
    return newparams

def change_settings(params,name,val,fix):
    if not isinstance(fix, bool):
                raise ValueError('fix must be boolean value')
    newparams = deepcopy(params)
    newparams[name]["value"] = val
    newparams[name]["fixed"] = fix
    return newparams

def get_fixed_params(params):
    """
    Finds all fixed parameters in params dict and returns them in a
    new dictionary.
    """

    return deepcopy({ key: value for key, value in params.items() if value['fixed']})


def get_free_params(params):
    """
    Finds all free parameters in params dict and returns them in a new
    dictionary.
    """

    return deepcopy({ key: value for key, value in params.items() if not value['fixed']})

def get_param_values(params):
    """
    Returns a list of parameter values
    """
    return [ deepcopy(val['value']) for key,val in sorted(params.items()) ]


def get_param_scales(params):
    """
    Returns a list of parameter scales
    """
    return [ deepcopy(val['scale']) for key,val in sorted(params.items()) ]


def get_param_bounds(params):
    """
    Returns a list of parameter bounds where elements are (min,max) pairs
    """
    return [ deepcopy(val['range']) for key,val in sorted(params.items()) ]


def get_param_priors(params):
    """
    Returns a list of Prior objects, one for each param.
    """
    priors = []
    for pname,param in sorted(params.items()):
        try:
            prior = Prior(**param['prior'])
        except TypeError:
            logging.error("  Check template settings format, "
                          "may have old-style priors")
            raise
        priors.append(prior)
    return priors


def get_atm_params(params):
    """
    Returns dictionary of just the atmospheric parameters
    """
    atm_params = ['deltam31','theta23']
    return deepcopy({ key: value for key, value in params.items()
             for p in atm_params if p in key })


def fix_osc_params(params):
    """
    Returns dict identical to params dict but with all oscillation
    parameters set to "fixed"=True
    """
    new_params = {}
    # or initialize with new copy by dict(params)
    osc_params = ['deltam31', 'deltam21', 'theta23', 'theta13', 'theta12',
                  'deltacp']
    for key,value in params.items():
        new_params[key] = deepcopy(value)
        for okey in osc_params:
            if okey in key:
                new_params[key]['fixed'] = True

    return new_params

def float_param(params,param):
    """
    returns dict with param set to 'fixed'=False
    """
    new_params = {}
    # or initialize with new copy by dict(params)
    for key,value in params.items():
        new_params[key] = deepcopy(value)
        if param in key:
            new_params[key]['fixed'] = False

    return new_params

def fix_param(params,param):
    """
    returns dict with param set to 'fixed'=True
    """
    new_params = {}
    # or initialize with new copy by dict(params)
    for key,value in params.items():
        new_params[key] = deepcopy(value)
        if param in key:
            new_params[key]['fixed'] = True

    return new_params
     
def fix_all_param(params,param):
    """
    returns dict with param set to 'fixed'=True
    """
    new_params = {}
    # or initialize with new copy by dict(params)
    for key,value in params.items():
        new_params[key] = deepcopy(value)
        if param[key].has_key('fixed'):
            new_params[key]['fixed'] = True

    return new_params


def fix_atm_params(params):
    """
    Returns dict identical to params dict but with all atmospheric
    oscillation parameters fixed.
    """
    new_params = {}
    # or initialize with new copy by dict(params)
    atm_params = ['deltam31', 'theta23']
    for key,value in params.items():
        new_params[key] = deepcopy(value)
        #for akey in atm_params:
        if (bool(re.match('^theta23', key)) or bool(re.match('^deltam31', key))):
            new_params[key]['fixed'] = True

    return new_params

def fix_non_atm_params(params):
    """
    Returns dict identical to params dict, except that it fixes all
    parameters besides that atmospheric mixing params:
    theta23/deltam31. Does not modify atm mix params, leaving them
    fixed or not, as they were..
    """

    new_params = {}
    # or initialize with new copy by dict(params)
    for key,value in params.items():
        new_params[key] = deepcopy(value)
        if (bool(re.match('^theta23', key)) or bool(re.match('^deltam31', key))):
            continue
        else:
            new_params[key]['fixed'] = True

    return new_params

def fix_all_params(params):
    """
    Returns dictionary identical to params, but with all
    parameters fixed.
    """
    new_params = {}
    for k,v in params.items():
        new_params[k] = deepcopy(v)
        new_params[k]['fixed'] = True

    return new_params

def test_Prior(ts_fname, param_name='theta23'):
    '''Produce plots roughly like NuFIT's 1D chi-squared projections'''
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    sigma = [1, 2, 3, 4, 5]
    chi2 =  [s**2 for s in sigma]

    ts = fileio.from_file(resources.find_resource(ts_fname))
    f1 = plt.figure(1) #,figsize=(8,14),dpi=60)
    f2 = plt.figure(2) #,figsize=(8,14),dpi=60)
    f1.clf();f2.clf()
    ax1 = f1.add_subplot(111)
    ax2 = f2.add_subplot(111)

    # Defaults
    x_xform = None
    xlabel = param_name
    xlim = None
    ylim = 0, 15

    # Special cases
    if param_name == 'theta12':
        x_xform = lambda x: np.sin(x)**2
        xlabel = r'$\sin^2\theta_{12}$'
        xlim = 0.2, 0.42
    elif param_name == 'theta23':
        x_xform = lambda x: np.sin(x)**2
        xlabel = r'$\sin^2\theta_{23}$'
        xlim = 0.26, 0.74
    elif param_name == 'theta13':
        x_xform = lambda x: np.sin(x)**2
        xlabel = r'$\sin^2\theta_{13}$'
        xlim = 0.012, 0.032
    elif param_name == 'deltam21':
        x_xform = lambda x: x*1e5
        xlabel = r'$\Delta m^2_{21} \; {\rm[10^{-5}\;eV^2]}$'
        xlim = 6.5, 8.7
    elif param_name == 'deltam31':
        x_xform = lambda x: np.abs(x)*1e3
        xlabel = r'$|\Delta m^2_{31}| \; {\rm[10^{-3}\;eV^2]}$'
        xlim = 2.15, 2.8
    elif param_name == 'deltacp':
        xlabel = r'$\delta_{\rm CP} \; {\rm [deg]}$'
    plot_prior(select_hierarchy(ts['params'], normal_hierarchy=True),
               param=param_name,
               x_xform=x_xform, ax1=ax1, ax2=ax2,
               color='r', label=r'${\rm NH}$')
    plot_prior(select_hierarchy(ts['params'], normal_hierarchy=False),
               param=param_name,
               x_xform=x_xform, ax1=ax1, ax2=ax2,
               color='b', linestyle='--', label=r'${\rm IH}$')

    ax1.set_ylim([-0.5*y for y in ylim[::-1]])
    ax2.set_ylim(ylim)
    plt.tight_layout()

    for ax in [ax1, ax2]:
        ax.legend(loc='best',frameon=False)
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel)
        ax.grid(which='both', b=True)
        ax.set_title('')

    for c2 in chi2:
        ax2.plot(xlim, [c2,c2], 'k-', lw=1.0, alpha=0.4)

    plt.draw();plt.show()

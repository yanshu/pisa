import scipy as sp; from scipy import stats
from NeutrinoPath import NeutrinoPath
class SimpleAtmosphere():
    def __init__(self, earth_radius=6371., nue_height=14., numu_height=17.6, nutau_height=0.):
        self.num_sample_points = 1
        self.earth_radius = earth_radius
        self.nue_height  = nue_height    #< [km]
        self.numu_height = numu_height   #< [km]
        self.nutau_height = nutau_height #< [km]
        self.path = None

    def set_zenith(self, zenith):
        self.path = NeutrinoPath(0., -(self.earth_radius - 2.), zenith)

    def get_nue_lengths(self):
        return self.__path_lengths__(self.nue_height)

    def get_numu_lengths(self):
        return self.__path_lengths__(self.numu_height)

    def get_nutau_lengths(self):
        return self.__path_lengths__(self.nutau_height)

    def __path_lengths__(self, height):
        r = self.earth_radius + height
        lengths = [self.path.get_separation_length(r, self.earth_radius)]
        return sp.asarray(lengths)


class NormalAtmosphere():
    def __init__(self, num_sample_points=8, earth_radius=6371., max_height=110.):
        self.num_sample_points = num_sample_points
        self.earth_radius = earth_radius #< [km]
        self.nue_height  = sp.stats.norm(loc=14.0, scale=9.3) #< [km]
        self.numu_height = sp.stats.norm(loc=17.6, scale=8.9) #< [km]
        self.top_of_atmosphere = max_height #< max height allowed for neutrinon production [km]
        self.path = None

    def set_zenith(self, zenith):
        self.path = NeutrinoPath(0., -(self.earth_radius - 2.), zenith)

    def get_nue_lengths(self):
        return self.__path_lengths__(self.nue_height)

    def get_numu_lengths(self):
        return self.__path_lengths__(self.numu_height)

    def get_nutau_lengths(self):
        return self.get_numu_lengths()
        #return sp.zeros(self.num_sample_points)

    def __path_lengths__(self, dist):
        heights = self.__inverse_transform_sampler__(dist, self.num_sample_points)
        radii = self.earth_radius + heights
        lengths = []
        for r in radii:
            path_length = self.path.get_separation_length(r, self.earth_radius)
            lengths.append(path_length)
        return sp.asarray(lengths)

    def __inverse_transform_sampler__(self, dist, N):
        """ Skip the first and last points to avoid infinities. """
        pseudo_uniform_points = sp.linspace(dist.cdf(0.), dist.cdf(self.top_of_atmosphere), N+2)[1:-1]
        # PPF = inverse of CDF, e.g. PPF(0.5) = median of distribution
        return dist.ppf(pseudo_uniform_points)

import math
import numpy as np
class NuCraftAtmosphere():
    def __init__(self, num_sample_points=8, earth_radius=6371.):
        self.num_sample_points = num_sample_points
        self.earth_radius = earth_radius
        self.top_of_atmosphere = np.inf

    def cos_theta_eff(self, cz, h=5.):
        R = self.earth_radius
        r = R / (R+h)
        return np.sqrt(1 - (r*r)*(1 - cz*cz))

    def set_zenith(self, zenith):
        self.path = NeutrinoPath(0., -(self.earth_radius - 2.), zenith)
        self.cosZen = self.cos_theta_eff(np.cos(zenith))
        #self.cosZen = (np.fabs(math.cos(zenith))**3 + 0.000125)**0.333333333

    def get_nue_lengths(self):
        mu = 1.285e-9*(self.cosZen-4.677)**14. + 2.581
        sigma = 0.6048*self.cosZen**0.7667 - 0.5308*self.cosZen + 0.1823
        logn = sp.stats.lognorm(sigma, scale=2*np.exp(mu), loc=-12)
        return self.__path_lengths__(logn)

    def get_numu_lengths(self):
        mu = 1.546e-9*(self.cosZen-4.618)**14. + 2.553
        sigma = 1.729*self.cosZen**0.8938 - 1.634*self.cosZen + 0.1844
        logn = sp.stats.lognorm(sigma, scale=2*np.exp(mu), loc=-12)
        return self.__path_lengths__(logn)

    def get_nutau_lengths(self):
        return self.get_numu_lengths()

    def __path_lengths__(self, dist):
        heights = self.__inverse_transform_sampler__(dist, self.num_sample_points)
        radii = self.earth_radius + heights
        lengths = []
        for r in radii:
            path_length = self.path.get_separation_length(r, self.earth_radius)
            lengths.append(path_length)
        return sp.asarray(lengths)

    #def __inverse_transform_sampler__(self, dist):
    #    cdf0 = dist.cdf(0)
    #    qList = cdf0 + np.array([0.0625,0.1875,0.3125,0.4375,0.5625,0.6875,0.8125,0.9375])*(1.-cdf0)
    #    return dist.ppf(qList)*self.cosZen

    def __inverse_transform_sampler__(self, dist, N):
        """ Skip the first and last points to avoid infinities. """
        pseudo_uniform_points = sp.linspace(dist.cdf(0.), dist.cdf(self.top_of_atmosphere), N+2)[1:-1]
        # PPF = inverse of CDF, e.g. PPF(0.5) = median of distribution
        return dist.ppf(pseudo_uniform_points) * self.cosZen


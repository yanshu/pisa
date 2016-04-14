from EarthLayer import EarthLayer
from NeutrinoPath import NeutrinoPath
from numpy.polynomial import Polynomial as P
import scipy as sp
from scipy import integrate
import numpy as np
import warnings

class PREM():
    """
    The Preliminary Reference Earth Model.

    Functions for getting the density at various positions and paths
    through the earth.

    """
    
    def __init__(self, step_size=500.):
        self.earth_radius = 6371. #km
        self.production_radius = self.earth_radius + 20. #km
        self.elevation_icecube = -2.00 #km
        self.detector_position = (0., -(self.earth_radius + self.elevation_icecube))
        self.km_per_cm = 1.e-5
        self.step_size = step_size

        self.layers = [
            #EarthLayer(name="atmosphere",   density_function=P([     0.,      0.,      0.,      0.]), r_min=self.earth_radius, r_max=self.production_radius, scale_radius=self.earth_radius),
            EarthLayer(name="ocean",        density_function=P([ 1.0200,      0.,      0.,      0.]), r_min=6368.0, r_max=self.earth_radius, scale_radius=self.earth_radius, electron_fraction=0.4957),
            EarthLayer(name="crust_2",      density_function=P([ 2.6000,      0.,      0.,      0.]), r_min=6356.0, r_max=6368.0,            scale_radius=self.earth_radius, electron_fraction=0.4957),
            EarthLayer(name="crust_1",      density_function=P([ 2.9000,      0.,      0.,      0.]), r_min=6346.6, r_max=6356.0,            scale_radius=self.earth_radius, electron_fraction=0.4957),
            EarthLayer(name="lvz_lid",      density_function=P([ 2.6910,  0.6924,      0.,      0.]), r_min=6151.0, r_max=6346.6,            scale_radius=self.earth_radius, electron_fraction=0.4957),
            EarthLayer(name="transition_3", density_function=P([ 7.1089, -3.8045,      0.,      0.]), r_min=5971.0, r_max=6151.0,            scale_radius=self.earth_radius, electron_fraction=0.4957),
            EarthLayer(name="transition_2", density_function=P([11.2494, -8.0298,      0.,      0.]), r_min=5771.0, r_max=5971.0,            scale_radius=self.earth_radius, electron_fraction=0.4957),
            EarthLayer(name="transition_1", density_function=P([ 5.3197, -1.4836,      0.,      0.]), r_min=5701.0, r_max=5771.0,            scale_radius=self.earth_radius, electron_fraction=0.4957),
            EarthLayer(name="lower_mantle", density_function=P([ 7.9565, -6.4761,  5.5283, -3.0807]), r_min=3480.0, r_max=5701.0,            scale_radius=self.earth_radius, electron_fraction=0.4957),
            EarthLayer(name="outer_core",   density_function=P([12.5815, -1.2638, -3.6426, -5.5281]), r_min=1221.5, r_max=3480.0,            scale_radius=self.earth_radius, electron_fraction=0.4656),
            EarthLayer(name="inner_core",   density_function=P([13.0885,      0., -8.8381,      0.]), r_min=   0.0, r_max=1221.5,            scale_radius=self.earth_radius, electron_fraction=0.4656),
            ]

        # Don't take chances...
        self.max_layers = np.ceil(2.*self.production_radius / self.step_size) + 2*len(self.layers)

    def get_density(self, radius):
        density = 0. #< vacuum if not in any of the layers
        for earth_layer in self.layers:
                # The (2.*Y_e) produces a density that is approximately equal to the matter density
                #   and allows us to include a pseudo-Y_e in the matter potential
                density += (2.*earth_layer.electron_fraction) * earth_layer.get_density(radius)

        return density

    def get_earthy_stuff(self, zenith):
        # output lists
        path_lengths = [] #< km
        integrated_density = [] #< g/cm^2

        (detector_x, detector_y) = self.detector_position
        path = NeutrinoPath(detector_x, detector_y, zenith)

        still_integrating = True
        going_in = True
        (x_i, y_i) = path.get_intersection_point(self.layers[0].r_max, going_in) #< start at the edge of the ocean

        j = 0 #< earth layer
        while still_integrating:
            layer = self.layers[j]

            if (j == len(self.layers)-1):
                #print "Bypass the center of the earth."
                going_in = False

            (x_iplus1, y_iplus1) = self.__get_trial_points__(path, layer, going_in)
            if not np.isrealobj((x_iplus1, y_iplus1)):
                #print "going out..."
                going_in = False
                (x_iplus1, y_iplus1) = self.__get_trial_points__(path, layer, going_in)

            if (layer.name == "ocean") and ((x_iplus1 < 0.) or (not going_in)):
                #print "Arriving at the detector."
                (x_iplus1, y_iplus1) = self.detector_position #< end at the detector
                still_integrating = False

            lengths = get_lengths(x_i, y_i, x_iplus1, y_iplus1, self.step_size)
            path_lengths.extend(np.diff(lengths))

            if np.fabs(x_iplus1 - x_i) > np.fabs(y_iplus1 - y_i):
                #integrate with respect to 'X'
                derivative = -path.sine * self.km_per_cm #< dx/dL
                bounds = x_i - lengths * path.sine
                density_function = get_density_from_x
            else:
                #integrate with respect to 'Y'
                derivative = -path.cosine * self.km_per_cm #< dy/dL
                bounds = y_i - lengths * path.cosine
                density_function = get_density_from_y

            for k in xrange(0, len(bounds)-1):
                integral, error = sp.integrate.quad(
                        lambda z: density_function(z, path, layer) / derivative,
                        bounds[k], bounds[k+1])

                integrated_density.append(integral)
                length = lengths[k+1] - lengths[k]
                length_in_cm =  length / self.km_per_cm
                #print "{:>12s}: L = {length:6.0f} km,      rho = {rho:6.3f} +/- {err:5.2e} g/cm^3".format(
                #        layer.name, length=length, rho=integral/length_in_cm, err=error)

            # Get ready for the next layer
            (x_i, y_i) = (x_iplus1, y_iplus1)

            # Handle the iteration index
            if going_in:
                j += 1
            else:
                j -= 1


        #print "\n"
        return (np.asarray(path_lengths), np.asarray(integrated_density))


    def __get_trial_points__(self, path, layer, going_in):
        if not going_in:
            return path.get_intersection_point(layer.r_max, False)
        else:
            return path.get_intersection_point(layer.r_min, True)


def get_lengths(x_start, y_start, x_stop, y_stop, step_size):
    length = get_length(x_start, y_start, x_stop, y_stop)
    num_steps = 1 + np.ceil(length / step_size)
    lengths = np.linspace(0., length, num_steps)
    return lengths

def get_length(x_start, y_start, x_stop, y_stop):
    dx = x_stop - x_start
    dy = y_stop - y_start
    length = np.sqrt(dx*dx + dy*dy)
    #print "L = {l:8.2f} = sqrt(({x2: 7.1f} - {x1: 7.1f})**2 + ({y2: 7.1f} - {y1: 7.1f})**2)".format(
    #        l=length, x1=x_start, x2=x_stop, y1=y_start, y2=y_stop)
    return length

def get_density_from_x(x, path, layer):
    y = path.get_y_from_x(x)
    radius = get_radius(x, y)

    # The (2.*Y_e) produces a density that is approximately equal to the matter density
    #   and allows us to include a pseudo-Y_e in the matter potential
    density = (2.*layer.electron_fraction) * layer.get_density(radius)
    #print "rho(x={x: 7.2f}, {zen:4.2f}) = {value:0.2e} g/cm^3".format(x=x, zen=zenith, value=density)
    return density

def get_density_from_y(y, path, layer):
    x = path.get_x_from_y(y)
    radius = get_radius(x, y)

    # The (2.*Y_e) produces a density that is approximately equal to the matter density
    #   and allows us to include a pseudo-Y_e in the matter potential
    density = (2.*layer.electron_fraction) * layer.get_density(radius)
    #print "rho(y={y: 7.2f}, {zen:4.2f}) = {value:0.2e} g/cm^3".format(y=y, zen=zenith, value=density)
    return density

def get_radius(x, y):
    return np.sqrt(x*x + y*y)


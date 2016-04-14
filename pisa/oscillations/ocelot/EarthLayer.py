import numpy as np

class EarthLayer():
    def __init__(self,
            name = "",
            electron_fraction = 0.5,
            density_function = lambda r: 0., 
            r_min = 0.,
            r_max = 0.,
            scale_radius = 1.):

        self.name = name
        self.electron_fraction = electron_fraction
        self.density_function = density_function
        self.r_min = r_min
        self.r_max = r_max
        self.scale_radius = scale_radius

    def get_density(self, r):
        if self.__legal_radius__(r):
            return self.density_function( r / self.scale_radius )
        else:
            return 0.

    def __legal_radius__(self, r):
        return (self.r_min <= r) and (r <= self.r_max)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "\n{:>20s}: {mi:6.1f} <= r < {ma:6.1f}".format(self.name, mi=self.r_min, ma=self.r_max)


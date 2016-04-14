import numpy as np
from EarthLayer import EarthLayer

class NeutrinoPath():
    def __init__(self, detector_x, detector_y, zenith):
        self.x_det = detector_x
        self.y_det = detector_y
        self.R = np.sqrt(detector_x*detector_x + detector_y*detector_y)
        nadir = np.pi - zenith
        self.sine = np.sin(nadir)
        self.cosine = np.cos(nadir)

    def get_separation_length(self, r, R, entry=True):
        x, y = self.get_intersection_point(r, entry)
        X, Y = self.get_intersection_point(R, entry)
        dx = x - X
        dy = y - Y
        distance = np.sqrt(dx*dx + dy*dy)
        return distance

    def get_distance_to_detector(self, r, entry=True):
        (x_shell, y_shell) = self.get_intersection_point(r, entry)
        dx = x_shell - self.x_det
        dy = y_shell - self.y_det
        distance = np.sqrt(dx*dx + dy*dy)
        return distance

    def get_intersection_point(self, r, entry=True):
        """Returns complex numbers if there is no physical intersection point."""
        sqrt_term = np.lib.scimath.sqrt((r/self.R - self.sine) * (r/self.R + self.sine))

        x_a = self.R * self.sine * self.cosine
        x_b = self.R * self.sine * sqrt_term

        y_a = -self.R * self.sine * self.sine
        y_b = self.R * self.cosine * sqrt_term

        if entry:
            x = x_a + x_b
            y = y_a + y_b
        else:
            x = x_a - x_b
            y = y_a - y_b

        return (x,y)
    
    def get_x_from_y(self, y):
        return (y + self.R) * (self.sine / self.cosine)

    def get_y_from_x(self, x):
        return (x * self.cosine / self.sine) - self.R


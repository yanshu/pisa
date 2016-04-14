import numpy as np
import scipy as sp
from scipy import linalg
import math

from scipy.constants import codata
G_Fermi = codata.value('Fermi coupling constant') # actually G_Fermi / (h_bar*c)**3 to get 1/[GeV]^2
N_A     = codata.value('Avogadro constant')
h_bar   = codata.value('Planck constant over 2 pi in eV s')
c       = codata.value('speed of light in vacuum')

class MixingParameters():
    def __init__(self, delta_M21_sq=0., delta_M31_sq=0.,
                 theta_12=0., theta_23=0., theta_13=0., delta_cp=0.,
                 num_neutrinos=3):
        """
        delta_M21_sq [eV^2] (signed)
        delta_M31_sq [eV^2] (signed)
        theta_12 [rad]
        theta_23 [rad]
        theta_13 [rad]
        delta_cp [rad]
        num_neutrinos (allows for future extensions)

        """

        self.delta_M21_sq = delta_M21_sq
        self.delta_M31_sq = delta_M31_sq
        self.theta_12 = theta_12
        self.theta_23 = theta_23
        self.theta_13 = theta_13
        self.delta_cp = delta_cp
        self.num_neutrinos = num_neutrinos


def init_oscillation_constants(normal_hierarchy=True):
    """Best fit values from arXiv:1205.4018v1, Table 1"""

    if normal_hierarchy:
        params = MixingParameters(delta_M21_sq =  7.62e-5,
                                    delta_M31_sq =  2.53e-3,
                                    theta_12 = math.asin(math.sqrt(0.320)),
                                    theta_23 = math.asin(math.sqrt(0.49)),
                                    theta_13 = math.asin(math.sqrt(0.026)),
                                    delta_cp = 0.83 * math.pi)

    else:
        params = MixingParameters(delta_M21_sq =  7.62e-5,
                                    delta_M31_sq = -2.40e-3,
                                    theta_12 = math.asin(math.sqrt(0.320)),
                                    theta_23 = math.asin(math.sqrt(0.53)),
                                    theta_13 = math.asin(math.sqrt(0.027)),
                                    delta_cp = 0.07 * math.pi)
    return params



class NeutrinoMixing():
    """
    """

    def __init__(self, params=MixingParameters()):

        # Store the mixing angles, and mass splittings
        self.params = params

        # Conversion factors
        self.eV_per_GeV = 1.e+9          # 1GeV * eV_per_GeV => eV
        self.cm_eV = 1.e+2*(h_bar * c)   # 1cm / cm_eV => eV^{-1}
        self.km_eV = 1.e-3*(h_bar * c)   # 1eV / km_eV => km^{-1}

        # Non-trivial class members
        self.U_pmns = self.__construct_mixing_matrix()
        self.dtype = self.U_pmns.dtype
        self.M_sq = self.__construct_mass_mixing_matrix()
        self.M_sq_flavor = self.mass_to_flavor(self.M_sq)
        self.A_matter = self.__construct_matter_potential()
        self.A_matter_mass = self.flavor_to_mass(self.A_matter)

    def updated(self):
        """ Needs to be called after updating any values in self.params """
        self.U_pmns = self.__construct_mixing_matrix()
        self.dtype = self.U_pmns.dtype
        self.M_sq = self.__construct_mass_mixing_matrix()
        self.M_sq_flavor = self.mass_to_flavor(self.M_sq)
        #! independent of mixing parameters: self.A_matter = self.A_matter
        self.A_matter_mass = self.flavor_to_mass(self.A_matter)

    def flavor_to_mass(self, flavor_state):
        mass_state = self.U_pmns.H * sp.asmatrix(flavor_state) * self.U_pmns
        return mass_state

    def mass_to_flavor(self, mass_state):
        flavor_state = self.U_pmns * sp.asmatrix(mass_state) * self.U_pmns.H
        return flavor_state

    def __construct_sub_mixing_matrix(self, state_m, state_n, angle, phase_angle=0.):
        # Numpy arrays go from [0...N-1], mass states go from [1...N]
        index_m = state_m - 1
        index_n = state_n - 1

        cosine = sp.cos(angle)
        sine = sp.sin(angle)
        phase = sp.exp(-1.j * phase_angle)

        mixing_matrix = sp.identity(self.params.num_neutrinos, complex)
        mixing_matrix[index_m, index_m] = cosine
        mixing_matrix[index_m, index_n] = sine * phase
        mixing_matrix[index_n, index_n] = cosine
        mixing_matrix[index_n, index_m] = -sine * sp.conjugate(phase)

        return sp.asmatrix(mixing_matrix)

    def __construct_mixing_matrix(self):
        """ U_pmns = U_{23} U_{13} U_{12} """

        U_23 = self.__construct_sub_mixing_matrix(2, 3, self.params.theta_23)
        U_13 = self.__construct_sub_mixing_matrix(1, 3, self.params.theta_13, self.params.delta_cp)
        U_12 = self.__construct_sub_mixing_matrix(1, 2, self.params.theta_12)

        # CPT requires cp_violating_phase to be flipped for anti-neutrinos
        U_neutrinos = U_23 * U_13 * U_12
        U_antineutrinos = sp.conjugate(U_neutrinos)

        U_pmns = sp.zeros((2*self.params.num_neutrinos,2*self.params.num_neutrinos), U_neutrinos.dtype)
        U_pmns[ :self.params.num_neutrinos,  :self.params.num_neutrinos] = U_neutrinos
        U_pmns[-self.params.num_neutrinos:, -self.params.num_neutrinos:] = U_antineutrinos

        return sp.asmatrix(U_pmns)


    def __construct_mass_mixing_matrix(self):
        """ Create the standard mass-mixing matrix. """

        # N.B. If m1_sq = 0, then 2-flavor mixing fails.
        #       The matricies need non-zero values to avoid
        #       the trivial solution where nothing happens.
        m1_sq = 1.e-6 #< arbitrary value
        m2_sq = self.params.delta_M21_sq + m1_sq
        m3_sq = self.params.delta_M31_sq + m1_sq
        M_sq_temp = sp.diagflat([m1_sq, m2_sq, m3_sq, m1_sq, m2_sq, m3_sq])

        # Ensure that all elements are positive
        M_sq = sp.diagflat(M_sq_temp.diagonal() - M_sq_temp.min())
        return sp.asmatrix(M_sq)


    def __construct_matter_potential(self):
        """
        Define matter potential, units are:
        [eV^-2]*[mol^-1]*[mol/g] = [g eV^2]^-1
        
        """

        # A more real value for Y_e is included in the density, but I also include
        #  a factor of 2 to make the density look more like expected values, this
        #  factor of 0.5 removes the factor of 2 from the density to make it correct.
        Y_e = 0.5   # electron fraction in the earth [mol/g?]

        # Keep the constants in eV
        # N.B. G_Fermi = G_F/(h_bar*c)**3, with units [GeV^-2]
        G_Fermi_in_eV = G_Fermi / (self.eV_per_GeV ** 2) #< [eV^-2]

        A_matter_ee = sp.sqrt(2.) * G_Fermi_in_eV * N_A * Y_e
        A_matter = sp.diagflat([A_matter_ee, 0. , 0., -A_matter_ee, 0. , 0.])
        return sp.asmatrix(A_matter)

    def __str__(self):
        return """
Neutrino Oscillation Parameters
============================================================
{constants:s}
{matrices:s}
        """.format(constants = self.__stringify_the_constants(),
                   matrices = self.__stringify_the_matrices())


    def __stringify_the_constants(self):
        constants_string = """
Mass Splittings [eV^2]
----------------------------------------
{Delta:s}(m_21)^2 = {delta_M21_sq: 9.2e}
{Delta:s}(m_31)^2 = {delta_M31_sq: 9.2e}
    
Mixing Angles [rad]
----------------------------------------
{theta:s}_12 = {theta_12: f}
{theta:s}_23 = {theta_23: f}
{theta:s}_13 = {theta_13: f}
{delta:s} = {delta_cp: f}"""

        return constants_string.format(
            theta="\\theta", #u"\u03b8",
            Delta="\\Delta", #u"\u0394",
            delta="\\delta", #u"\u03b4",
            **vars(self.params))


    def __stringify_the_matrices(self):
        matrices_string = """
PMNS Mixing Matrix
----------------------------------------
{U_pmns:s}

Mass Squared Matrix
----------------------------------------
{M_sq:s}

Mass Squared Matrix (Flavor Basis)
----------------------------------------
{M_sq_flavor:s}"""
        
        np.set_printoptions(precision=5,
                            linewidth=130,
                            suppress=True)

        return matrices_string.format(
            U_pmns = self.U_pmns.__str__(),
            M_sq = self.M_sq.__str__(),
            M_sq_flavor = self.M_sq_flavor.__str__())


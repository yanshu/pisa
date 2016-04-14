import numpy as np
np.set_printoptions(precision=3, linewidth=130, suppress=True)
import scipy as sp
import warnings

from scipy.constants import codata
G_Fermi = codata.value('Fermi coupling constant') # actually G_Fermi / (h_bar*c)**3 to get 1/[GeV]^2
N_A     = codata.value('Avogadro constant')
h_bar   = codata.value('Planck constant over 2 pi in eV s')
c       = codata.value('speed of light in vacuum')


class BargerProbabilities():
    def __init__(self, neutrino_mixing, earth_model):
        self.mixing = neutrino_mixing
        self.earth = earth_model

        self.energy = 0. #< eV
        self.zenith = 0. #< radians

        self.nue = 0; self.nue_bar = self.nue + self.mixing.params.num_neutrinos
        self.nu1 = 0; self.nu1_bar = self.nu1 + self.mixing.params.num_neutrinos
        self.nu2 = 1; self.nu2_bar = self.nu2 + self.mixing.params.num_neutrinos
        self.nu3 = 2; self.nu3_bar = self.nu3 + self.mixing.params.num_neutrinos

    def updated(self):
        """ Needs to be called after updating any values in self.mixing.params """
        self.mixing.updated()

    def set_zenith(self, zenith):
        if (zenith == self.zenith):
            # values are cached, no need to update
            pass
        else:
            self.zenith = zenith
            (self.path_lengths, self.path_densities) = self.earth.get_earthy_stuff(zenith)
            self.path_lengths /= self.mixing.km_eV #< [eV^-1]
            self.path_densities *= self.mixing.cm_eV**2 #< [g eV^2]
            self.average_densities = self.path_densities / self.path_lengths #< [g eV^3]
            #self.average_densities = self.path_densities / (self.path_lengths / self.earth.km_per_cm) #< [g/cm^3]

    def set_energy(self, energy):
        if (energy*self.mixing.eV_per_GeV == self.energy):
            # values are cached, no need to update
            pass
        else:
            self.energy = energy * self.mixing.eV_per_GeV   #< [eV]
            self.eigenvalue_order = self.__get_eigenvalue_order()

    def vacuum_probabilities(self, zenith=None, energy=None):
        if (zenith is not None):
            self.set_zenith(zenith)
        if (energy is not None):
            self.set_energy(energy)

        total_length = sp.sum(self.path_lengths)
        total_length_in_eV = total_length / self.mixing.km_eV

        warnings.simplefilter("ignore")
        amplitudes = sp.linalg.expm2(-1.j * self.mixing.M_sq_flavor * total_length_in_eV / (2.*self.energy))
        warnings.simplefilter("default")

        amplitudes = sp.asarray(amplitudes)
        probabilities = amplitudes * sp.conjugate(amplitudes)
        return probabilities.real


    def matter_probabilities(self, zenith=None, energy=None):
        if (zenith is not None):
            self.set_zenith(zenith)
        if (energy is not None):
            self.set_energy(energy)

        n_nu = self.mixing.params.num_neutrinos 
        amplitudes = sp.asmatrix(sp.identity(2*n_nu, complex))

        # Compute the amplitude in each layer where the density is integrable
        for j, L_j in enumerate(self.path_lengths):
            # Local aliases are the best kind of aliases
            rho_j = self.average_densities[j]
            #print "\n\nStep {:2d}:  density = {:5.2f} g/cm^3, length = {:6.2f} km".format(layer, rho_j, L_j)

            amp_j = sp.zeros_like(self.mixing.U_pmns)
            amp_j[ :n_nu, :n_nu] = self.__get_matrix_exponential(rho_j, L_j, anti_neutrinos=False)
            amp_j[-n_nu:,-n_nu:] = self.__get_matrix_exponential(rho_j, L_j, anti_neutrinos=True)

            # Matrix multiplication satisfies the boundary conditions between layers
            amplitudes = amplitudes * sp.asmatrix(amp_j)
            #print amplitudes

        # Probabilities are calculated by taking the elementwise complex modulus in flavor states
        amplitudes = sp.asarray(self.mixing.mass_to_flavor(amplitudes))
        probabilities = amplitudes * sp.conjugate(amplitudes)

        return probabilities.real

    def __get_matrix_exponential(self, density, length, anti_neutrinos=False):
        """Using Sylvester's formula (valid for diagonalizable matricies)"""

        Kamiltonian = self.__get_kamiltonian(density, anti_neutrinos)
        eigenvalues = self.__get_eigenvalues(density, anti_neutrinos)

        result = sp.zeros_like(Kamiltonian)
        for k, lambda_k in enumerate(eigenvalues):
            f_of_lambda_k = sp.exp(-1.j * lambda_k * length / (2. * self.energy))
            frobenius_covariant_k = self.__get_frobenius_covariant_matrix(k, eigenvalues, Kamiltonian)
            result += f_of_lambda_k * frobenius_covariant_k

        return result

    def __get_kamiltonian(self, density, anti_neutrinos=False):
        """K = 2E*H = \\Delta M^2 + 2E*V."""
        full_kamiltonian = self.mixing.M_sq + (2.*self.energy * self.mixing.A_matter_mass * density)
        n_nu = self.mixing.params.num_neutrinos 
        if anti_neutrinos:
            return full_kamiltonian[-n_nu:, -n_nu:]
        else:
            return full_kamiltonian[ :n_nu,  :n_nu]


    def __get_frobenius_covariant_matrix(self, i, eigenvalues, matrix):
        lambda_i = eigenvalues[i]
        identity = sp.asmatrix(sp.identity(self.mixing.params.num_neutrinos, complex))
        frobenius_covariant_i = identity
        for j, lambda_j in enumerate(eigenvalues):
            if (j == i):
                continue
            else:
                covariant_ij = (matrix - lambda_j*identity) / (lambda_i - lambda_j)
                frobenius_covariant_i = frobenius_covariant_i * covariant_ij

        return frobenius_covariant_i

    def __get_eigenvalues(self, density, anti_neutrinos=False):
        unordered_eigenvalues = self.__get_unordered_eigenvalues(density, anti_neutrinos)
        return self.__sort_eigenvalues(unordered_eigenvalues)

    def __sort_eigenvalues(self, unordered_eigenvalues):
        sorted_eigenvalues = sp.zeros_like(unordered_eigenvalues)
        
        for i_unsorted, j_sorted in enumerate(self.eigenvalue_order):
            sorted_eigenvalues[j_sorted] = unordered_eigenvalues[i_unsorted]

        return sorted_eigenvalues

    def __get_eigenvalue_order(self):
        vacuum_density = 0.
        unordered_vacuum_values = self.__get_unordered_eigenvalues(vacuum_density)

        n_nu = self.mixing.params.num_neutrinos 
        ordered_masses = sp.diagonal(self.mixing.M_sq[:n_nu, :n_nu])

        unshuffler = sp.zeros(n_nu, dtype=int)
        for i, eigenvalue in enumerate(unordered_vacuum_values):
            deltas = sp.absolute(eigenvalue - ordered_masses)
            unshuffler[i] = sp.argmin(deltas)

        return unshuffler

    def __get_unordered_eigenvalues(self, density, anti_neutrinos=False):
        matter_coefficient = self.__get_matter_coefficient(density, anti_neutrinos)
        alpha = self.__get_alpha(matter_coefficient) #< alpha is not explicitly dependent of neutrino type
        beta = self.__get_beta(matter_coefficient, anti_neutrinos)
        gamma = self.__get_gamma(matter_coefficient, anti_neutrinos)

        numerator = 2.*alpha**3 - 9.*alpha*beta + 27.*gamma
        denominator = 2. * sp.sqrt((alpha**2 - 3.*beta)**3)
        fraction = numerator / denominator
        if (sp.absolute(fraction) > 1.):
            # Don't let numerical stability stand in your way...
            fraction = fraction / sp.absolute(fraction)

        # these are the roots the paper refers to...
        theta_1 = sp.arccos(fraction) / 3.
        theta_2 = theta_1 - 2.*sp.pi/3.
        theta_3 = theta_1 + 2.*sp.pi/3.
        angles = sp.array([theta_1, theta_2, theta_3])

        m1_sq = self.mixing.M_sq[self.nu1, self.nu1] 
        eigenvalues = (-2./3. * sp.sqrt(alpha**2 - 3.*beta) * sp.cos(angles)) + m1_sq - alpha/3.
        return eigenvalues

    def __get_matter_coefficient(self, density, anti_neutrinos=False):
        if anti_neutrinos:
            matter_coefficient = 2.*self.energy * self.mixing.A_matter[self.nue_bar, self.nue_bar] * density
        else:
            matter_coefficient = 2.*self.energy * self.mixing.A_matter[self.nue, self.nue] * density

        # Don't forget the magical negative sign!!!
        matter_coefficient *= -1.

        return (matter_coefficient).real

    def __get_alpha(self, matter_coefficient):
        delta_m12_sq = -self.mixing.params.delta_M21_sq
        delta_m13_sq = -self.mixing.params.delta_M31_sq

        alpha_1 = matter_coefficient
        alpha_2 = delta_m12_sq
        alpha_3 = delta_m13_sq
        return (alpha_1 + alpha_2 + alpha_3).real

    def __get_beta(self, matter_coefficient, anti_neutrinos=False):
        delta_m12_sq = -self.mixing.params.delta_M21_sq
        delta_m13_sq = -self.mixing.params.delta_M31_sq

        if anti_neutrinos:
            U_e2 = self.mixing.U_pmns[self.nue_bar, self.nu2_bar] 
            U_e3 = self.mixing.U_pmns[self.nue_bar, self.nu3_bar]
        else:
            U_e2 = self.mixing.U_pmns[self.nue, self.nu2]
            U_e3 = self.mixing.U_pmns[self.nue, self.nu3]

        beta_1 = delta_m12_sq * delta_m13_sq
        beta_2a = matter_coefficient
        beta_2b = delta_m12_sq * (1. - U_e2*sp.conjugate(U_e2))
        beta_2c = delta_m13_sq * (1. - U_e3*sp.conjugate(U_e3))
        return (beta_1 + beta_2a*(beta_2b + beta_2c)).real

    def __get_gamma(self, matter_coefficient, anti_neutrinos=False):
        delta_m12_sq = -self.mixing.params.delta_M21_sq
        delta_m13_sq = -self.mixing.params.delta_M31_sq

        if anti_neutrinos:
            U_e1 = self.mixing.U_pmns[self.nue_bar, self.nu1_bar]
        else:
            U_e1 = self.mixing.U_pmns[self.nue, self.nu1]

        gamma_1a = matter_coefficient
        gamma_1b = delta_m12_sq * delta_m13_sq * U_e1*sp.conjugate(U_e1)
        return (gamma_1a * gamma_1b).real


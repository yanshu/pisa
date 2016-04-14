import numpy as np
np.set_printoptions(precision=3, linewidth=130, suppress=True)
import scipy as sp; from scipy import linalg
import warnings

def check_unitarity(probability_matrix, precision=5.e-4):
    for j_flavor in xrange(0, 6):
        measurement = np.fabs(1. - np.sum(probability_matrix[j_flavor,:])) 
        if (measurement > precision):
            warnings.warn("The computed unitarity does not meet the specified precision: {:0.2e} > {:0.2e}".format(measurement, precision))

        measurement = np.fabs(1. - np.sum(probability_matrix[:,j_flavor])) 
        if (measurement > precision):
            warnings.warn("The computed unitarity does not meet the specified precision: {:0.2e} > {:0.2e}".format(measurement, precision))

    return

class AtmosphericProbabilities():
    def __init__(self, neutrino_mixing, atmospheric_model):
        self.mixing = neutrino_mixing
        self.atmosphere = atmospheric_model

        self.energy = -1. #< eV
        self.zenith = -1. #< radians

    def updated(self):
        """ Needs to be called after updating any values in self.mixing.params """
        # update the mixing matricies
        self.mixing.updated()

        # ignore any cached values
        self.energy = -1. #< eV
        self.zenith = -1. #< radians

    def set_zenith(self, zenith):
        if (zenith == self.zenith) and (zenith != -1.):
            # values are cached, no need to update
            pass
        else:
            self.zenith = zenith
            self.atmosphere.set_zenith(zenith)
            self.atm_nue_lengths = self.atmosphere.get_nue_lengths() / self.mixing.km_eV #< [eV^-1]
            self.atm_numu_lengths = self.atmosphere.get_numu_lengths() / self.mixing.km_eV #< [eV^-1]

    def set_energy(self, energy):
        if (energy*self.mixing.eV_per_GeV == self.energy):
            # values are cached, no need to update
            pass
        else:
            self.energy = energy * self.mixing.eV_per_GeV   #< [eV]

    def vacuum_probabilities(self, zenith=None, energy=None):
        """ Calculate vacuum probabilities using eigenvalue decomposition of H_vacuum*L.  """
        if (energy is not None):
            self.set_energy(energy)
        if (zenith is not None):
            self.set_zenith(zenith)

        nue_L_over_2E = self.atm_nue_lengths / (2. * self.energy)
        numu_L_over_2E = self.atm_numu_lengths / (2. * self.energy)

        num_neutrinos = self.mixing.params.num_neutrinos
        nue  = 0;   nuebar  = nue + num_neutrinos;
        numu = 1;   numubar = numu + num_neutrinos;

        average_probability = sp.identity(2*num_neutrinos)
        avg__nue_amp = sp.zeros_like(average_probability, dtype=complex)
        avg_numu_amp = sp.zeros_like(average_probability, dtype=complex)

        for (nue_Lo2E, numu_Lo2E) in zip(nue_L_over_2E, numu_L_over_2E):
            avg__nue_amp += self.__get_probability_amplitudes__(nue_Lo2E)
            avg_numu_amp += self.__get_probability_amplitudes__(numu_Lo2E)
        avg__nue_amp /= self.atmosphere.num_sample_points
        avg_numu_amp /= self.atmosphere.num_sample_points

        avg__nue_probability = self.__get_probabilities__(avg__nue_amp)
        avg_numu_probability = self.__get_probabilities__(avg_numu_amp)

        average_probability[:,    nue] = avg__nue_probability[:,    nue]
        average_probability[:,   numu] = avg_numu_probability[:,   numu]
        average_probability[:, nuebar] = avg__nue_probability[:, nuebar]
        average_probability[:,numubar] = avg_numu_probability[:,numubar]

        return sp.asmatrix(average_probability)

    def __get_probabilities__(self, amplitudes):
        probabilities = (amplitudes * sp.conjugate(amplitudes)).real
        #check_unitarity(probabilities)
        return probabilities

    def __get_probability_amplitudes__(self, L_over_2E):
        # vacuum probabilities
        argument = -1.j * sp.diagonal(self.mixing.M_sq) * L_over_2E    #< collapase to diagonal
        amplitudes_mass = sp.diagflat(sp.exp(argument.astype(complex)))  #< exponentiate, then expand to full matrix
        amplitudes = self.mixing.mass_to_flavor(amplitudes_mass)    #< convert to flavor eigenstates

        amplitudes = sp.asarray(amplitudes)
        return amplitudes

class Probabilities():
    def __init__(self, neutrino_mixing, earth_model, atmospheric_model):
        self.mixing = neutrino_mixing
        self.earth = earth_model
        self.atmosphere = atmospheric_model

        self.energy = 0. #< eV
        self.zenith = 0. #< radians

    def updated(self):
        """ Needs to be called after updating any values in self.mixing.params """
        self.mixing.updated()
        if (self.energy != 0.):
            self.H_vacuum = self.mixing.M_sq_flavor / (2.*self.energy) #< [GeV]

    def set_zenith(self, zenith):
        if (zenith == self.zenith):
            # values are cached, no need to update
            pass
        else:
            self.zenith = zenith
            self.atmosphere.set_zenith(zenith)
            self.atm_nue_lengths = self.atmosphere.get_nue_lengths() / self.mixing.km_eV #< [eV^-1]
            self.atm_numu_lengths = self.atmosphere.get_numu_lengths() / self.mixing.km_eV #< [eV^-1]
            self.atm_nutau_lengths = self.atmosphere.get_nutau_lengths() / self.mixing.km_eV #< [eV^-1]

            (self.path_lengths, self.path_densities) = self.earth.get_earthy_stuff(zenith)
            self.path_lengths /= self.mixing.km_eV #< [eV^-1]
            self.path_densities *= self.mixing.cm_eV**2 #< [g eV^2]

    def set_energy(self, energy):
        if (energy*self.mixing.eV_per_GeV == self.energy):
            # values are cached, no need to update
            pass
        else:
            self.energy = energy * self.mixing.eV_per_GeV   #< [eV]
            self.H_vacuum = self.mixing.M_sq_flavor / (2.*self.energy) #< [GeV]

    def vacuum_probabilities(self, length=None, zenith=None, energy=None):
        """ Calculate vacuum probabilities using eigenvalue decomposition of H_vacuum*L.  """

        if (length is not None):
            total_length = length / self.mixing.km_eV #< [eV^-1]
        else:
            if (zenith is not None):
                self.set_zenith(zenith)
            total_length = sp.sum(self.path_lengths)

        if (energy is not None):
            self.set_energy(energy)

        L_over_2E = total_length / (2. * self.energy)
        argument = -1.j * sp.diagonal(self.mixing.M_sq) * L_over_2E    #< collapase to diagonal
        amplitudes_mass = sp.diagflat(sp.exp(argument.astype(complex)))  #< exponentiate, then expand to full matrix
        amplitudes = self.mixing.mass_to_flavor(amplitudes_mass)    #< convert to flavor eigenstates

        amplitudes = sp.asarray(amplitudes)
        probabilities = (amplitudes * sp.conjugate(amplitudes)).real
        check_unitarity(probabilities)

        return sp.asmatrix(probabilities)

    def non_parametric_matter_probabilities(self, zenith=None, energy=None):
        """ Calculate matter probabilities using eigenvalue decomposition of (H_vacuum*L + A*rho).  """
        if (zenith is not None):
            self.set_zenith(zenith)
        if (energy is not None):
            self.set_energy(energy)

        total_length = sp.sum(self.path_lengths) #< 1/eV
        total_density = sp.sum(self.path_densities) #< g eV^2
        
        amplitudes = sp.asmatrix(sp.identity(2*self.mixing.params.num_neutrinos, complex))
        argument = -1.j * (self.H_vacuum*total_length + self.mixing.A_matter*total_density)

        warnings.simplefilter("ignore")
        #! scipy 1.13 says expm2 is deprecated... ignore that and carry on using it.
        #   expm is ~3x slower than expm2, but is not deprecated
        amplitudes = sp.linalg.expm2(argument.astype(complex))
        warnings.simplefilter("default")

        # Probabilities are calculated by taking the elementwise complex modulus
        amplitudes = sp.asarray(amplitudes)
        probabilities = (amplitudes * sp.conjugate(amplitudes)).real

        check_unitarity(probabilities)
        return sp.asmatrix(probabilities)


    def matter_probabilities(self, zenith=None, energy=None, quick_and_dirty=False):
        """ Calculate matter probabilities using eigenvalue decomposition of (H_vacuum*L + A*rho).  """
        if (zenith is not None):
            self.set_zenith(zenith)
        if (energy is not None):
            self.set_energy(energy)

        if quick_and_dirty:
            L_in_km = sp.sum(self.path_lengths) / self.mixing.km_eV
            E_in_GeV = self.energy / self.mixing.eV_per_GeV
            matter_limit = 250. #< km/GeV
            if ((L_in_km / E_in_GeV) <= matter_limit):
                return self.vacuum_probabilities()
        
        earth_amplitudes = self.__get_earth_amplitudes__()
        nue_probs = self.__get_atm_probs__(self.atm_nue_lengths, earth_amplitudes)
        numu_probs = self.__get_atm_probs__(self.atm_numu_lengths, earth_amplitudes)
        nutau_probs = self.__get_atm_probs__(self.atm_nutau_lengths, earth_amplitudes)

        num_neutrinos = self.mixing.params.num_neutrinos
        nue  = 0;   nuebar  = nue + num_neutrinos;
        numu = 1;   numubar = numu + num_neutrinos;
        probabilities = nutau_probs
        probabilities[:,    nue] =  nue_probs[:,    nue]
        probabilities[:,   numu] = numu_probs[:,   numu]
        probabilities[:, nuebar] =  nue_probs[:, nuebar]
        probabilities[:,numubar] = numu_probs[:,numubar]

        return probabilities

    def __get_atm_probs__(self, lengths, earth_amplitudes):
        dim = 2*self.mixing.params.num_neutrinos
        probabilities = sp.zeros((dim, dim), float)
        for L_k in lengths:
            L_over_2E = L_k / (2. * self.energy)
            argument = -1.j * sp.diagonal(self.mixing.M_sq) * L_over_2E #< collapase to diagonal
            amplitudes_mass = sp.diagflat(sp.exp(argument.astype(complex))) #< exponentiate, then expand to full matrix
            atm_amplitudes = self.mixing.mass_to_flavor(amplitudes_mass) #< convert to flavor eigenstates

            # Probabilities are calculated by taking the elementwise complex modulus
            full_amplitudes = sp.asarray(atm_amplitudes * earth_amplitudes)
            prob_k = (full_amplitudes * sp.conjugate(full_amplitudes)).real

            # Take the average probability over all the lengths
            probabilities += (1. / self.atmosphere.num_sample_points) * prob_k

        check_unitarity(probabilities)
        return probabilities

    def __get_earth_amplitudes__(self):
        amplitudes = sp.asmatrix(sp.identity(2*self.mixing.params.num_neutrinos, complex))
        warnings.simplefilter("ignore")
        # Compute the amplitude in each layer where the density is integrable
        for (L_j, rho_j) in zip(self.path_lengths, self.path_densities):
            argument = -1.j * (self.H_vacuum*L_j + self.mixing.A_matter*rho_j)

            #! scipy 1.13 says expm2 is deprecated... ignore that and carry on using it.
            #   expm is ~3x slower than expm2, but is not deprecated
            amp_j = sp.linalg.expm2(argument.astype(complex))

            # Matrix multiplication satisfies the boundary conditions between layers
            amp_j = sp.asmatrix(amp_j)
            amplitudes = amplitudes * amp_j

        warnings.simplefilter("default")

        return amplitudes


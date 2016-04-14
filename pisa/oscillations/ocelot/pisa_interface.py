import h5py
import numpy as np

def pisa_from_file(filename):
    """
    Read in an h5py file containing probabilities in ndarrays and return them
    in a dictionary that PISA can understand.

    """

    fh = h5py.File(filename, "r")
    cos_zenith_bins = np.array(fh['cos_zenith_bins'])
    energy_bins = np.array(fh['energy_bins'])
    probs = np.array(fh['probabilities'])
    fh.close()

    pisa_dict = pisa_interface(cos_zenith_bins, energy_bins, probs)
    return pisa_dict

def pisa_interface(cos_zenith_bins, energy_bins, raw_probs, params=None):
    # allows faster dict creation
    probs = np.transpose(raw_probs)

    # List the array indicies
    nue = 0; numu = 1; nutau = 2; nuebar = 3; numubar = 4; nutaubar = 5

    pisa = {}
    pisa['czbins'] = cos_zenith_bins
    pisa['ebins'] = energy_bins
    pisa['nue_maps']     = {'nue':probs[nue, nue], 'numu':probs[nue, numu], 'nutau':probs[nue, nutau]}
    pisa['numu_maps']    = {'nue':probs[numu, nue], 'numu':probs[numu, numu], 'nutau':probs[numu, nutau]}
    pisa['nue_bar_maps'] = {'nue_bar':probs[nuebar, nuebar], 'numu_bar':probs[nuebar, numubar], 'nutau_bar':probs[nuebar,nutaubar]}
    pisa['numu_bar_maps']= {'nue_bar':probs[numubar, nuebar], 'numu_bar':probs[numubar, numubar], 'nutau_bar':probs[numubar,nutaubar]}
    if params is not None:
        pisa['params'] = {'deltam21':params.delta_M21_sq,
                          'deltam31':params.delta_M31_sq,
                          'theta12':params.theta_12,
                          'theta23':params.theta_23,
                          'theta13':params.theta_13,
                          'deltacp':params.delta_cp}

    
    return pisa

def pisa_to_file(pisa_dict, filename):
    pisa = h5py.File(filename, "w")
    for (key, value) in pisa_dict.iteritems():
        if ("map" in key) or (key == "params"):
            for (inner_key, inner_value) in value.iteritems():
                pisa.create_dataset(key+'/'+inner_key, data=inner_value)
        else:
            pisa.create_dataset(key, data=value)
    pisa.close()


#! /usr/bin/env python
#
# TemplateMaker_MC_functions.py
#
# Functions used for TemplateMaker_MC.py

import numpy as np

def apply_ratio_scale(flux1, flux2, ratio_scale):
    # keep sum of flux1, flux2 constant
    #orig_sum = flux1 + flux2
    #orig_ratio = flux1/flux2
    #scaled_flux2 = orig_sum / (1 + ratio_scale*orig_ratio)
    #scaled_flux1 = ratio_scale*orig_ratio*scaled_flux2
    #return scaled_flux1, scaled_flux2

    # don't keep sum of flux1, flux2 constant
    orig_ratio = flux1/flux2
    scaled_flux1 = ratio_scale*orig_ratio*flux2
    return scaled_flux1, flux2

def apply_reco_sys(true_energy, true_coszen, reco_energy, reco_coszen, e_reco_precision_up, e_reco_precision_down, cz_reco_precision_up, cz_reco_precision_down):
    print "Apply reco precisions..."
    if e_reco_precision_up != 1:
        delta = reco_energy[true_coszen<=0] - true_energy[true_coszen<=0]
        change = delta/true_energy[true_coszen<=0]
        print 'more than 100 %% delta for %s %% of the events '%(np.count_nonzero(change[change>1.])/float(len(change))*100)
        delta *= e_reco_precision_up
        reco_energy[true_coszen<=0] = true_energy[true_coszen<=0] + delta

    if e_reco_precision_down != 1:
        reco_energy[true_coszen>0] *= e_reco_precision_down
        reco_energy[true_coszen>0] -= (e_reco_precision_down - 1) * true_energy[true_coszen>0]

    if cz_reco_precision_up != 1:
        reco_coszen[true_coszen<=0] *= cz_reco_precision_up
        reco_coszen[true_coszen<=0] -= (cz_reco_precision_up - 1) * true_coszen[true_coszen<=0]

    if cz_reco_precision_down != 1:
        reco_coszen[true_coszen>0] *= cz_reco_precision_down
        reco_coszen[true_coszen>0] -= (cz_reco_precision_down - 1) * true_coszen[true_coszen>0]

    while np.any(reco_coszen<-1) or np.any(reco_coszen>1):
        reco_coszen[reco_coszen>1] = 2-reco_coszen[reco_coszen>1]
        reco_coszen[reco_coszen<-1] = -2-reco_coszen[reco_coszen<-1]
    return reco_energy, reco_coszen

def get_osc_probs(evts, params, osc_service, use_cut_on_trueE, ebins):
    osc_probs = {}
    for prim in ['nue', 'nue_bar', 'numu', 'numu_bar', 'nutau', 'nutau_bar']:
        osc_probs[prim]= {}
        for int_type in ['cc', 'nc']:
            osc_probs[prim][int_type] = {}
            for nu in ['nue','numu','nue_bar','numu_bar']:
                osc_probs[prim][int_type] = {'nue_maps': [], 'numu_maps' :[]}
    for prim in ['nue', 'nue_bar', 'numu', 'numu_bar', 'nutau', 'nutau_bar']:
        for int_type in ['cc', 'nc']:
            true_e = evts[prim][int_type]['true_energy']
            true_cz = evts[prim][int_type]['true_coszen']
            if use_cut_on_trueE:
                cut = np.logical_and(true_e < ebins[-1], true_e>= ebins[0])
                true_e = true_e[cut]
                true_cz = true_cz[cut]
            osc_probs[prim][int_type] = osc_service.fill_osc_prob(true_e, true_cz, prim, event_by_event=True, **params)
    return osc_probs

def apply_flux_sys(nue_flux, numu_flux, oppo_nue_flux, oppo_numu_flux, true_e, params):
    # nue_numu_ratio
    if params['nue_numu_ratio'] != 1:
        scaled_nue_flux, scaled_numu_flux = apply_ratio_scale(nue_flux, numu_flux, params['nue_numu_ratio'])
        nue_flux = scaled_nue_flux
        numu_flux = scaled_numu_flux

    # nu_nubar_ratio
    if params['nu_nubar_ratio'] != 1:
        if 'bar' not in prim:
            scaled_nue_flux,_ = apply_ratio_scale(nue_flux, oppo_nue_flux, params['nu_nubar_ratio'])
            scaled_numu_flux,_ = apply_ratio_scale(numu_flux, oppo_numu_flux, params['nu_nubar_ratio'])
        else:
            #nue(mu)_flux is actually nue(mu)_bar because prim has '_bar' in it
            _, scaled_nue_flux = apply_ratio_scale(oppo_nue_flux, nue_flux, params['nu_nubar_ratio'])
            _, scaled_numu_flux = apply_ratio_scale(oppo_numu_flux, numu_flux, params['nu_nubar_ratio'])
        nue_flux = scaled_nue_flux
        numu_flux = scaled_numu_flux

    #numu delta spectral index 
    if params['atm_delta_index'] != 1:
        delta_index = params['atm_delta_index']
        egy_med = np.median(true_e) 
        scale = np.power((true_e/egy_med),delta_index)

        total_nue_flux = nue_flux.sum()
        scaled_nue_flux = nue_flux*scale
        scaled_nue_flux *= (total_nue_flux/scaled_nue_flux.sum())
        nue_flux = scaled_nue_flux

        total_numu_flux = numu_flux.sum()
        scaled_numu_flux = numu_flux*scale
        scaled_numu_flux *= (total_numu_flux/scaled_numu_flux.sum())
        numu_flux = scaled_numu_flux
    return nue_flux, numu_flux

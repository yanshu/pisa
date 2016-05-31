#! /usr/bin/env python
#
# TemplateMaker_MC_functions.py
#
# Functions used for TemplateMaker_MC.py

import numpy as np
from pisa.utils.shape_mc import SplineService
from pisa.utils.params import construct_genie_dict
from pisa.utils.params import construct_shape_dict
from pisa.utils.log import physics, profile, set_verbosity, logging
from pisa.utils.utils import Timer
#import systematicFunctions as sf

def apply_ratio_scale(flux1, flux2, ratio_scale, sum_const):
    if sum_const:
        # keep sum of flux1, flux2 constant
        orig_sum = flux1 + flux2
        orig_ratio = flux1/flux2
        scaled_flux2 = orig_sum / (1 + ratio_scale*orig_ratio)
        scaled_flux1 = ratio_scale*orig_ratio*scaled_flux2
        return scaled_flux1, scaled_flux2
    else:
        # don't keep sum of flux1, flux2 constant
        orig_ratio = flux1/flux2
        scaled_flux1 = ratio_scale*orig_ratio*flux2
        return scaled_flux1, flux2

def apply_reco_sys(true_energy, true_coszen, reco_energy, reco_coszen, e_reco_precision_up, e_reco_precision_down, cz_reco_precision_up, cz_reco_precision_down):
    if e_reco_precision_up != 1:
        delta = reco_energy[true_coszen<=0] - true_energy[true_coszen<=0]
        change = delta/true_energy[true_coszen<=0]
        #print 'more than 100 %% delta for %s %% of the events '%(np.count_nonzero(change[change>1.])/float(len(change))*100)
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

def get_osc_probs(evts, params, osc_service, use_cut_on_trueE, ebins, turn_off_osc_NC=False):
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
            if turn_off_osc_NC and int_type=='nc':
                print "NO OSC FOR NC"
                if prim in ['nutau', 'nutau_bar']:
                    osc_probs[prim][int_type]['nue_maps'] = np.zeros(np.shape(osc_probs[prim][int_type]['nue_maps']))
                    osc_probs[prim][int_type]['numu_maps'] = np.zeros(np.shape(osc_probs[prim][int_type]['numu_maps']))
                elif prim in ['nue', 'nue_bar']:
                    osc_probs[prim][int_type]['nue_maps'] = np.ones(np.shape(osc_probs[prim][int_type]['nue_maps']))
                    osc_probs[prim][int_type]['numu_maps'] = np.zeros(np.shape(osc_probs[prim][int_type]['numu_maps']))
                else:
                    osc_probs[prim][int_type]['nue_maps'] = np.zeros(np.shape(osc_probs[prim][int_type]['nue_maps']))
                    osc_probs[prim][int_type]['numu_maps'] = np.ones(np.shape(osc_probs[prim][int_type]['numu_maps']))
    return osc_probs

def apply_flux_ratio(prim, nue_flux, numu_flux, oppo_nue_flux, oppo_numu_flux, true_e, params, flux_sys_renorm):
    # nue_numu_ratio
    if params['nue_numu_ratio'] != 1:
        scaled_nue_flux, scaled_numu_flux = apply_ratio_scale(nue_flux, numu_flux, params['nue_numu_ratio'], sum_const=flux_sys_renorm)
        nue_flux = scaled_nue_flux
        numu_flux = scaled_numu_flux

    # nu_nubar_ratio
    if params['nu_nubar_ratio'] != 1:
        if 'bar' not in prim:
            scaled_nue_flux,_ = apply_ratio_scale(nue_flux, oppo_nue_flux, params['nu_nubar_ratio'], sum_const=flux_sys_renorm)
            scaled_numu_flux,_ = apply_ratio_scale(numu_flux, oppo_numu_flux, params['nu_nubar_ratio'], sum_const=flux_sys_renorm)
        else:
            #nue(mu)_flux is actually nue(mu)_bar because prim has '_bar' in it
            _, scaled_nue_flux = apply_ratio_scale(oppo_nue_flux, nue_flux, params['nu_nubar_ratio'], sum_const=flux_sys_renorm)
            _, scaled_numu_flux = apply_ratio_scale(oppo_numu_flux, numu_flux, params['nu_nubar_ratio'], sum_const=flux_sys_renorm)
        nue_flux = scaled_nue_flux
        numu_flux = scaled_numu_flux

    #if params['Barr_nu_nubar_ratio']!=0:
    #    scale = sf.modRatioNuEBar(nue_cc, params['Barr_nu_nubar_ratio'], params['Barr_nu_nubar_ratio'])*\
    #            sf.modRatioUpHor_NuE(nue_cc, params['Barr_uphor_ratio'])
    return nue_flux, numu_flux

def apply_spectral_index(nue_flux, numu_flux, true_e, egy_pivot, aeff_weights, params, flux_sys_renorm):
    if params['atm_delta_index'] != 0:
        delta_index = params['atm_delta_index']
        egy_med = np.median(true_e) 
        egy_mean = np.mean(true_e) 
        #egy_pivot = egy_med
        #egy_pivot = egy_mean
        scale = np.power((true_e/egy_pivot),delta_index)
        if flux_sys_renorm:
            # keep weighted flux constant
            weighted_nue_flux = nue_flux * aeff_weights
            total_nue_flux = weighted_nue_flux.sum() 
            scaled_nue_flux = nue_flux*scale
            weighted_scaled_nue_flux = scaled_nue_flux * aeff_weights
            total_scaled_nue_flux = weighted_scaled_nue_flux.sum()
            scaled_nue_flux *= (total_nue_flux/total_scaled_nue_flux)
            nue_flux = scaled_nue_flux

            weighted_numu_flux = numu_flux * aeff_weights
            total_numu_flux = weighted_numu_flux.sum() 
            scaled_numu_flux = numu_flux*scale
            weighted_scaled_numu_flux = scaled_numu_flux * aeff_weights
            total_scaled_numu_flux = weighted_scaled_numu_flux.sum()
            scaled_numu_flux *= (total_numu_flux/total_scaled_numu_flux)
            numu_flux = scaled_numu_flux
        else:
            # do not keep sum constant
            scaled_nue_flux = nue_flux*scale
            nue_flux = scaled_nue_flux
            scaled_numu_flux = numu_flux*scale
            numu_flux = scaled_numu_flux
    return nue_flux, numu_flux

def apply_Barr_mod(prim, int_type, ebins, nue_flux, numu_flux, true_e, true_cz, barr_splines, **params):
    '''
    Taking Joakim's shape mod functionality and applying it generally
    to the flux_maps, regardless of the flux_service used
    '''

    #make flux_mod_dict and add it to the list of params.
    Flux_Mod_Dict = construct_shape_dict('flux', params)

    if np.all([Flux_Mod_Dict[key] == 0 for key in Flux_Mod_Dict.keys()]):
        return nue_flux, numu_flux 

    ### make spline service for Barr parameters ###
    flux_spline_service = SplineService(ebins=ebins, evals=true_e, dictFile = params['flux_uncertainty_inputs'])

    ### FORM A TABLE FROM THE UNCERTAINTY WEIGHTS AND THE SPLINED MAPS CORRESPONDING TO THEM - WE DISCUSSED THIS SHOUD BE DONE EXPLICITLY FOR EASIER UNDERSTANDING###
    #Now apply all the shape modification for each of the flux uncertainties
    #Modellling of the uncertainties follows the discussion
    #in Barr et al. (2006)
    mod_table = np.zeros(len(true_e))
    #print "for testing purposes, here is flux_hadronic_H: ", flux_hadronic_H
    # here I want a dictionary named Flux_Mod_Dict containing the mod factors as keys and UNCF_X files as entries, then I can modify the flux by:
    logging.info("now reaching the flux_mod_dict stage: \n %s"%Flux_Mod_Dict)
    with Timer(verbose=False) as t:
        for entry in Flux_Mod_Dict:
            if entry == 'fixed': continue
            logging.info("testing for: %s" %entry)
            if Flux_Mod_Dict[entry] != 0.0:
                mod_table += flux_spline_service.modify_shape(true_e, true_cz, Flux_Mod_Dict[entry], entry, event_by_event=True, pre_saved_splines=barr_splines)
    print("==> time barr_spline_service.modify_shape : %s sec"%t.secs)

    if mod_table[mod_table<0].any():
        #remember: mod_table contains the 1 sigma modification of the flux squared and multiplied by the modification factor -
        # - this can of course be negative, but is not unphysical. It just represents the minus part of a +/- uncertainty. As such we should still assign it as an uncertainty, and just need to take the sqrt of the absolute value
        # - and then remember that this is to be applied in the negative direction, if it was negative.
        mod_table = -1 * np.sqrt(-1 * mod_table)
    else:
        mod_table = np.sqrt(mod_table) # just take the sqrt of the + side of the +/- uncertainty.

    modified_nue_flux = nue_flux * (1 + mod_table) #this is where the actual modification happens
    modified_numu_flux = numu_flux * (1 + mod_table) #this is where the actual modification happens
    return modified_nue_flux, modified_numu_flux


def apply_GENIE_mod(prim, int_type, ebins, true_e, true_cz, aeff_weights, gensys_splines, **params):

    # code modified from Ste's apply_shape_mod() in Aeff.py

    print "apply_GENIE_mod for ", prim , " ", int_type
    ### make dict of genie parameters ###
    GENSYS = construct_genie_dict(params)

    if np.all([GENSYS[key] == 0 for key in GENSYS.keys()]):
        return aeff_weights

    ### make spline service for genie parameters ###
    with Timer(verbose=False) as t:
        genie_spline_service = SplineService(ebins=ebins, evals=true_e, dictFile = params['GENSYS_files'])
    print("==> time initialize SplineService : %s sec"%t.secs)

    mod_table = np.zeros(len(true_e))

    # FILL THE SHAPE MODIFICATION TABLES
    with Timer(verbose=False) as t:
        for entry in GENSYS:
            if GENSYS[entry] != 0.0:
                #print "we are now passing onto modify shape: ", GENSYS[entry]
                if entry == "MaCCQE" and int_type=='nc': continue
                mod_table += genie_spline_service.modify_shape(true_e, true_cz, GENSYS[entry], str(entry)+"_"+str(prim), event_by_event=True, pre_saved_splines=gensys_splines)
    print("==> time genie_spline_service.modify_shape : %s sec"%t.secs)

    ### THIS FOLLOWING SECTION HAS DELIBERATE BEEN MADE THIS COMPLICATED - THE ASIMOV METHOD BREAKS THINGS OTHERWISE (ask me about it if interested) ###
    with Timer(verbose=False) as t:
        if mod_table[mod_table<0].any():
            for i in range(len(mod_table)):
                if mod_table[i] < 0.0:
                    mod_table[i] = -1.0 * (np.sqrt(-1.0 * mod_table[i]))
                else:
                    mod_table[i] = np.sqrt(mod_table[i])
        else:
            mod_table = np.sqrt(mod_table)
        
        modified_aeff_weights = aeff_weights * ( mod_table + 1.0)
        
        #TEST FOR 0 AND OR NEGATIVE VALEUS #
        if modified_aeff_weights[modified_aeff_weights == 0.0].any():
            raise ValueError("Modified aeff_weights must have all bins > 0")
    print("==> time for the rest process : %s sec"%t.secs)
    return modified_aeff_weights


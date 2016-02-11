#
# Maps_i3.py
#
# Functions to get maps from MC directly. 
#
# author: Feifei Huang <fxh140@psu.edu>
# date:   2016-02-03
#

import os
import numpy as np
import h5py
from scipy.constants import Julian_year
from pisa.analysis.stats.LLHStatistics import get_random_map
from pisa.utils.log import logging
from pisa.utils.jsons import from_json,to_json
from pisa.resources.resources import find_resource
import pisa.analysis.stats.Maps as Maps

def get_i3_maps(output_form, cut_level, year, anlys_ebins, czbins):
    anlys_bins = (anlys_ebins, czbins)
    num_nue_files = 2700
    num_numu_files = 4000
    num_nutau_files = 1400
    livetime_in_s = Julian_year
    #livetime_in_s = 27920000  # (DC12: 1 livetime year = 27920000 s)

    # read MC hdf5 files directly
    #MC_file_nue = h5py.File(find_resource('/Users/feifeihuang/Desktop/Matt_data/DOMEff_HoleIce/DC12_nue.hd5','r'))
    #MC_file_nutau = h5py.File(find_resource('/Users/feifeihuang/Desktop/Matt_data/DOMEff_HoleIce/DC12_nutau.hd5','r'))
    #MC_file_numu = h5py.File(find_resource('/Users/feifeihuang/Desktop/Matt_data/DOMEff_HoleIce/DC12_numu.hd5','r'))
    MC_file_nue = h5py.File(find_resource('/Users/feifeihuang/Desktop/Matt_data/Matt_L5b_mc_with_weights/Matt_L5b_mc_with_weights_nue.hdf5','r'))
    MC_file_nutau = h5py.File(find_resource('/Users/feifeihuang/Desktop/Matt_data/Matt_L5b_mc_with_weights/Matt_L5b_mc_with_weights_nutau.hdf5','r'))
    MC_file_numu = h5py.File(find_resource('/Users/feifeihuang/Desktop/Matt_data/Matt_L5b_mc_with_weights/Matt_L5b_mc_with_weights_numu.hdf5','r'))

    L6_result = {}
    L6_result['nue'] = MC_file_nue['IC86_Dunkman_L6']['result']
    L6_result['numu'] = MC_file_numu['IC86_Dunkman_L6']['result']
    L6_result['nutau'] = MC_file_nutau['IC86_Dunkman_L6']['result']

    ExpectedNumber = {}
    ExpectedNumber['nue'] = year * MC_file_nue['NeutrinoWeights_nufit']['OscillatedRate']*livetime_in_s/(num_nue_files/2)
    ExpectedNumber['numu'] = year * MC_file_numu['NeutrinoWeights_nufit']['OscillatedRate']*livetime_in_s/(num_numu_files/2)
    ExpectedNumber['nutau'] = year * MC_file_nutau['NeutrinoWeights_nufit']['OscillatedRate']*livetime_in_s/(num_nutau_files/2)

    MC_true_x = {}
    MC_true_x['nue'] = MC_file_nue['trueNeutrino']['x']
    MC_true_x['numu'] = MC_file_numu['trueNeutrino']['x']
    MC_true_x['nutau'] = MC_file_nutau['trueNeutrino']['x']

    MC_true_y = {}
    MC_true_y['nue'] = MC_file_nue['trueNeutrino']['y']
    MC_true_y['numu'] = MC_file_numu['trueNeutrino']['y']
    MC_true_y['nutau'] = MC_file_nutau['trueNeutrino']['y']

    MC_true_z = {}
    MC_true_z['nue'] = MC_file_nue['trueNeutrino']['z']
    MC_true_z['numu'] = MC_file_numu['trueNeutrino']['z']
    MC_true_z['nutau'] = MC_file_nutau['trueNeutrino']['z']

    MC_true_t = {}
    MC_true_t['nue'] = MC_file_nue['trueNeutrino']['time']
    MC_true_t['numu'] = MC_file_numu['trueNeutrino']['time']
    MC_true_t['nutau'] = MC_file_nutau['trueNeutrino']['time']

    MC_true_energy = {}
    MC_true_energy['nue'] = MC_file_nue['trueNeutrino']['energy']
    MC_true_energy['numu'] = MC_file_numu['trueNeutrino']['energy']
    MC_true_energy['nutau'] = MC_file_nutau['trueNeutrino']['energy']

    MC_true_coszen = {}
    MC_true_coszen['nue'] = np.cos(MC_file_nue['trueNeutrino']['zenith'])
    MC_true_coszen['numu'] = np.cos(MC_file_numu['trueNeutrino']['zenith'])
    MC_true_coszen['nutau'] = np.cos(MC_file_nutau['trueNeutrino']['zenith'])

    MN_reco_x = {}
    MN_reco_x['nue'] = MC_file_nue['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['x']
    MN_reco_x['numu'] = MC_file_numu['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['x']
    MN_reco_x['nutau'] = MC_file_nutau['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['x']

    MN_reco_y = {}
    MN_reco_y['nue'] = MC_file_nue['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['y']
    MN_reco_y['numu'] = MC_file_numu['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['y']
    MN_reco_y['nutau'] = MC_file_nutau['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['y']

    MN_reco_z = {}
    MN_reco_z['nue'] = MC_file_nue['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['z']
    MN_reco_z['numu'] = MC_file_numu['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['z']
    MN_reco_z['nutau'] = MC_file_nutau['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['z']

    MN_reco_t = {}
    MN_reco_t['nue'] = MC_file_nue['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['time']
    MN_reco_t['numu'] = MC_file_numu['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['time']
    MN_reco_t['nutau'] = MC_file_nutau['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['time']

    MN_reco_energy = {}
    MN_reco_energy['nue'] = MC_file_nue['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['energy']
    MN_reco_energy['numu'] = MC_file_numu['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['energy']
    MN_reco_energy['nutau'] = MC_file_nutau['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['energy']

    MN_reco_coszen = {}
    MN_reco_coszen['nue'] = np.cos(MC_file_nue['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['zenith'])
    MN_reco_coszen['numu'] = np.cos(MC_file_numu['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['zenith'])
    MN_reco_coszen['nutau'] = np.cos(MC_file_nutau['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['zenith'])

    MN_reco_trck_len = {}
    MN_reco_trck_len['nue'] = MC_file_nue['IC86_Dunkman_L6_MultiNest8D_PDG_Track']['length']
    MN_reco_trck_len['numu'] = MC_file_numu['IC86_Dunkman_L6_MultiNest8D_PDG_Track']['length']
    MN_reco_trck_len['nutau'] = MC_file_nutau['IC86_Dunkman_L6_MultiNest8D_PDG_Track']['length']

    InteractionType = {}
    InteractionType['nue'] = MC_file_nue['I3MCWeightDict']['InteractionType']
    InteractionType['numu'] = MC_file_numu['I3MCWeightDict']['InteractionType']
    InteractionType['nutau'] = MC_file_nutau['I3MCWeightDict']['InteractionType']

    TrueNeutrino_pdg = {}
    TrueNeutrino_pdg['nue'] = MC_file_nue['trueNeutrino']['pdg_encoding']
    TrueNeutrino_pdg['numu'] = MC_file_numu['trueNeutrino']['pdg_encoding']
    TrueNeutrino_pdg['nutau'] = MC_file_nutau['trueNeutrino']['pdg_encoding']

    deltaLLH = {}
    deltaLLH['nue'] = MC_file_nue['IC86_Dunkman_L6']['delta_LLH']
    deltaLLH['numu'] = MC_file_numu['IC86_Dunkman_L6']['delta_LLH']
    deltaLLH['nutau'] = MC_file_nutau['IC86_Dunkman_L6']['delta_LLH']

    print "len MC_true_energy['nue'] ", len(MC_true_energy['nue'] )
    print "len MC_true_energy['numu'] ", len(MC_true_energy['numu'] )
    print "len MC_true_energy['nutau'] ", len(MC_true_energy['nutau'] )
    print "totl = ", len(MC_true_energy['nue'] )+ len(MC_true_energy['numu'])+ len(MC_true_energy['nutau'])
    x = len(MC_true_energy['nue'][L6_result['nue'] ==1] )+ len(MC_true_energy['numu'][L6_result['numu'] ==1])+ len(MC_true_energy['nutau'][L6_result['nutau'] ==1])
    print "L6 totl = ", x

    nuDict = {'nue':12,'numu':14,'nutau':16,'nue_bar':-12,'numu_bar':-14,'nutau_bar':-16}
    inttypeDict = {'cc':1, 'nc':2}

    # get aeff maps from i3 files
    cut_aeff = {}
    for flavor in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
        cut_aeff[flavor] = {}
        for int_type in ['cc','nc']:
            if cut_level == 'L6':
                cut_aeff[flavor][int_type] = np.logical_and(L6_result[flavor.split('_bar')[0]] ==1 , np.logical_and(TrueNeutrino_pdg[flavor.split('_bar')[0]] == nuDict[flavor], InteractionType[flavor.split('_bar')[0]] == inttypeDict[int_type]))
            elif cut_level == 'L5':
                cut_aeff[flavor][int_type] = np.logical_and(TrueNeutrino_pdg[flavor.split('_bar')[0]] == nuDict[flavor], InteractionType[flavor.split('_bar')[0]] == inttypeDict[int_type])
            else:
                #TODO
                print "cut level above L5 is not available"

    aeff_maps_from_i3 = {}
    true_coszen_from_i3 = {}
    true_energy_from_i3 = {}
    true_xyzt_from_i3 = {}
    reco_coszen_from_i3 = {}
    reco_energy_from_i3 = {}
    reco_xyzt_from_i3 = {}
    trck_len_from_i3 = {}
    weights = {}
    for flavor in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
        aeff_maps_from_i3[flavor] = {}
        true_coszen_from_i3[flavor] = {}
        true_energy_from_i3[flavor] = {}
        true_xyzt_from_i3[flavor] = {}
        reco_coszen_from_i3[flavor] = {}
        reco_energy_from_i3[flavor] = {}
        reco_xyzt_from_i3[flavor] = {}
        trck_len_from_i3[flavor] = {}
        weights[flavor] = {}
        for int_type in ['cc','nc']:
            oscillated_rate = ExpectedNumber[flavor.split('_bar')[0]][cut_aeff[flavor][int_type]]
            true_energy = MC_true_energy[flavor.split('_bar')[0]][cut_aeff[flavor][int_type]] 
            true_coszen = MC_true_coszen[flavor.split('_bar')[0]][cut_aeff[flavor][int_type]]
            reco_energy = MN_reco_energy[flavor.split('_bar')[0]][cut_aeff[flavor][int_type]] 
            reco_coszen = MN_reco_coszen[flavor.split('_bar')[0]][cut_aeff[flavor][int_type]]

            weights[flavor][int_type] = oscillated_rate
            true_coszen_from_i3[flavor][int_type] = true_coszen
            true_energy_from_i3[flavor][int_type] = true_energy
            reco_coszen_from_i3[flavor][int_type] = reco_coszen
            reco_energy_from_i3[flavor][int_type] = reco_energy

            aeff_hist,_,_ = np.histogram2d(true_energy,true_coszen, weights=oscillated_rate,bins=anlys_bins)
            aeff_maps_from_i3[flavor][int_type] = {'map':aeff_hist,
                                                   'ebins':anlys_ebins,
                                                   'czbins':czbins}

            if output_form == 'reco_info':
                reco_x = MN_reco_x[flavor.split('_bar')[0]][cut_aeff[flavor][int_type]] 
                reco_y = MN_reco_y[flavor.split('_bar')[0]][cut_aeff[flavor][int_type]] 
                reco_z = MN_reco_z[flavor.split('_bar')[0]][cut_aeff[flavor][int_type]] 
                reco_t = MN_reco_t[flavor.split('_bar')[0]][cut_aeff[flavor][int_type]] 
                reco_trck_len = MN_reco_trck_len[flavor.split('_bar')[0]][cut_aeff[flavor][int_type]]
                reco_xyzt_from_i3[flavor][int_type] = {}
                reco_xyzt_from_i3[flavor][int_type]['x'] = reco_x 
                reco_xyzt_from_i3[flavor][int_type]['y'] = reco_y
                reco_xyzt_from_i3[flavor][int_type]['z'] = reco_z  
                reco_xyzt_from_i3[flavor][int_type]['time'] = reco_t 
                trck_len_from_i3[flavor][int_type] = reco_trck_len


            if output_form == 'true_info':
                true_x = MC_true_x[flavor.split('_bar')[0]][cut_aeff[flavor][int_type]] 
                true_y = MC_true_y[flavor.split('_bar')[0]][cut_aeff[flavor][int_type]] 
                true_z = MC_true_z[flavor.split('_bar')[0]][cut_aeff[flavor][int_type]] 
                true_t = MC_true_t[flavor.split('_bar')[0]][cut_aeff[flavor][int_type]] 
                true_xyzt_from_i3[flavor][int_type] = {}
                true_xyzt_from_i3[flavor][int_type]['x'] = true_x 
                true_xyzt_from_i3[flavor][int_type]['y'] = true_y
                true_xyzt_from_i3[flavor][int_type]['z'] = true_z  
                true_xyzt_from_i3[flavor][int_type]['time'] = true_t 

    if output_form == 'true_info':
        return (true_xyzt_from_i3, true_energy_from_i3, true_coszen_from_i3, weights)

    if output_form == 'reco_info':
        return (reco_xyzt_from_i3, reco_energy_from_i3, reco_coszen_from_i3, trck_len_from_i3, weights)

    if output_form == 'aeff_map':
        return aeff_maps_from_i3

    if output_form == 'final_map':
        final_maps_from_i3 = {}
        cut_pid = {}
        for flavor in ['nue', 'numu', 'nutau']:
            cut_pid[flavor]={}
            cut_pid[flavor]['trck'] = np.logical_and(L6_result[flavor] ==1 , deltaLLH[flavor]>= 3.0)
            cut_pid[flavor]['cscd'] = np.logical_and(L6_result[flavor] ==1 , deltaLLH[flavor]< 3.0)

        for channel in ['cscd','trck']:
            reco_energy_pid = np.array([])
            reco_coszen_pid = np.array([]) 
            oscillated_rate_pid = np.array([]) 
            for flavor in ['nue', 'numu', 'nutau']:
                oscillated_rate = ExpectedNumber[flavor][cut_pid[flavor][channel]]/2
                print "len(oscillated_rate) = ", len(oscillated_rate)
                reco_energy = MN_reco_energy[flavor][cut_pid[flavor][channel]] 
                reco_coszen = MN_reco_coszen[flavor][cut_pid[flavor][channel]]
                oscillated_rate_pid = np.concatenate([oscillated_rate_pid, oscillated_rate])
                reco_energy_pid = np.concatenate([reco_energy_pid, reco_energy]) 
                reco_coszen_pid = np.concatenate([reco_coszen_pid, reco_coszen])

            pid_hist,_,_ = np.histogram2d(reco_energy_pid,reco_coszen_pid, weights=oscillated_rate_pid,bins=anlys_bins)
            final_maps_from_i3[channel] = {'map':pid_hist,
                                           'ebins':anlys_ebins,
                                           'czbins':czbins}
        return final_maps_from_i3



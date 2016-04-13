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

def get_i3_maps(nue_file, numu_file, nutau_file, n_nue_files, n_numu_files, n_nutau_files, output_form, cut_level, year, ebins, anlys_ebins, czbins, sim_version, use_cut_on_trueE=True):
    anlys_bins = (anlys_ebins, czbins)
    livetime_in_s = Julian_year
    #livetime_in_s = 27920000  # (DC12: 1 livetime year = 27920000 s)
    if sim_version == "4digit":
        Reco_Neutrino_Name = 'IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino'
        Reco_Track_Name = 'IC86_Dunkman_L6_MultiNest8D_PDG_Track'
    elif sim_version == "5digit":
        Reco_Neutrino_Name = 'IC86_Dunkman_L6_PegLeg_MultiNest8D_NuMuCC'
        Reco_Track_Name = 'IC86_Dunkman_L6_PegLeg_MultiNest8D_Track'

    # read MC hdf5 files directly
    MC_file_nue = h5py.File(find_resource(nue_file,'r'))
    MC_file_numu = h5py.File(find_resource(numu_file,'r'))
    MC_file_nutau = h5py.File(find_resource(nutau_file,'r'))

    L6_result = {}
    L6_result['nue'] = MC_file_nue['IC86_Dunkman_L6']['result']
    L6_result['numu'] = MC_file_numu['IC86_Dunkman_L6']['result']
    L6_result['nutau'] = MC_file_nutau['IC86_Dunkman_L6']['result']

    Oscillated_ExpectedNumber = {}
    Oscillated_ExpectedNumber['nue'] = year * MC_file_nue['NeutrinoWeights_nufit']['OscillatedRate']*livetime_in_s/(n_nue_files)
    Oscillated_ExpectedNumber['numu'] = year * MC_file_numu['NeutrinoWeights_nufit']['OscillatedRate']*livetime_in_s/(n_numu_files)
    Oscillated_ExpectedNumber['nutau'] = year * MC_file_nutau['NeutrinoWeights_nufit']['OscillatedRate']*livetime_in_s/(n_nutau_files)

    #UnOscillated_ExpectedNumber = {}
    #UnOscillated_ExpectedNumber['nue'] = year * MC_file_nue['NeutrinoWeights_nufit']['UnoscillatedRate']*livetime_in_s/(n_nue_files)
    #UnOscillated_ExpectedNumber['numu'] = year * MC_file_numu['NeutrinoWeights_nufit']['UnoscillatedRate']*livetime_in_s/(n_numu_files)
    #UnOscillated_ExpectedNumber['nutau'] = year * MC_file_nutau['NeutrinoWeights_nufit']['UnoscillatedRate']*livetime_in_s/(n_nutau_files)

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
    MN_reco_x['nue'] = MC_file_nue[Reco_Neutrino_Name]['x']
    MN_reco_x['numu'] = MC_file_numu[Reco_Neutrino_Name]['x']
    MN_reco_x['nutau'] = MC_file_nutau[Reco_Neutrino_Name]['x']

    MN_reco_y = {}
    MN_reco_y['nue'] = MC_file_nue[Reco_Neutrino_Name]['y']
    MN_reco_y['numu'] = MC_file_numu[Reco_Neutrino_Name]['y']
    MN_reco_y['nutau'] = MC_file_nutau[Reco_Neutrino_Name]['y']

    MN_reco_z = {}
    MN_reco_z['nue'] = MC_file_nue[Reco_Neutrino_Name]['z']
    MN_reco_z['numu'] = MC_file_numu[Reco_Neutrino_Name]['z']
    MN_reco_z['nutau'] = MC_file_nutau[Reco_Neutrino_Name]['z']

    MN_reco_t = {}
    MN_reco_t['nue'] = MC_file_nue[Reco_Neutrino_Name]['time']
    MN_reco_t['numu'] = MC_file_numu[Reco_Neutrino_Name]['time']
    MN_reco_t['nutau'] = MC_file_nutau[Reco_Neutrino_Name]['time']

    MN_reco_energy = {}
    MN_reco_energy['nue'] = MC_file_nue[Reco_Neutrino_Name]['energy']
    MN_reco_energy['numu'] = MC_file_numu[Reco_Neutrino_Name]['energy']
    MN_reco_energy['nutau'] = MC_file_nutau[Reco_Neutrino_Name]['energy']

    MN_reco_coszen = {}
    MN_reco_coszen['nue'] = np.cos(MC_file_nue[Reco_Neutrino_Name]['zenith'])
    MN_reco_coszen['numu'] = np.cos(MC_file_numu[Reco_Neutrino_Name]['zenith'])
    MN_reco_coszen['nutau'] = np.cos(MC_file_nutau[Reco_Neutrino_Name]['zenith'])

    MN_reco_trck_len = {}
    MN_reco_trck_len['nue'] = MC_file_nue[Reco_Track_Name]['length']
    MN_reco_trck_len['numu'] = MC_file_numu[Reco_Track_Name]['length']
    MN_reco_trck_len['nutau'] = MC_file_nutau[Reco_Track_Name]['length']

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

    #print "totl = ", len(MC_true_energy['nue'] )+ len(MC_true_energy['numu'])+ len(MC_true_energy['nutau'])
    #x = len(MC_true_energy['nue'][L6_result['nue'] ==1] )+ len(MC_true_energy['numu'][L6_result['numu'] ==1])+ len(MC_true_energy['nutau'][L6_result['nutau'] ==1])
    #print "L6 totl = ", x

    nuDict = {'nue':12,'numu':14,'nutau':16,'nue_bar':-12,'numu_bar':-14,'nutau_bar':-16}
    inttypeDict = {'cc':1, 'nc':2}

    # cut to get aeff maps and osc_flux maps from i3 files
    #cut_osc_flux = {}
    #for flavor in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
    #    cut_osc_flux[flavor] = {}
    #    for int_type in ['cc','nc']:
    #        cut_osc_flux[flavor][int_type] = TrueNeutrino_pdg[flavor.split('_bar')[0]] == nuDict[flavor][int_type]
    #            cut_aeff[flavor][int_type] = np.logical_and(L6_result[flavor.split('_bar')[0]] ==1 , np.logical_and(TrueNeutrino_pdg[flavor.split('_bar')[0]] == nuDict[flavor], InteractionType[flavor.split('_bar')[0]] == inttypeDict[int_type]))

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
                raise ValueError("cut level above L5 is not available")

    #osc_weights = {}
    #osc_flux_maps_from_i3 = {}
    #for flavor in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
    #    for int_type in ['cc','nc']:
    #        oscillated_rate = Oscillated_ExpectedNumber[flavor.split('_bar')[0]][cut_osc_flux[flavor][int_type]]
    #        osc_weights[flavor] = oscillated_rate
    #        true_energy = MC_true_energy[flavor.split('_bar')[0]][cut_osc_flux[flavor][int_type]] 
    #        true_coszen = MC_true_coszen[flavor.split('_bar')[0]][cut_osc_flux[flavor][int_type]]
    #        osc_flux_hist,_,_ = np.histogram2d(true_energy,true_coszen, weights=oscillated_rate,bins=anlys_bins)
    #        osc_flux_maps_from_i3[flavor] = {'map':osc_flux_hist,
    #                                     'ebins':anlys_ebins,
    #                                     'czbins':czbins}


    aeff_maps_from_i3 = {}
    true_coszen_from_i3 = {}
    true_energy_from_i3 = {}
    true_xyzt_from_i3 = {}
    reco_coszen_from_i3 = {}
    reco_energy_from_i3 = {}
    reco_xyzt_from_i3 = {}
    trck_len_from_i3 = {}
    osc_weights = {}
    for flavor in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
        aeff_maps_from_i3[flavor] = {}
        true_coszen_from_i3[flavor] = {}
        true_energy_from_i3[flavor] = {}
        true_xyzt_from_i3[flavor] = {}
        reco_coszen_from_i3[flavor] = {}
        reco_energy_from_i3[flavor] = {}
        reco_xyzt_from_i3[flavor] = {}
        trck_len_from_i3[flavor] = {}
        osc_weights[flavor] = {}
        for int_type in ['cc','nc']:
            oscillated_rate = Oscillated_ExpectedNumber[flavor.split('_bar')[0]][cut_aeff[flavor][int_type]]
            true_energy = MC_true_energy[flavor.split('_bar')[0]][cut_aeff[flavor][int_type]] 
            true_coszen = MC_true_coszen[flavor.split('_bar')[0]][cut_aeff[flavor][int_type]]
            reco_energy = MN_reco_energy[flavor.split('_bar')[0]][cut_aeff[flavor][int_type]] 
            reco_coszen = MN_reco_coszen[flavor.split('_bar')[0]][cut_aeff[flavor][int_type]]
            cut_on_true_E = np.logical_and(true_energy >= ebins[0], true_energy <= ebins[-1])
            if use_cut_on_trueE:
                true_energy = true_energy[cut_on_true_E]
                true_coszen = true_coszen[cut_on_true_E]
                reco_energy = reco_energy[cut_on_true_E]
                reco_coszen = reco_coszen[cut_on_true_E]
                oscillated_rate = oscillated_rate[cut_on_true_E]
            osc_weights[flavor][int_type] = oscillated_rate
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

                if use_cut_on_trueE:
                    reco_x = reco_y[cut_on_true_E]
                    reco_y = reco_y[cut_on_true_E]
                    reco_z = reco_z[cut_on_true_E]
                    reco_t = reco_t[cut_on_true_E]
                    reco_trck_len = reco_t[cut_on_true_E]

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

                if use_cut_on_trueE:
                    true_x = true_y[cut_on_true_E]
                    true_y = true_y[cut_on_true_E]
                    true_z = true_z[cut_on_true_E]
                    true_t = true_t[cut_on_true_E]
                    true_trck_len = true_t[cut_on_true_E]

                true_xyzt_from_i3[flavor][int_type] = {}
                true_xyzt_from_i3[flavor][int_type]['x'] = true_x 
                true_xyzt_from_i3[flavor][int_type]['y'] = true_y
                true_xyzt_from_i3[flavor][int_type]['z'] = true_z  
                true_xyzt_from_i3[flavor][int_type]['time'] = true_t 

    MC_file_nue.close()
    MC_file_numu.close()
    MC_file_nutau.close()

    if output_form == 'true_info':
        return (true_xyzt_from_i3, true_energy_from_i3, true_coszen_from_i3, osc_weights)

    if output_form == 'reco_info':
        return (reco_xyzt_from_i3, reco_energy_from_i3, reco_coszen_from_i3, trck_len_from_i3, osc_weights)

    #if output_form == 'osc_flux_map':
    #    return osc_flux_maps_from_i3

    if output_form == 'aeff_and_final_map':
        final_maps_from_i3 = {}
        cut_pid = {}
        for flavor in ['nue', 'numu', 'nutau']:
            cut_pid[flavor]={}
            cut_pid[flavor]['trck'] = np.logical_and(L6_result[flavor] ==1 , deltaLLH[flavor]>= 3.0)

            # This is the correct way, but right now in PID stage, PISA couldn't throw away events with deltaLLH < -3.
            cut_pid[flavor]['cscd'] = np.logical_and(np.logical_and(L6_result[flavor] ==1 , deltaLLH[flavor]< 3.0), deltaLLH[flavor]>= -3.0)

        for channel in ['cscd','trck']:
            reco_energy_pid = np.array([])
            reco_coszen_pid = np.array([]) 
            oscillated_rate_pid = np.array([]) 
            for flavor in ['nue', 'numu', 'nutau']:
                oscillated_rate = Oscillated_ExpectedNumber[flavor][cut_pid[flavor][channel]]
                reco_energy = MN_reco_energy[flavor][cut_pid[flavor][channel]] 
                reco_coszen = MN_reco_coszen[flavor][cut_pid[flavor][channel]]
                true_energy = MC_true_energy[flavor][cut_pid[flavor][channel]] 
                if use_cut_on_trueE:
                    cut_on_true_E = np.logical_and(true_energy >= ebins[0], true_energy <= ebins[-1])
                    reco_energy = reco_energy[cut_on_true_E]
                    reco_coszen = reco_coszen[cut_on_true_E]
                    oscillated_rate = oscillated_rate[cut_on_true_E]

                oscillated_rate_pid = np.concatenate([oscillated_rate_pid, oscillated_rate])
                reco_energy_pid = np.concatenate([reco_energy_pid, reco_energy]) 
                reco_coszen_pid = np.concatenate([reco_coszen_pid, reco_coszen])

            pid_hist,_,_ = np.histogram2d(reco_energy_pid,reco_coszen_pid, weights=oscillated_rate_pid,bins=anlys_bins)
            final_maps_from_i3[channel] = {'map':pid_hist,
                                           'ebins':anlys_ebins,
                                           'czbins':czbins}
        return aeff_maps_from_i3, final_maps_from_i3



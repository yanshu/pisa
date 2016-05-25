#! /usr/bin/env python# #
# author: Timothy C. Arlen
#         Feifei Huang
#
# date: 20 Jan 2016
#
#   Compare maps from default PISA and event-by-event PISA.
#

import copy
import numpy as np
import os
import h5py
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy import stats
from matplotlib.offsetbox import AnchoredText
from scipy.constants import Julian_year
from pisa.analysis.TemplateMaker_nutau import TemplateMaker as TemplateMaker_default
from pisa.analysis.TemplateMaker_MC import TemplateMaker as TemplateMaker_event
import pisa.utils.utils as utils
from pisa.utils.params import get_values, select_hierarchy_and_nutau_norm, select_hierarchy
from pisa.utils.log import set_verbosity,logging,profile
from pisa.resources.resources import find_resource
from pisa.utils.utils import Timer, is_linear, is_logarithmic, get_bin_centers
from pisa.utils.jsons import from_json
from pisa.utils.plot import show_map_swap, sum_map, ratio_map, delta_map, plot_one_map
from pisa.analysis.stats.Maps_i3 import get_i3_maps
from pisa.background.BackgroundServiceICC import BackgroundServiceICC
import pandas as pd

def get_1D_projection(map_2d, axis):
    if axis == 'coszen':
        output_array = np.zeros(map_2d.shape[1])
        for i in range(0, map_2d.shape[0]):
            output_array += map_2d[i,:]
    if axis == 'energy':
        output_array = np.zeros(map_2d.shape[0])
        for i in range(0, map_2d.shape[1]):
            output_array += map_2d[:,i]
    return output_array

def plot_i3_MC_comparison(MC_nutau, MC_no_nutau, MC_i3, MC_nutau_name, MC_no_nutau_name, MC_i3_name, x_bin_centers, x_bin_edges, channel, x_label):
    fig = plt.figure(figsize=(8,8))
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    hist_MC_tau,_,_ = ax1.hist(x_bin_centers,weights= MC_nutau,bins=x_bin_edges,histtype='step',lw=2,color='b',label= MC_nutau_name,linestyle='solid',normed=args.norm)
    hist_MC_notau,_,_ = ax1.hist(x_bin_centers,weights=MC_no_nutau,bins=x_bin_edges,histtype='step',lw=2,color='g',label= MC_no_nutau_name,linestyle='dashed',normed=args.norm)
    hist_i3,_,_ = ax1.hist(x_bin_centers,weights=MC_i3,bins=x_bin_edges,histtype='step',lw=2,color='black',label= MC_i3_name ,linestyle='solid',normed=args.norm)
    hist_BS,_= np.histogram(x_bin_centers,weights=MC_i3,bins=x_bin_edges)
    #ax1.errorbar(x_bin_centers,hist_BS,yerr=np.sqrt(hist_BS),fmt='o',color='black',label='MC_i3')
    #ax1.bar(x_bin_edges[:-1],2*error, bottom=map-error, width=x_bin_width, color=color, alpha=0.25, linewidth=0)
    #if (channel == 'cscd' or channel == 'cscd+trck') and x_label == 'energy':
    ax1.legend(loc='upper right',ncol=1, frameon=False,numpoints=1)
    plt.title(r'${\rm  \, %s }$'%(channel), fontsize='large')
    min_hist = min(np.min(hist_BS), np.min(hist_MC_notau), np.min(hist_MC_tau))
    max_hist = max(np.max(hist_BS), np.max(hist_MC_notau), np.max(hist_MC_tau))
    ax1.set_ylim(min_hist - min_hist*0.4,max_hist + 0.4*max_hist)
    ax1.set_ylabel('$\#$ events')
    ax1.grid()

    x2,_ = stats.chisquare(MC_i3, f_exp=MC_nutau)
    x2_nutau = x2/len(MC_i3)
    x2,_ = stats.chisquare(MC_i3, f_exp=MC_no_nutau)
    x2_no_nutau = x2/len(MC_i3)

    ax2 = plt.subplot2grid((3,1), (2,0),sharex=ax1)
    #print "hist_MC_tau = ", hist_MC_tau
    #print "hist_BS =     ", hist_BS
    ratio_MC_to_BS_tau = np.zeros_like(hist_MC_tau)
    for i in range(0,len(hist_MC_tau)):
        if hist_MC_tau[i]==0 and hist_BS[i]==0:
            ratio_MC_to_BS_tau[i] = 1
        elif hist_BS[i]==0 and hist_MC_tau[i]!=0:
            print " non zero divided by 0 !!!"
        else:
            ratio_MC_to_BS_tau[i] = hist_MC_tau[i]/hist_BS[i]
    ratio_MC_to_BS_notau = np.zeros_like(hist_MC_notau)
    for i in range(0,len(hist_MC_notau)):
        if hist_MC_notau[i]==0 and hist_BS[i]==0:
            ratio_MC_to_BS_notau[i] = 1
        elif hist_BS[i]==0 and hist_MC_notau[i]!=0:
            print " non zero divided by 0 !!!"
        else:
            ratio_MC_to_BS_notau[i] = hist_MC_notau[i]/hist_BS[i]

    hist_ratio_MC_to_BS_tau = ax2.hist(x_bin_centers, weights=ratio_MC_to_BS_tau,bins=x_bin_edges,histtype='step',lw=2,color='b', linestyle='solid', label='PISA tau/I3')
    hist_ratio_MC_to_BS_notau = ax2.hist(x_bin_centers, weights=ratio_MC_to_BS_notau, bins=x_bin_edges,histtype='step',lw=2,color='g', linestyle='dashed', label = 'PISA notau/I3')
    if x_label == 'energy':
        ax2.set_xlabel('energy [GeV]')
    if x_label == 'coszen':
        ax2.set_xlabel('coszen')
    ax2.set_ylabel('ratio (PISA/I3)')
    ax2.set_ylim(min(min(ratio_MC_to_BS_notau),min(ratio_MC_to_BS_tau))-0.1,max(max(ratio_MC_to_BS_notau),max(ratio_MC_to_BS_tau))+0.2)
    ax2.axhline(y=1,linewidth=1, color='r')
    #ax2.legend(loc='upper center',ncol=1, frameon=False)
    a_text = AnchoredText('nutau x2/NDF=%.2f\nno nutau x2/NDF=%.2f'%(x2_nutau,x2_no_nutau), loc=2)
    ax2.add_artist(a_text)
    ax2.grid()
    fig.subplots_adjust(hspace=0)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.savefig(args.outdir+'I3_MC_%s_%s_distribution_0318.png' % (channel, x_label),dpi=150)
    plt.clf()

def plot_1D_distribution_comparison(i3_maps, nominal_nutau, nominal_no_nutau, png_name_root):

    print "Entering plot_1D_distribution_comparison "

    # get 1D energy (coszen) distribution
    i3_cscd_map = i3_maps['cscd']['map']
    i3_trck_map = i3_maps['trck']['map']

    I3_RecoEnergy_cscd = get_1D_projection(i3_cscd_map, 'energy')
    I3_RecoEnergy_trck = get_1D_projection(i3_trck_map, 'energy')
    I3_RecoCoszen_cscd = get_1D_projection(i3_cscd_map, 'coszen')
    I3_RecoCoszen_trck = get_1D_projection(i3_trck_map, 'coszen')

    nominal_nutau_cscd_map = nominal_nutau['cscd']['map']
    MC_RecoEnergy_nominal_nutau_cscd = get_1D_projection(nominal_nutau_cscd_map, 'energy')
    MC_RecoCoszen_nominal_nutau_cscd = get_1D_projection(nominal_nutau_cscd_map, 'coszen')
    nominal_nutau_trck_map = nominal_nutau['trck']['map']
    MC_RecoEnergy_nominal_nutau_trck = get_1D_projection(nominal_nutau_trck_map, 'energy')
    MC_RecoCoszen_nominal_nutau_trck = get_1D_projection(nominal_nutau_trck_map, 'coszen')

    nominal_no_nutau_cscd_map = nominal_no_nutau['cscd']['map']
    MC_RecoEnergy_nominal_no_nutau_cscd = get_1D_projection(nominal_no_nutau_cscd_map, 'energy')
    MC_RecoCoszen_nominal_no_nutau_cscd = get_1D_projection(nominal_no_nutau_cscd_map, 'coszen')
    nominal_no_nutau_trck_map = nominal_no_nutau['trck']['map']
    MC_RecoEnergy_nominal_no_nutau_trck = get_1D_projection(nominal_no_nutau_trck_map, 'energy')
    MC_RecoCoszen_nominal_no_nutau_trck = get_1D_projection(nominal_no_nutau_trck_map, 'coszen')

    MC_RecoEnergy_nominal_nutau_all_chan = MC_RecoEnergy_nominal_nutau_cscd + MC_RecoEnergy_nominal_nutau_trck
    MC_RecoCoszen_nominal_nutau_all_chan = MC_RecoCoszen_nominal_nutau_cscd + MC_RecoCoszen_nominal_nutau_trck

    MC_RecoEnergy_nominal_no_nutau_all_chan = MC_RecoEnergy_nominal_no_nutau_cscd + MC_RecoEnergy_nominal_no_nutau_trck
    MC_RecoCoszen_nominal_no_nutau_all_chan = MC_RecoCoszen_nominal_no_nutau_cscd + MC_RecoCoszen_nominal_no_nutau_trck

    I3_RecoEnergy_all_chan = I3_RecoEnergy_cscd + I3_RecoEnergy_trck
    I3_RecoCoszen_all_chan = I3_RecoCoszen_cscd + I3_RecoCoszen_trck

    # plot 1D energy (coszen) distribution
    plot_i3_MC_comparison( MC_RecoEnergy_nominal_nutau_cscd, MC_RecoEnergy_nominal_no_nutau_cscd, I3_RecoEnergy_cscd, 'MC cscd (nutau)', 'MC cscd (no nutau)', 'I3 cscd', E_bin_centers, anlys_ebins, 'cscd', png_name_root+'energy')

    plot_i3_MC_comparison( MC_RecoCoszen_nominal_nutau_cscd, MC_RecoCoszen_nominal_no_nutau_cscd, I3_RecoCoszen_cscd, 'MC cscd (nutau)', 'MC cscd (no nutau)', 'I3 cscd', CZ_bin_centers, czbins, 'cscd', png_name_root+'coszen')

    plot_i3_MC_comparison( MC_RecoEnergy_nominal_nutau_trck, MC_RecoEnergy_nominal_no_nutau_trck, I3_RecoEnergy_trck, 'MC trck (nutau)', 'MC trck (no nutau)', 'I3 trck', E_bin_centers, anlys_ebins, 'trck', png_name_root+'energy')

    plot_i3_MC_comparison( MC_RecoCoszen_nominal_nutau_trck, MC_RecoCoszen_nominal_no_nutau_trck, I3_RecoCoszen_trck, 'MC trck (nutau)', 'MC trck (no nutau)', 'I3 trck', CZ_bin_centers, czbins, 'trck', png_name_root+'coszen')

    plot_i3_MC_comparison( MC_RecoEnergy_nominal_nutau_all_chan, MC_RecoEnergy_nominal_no_nutau_all_chan, I3_RecoEnergy_all_chan, 'MC cscd+trck (nutau)', 'MC cscd+trck (no nutau)', 'I3 cscd+trck', E_bin_centers, anlys_ebins, 'cscd+trck', png_name_root+'energy')

    plot_i3_MC_comparison( MC_RecoCoszen_nominal_nutau_all_chan, MC_RecoCoszen_nominal_no_nutau_all_chan, I3_RecoCoszen_all_chan, 'MC cscd+trck (nutau)', 'MC cscd+trck (no nutau)', 'I3 cscd+trck', CZ_bin_centers, czbins, 'cscd+trck', png_name_root+'coszen')

    # make 1-d slices
    i3_map = i3_trck_map + i3_cscd_map
    nominal_nutau_map = nominal_nutau_trck_map + nominal_nutau_cscd_map
    nominal_no_nutau_map = nominal_no_nutau_trck_map + nominal_no_nutau_cscd_map

    for i in range(0, i3_trck_map.shape[0]):
        plot_i3_MC_comparison( nominal_nutau_trck_map[i,:], nominal_no_nutau_trck_map[i,:], i3_trck_map[i,:], 'MC trck (nutau)', 'MC trck (no nutau)', 'I3 trck', CZ_bin_centers, czbins, 'trck', png_name_root+'coszen_E_bin%s'%i)
        plot_i3_MC_comparison( nominal_nutau_cscd_map[i,:], nominal_no_nutau_cscd_map[i,:], i3_cscd_map[i,:], 'MC cscd (nutau)', 'MC cscd (no nutau)', 'I3 cscd', CZ_bin_centers, czbins, 'cscd', png_name_root+'coszen_E_bin%s'%i)
        plot_i3_MC_comparison( nominal_nutau_map[i,:], nominal_no_nutau_map[i,:], i3_map[i,:], 'MC cscd+trck (nutau)', 'MC cscd+trck (no nutau)', 'I3 cscd+trck', CZ_bin_centers, czbins, 'cscd+trck', png_name_root+'coszen_E_bin%s'%i)


if __name__ == '__main__':

    set_verbosity(0)
    parser = ArgumentParser(description='''Quick check if all components are working reasonably well, by
making the final level hierarchy asymmetry plots from the input
settings file. ''')
    parser.add_argument('template_settings',metavar='JSON',
                        help='Settings file to use for template generation')
    parser.add_argument('--background_file',metavar='FILE',type=str, required=True,
                        default='background/Matt_L5b_icc_data_IC86_2_3_4.hdf5',
                        help='''HDF5 File containing atmospheric background from 3 years'
                        inverted corridor cut data''')
    parser.add_argument('--sim_version',metavar='str',default='dima',type=str,
                        help='''name of the simulation version, can only be '4digit', '5digit' or 'dima'.''')
    parser.add_argument('--plot_aeff',action='store_true',default=False,
                        help='Plot Aeff stage comparisons between PISA map and I3 map.')
    parser.add_argument('--cmpr_oscfit_pisa',action='store_true',default=False,
                        help='compare oscFit and PISA histograms for final cscd and trck')
    parser.add_argument('--cmpr_grouped_flavor',action='store_true',default=False,
                        help='compare oscFit and PISA histograms (true and reco) for grouped flavor')
    parser.add_argument('--no_flux_sys_renorm',action='store_true',default=False,
                        help='Use no flux renormalization when applying flux systematics, only use for event-based PISA.')
    parser.add_argument('--no_NC_osc',action='store_true',default=False,
                        help='Use no osc. for NC, for comparison with oscFit.')
    parser.add_argument('--plot_diff_with_nominal',metavar='str(list)',default="[]",type=str,
                        help='''Give a list of systematics, plot the difference between the nominal map
                        and map at systematic+=delta, delta is defined in the delta_val dict.''')
    parser.add_argument('--plot_aeff_1D',action='store_true',default=False,
                        help='''Plot Aeff stage comparisons in 1D distributions.''')
    parser.add_argument('--plot_map_2d_3d',action='store_true',default=False,
                        help='Plot Aeff and final stage comparisons between using NuFlux 2d and NuFlux 3d.')
    parser.add_argument('-no_logE','--no_logE',action='store_true',default=False,
                        help='Do not use log scale for energy.')
    parser.add_argument('-norm','--norm',action='store_true',default=False,
                        help='Normalize the histogram.')
    parser.add_argument('-a','--all',action='store_true',default=False,
                        help='Plot all stages 1-5 of templates and Asymmetry')
    parser.add_argument('--title',metavar='str',default='',
                        help='Title of the geometry or test in plots')
    parser.add_argument('--save',action='store_true',default=False,
                        help='Save plots in outdir')
    parser.add_argument('--savecsv',action='store_true',default=False,
                        help='Save csv files in outdir')
    parser.add_argument('-o','--outdir',metavar='DIR',default='',
                        help='Directory to save the output figures.')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='set verbosity level')
    args = parser.parse_args()
    sim_version = args.sim_version
    set_verbosity(args.verbose)

    # get basic information
    template_settings = from_json(args.template_settings)
    template_settings['params']['icc_bg_file']['value'] = find_resource(args.background_file)
    # for this study, make atmos background = 0
    template_settings['params']['atmos_mu_scale']['value'] = 0
    template_settings['params']['atmmu_f']['value'] = 0
    livetime = 2.5
    template_settings['params']['livetime']['value'] = livetime
    print "template_settings['params']['sim_ver']['value'] = ", template_settings['params']['sim_ver']['value']
    if sim_version!=template_settings['params']['sim_ver']['value']:
        print "Use sim_version defined here:" , sim_version
        template_settings['params']['sim_ver']['value'] = sim_version 

    ebins = template_settings['binning']['ebins']
    anlys_ebins = template_settings['binning']['anlys_ebins']
    czbins = template_settings['binning']['czbins']
    anlys_bins = (anlys_ebins,czbins)
    CZ_bin_centers = get_bin_centers(czbins)
    E_bin_centers = get_bin_centers(anlys_ebins)
    print 'E_bin_centers = ', E_bin_centers
    print 'CZ_bin_centers = ', CZ_bin_centers
    utils.mkdir(args.outdir)
    utils.mkdir(args.outdir+'/pdf/')

    # when plotting comparisons, use same vmax
    cscd_max = int(400 * livetime)
    trck_max = int(250 * livetime)
    all_max = int(650 * livetime)
    #cscd_max = 250
    #trck_max = 150
    #all_max = 400

    # systematics' name in oscFit
    sys_jp_name = {'theta23': 'theta23',
                 'theta13': 'theta13',
                 'deltam31': 'dm31',
                 'aeff_scale': 'norm_nu',
                 'nutau_norm': 'norm_tau',
                 'nc_norm': 'norm_nc',
                 'atmmu_f': 'atmmu_f',
                 'nue_numu_ratio': 'norm_e',
                 'atm_delta_index': 'gamma',
                 'dom_eff': 'domeff',
                 'hole_ice': 'hole_ice',
                 'hole_ice_fwd': 'hi_fwd',
                 'GENSYS_MaCCQE': 'axm_qe',
                 'GENSYS_MaRES': 'axm_res'
                 }

    # define delta values for systematics:
    delta_val = {'theta23': 0.1,
                 'theta13': 0.008,
                 'deltam31': 0.0002,
                 'aeff_scale': 0.1199999999999999,
                 'nutau_norm': 0.5,
                 'nc_norm': 0.2,
                 'atmos_mu_scale': 0.1,
                 'atmmu_f': 0.1,
                 'nue_numu_ratio': 0.05,
                 'atm_delta_index': 0.1,
                 'nu_nubar_ratio':0.1,
                 'dom_eff': 0.1,
                 'hole_ice': 0.1,
                 'hole_ice_fwd': -1.0, 
                 'e_reco_precision_up': 0.05,
                 'cz_reco_precision_up': 0.05,
                 'e_reco_precision_down': 0.05,
                 'cz_reco_precision_down': 0.05,
                 'GENSYS_MaCCQE': 1,
                 'GENSYS_MaRES': 1,
                 'GENSYS_AhtBY': 1,
                 'GENSYS_BhtBY': 1,
                 'GENSYS_CV1uBY': 1,
                 'GENSYS_CV2uBY': 1,
                 'flux_hadronic_A': 1,
                 'flux_hadronic_B': 1,
                 'flux_hadronic_C': 1,
                 'flux_hadronic_D': 1,
                 'flux_hadronic_E': 1,
                 'flux_hadronic_F': 1,
                 'flux_hadronic_G': 1,
                 'flux_hadronic_H': 1,
                 'flux_hadronic_I': 1,
                 'flux_hadronic_W': 1,
                 'flux_hadronic_X': 1,
                 'flux_hadronic_Y': 1,
                 'flux_hadronic_Z': 1,
                 'flux_pion_chargeratio_Chg': 1,
                 'flux_prim_norm_a': 1,
                 'flux_prim_exp_norm_b': 1,
                 'flux_prim_exp_factor_c': 1,
                 'flux_spectral_index_d': 1,
                 }
    list_sys = eval(args.plot_diff_with_nominal)
    print "list_sys = ", list_sys


    ##################### Get Maps from PISA #######################

    nominal_template_settings_default = copy.deepcopy(template_settings)
    if args.no_NC_osc:
        sys_file_name_end = 'fitter_10_by_10_no_NC_osc'
    else:
        sys_file_name_end = 'fitter_10_by_10'
    nominal_template_settings_event = copy.deepcopy(template_settings)
    nominal_template_settings_event['params']['domeff_slope_file']['value'] = "domeff_holeice/dima_p1_event_DomEff_fits_%s.json" % sys_file_name_end
    nominal_template_settings_event['params']['holeice_slope_file']['value'] = "domeff_holeice/dima_p1_event_HoleIce_fits_%s.json" % sys_file_name_end
    nominal_template_settings_event['params']['holeice_fwd_slope_file']['value'] = "domeff_holeice/dima_p2_event_HoleIce_fwd_fits_%s.json" % sys_file_name_end
    nominal_template_settings_event['params']['reco_prcs_coeff_file']['value'] = "reco_prcs/dima_p1_event_RecoPrecisionCubicFitCoefficients_0.7_1.3_data_tau_special_binning.json"
    # for default, right now can't turn off only NC
    #sys_file_name_end_with_osc = 'fitter_10_by_10'   # right now, fitter_10_by_10_no_NC_osc and fitter_10_by_10 are the same; will change the name
    sys_file_name_end_with_osc = 'fitter_10_by_10_no_NC_osc'
    nominal_template_settings_default['params']['domeff_slope_file']['value'] = "domeff_holeice/dima_p1_hist_DomEff_fits_%s.json" % sys_file_name_end_with_osc
    nominal_template_settings_default['params']['holeice_slope_file']['value'] = "domeff_holeice/dima_p1_hist_HoleIce_fits_%s.json" % sys_file_name_end_with_osc
    nominal_template_settings_default['params']['holeice_fwd_slope_file']['value'] = "domeff_holeice/dima_p2_hist_HoleIce_fwd_fits_%s.json" % sys_file_name_end_with_osc
    nominal_template_settings_default['params']['reco_prcs_coeff_file']['value'] = "reco_prcs/dima_p1_hist_RecoPrecisionCubicFitCoefficients_0.7_1.3_data_tau_special_binning.json"

    with Timer() as t:
        nominal_nutau_params_default = copy.deepcopy(select_hierarchy_and_nutau_norm( nominal_template_settings_default['params'],True,1.0))
        nominal_no_nutau_params_default = copy.deepcopy(select_hierarchy_and_nutau_norm( nominal_template_settings_default['params'],True,0.0))
        nominal_nutau_params_event = copy.deepcopy(select_hierarchy_and_nutau_norm( nominal_template_settings_event['params'],True,1.0))
        nominal_no_nutau_params_event = copy.deepcopy(select_hierarchy_and_nutau_norm( nominal_template_settings_event['params'],True,0.0))

        nominal_template_maker_default = TemplateMaker_default(get_values(nominal_nutau_params_default), no_sys_maps=True, **nominal_template_settings_default['binning'])
        nominal_template_maker_event = TemplateMaker_event(get_values(nominal_nutau_params_event), no_sys_maps=True, **nominal_template_settings_event['binning'])

        no_nutau_nominal_template_maker_default = TemplateMaker_default(get_values(nominal_no_nutau_params_default), no_sys_maps=True, **nominal_template_settings_default['binning'])
        no_nutau_nominal_template_maker_event = TemplateMaker_event(get_values(nominal_no_nutau_params_event), no_sys_maps=True, **nominal_template_settings_event['binning'])

    profile.info('==> elapsed time to initialize templates: %s sec'%t.secs)


    # Make nutau template:
    with Timer(verbose=False) as t:
        nominal_nutau_all_stages = nominal_template_maker_default.get_template(get_values(nominal_nutau_params_default),return_stages=True, no_sys_maps=True)
    profile.info('==> elapsed time to get NUTAU template (default PISA): %s sec'%t.secs)
    nominal_no_nutau_all_stages = no_nutau_nominal_template_maker_default.get_template(get_values(nominal_no_nutau_params_default),return_stages=True, no_sys_maps=True)
    flux_map_nominal_nutau = nominal_nutau_all_stages[0]
    osc_flux_map_nominal_nutau = nominal_nutau_all_stages[1]
    evt_rate_map_nominal_nutau = nominal_nutau_all_stages[2]
    evt_rate_map_nominal_no_nutau = nominal_no_nutau_all_stages[2]
    reco_rate_map_nominal_nutau = nominal_nutau_all_stages[3]
    reco_rate_map_nominal_no_nutau = nominal_no_nutau_all_stages[3]
    nominal_nutau = nominal_nutau_all_stages[5]
    nominal_no_nutau = nominal_no_nutau_all_stages[5]

    with Timer(verbose=False) as t:
        nominal_nutau_event_all_stages = nominal_template_maker_event.get_template(get_values(nominal_nutau_params_event),return_stages=True, no_sys_maps=True, use_cut_on_trueE=False, flux_sys_renorm=not(args.no_flux_sys_renorm), turn_off_osc_NC=args.no_NC_osc)
        #nominal_nutau_event_all_stages = nominal_template_maker_event.get_template(get_values(nominal_nutau_params_event),return_stages=True, no_sys_maps=True, use_cut_on_trueE=True, flux_sys_renorm=not(args.no_flux_sys_renorm), turn_off_osc_NC=args.no_NC_osc)
    profile.info('==> elapsed time to get NUTAU template (event-by-event PISA): %s sec'%t.secs)
    with Timer(verbose=False) as t:
        nominal_no_nutau_event_all_stages = no_nutau_nominal_template_maker_event.get_template(get_values(nominal_no_nutau_params_event),return_stages=True, no_sys_maps=True, use_cut_on_trueE=False, flux_sys_renorm=not(args.no_flux_sys_renorm), turn_off_osc_NC=args.no_NC_osc)
        #nominal_no_nutau_event_all_stages = no_nutau_nominal_template_maker_event.get_template(get_values(nominal_no_nutau_params_event),return_stages=True, no_sys_maps=True, use_cut_on_trueE=True, flux_sys_renorm=not(args.no_flux_sys_renorm), turn_off_osc_NC=args.no_NC_osc)
    profile.info('==> elapsed time to get NUTAU template (event-by-event PISA): %s sec'%t.secs)
    evt_rate_map_nominal_nutau_event = nominal_nutau_event_all_stages[1]
    evt_rate_map_nominal_no_nutau_event = nominal_no_nutau_event_all_stages[1]
    reco_rate_map_nominal_nutau_event = nominal_nutau_event_all_stages[2]
    reco_rate_map_nominal_no_nutau_event = nominal_no_nutau_event_all_stages[2]
    nominal_nutau_event = nominal_nutau_event_all_stages[4]
    nominal_no_nutau_event = nominal_no_nutau_event_all_stages[4]
    nominal_nutau_event_grouped = nominal_nutau_event_all_stages[5]
    true_nominal_nutau_event_grouped = nominal_nutau_event_all_stages[6]

    # COMPARE nominal maps and maps at sys = nominal sys + delta:
    if args.plot_diff_with_nominal and list_sys!=[]:
        print "Plotting diff plots with systematics += delta"
        for sys in list_sys:
            one_sigma_template_settings_default = copy.deepcopy(nominal_template_settings_default)
            one_sigma_template_settings_event = copy.deepcopy(nominal_template_settings_event)
            if sys == 'theta23' or sys=='deltam31':
                one_sigma_template_settings_default['params'][sys+'_nh']['value'] += delta_val[sys]
                one_sigma_template_settings_event['params'][sys+'_nh']['value'] += delta_val[sys]
            else:
                one_sigma_template_settings_default['params'][sys]['value'] += delta_val[sys]
                one_sigma_template_settings_event['params'][sys]['value'] += delta_val[sys]

            one_sigma_params_default = copy.deepcopy(select_hierarchy(one_sigma_template_settings_default['params'],True))
            one_sigma_params_event = copy.deepcopy(select_hierarchy(one_sigma_template_settings_event['params'],True))
            sys_val = one_sigma_params_default[sys]['value']
            print sys , " sys_val is : ", sys_val

            # for some systematics, need different settings
            if sys== 'atmmu_f':
                compare_atmmu_f = True
                print "in atmmu_f>>>>"
                nominal_nutau_event = nominal_template_maker_event.get_template(get_values(nominal_nutau_params_event),return_stages=False, no_sys_maps=True, use_cut_on_trueE=False, flux_sys_renorm=not(args.no_flux_sys_renorm), use_atmmu_f=compare_atmmu_f, turn_off_osc_NC=args.no_NC_osc)
            else:
                compare_atmmu_f = False

            if sys not in ['hole_ice', 'hole_ice_fwd', 'dom_eff', 'cz_reco_precision_down', 'cz_reco_precision_up', 'e_reco_precision_down', 'e_reco_precision_up']:
                one_sigma_template_maker_default = TemplateMaker_default(get_values(one_sigma_params_default), no_sys_maps=True, **one_sigma_template_settings_default['binning'])
                one_sigma_template_maker_event = TemplateMaker_event(get_values(one_sigma_params_event), no_sys_maps=True, **one_sigma_template_settings_event['binning'])
                one_sigma_maps_default_all_stages = one_sigma_template_maker_default.get_template(get_values(one_sigma_params_default),return_stages=True, no_sys_maps=True, use_atmmu_f=compare_atmmu_f)
                one_sigma_maps_event_all_stages = one_sigma_template_maker_event.get_template(get_values(one_sigma_params_event),return_stages=True, no_sys_maps=True,use_cut_on_trueE=False, flux_sys_renorm=not(args.no_flux_sys_renorm), use_atmmu_f=compare_atmmu_f, turn_off_osc_NC=args.no_NC_osc)
            else:
                one_sigma_template_maker_default = TemplateMaker_default(get_values(one_sigma_params_default), no_sys_maps=False, **one_sigma_template_settings_default['binning'])
                one_sigma_template_maker_event = TemplateMaker_event(get_values(one_sigma_params_event), no_sys_maps=False, **one_sigma_template_settings_event['binning'])
                one_sigma_maps_default_all_stages = one_sigma_template_maker_default.get_template(get_values(one_sigma_params_default),return_stages=True, no_sys_maps=False, use_atmmu_f=compare_atmmu_f)
                one_sigma_maps_event_all_stages = one_sigma_template_maker_event.get_template(get_values(one_sigma_params_event),return_stages=True, no_sys_maps=False, use_cut_on_trueE=False, flux_sys_renorm=not(args.no_flux_sys_renorm), use_atmmu_f=compare_atmmu_f, turn_off_osc_NC=args.no_NC_osc)
            one_sigma_map_event = one_sigma_maps_event_all_stages[4]
            one_sigma_map_default = one_sigma_maps_default_all_stages[5]

            for channel in ['cscd','trck']:
                diff_hist = (one_sigma_map_event[channel]['map'] - nominal_nutau_event[channel]['map']) #/nominal_nutau_event[channel]['map']
                diff_map = {'map' : diff_hist, 'ebins': anlys_ebins, 'czbins': czbins}
                plot_one_map(one_sigma_map_event[channel], args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_systematics_PISA_event_final_%s_%s'% (channel, sys), fig_title=r'${\rm %s \, yr \, event \, PISA \, %s \, (%s \, = \, %s, \, Nevts: \, %.1f) }$'%(livetime, channel, sys.replace('_', '\, '), str(sys_val), np.sum(one_sigma_map_event[channel]['map'])), save=args.save, max=cscd_max if channel=='cscd' else trck_max)
                plot_one_map(diff_map, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_systematics_diff_with_nominal_PISA_event_final_%s_%s'% (channel, sys), fig_title=r'${\rm %s \, yr \, event \, PISA \, %s \, map \, diff \, (\Delta \, %s \, = \, %s, \, Nevts: \, %.1f) }$'%(livetime, channel, sys.replace('_', '\, '), str(delta_val[sys]), np.sum(diff_map['map'])), save=args.save)

                diff_hist_default = (one_sigma_map_default[channel]['map'] - nominal_nutau[channel]['map']) #/nominal_nutau[channel]['map']
                diff_map_default = {'map' : diff_hist_default, 'ebins': anlys_ebins, 'czbins': czbins}
                plot_one_map(one_sigma_map_default[channel], args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_systematics_PISA_default_final_%s_%s'% (channel, sys), fig_title=r'${\rm %s \, yr \, default \, PISA \, %s \, (%s \, = \, %s, \, Nevts: \, %.1f) }$'%(livetime, channel, sys.replace('_', '\, '), str(sys_val), np.sum(one_sigma_map_default[channel]['map'])), save=args.save, max=cscd_max if channel=='cscd' else trck_max)
                plot_one_map(diff_map_default, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_systematics_diff_with_nominal_PISA_default_final_%s_%s'% (channel, sys), fig_title=r'${\rm %s \, yr \, default \, PISA \, %s \, map \, diff \, (\Delta \, %s \, = \, %s, \, Nevts: \, %.1f) }$'%(livetime, channel, sys.replace('_', '\, '), str(delta_val[sys]), np.sum(diff_map_default['map'])), save=args.save)

            # get data from JP's csv file
                channel_jp = 'cascade' if channel=='cscd' else 'track'
                if sys in ['dom_eff', 'hole_ice', 'hole_ice_fwd']:
                    #file_name = '/Users/feifeihuang/pisa/pisa/i3utils/OscFit_PISA_cmpr/JP/1X600_newmc/diff_%s_baseline_%s.csv'% (sys_jp_name[sys], channel_jp)
                    #file_name = '/Users/feifeihuang/pisa/pisa/i3utils/OscFit_PISA_cmpr/JP/1X600_newmc_forceBaselineCrossing/diff_%s_baseline_%s.csv'% (sys_jp_name[sys], channel_jp)
                    file_name = '/Users/feifeihuang/pisa/pisa/i3utils/OscFit_PISA_cmpr/JP/1X600_newmc_forceBaselineCrossing_detailedSystematics/diff_%s_baseline_%s.csv'% (sys_jp_name[sys], channel_jp)
                elif sys in ['atmmu_f']:
                    file_name = '/Users/feifeihuang/pisa/pisa/i3utils/OscFit_PISA_cmpr/JP/1X600_newDataForIC/diff_%s_baseline_%s.csv'% (sys_jp_name[sys], channel_jp)
                else:
                    file_name = '/Users/feifeihuang/pisa/pisa/i3utils/OscFit_PISA_cmpr/JP/1X600_diff_csv/diff_%s_baseline_%s.csv'% (sys_jp_name[sys], channel_jp)
                if sys == 'atm_delta_index':
                    file_name = '/Users/feifeihuang/pisa/pisa/i3utils/OscFit_PISA_cmpr/JP/1X600_SI_pivotE27.2602103972/diff_%s_baseline_%s.csv'% (sys_jp_name[sys], channel_jp)
                oscFit_data = pd.read_csv(file_name, sep=',',header=None)
                oscFit_data_x = oscFit_data[0].values
                oscFit_data_y = oscFit_data[1].values
                oscFit_data_z = oscFit_data[2].values
                oscFit_hist, x_edges, y_edges = np.histogram2d(oscFit_data_x, oscFit_data_y, weights = oscFit_data_z)
                print "sum of oscFit_hist = ", np.sum(oscFit_hist)
                oscFit_map = {'map' : oscFit_hist, 'ebins': anlys_ebins, 'czbins': czbins}
                perc_diff_hist = (oscFit_hist - diff_map['map'])/diff_map['map']
                perc_diff_map = {'map' : perc_diff_hist, 'ebins': anlys_ebins, 'czbins': czbins}

                ratio_hist = oscFit_hist/diff_map['map']
                ratio_diff_map = {'map' : ratio_hist, 'ebins': anlys_ebins, 'czbins': czbins}

                delta_hist = oscFit_hist-diff_map['map']
                delta_diff_map = {'map' : delta_hist, 'ebins': anlys_ebins, 'czbins': czbins}

                plot_one_map(oscFit_map, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_systematics_diff_with_nominal_OscFit_final_%s_%s'% (channel, sys), fig_title=r'${\rm %s \, yr \, OscFit \, %s \, map \, diff \, ( \Delta \, %s \, = \, %s, \, Nevts: \, %.1f) }$'%(livetime, channel, sys.replace('_', '\, '), str(delta_val[sys]), np.sum(oscFit_map['map'])), save=args.save)

                abs_max = np.max(abs(perc_diff_map['map']))
                plot_one_map(perc_diff_map, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_systematics_diff_with_nominal_OscFit_PISA_event_cmpr_%s_%s_PercentDifference'% (sys, channel), fig_title=r'${\rm %s \, yr \, %s \, plot \, (OscFit\, - \, PISA) \, / \,PISA \, %s }$'%(livetime, sys.replace('_', '\, '), channel), save=args.save,min= -abs_max, max = abs_max, annotate_prcs=3,cmap='RdBu_r')
                #plot_one_map(perc_diff_map, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_systematics_diff_with_nominal_OscFit_PISA_event_cmpr_%s_%s_PercentDifference'% (sys, channel), fig_title=r'${\rm %s \, yr \, %s \, plot \, (OscFit\, - \, PISA) \, / \,PISA \, %s }$'%(livetime, sys.replace('_', '\, '), channel), save=args.save, annotate_prcs=3,cmap='RdBu_r')
                plot_one_map(ratio_diff_map, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_systematics_diff_with_nominal_OscFit_PISA_event_cmpr_%s_%s_Ratio'% (sys, channel), fig_title=r'${\rm %s \, yr \, %s \, plot \, (OscFit \, / \, PISA )\, %s }$'%(livetime, sys.replace('_', '\, '), channel), save=args.save,annotate_prcs=3)
                plot_one_map(delta_diff_map, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_systematics_diff_with_nominal_OscFit_PISA_event_cmpr_%s_%s_Delta'% (sys, channel), fig_title=r'${\rm %s \, yr \, %s \, plot \, (OscFit \, - \, PISA )\, %s }$'%(livetime, sys.replace('_', '\, '), channel), save=args.save,annotate_prcs=3, cmap='RdBu_r')



    #############  Compare Aeff Stage Template ############
    if args.plot_aeff:
        for flavor in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
            for int_type in ['cc','nc']:
                # Plot Aeff map from default PISA
                plot_one_map(evt_rate_map_nominal_nutau[flavor][int_type], args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_default_PISA_aeff_%s_%s'% (flavor, int_type), fig_title=r'${\rm %s \, yr \, default \, PISA \, aeff \, map \, %s \, %s \, (Nevts: \, %.1f) }$'%(livetime, flavor, int_type, np.sum(evt_rate_map_nominal_nutau[flavor][int_type]['map'])), save=args.save)

                # Plot Aeff map from event-by-event PISA
                plot_one_map(evt_rate_map_nominal_nutau_event[flavor][int_type], args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_event_by_event_PISA_aeff_%s_%s'% (flavor, int_type), fig_title=r'${\rm %s \, yr \, event \, PISA \, aeff \, map \, %s \, %s \, (Nevts: \, %.1f) }$'%(livetime, flavor, int_type, np.sum(evt_rate_map_nominal_nutau_event[flavor][int_type]['map'])), save=args.save)

                # Plot Aeff map (from default PISA) / Aeff map (from event by event PISA)
                ratio_aeff_default_event = ratio_map(evt_rate_map_nominal_nutau[flavor][int_type], evt_rate_map_nominal_nutau_event[flavor][int_type])
                plot_one_map(ratio_aeff_default_event, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_Ratio_Aeff_default_event_%s_%s'% (flavor, int_type), fig_title=r'${\rm Ratio \, of \, Aeff \,map \,(default \, / \,event \, %s \, %s }$'%(flavor, int_type), save=args.save,annotate_prcs=3)

                # Plot Aeff map (from default PISA) - Aef map (from event by event PISA)
                delta_aeff_default_event = delta_map(evt_rate_map_nominal_nutau[flavor][int_type], evt_rate_map_nominal_nutau_event[flavor][int_type])
                plot_one_map(delta_aeff_default_event, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_Delta_Aeff_default_event_%s_%s'% (flavor, int_type), fig_title=r'${\rm Delta \, of \, Aeff \,map \,(default \, - \,event \, %s \, %s }$'%(flavor, int_type), save=args.save)



    #############  Compare PID Template for grouped flavors ############

    for channel in ['cscd','trck']:

        if args.cmpr_oscfit_pisa:
            # get data from JP's csv file
            flav = 'cascade' if channel=='cscd' else 'track'
            file_name = '/Users/feifeihuang/pisa/pisa/i3utils/OscFit_PISA_cmpr/JP/1X600_reco/sample_baseline_%s_histo.csv'%flav
            oscFit_data = pd.read_csv(file_name, sep=',',header=None)
            oscFit_data_x = oscFit_data[0].values
            oscFit_data_y = oscFit_data[1].values
            oscFit_data_z = oscFit_data[2].values
            oscFit_hist, x_edges, y_edges = np.histogram2d(oscFit_data_x, oscFit_data_y, weights = oscFit_data_z)
            oscFit_map = {'map' : oscFit_hist, 'ebins': anlys_ebins, 'czbins': czbins}

            perc_diff_hist = (oscFit_hist - nominal_nutau_event[channel]['map'])/nominal_nutau_event[channel]['map']
            perc_diff_map = {'map' : perc_diff_hist, 'ebins': anlys_ebins, 'czbins': czbins}

            ratio_hist = oscFit_hist/nominal_nutau_event[channel]['map']
            ratio_diff_map = {'map' : ratio_hist, 'ebins': anlys_ebins, 'czbins': czbins}

            plot_one_map(oscFit_map, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_OscFit_event_final_NutauCCNorm_1_%s'% (channel), fig_title=r'${\rm %s \, yr \, event \, OscFit \, map \, %s \, (true \, Nevts: \, %.1f) }$'%(livetime, channel, np.sum(oscFit_map['map'])), save=args.save)

            abs_max = np.max(abs(perc_diff_map['map']))
            plot_one_map(perc_diff_map, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_PISA_reco_info_event_final_%s_PercentDifference'% (channel), fig_title=r'${\rm %s \, yr \, (OscFit\, - \, PISA) \, / \,PISA \, %s }$'%(livetime, channel), save=args.save,min= -abs_max, max = abs_max, annotate_prcs=3,cmap='RdBu_r')
            #plot_one_map(perc_diff_map, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_PISA_reco_info_event_final_%s_PercentDifference'% (channel), fig_title=r'${\rm %s \, yr \, (OscFit\, - \, PISA) \, / \,PISA \, %s }$'%(livetime, channel), save=args.save, annotate_prcs=3,cmap='RdBu_r')
            plot_one_map(ratio_diff_map, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_PISA_reco_info_event_final_%s_Ratio'% (channel), fig_title=r'${\rm %s \, yr \, OscFit \, / \, PISA \, %s }$'%(livetime, channel), save=args.save,annotate_prcs=3)

        # plot nominal map from event-based and hist-based PISA
        plot_one_map(nominal_nutau[channel], args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_PISA_default_final_NutauCCNorm_1_%s'% (channel), fig_title=r'${\rm %s \, yr \, default \, PISA \, map \, %s \, (Nevts: \, %.1f) }$'%(livetime, channel, np.sum(nominal_nutau[channel]['map'])), save=args.save, max=cscd_max if channel=='cscd' else trck_max)

        plot_one_map(nominal_nutau_event[channel], args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_PISA_event_final_NutauCCNorm_1_%s'% (channel), fig_title=r'${\rm %s \, yr \, event \, PISA \, map \, %s \, (Nevts: \, %.1f) }$'%(livetime, channel, np.sum(nominal_nutau_event[channel]['map'])), save=args.save, max=cscd_max if channel=='cscd' else trck_max)

        # Plot Ratio of PID map (from PISA) to PID map (from event by event PISA)
        ratio_pid_default_event = ratio_map(nominal_nutau[channel], nominal_nutau_event[channel])
        plot_one_map(ratio_pid_default_event, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_Ratio_default_event_final_map_%s'% (channel), fig_title=r'${\rm Ratio \, of \, (default \, / \, event ) \, %s }$'%(channel), save=args.save,annotate_prcs=3) 

        delta_pid_default_event = delta_map(nominal_nutau[channel], nominal_nutau_event[channel])
        plot_one_map(delta_pid_default_event, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_Delta_default_event_final_map_%s'% (channel), fig_title=r'${\rm Delta \, of \, (default \, - \, event ) \, %s }$'%(channel), save=args.save) 

    print "In Final stage, from event-by-event PISA: \n",
    print "no.of cscd (f = 1) in event-by-event PISA = ", np.sum(nominal_nutau_event['cscd']['map'])
    print "no.of trck (f = 1) in event-by-event PISA = ", np.sum(nominal_nutau_event['trck']['map'])
    print "total no. of cscd and trck  = ", np.sum(nominal_nutau_event['cscd']['map'])+np.sum(nominal_nutau_event['trck']['map'])
    print "\n"
    print "no.of cscd (f = 0) in event-by-event PISA = ", np.sum(nominal_no_nutau_event['cscd']['map'])
    print "no.of trck (f = 0) in event-by-event PISA = ", np.sum(nominal_no_nutau_event['trck']['map'])
    print "total no. of cscd and trck  = ", np.sum(nominal_no_nutau_event['cscd']['map'])+np.sum(nominal_no_nutau_event['trck']['map'])
    print ' \n'

    print "In Final stage, from histogram-based PISA :"
    print 'no. of cscd (f = 1) = ', np.sum(nominal_nutau['cscd']['map'])
    print 'no. of trck (f = 1) = ', np.sum(nominal_nutau['trck']['map'])
    print ' total of the above two : ', np.sum(nominal_nutau['cscd']['map'])+np.sum(nominal_nutau['trck']['map'])
    print ' \n'
    print 'no. of cscd (f = 0) = ', np.sum(nominal_no_nutau['cscd']['map'])
    print 'no. of trck (f = 0) = ', np.sum(nominal_no_nutau['trck']['map'])
    print ' total of the above two : ', np.sum(nominal_no_nutau['cscd']['map'])+np.sum(nominal_no_nutau['trck']['map'])
    print ' \n'
    # plot 1d comparison
    #plot_1D_distribution_comparison(nominal_nutau_event, nominal_nutau,  nominal_no_nutau, png_name_root = 'whole_region_')

    if args.cmpr_grouped_flavor:
        for channel in ['cscd','trck']:
            for group in ['numu_cc', 'nue_cc', 'nutau_cc', 'nuall_nc']:
                flav = group.split('_')[0]
                if group == 'nuall_nc':
                    flav = 'nc'
                channel_jp = 'cascade' if channel=='cscd' else 'track'
                file_name = '/Users/feifeihuang/pisa/pisa/i3utils/OscFit_PISA_cmpr/JP/1X600_reco/sample_baseline_%s_%s_histo.csv'%(channel_jp, flav)
                oscFit_data = pd.read_csv(file_name, sep=',',header=None)
                oscFit_data_x = oscFit_data[0].values
                oscFit_data_y = oscFit_data[1].values
                oscFit_data_z = oscFit_data[2].values
                oscFit_hist, x_edges, y_edges = np.histogram2d(oscFit_data_x, oscFit_data_y, weights = oscFit_data_z)
                oscFit_map = {'map' : oscFit_hist, 'ebins': anlys_ebins, 'czbins': czbins}

                perc_diff_hist = (oscFit_hist - nominal_nutau_event_grouped[channel][group]['map'])/nominal_nutau_event_grouped[channel][group]['map']
                #perc_diff_hist = np.nan_to_num(perc_diff_hist)
                perc_diff_map = {'map' : perc_diff_hist, 'ebins': anlys_ebins, 'czbins': czbins}

                ratio_hist = oscFit_hist/nominal_nutau_event_grouped[channel][group]['map']
                ratio_diff_map = {'map' : ratio_hist, 'ebins': anlys_ebins, 'czbins': czbins}

                plot_one_map(nominal_nutau_event_grouped[channel][group], args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_PISA_event_grouped_%s_NutauCCNorm_1_%s'% (group, channel), fig_title=r'${\rm %s \, yr \, event \, PISA \, map \, %s \, %s \, (Nevts: \, %.1f) }$'%(livetime, group.replace('_', '\, '), channel, np.sum(nominal_nutau_event_grouped[channel][group]['map'])), save=args.save)

                plot_one_map(oscFit_map, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_OscFit_event_grouped_%s_NutauCCNorm_1_%s'% (group, channel), fig_title=r'${\rm %s \, yr \, event \, OscFit \, map \, %s \, %s \, (Nevts: \, %.1f) }$'%(livetime, group.replace('_', '\, '), channel, np.sum(oscFit_map['map'])), save=args.save)

                abs_max = np.max(abs(perc_diff_map['map']))
                #plot_one_map(perc_diff_map, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_PISA_event_grouped_%s_%s_PercentDifference'% (group, channel), fig_title=r'${\rm %s \, yr \, (OscFit\, - \, PISA) \, / \,PISA \, %s \, %s \, }$'%(livetime, group.replace('_', '\, '), channel), save=args.save,min= -abs_max, max = abs_max, annotate_prcs=3,cmap='RdBu_r')
                plot_one_map(perc_diff_map, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_PISA_event_grouped_%s_%s_PercentDifference'% (group, channel), fig_title=r'${\rm %s \, yr \, (OscFit\, - \, PISA) \, / \,PISA \, %s \, %s \, }$'%(livetime, group.replace('_', '\, '), channel), save=args.save, annotate_prcs=3,cmap='RdBu_r')
                plot_one_map(ratio_diff_map, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_PISA_event_grouped_%s_%s_Ratio'% (group, channel), fig_title=r'${\rm %s \, yr \, OscFit \, / \, PISA \, %s \, %s }$'%(livetime, group.replace('_', '\, '), channel), save=args.save,annotate_prcs=3)
                print "no of evts in ", group, " ", channel, " (reco E and CZ) =  ", np.sum(nominal_nutau_event_grouped[channel][group]['map'])
        print "\n"


        for channel in ['cscd','trck']:
            for group in ['numu_cc', 'nue_cc', 'nutau_cc', 'nuall_nc']:
                flav = group.split('_')[0]
                if group == 'nuall_nc':
                    flav = 'nc'
                channel_jp = 'cascade' if channel=='cscd' else 'track'
                #file_name = '/Users/feifeihuang/pisa/pisa/i3utils/OscFit_PISA_cmpr/JP/1X600_truth/sample_baseline_%s_%s_histo.csv'%(channel_jp, flav)
                #oscFit_data = pd.read_csv(file_name, sep=',',header=None)
                #oscFit_data_x = oscFit_data[0].values
                #oscFit_data_y = oscFit_data[1].values
                #oscFit_data_z = oscFit_data[2].values
                #oscFit_hist, x_edges, y_edges = np.histogram2d(oscFit_data_x, oscFit_data_y, weights = oscFit_data_z)
                #oscFit_map = {'map' : oscFit_hist, 'ebins': anlys_ebins, 'czbins': czbins}

                #perc_diff_hist = (oscFit_hist - true_nominal_nutau_event_grouped[channel][group]['map'])/true_nominal_nutau_event_grouped[channel][group]['map']
                #perc_diff_map = {'map' : perc_diff_hist, 'ebins': anlys_ebins, 'czbins': czbins}

                #ratio_hist = oscFit_hist/true_nominal_nutau_event_grouped[channel][group]['map']
                #ratio_diff_map = {'map' : ratio_hist, 'ebins': anlys_ebins, 'czbins': czbins}

                plot_one_map(true_nominal_nutau_event_grouped[channel][group], args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_PISA_true_info_event_grouped_%s_NutauCCNorm_1_%s'% (group, channel), fig_title=r'${\rm %s \, yr \, event \, PISA \, map \, %s \, %s \, (true \, Nevts: \, %.1f) }$'%(livetime, group.replace('_', '\, '), channel, np.sum(true_nominal_nutau_event_grouped[channel][group]['map'])), save=args.save)

                #plot_one_map(oscFit_map, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_OscFit_true_info_event_grouped_%s_NutauCCNorm_1_%s'% (group, channel), fig_title=r'${\rm %s \, yr \, event \, OscFit \, map \, %s \, %s \, (true \, Nevts: \, %.1f) }$'%(livetime, group.replace('_', '\, '), channel, np.sum(oscFit_map['map'])), save=args.save)

                #abs_max = np.max(abs(perc_diff_map['map']))
                ##plot_one_map(perc_diff_map, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_PISA_true_info_event_grouped_%s_%s_PercentDifference'% (group, channel), fig_title=r'${\rm %s \, yr \, (OscFit\, - \, PISA) \, / \,PISA \, %s \, %s \, (true) }$'%(livetime, group.replace('_', '\, '), channel), save=args.save,min= -abs_max, max = abs_max, annotate_prcs=3,cmap='RdBu_r')
                #plot_one_map(perc_diff_map, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_PISA_true_info_event_grouped_%s_%s_PercentDifference'% (group, channel), fig_title=r'${\rm %s \, yr \, (OscFit\, - \, PISA) \, / \,PISA \, %s \, %s \, (true) }$'%(livetime, group.replace('_', '\, '), channel), save=args.save, annotate_prcs=3,cmap='RdBu_r')

                #plot_one_map(ratio_diff_map, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_PISA_true_info_event_grouped_%s_%s_Ratio'% (group, channel), fig_title=r'${\rm %s \, yr \, OscFit \, / \, PISA \, %s \, %s \, (true) }$'%(livetime, group.replace('_', '\, '), channel), save=args.save,annotate_prcs=3)
                print "no of evts in ", group, " ", channel, " (true E and CZ) =  ", np.sum(true_nominal_nutau_event_grouped[channel][group]['map'])
        print "\n"


    # plot no_pid maps comparison
    nominal_nutau_no_pid = sum_map(nominal_nutau['cscd'], nominal_nutau['trck'])
    nominal_nutau_event_no_pid = sum_map(nominal_nutau_event['cscd'], nominal_nutau_event['trck'])
    ratio_no_pid = ratio_map(nominal_nutau_no_pid, nominal_nutau_event_no_pid)
    delta_no_pid = delta_map(nominal_nutau_no_pid, nominal_nutau_event_no_pid)

    plot_one_map(nominal_nutau_no_pid, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_PISA_default_final_NutauCCNorm_1_all_channel', fig_title=r'${\rm %s \, yr \, PISA \, map \, cscd \, + \, trck \, map \, (Nevts: \, %.1f) }$'%(livetime, np.sum(nominal_nutau_no_pid['map'])), save=args.save, max=all_max)
    plot_one_map(nominal_nutau_event_no_pid, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_PISA_event_final_NutauCCNorm_1_all_channel', fig_title=r'${\rm %s \, yr \, event \, PISA \, cscd \, + \, trck \, map \, (Nevts: \, %.1f) }$'%(livetime, np.sum(nominal_nutau_event_no_pid['map'])), save=args.save, max=all_max)
    plot_one_map(ratio_no_pid, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_Ratio_final_PISA_all_channel', fig_title=r'${\rm Ratio \, of \, final \, map (default \, / \, event ) \, cscd \, + \, trck }$', save=args.save,annotate_prcs=3) 
    plot_one_map(delta_no_pid, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_Delta_final_PISA_all_channel', fig_title=r'${\rm Delta \, of \, final \, map (default \, / \, event ) \, cscd \, + \, trck }$', save=args.save,annotate_prcs=3) 
    plt.figure()
    abs_max = np.max(abs(delta_no_pid['map']))
    show_map_swap(delta_no_pid, vmin= -abs_max, vmax = abs_max, annotate_prcs=3,cmap='RdBu_r',savetxt=args.savecsv)
    if args.save:
        filename = os.path.join(args.outdir,args.title+ '_Delta_defaultPISA_eventPISA_all_channel_RdBu_r.png')
        filename_pdf = os.path.join(args.outdir+'/pdf/',args.title+ '_Delta_defaultPISA_eventPISA_all_channel_RdBu_r.pdf')
        plt.title('Delta of PID map (default - event) cscd + trck')
        plt.savefig(filename,dpi=150)
        plt.savefig(filename_pdf,dpi=150)
        plt.clf()


    if not args.save: plt.show()
    else: print '\n-->>Saved all files to: ',args.outdir


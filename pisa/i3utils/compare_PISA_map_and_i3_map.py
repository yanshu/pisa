#! /usr/bin/env python
#
# author: Timothy C. Arlen
#         Feifei Huang
#
# date: 20 Jan 2016
#
#   Make PISA maps, compare with maps from I3 files directly. 
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
#from pisa.analysis.TemplateMaker_nutau import TemplateMaker
from TemplateMaker_MC import TemplateMaker
import pisa.utils.utils as utils
from pisa.utils.params import get_values, select_hierarchy_and_nutau_norm
from pisa.utils.log import set_verbosity,logging,profile
from pisa.resources.resources import find_resource
from pisa.utils.utils import Timer, is_linear, is_logarithmic, get_bin_centers
from pisa.utils.jsons import from_json
from pisa.utils.plot import show_map, sum_map, ratio_map, delta_map
from pisa.analysis.stats.Maps_i3 import get_i3_maps
from pisa.background.BackgroundServiceICC import BackgroundServiceICC

def plot_one_map(map_to_plot, outdir, logE, fig_title, fig_name, save, max=None, min=None):
    plt.figure()
    show_map(map_to_plot, vmin= min if min!=None else np.min(map_to_plot['map']), vmax= max if max!=None else np.max(map_to_plot['map']),annotate_prcs=2, logE=logE)
    if save:
        filename = os.path.join(outdir, fig_name + '.png')
        pdf_filename = os.path.join(outdir+'/pdf/', fig_name + '.pdf')
        plt.title(fig_title)
        plt.savefig(filename,dpi=150)
        plt.savefig(pdf_filename,dpi=150)
        plt.clf()

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
    plt.title(r'${\rm 1 \, yr \, %s }$'%(channel), fontsize='large')
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
    parser.add_argument('--background_file',metavar='FILE',type=str,
                        default='background/Matt_L5b_icc_data_IC86_2_3_4.hdf5',
                        help='''HDF5 File containing atmospheric background from 3 years'
                        inverted corridor cut data''')
    parser.add_argument('--sim_version',metavar='str',default='',type=str,
                        help='''name of the simulation version, can only be '4digit' or '5digit'.''')
    parser.add_argument('--plot_pisa_i3_cmpr',action='store_true',default=False,
                        help='Plot comparisons between PISA map and I3 map.')
    parser.add_argument('--plot_aeff',action='store_true',default=False,
                        help='Plot Aeff stage comparisons between PISA map and I3 map.')
    parser.add_argument('--plot_aeff_1D',action='store_true',default=False,
                        help='Plot Aeff stage comparisons in 1D distributions.')
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
    parser.add_argument('-o','--outdir',metavar='DIR',default='',
                        help='Directory to save the output figures.')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='set verbosity level')
    args = parser.parse_args()
    sim_version = args.sim_version
    print "type sim_version = ", type(sim_version)
    set_verbosity(args.verbose)

    # get basic information
    template_settings = from_json(args.template_settings)
    template_settings['params']['icc_bg_file']['value'] = find_resource(args.background_file)
    template_settings['params']['atmos_mu_scale']['value'] = 0
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


    ##################### Get Maps from PISA #######################

    nominal_template_settings = copy.deepcopy(template_settings)

    with Timer() as t:
        nominal_nutau_params = copy.deepcopy(select_hierarchy_and_nutau_norm( nominal_template_settings['params'],True,1.0))
        nominal_template_maker = TemplateMaker(get_values(nominal_nutau_params), **nominal_template_settings['binning'])
        nominal_no_nutau_params = copy.deepcopy(select_hierarchy_and_nutau_norm( template_settings['params'],True,0.0))
        no_nutau_nominal_template_maker = TemplateMaker(get_values(nominal_no_nutau_params), **template_settings['binning'])
    profile.info('==> elapsed time to initialize templates: %s sec'%t.secs)

    # Make nutau template:
    with Timer(verbose=False) as t:
        nominal_nutau_all_stages = nominal_template_maker.get_template(get_values(nominal_nutau_params),return_stages=args.all, no_sys_maps=True, read_flux_json=False, read_osc_json=True)
        nominal_no_nutau_all_stages = no_nutau_nominal_template_maker.get_template(get_values(nominal_no_nutau_params),return_stages=args.all, no_sys_maps=True,read_flux_json=False, read_osc_json=True)
        evt_rate_map_nominal_nutau = nominal_nutau_all_stages[2]
        evt_rate_map_nominal_no_nutau = nominal_no_nutau_all_stages[2]
        reco_rate_map_nominal_nutau = nominal_nutau_all_stages[3]
        reco_rate_map_nominal_no_nutau = nominal_no_nutau_all_stages[3]
        nominal_nutau = nominal_nutau_all_stages[5]
        nominal_no_nutau = nominal_no_nutau_all_stages[5]

    profile.info('==> elapsed time to get NUTAU template: %s sec'%t.secs)


    ##################### Plot Aeff Maps #######################

    # if plot aeff maps comparison, one using NuFLux 2d and one NuFLux 3d
    if args.plot_map_2d_3d:
        file_dir_2d = '/gpfs/group/icecube/Matt_level5b_mc_with_weights_2d_IPhonda_2014_spl_solmin/'
        file_dir_3d = '/gpfs/group/icecube/Matt_level5b_mc_with_weights_IPhonda2014/'
        n_nue_files = 2700
        n_numu_files = 4000
        n_nutau_files = 1400
        evt_rate_maps_from_i3_flux_2d, final_maps_from_i3_flux_2d = get_i3_maps(nue_file= file_dir_2d+'DC12_1260_2d.hdf5', numu_file = file_dir_2d+'DC12_1460_2d.hdf5', nutau_file=file_dir_2d+'DC12_1660_2d.hdf5', n_nue_files=n_nue_files, n_numu_files=n_numu_files, n_nutau_files=n_nutau_files, output_form = 'aeff_and_final_map', cut_level ='L6', year= 1, ebins = ebins, anlys_ebins = anlys_ebins, czbins=czbins, sim_version=args.sim_version)
        evt_rate_maps_from_i3_flux_3d, final_maps_from_i3_flux_3d = get_i3_maps(nue_file= file_dir_3d+'Matt_L5b_mc_with_weights_nue.hdf5', numu_file = file_dir_3d+'Matt_L5b_mc_with_weights_numu.hdf5', nutau_file=file_dir_3d+'Matt_L5b_mc_with_weights_nutau.hdf5', n_nue_files=n_nue_files, n_numu_files=n_numu_files, n_nutau_files=n_nutau_files, output_form = 'aeff_and_final_map', cut_level ='L6', year= 1, ebins = ebins, anlys_ebins = anlys_ebins, czbins=czbins, sim_version=args.sim_version)
        for flavor in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
            for int_type in ['cc','nc']:
                # Plot Aeff map (from PISA) to Aef map (from I3 files)
                plot_one_map(evt_rate_maps_from_i3_flux_3d[flavor][int_type], args.outdir+'flux_2d_3d/', fig_name=args.title+ '_Aeff_3d_%s_%s_'% (flavor, int_type), fig_title='Aeff map (3d) %s %s'% (flavor, int_type), save=args.save)
                plot_one_map(evt_rate_maps_from_i3_flux_2d[flavor][int_type], args.outdir+'flux_2d_3d/', fig_name=args.title+ '_Aeff_2d_%s_%s_'% (flavor, int_type), fig_title='Aeff map (2d) %s %s'% (flavor, int_type), save=args.save)
                delta_aeff_3d_2d = delta_map(evt_rate_maps_from_i3_flux_3d[flavor][int_type], evt_rate_maps_from_i3_flux_2d[flavor][int_type])
                plot_one_map(delta_aeff_3d_2d, args.outdir+'flux_2d_3d/', fig_name=args.title+ '_Delta_Aeff_3d_2d_%s_%s_'% (flavor, int_type), fig_title='Delta of Aeff map (3d - 2d flux) %s %s'% (flavor, int_type), save=args.save)

                ratio_aeff_3d_2d = ratio_map(evt_rate_maps_from_i3_flux_3d[flavor][int_type], evt_rate_maps_from_i3_flux_2d[flavor][int_type])
                plot_one_map(ratio_aeff_3d_2d, args.outdir+'flux_2d_3d/', fig_name=args.title+ '_Ratio_Aeff_3d_2d_%s_%s_'% (flavor, int_type), fig_title='Ratio of Aeff map (3d / 2d flux) %s %s'% (flavor, int_type), save=args.save)

        for flav in ['cscd', 'trck']:
            plot_one_map(final_maps_from_i3_flux_3d[flav], args.outdir+'flux_2d_3d/', fig_name=args.title+ '_PID_3d_%s_'% (flav), fig_title='PID map (3d) %s (Nevts: %s)'% (flav, np.sum(final_maps_from_i3_flux_3d[flav]['map'])), save=args.save)
            plot_one_map(final_maps_from_i3_flux_2d[flav], args.outdir+'flux_2d_3d/', fig_name=args.title+ '_PID_2d_%s_'% (flav), fig_title='PID map (2d) %s (Nevts: %s)'% (flav, np.sum(final_maps_from_i3_flux_2d[flav]['map'])), save=args.save)
            delta_pid_3d_2d = delta_map(final_maps_from_i3_flux_3d[flav], final_maps_from_i3_flux_2d[flav])
            plot_one_map(delta_pid_3d_2d, args.outdir+'flux_2d_3d/', fig_name=args.title+ '_Delta_PID_3d_2d_%s_'% (flav), fig_title='Delta of PID map (3d - 2d flux) %s'% (flav), save=args.save)
            ratio_pid_3d_2d = ratio_map(final_maps_from_i3_flux_3d[flav], final_maps_from_i3_flux_2d[flav])
            plot_one_map(ratio_pid_3d_2d, args.outdir+'flux_2d_3d/', fig_name=args.title+ '_Ratio_PID_3d_2d_%s_'% (flav), fig_title='Ratio of PID map (3d / 2d flux) %s'% (flav), save=args.save)
        print ' total no. of evts from 3d flux: ', np.sum(final_maps_from_i3_flux_3d['cscd']['map'])+np.sum(final_maps_from_i3_flux_3d['trck']['map'])
        print ' total no. of evts from 2d flux : ', np.sum(final_maps_from_i3_flux_2d['cscd']['map'])+np.sum(final_maps_from_i3_flux_2d['trck']['map'])


    if sim_version == "4digit":
        #file_dir = '/gpfs/group/icecube/Matt_level5b_mc_with_weights_IPhonda2014/'      # flux 3d
        #nue_file = file_dir + 'Matt_L5b_mc_with_weights_nue.hdf5'
        #numu_file = file_dir + 'Matt_L5b_mc_with_weights_numu.hdf5' 
        #nutau_file = file_dir + 'Matt_L5b_mc_with_weights_nutau.hdf5'
        file_dir = '/gpfs/group/icecube/Matt_level5b_mc_with_weights_2d_IPhonda_2014_spl_solmin/'     # flux 2d
        nue_file = file_dir + 'DC12_1260_2d.hdf5'
        numu_file = file_dir + 'DC12_1460_2d.hdf5'
        nutau_file = file_dir + 'DC12_1660_2d.hdf5'
        n_nue_files = 2700
        n_numu_files = 4000
        n_nutau_files = 1400
    elif sim_version == "5digit":
        #TODO
        print "to be done"
        nue_file = 'events/'
        numu_file = 'events/' 
        nutau_file = 'events/'
        n_nue_files = 750
        n_numu_files = 300
        n_nutau_files = 60
    else:
        print "sim_version given ", sim_version
        raise ValueError( "sim_version allowed: ['4digit', '5digit']")

    ##################### Plot PISA maps and I3 maps #######################
    if args.plot_pisa_i3_cmpr:
        evt_rate_maps_from_i3, final_maps_from_i3 = get_i3_maps(nue_file, numu_file, nutau_file, n_nue_files, n_numu_files, n_nutau_files,output_form = 'aeff_and_final_map', cut_level ='L6', year= 1, ebins = ebins, anlys_ebins = anlys_ebins, czbins=czbins, sim_version=args.sim_version)

        #if args.plot_aeff_1D:
        #    for flavor in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
        #        for int_type in ['cc','nc']:
        #            plot_maps = [evt_rate_map_nominal_nutau[flav][int_type]['map']]
        #            plot_sumw2 = []
        #            plot_colors = []
        #            plot_names = []
        #    myPlotter = plotter(livetime, args.outdir, logy=False)
        #    myPlotter.plot_1d_projection(plot_maps,plot_sumw2,plot_colors,plot_names ,axis, bins, channel)
        #    myPlotter.plot_1d_slices(plot_maps,plot_sumw2,plot_colors,plot_names ,axis, bins, channel)


        if args.plot_aeff:
            total_no_i3_aeff = 0
            for flavor in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
                for int_type in ['cc','nc']:
                    total_no_i3_aeff += np.sum(evt_rate_maps_from_i3[flavor][int_type]['map'])
                    plot_one_map(evt_rate_maps_from_i3[flavor][int_type], args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_I3_file_aeff_event_rate_%s_%s'% (flavor, int_type), fig_title=r'${\rm 1 \, yr \, I3 \, files \, aeff \, map \, %s \, %s \, (Nevts: \, %.1f) }$'%(flavor, int_type, np.sum(evt_rate_maps_from_i3[flavor][int_type]['map'])), save=args.save)
            print "After Effective Area stage, total no. of I3 events= ", total_no_i3_aeff
            print ' \n'

            # Get map in analysis ebins (smaller than ebins)
            anlys_elements = []
            assert(len(anlys_ebins) <= len(ebins))
            for i in range(0,len(ebins)):
                if ebins[i] in anlys_ebins:
                    anlys_elements.append(i)
            anlys_elements.pop()
            for flav in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
                for int_type in ['cc','nc']:
                    reco_event_rate = evt_rate_map_nominal_nutau[flav][int_type]['map']
                    reco_event_rate_anlys = evt_rate_map_nominal_nutau[flav][int_type]['map']
                    #reco_event_rate_anlys = reco_event_rate[:][anlys_elements]
                    evt_rate_map_nominal_nutau[flav][int_type] = {'map': reco_event_rate_anlys,
                                                                 'ebins': anlys_ebins,
                                                                 'czbins': czbins}

            total_no_PISA_aeff = 0
            for flavor in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
                for int_type in ['cc','nc']:
                    # Plot Aeff map from PISA
                    plot_one_map(evt_rate_map_nominal_nutau[flavor][int_type], args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_PISA_aeff_event_rate_%s_%s'% (flavor, int_type), fig_title=r'${\rm 1 \, yr \, PISA \, aeff \, map \, %s \, %s \, (Nevts: \, %.1f) }$'%(flavor, int_type, np.sum(evt_rate_map_nominal_nutau[flavor][int_type]['map'])), save=args.save)
                    total_no_PISA_aeff += np.sum(evt_rate_map_nominal_nutau[flavor][int_type]['map'])

                    # Plot Aeff map (from PISA) to Aef map (from I3 files)
                    ratio_aeff_pisa_aeff_i3 = ratio_map(evt_rate_map_nominal_nutau[flavor][int_type], evt_rate_maps_from_i3[flavor][int_type])
                    plot_one_map(ratio_aeff_pisa_aeff_i3, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_Ratio_Aeff_PISA_Aeff_I3_%s_%s'% (flavor, int_type), fig_title=r'${\rm Ratio \, of \, Aeff \,map \,(PISA \, / \,I3 \, %s \, %s }$'%(flavor, int_type), save=args.save)
                        #plt.title('Ratio of Aeff map (PISA/ I3 file) %s %s'% (flavor, int_type))

                    # Plot Aeff map (from PISA) to Aef map (from I3 files)
                    delta_aeff_pisa_aeff_i3 = delta_map(evt_rate_map_nominal_nutau[flavor][int_type], evt_rate_maps_from_i3[flavor][int_type])
                    plot_one_map(delta_aeff_pisa_aeff_i3, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_Delta_Aeff_PISA_Aeff_I3_%s_%s'% (flavor, int_type), fig_title=r'${\rm Delta \, of \, Aeff \,map \,(PISA \, - \,I3 \, %s \, %s }$'%(flavor, int_type), save=args.save)

            print "After Effective Area stage, total no. of PISA events = ", total_no_PISA_aeff
            print ' \n'


        #############  Compare PISA and I3Files Final Stage Template ############

        # plot trck/cscd maps from i3 files
        for channel in ['cscd','trck']:
            plot_one_map(final_maps_from_i3[channel], args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_I3_file_final_event_rate_%s'% (channel), fig_title=r'${\rm 1 \, yr \, I3 \, files \, map \, %s \, (Nevts: \, %.1f) }$'%(channel, np.sum(final_maps_from_i3[channel]['map'])), save=args.save, max=250 if channel=='cscd' else 150)
            plot_one_map(nominal_nutau[channel], args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_PISA_final_NutauCCNorm_1_%s'% (channel), fig_title=r'${\rm 1 \, yr \, PISA \, map \, %s \, (Nevts: \, %.1f) }$'%(channel, np.sum(nominal_nutau[channel]['map'])), save=args.save, max=250 if channel=='cscd' else 150)
            # Plot Ratio of Aeff map (from PISA) to Aef map (from I3 files)
            ratio_pid_pisa_pid_i3 = ratio_map(nominal_nutau[channel], final_maps_from_i3[channel])
            plot_one_map(ratio_pid_pisa_pid_i3, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_Ratio_PID_PISA_PID_I3_%s'% (channel), fig_title=r'${\rm Ratio \, of \, (PISA \, / \, I3 ) \, %s }$'%(channel), save=args.save) 
            #show_map(ratio_pid_pisa_pid_i3, vmin= np.min(ratio_pid_pisa_pid_i3['map']), vmax= np.max(ratio_pid_pisa_pid_i3['map']),annotate_prcs=2)
            #    filename = os.path.join(args.outdir,args.title+ '_Ratio_PID_PISA_PID_I3_%s.png'% channel)
            #    plt.title('Ratio of PID map (PISA/ I3 file) %s' % channel)

        print "In Final stage, from I3: \n",
        print "no.of cscd in I3 = ", np.sum(final_maps_from_i3['cscd']['map'])
        print "no.of trck in I3 = ", np.sum(final_maps_from_i3['trck']['map'])
        print "total no. of cscd and trck  = ", np.sum(final_maps_from_i3['cscd']['map'])+np.sum(final_maps_from_i3['trck']['map'])
        print ' \n'

        print "In Final stage, from PISA :"
        print 'no. of cscd (f = 1) = ', np.sum(nominal_nutau['cscd']['map'])
        print 'no. of trck (f = 1) = ', np.sum(nominal_nutau['trck']['map'])
        print ' total of the above two : ', np.sum(nominal_nutau['cscd']['map'])+np.sum(nominal_nutau['trck']['map'])
        print ' \n'
        print 'no. of cscd (f = 0) = ', np.sum(nominal_no_nutau['cscd']['map'])
        print 'no. of trck (f = 0) = ', np.sum(nominal_no_nutau['trck']['map'])
        print ' total of the above two : ', np.sum(nominal_no_nutau['cscd']['map'])+np.sum(nominal_no_nutau['trck']['map'])
        print ' \n'
        # plot 1d comparison
        #plot_1D_distribution_comparison(final_maps_from_i3, nominal_nutau,  nominal_no_nutau, png_name_root = 'whole_region_')


        # plot no_pid maps comparison
        nominal_nutau_no_pid = sum_map(nominal_nutau['cscd'], nominal_nutau['trck'])
        final_maps_from_i3_no_pid = sum_map(final_maps_from_i3['cscd'], final_maps_from_i3['trck'])
        ratio_pisa_i3_no_pid = ratio_map(nominal_nutau_no_pid, final_maps_from_i3_no_pid)
        delta_pisa_i3_no_pid = delta_map(nominal_nutau_no_pid, final_maps_from_i3_no_pid)
        plot_one_map(nominal_nutau_no_pid, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_PISA_final_NutauCCNorm_1_all_channel', fig_title=r'${\rm 1 \, yr \, PISA \, map \, cscd \, + \, trck \, map \, (Nevts: \, %.1f) }$'%(np.sum(nominal_nutau_no_pid['map'])), save=args.save, max=400)
        plot_one_map(final_maps_from_i3_no_pid, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_I3_file_final_NutauCCNorm_1_all_channel', fig_title=r'${\rm 1 \, yr \, I3 \, map \, cscd \, + \, trck \, map \, (Nevts: \, %.1f) }$'%(np.sum(final_maps_from_i3_no_pid['map'])), save=args.save, max=400)
        plot_one_map(ratio_pisa_i3_no_pid, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_Ratio_final_PISA_I3_all_channel', fig_title=r'${\rm Ratio \, of \, final map (PISA \, / \, I3 ) \, cscd \, + \, trck }$', save=args.save) 
        #    plt.title('Ratio of PID map (PISA/ I3 file) cscd + trck')
        plt.figure()
        abs_max = np.max(abs(delta_pisa_i3_no_pid['map']))
        show_map(delta_pisa_i3_no_pid, vmin= -abs_max, vmax = abs_max, annotate_prcs=2,cmap='RdBu_r')
        if args.save:
            filename = os.path.join(args.outdir,args.title+ '_Delta_PID_PISA_PID_I3_all_channel.png')
            plt.title('Delta of PID map (PISA - I3 file) cscd + trck')
            plt.savefig(filename,dpi=150)
            plt.clf()


    if not args.save: plt.show()
    else: print '\n-->>Saved all files to: ',args.outdir


#! /usr/bin/env python
#
# author: Timothy C. Arlen
#         Feifei Huang
#
# date: 20 Jan 2016
#
#   Compare maps of the burn sample in the form of the PISA final stage output with PISA MC expectation.
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
from scipy.special import gammaln
from scipy.stats import poisson
from scipy.constants import Julian_year
from pisa.analysis.TemplateMaker_nutau import TemplateMaker
from pisa.utils.params import get_values, select_hierarchy_and_nutau_norm, select_hierarchy
import pisa.utils.utils as utils
import pisa.analysis.stats.Maps as Maps
from pisa.analysis.stats.Maps_nutau import get_burn_sample_maps
from pisa.analysis.stats.Maps_nutau import get_low_level_quantities
from pisa.utils.log import set_verbosity,logging,profile
from pisa.resources.resources import find_resource
from pisa.utils.utils import Timer, is_linear, is_logarithmic, get_bin_centers, get_bin_sizes
from pisa.utils.jsons import from_json
from pisa.utils.plot import show_map, sum_map, ratio_map, delta_map
from pisa.background.BackgroundServiceICC_nutau import BackgroundServiceICC

def plot_param_distribution(data, bg, mc, mc_weights, nbins, outdir, x_label, fig_name, save, norm, livetime, atmos_mu_scale, aeff_scale, logy):
    mc_norm = aeff_scale*livetime*Julian_year
    bg_weights =np.ones(len(bg))*livetime*atmos_mu_scale
    data_counts,x_bin_edges = np.histogram(data, bins=nbins)
    mc_counts,_ = np.histogram(mc, bins=x_bin_edges)
    mc_event_counts,_ = np.histogram(mc, bins=x_bin_edges, weights= CMSQ_TO_MSQ*mc_weights*mc_norm)
    print "x_bin_edges = ", x_bin_edges
    bg_counts,_ = np.histogram(bg, bins=x_bin_edges, weights=bg_weights)
    fig = plt.figure(figsize=(8,8))
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    x_bin_centers = get_bin_centers(x_bin_edges)
    x_bin_width = get_bin_sizes(x_bin_edges)
    if norm==True:
        # plot data
        hist_data, x_bin_edges_3,_ = ax1.hist(data, bins=x_bin_edges, histtype='step',lw=2,color='k', normed=True, label='burn sample (0.045 yr)',log=logy)
        data_error = hist_data*(np.sqrt(data_counts)/data_counts)
        ax1.errorbar(x_bin_centers, hist_data, yerr=data_error,fmt='o',color='black')

        # normalize by the sum of the integral of mc and background 
        mc_bin_vals, mc_bin_edges = np.histogram(mc, weights = CMSQ_TO_MSQ*mc_weights*mc_norm, bins=x_bin_edges)
        bg_bin_vals, bg_bin_edges = np.histogram(bg, weights = bg_weights, bins=x_bin_edges)
        integral_mc = np.sum(np.diff(mc_bin_edges) * mc_bin_vals)
        integral_bg = np.sum(np.diff(bg_bin_edges) * bg_bin_vals)
        integral_total = integral_bg + integral_mc

        hist_bg, _, _ = ax1.hist(bg, bins=x_bin_edges, weights = bg_weights/integral_total, histtype='step', lw=2, color='b', label='ICC background',log=logy)
        bg_error = np.sqrt(bg_counts)/integral_total
        ax1.bar(x_bin_edges[:-1], 2*bg_error, bottom=hist_bg-bg_error, width=x_bin_width, color='b', alpha=0.25, linewidth=0)

        hist_mc, _, _ = ax1.hist(mc, weights = CMSQ_TO_MSQ*mc_weights*mc_norm/integral_total, bins=x_bin_edges, histtype='step', lw=2, color='g', label='MC nu',log=logy)
        mc_error = mc_event_counts/np.sqrt(mc_counts)/integral_total
        ax1.bar(x_bin_edges[:-1], 2*mc_error, bottom=hist_mc-mc_error, width=x_bin_width, color='g', alpha=0.25, linewidth=0)

        hist_mc_and_bg,_,_ = ax1.hist(x_bin_centers, weights= (hist_bg+hist_mc), bins=x_bin_edges, histtype='step', lw=2, color='r', linestyle='solid', label='nu + background',log=logy)
        mc_and_bg_error = np.sqrt(bg_error**2 + mc_error**2)
        ax1.bar(x_bin_edges[:-1], 2*mc_and_bg_error, bottom=hist_mc_and_bg - mc_and_bg_error, width=x_bin_width, color='r', alpha=0.25, linewidth=0)
    else:
        hist_data, x_bin_edges_3,_ = ax1.hist(data, bins=x_bin_edges, histtype='step',lw=2,color='k', normed=False, label='burn sample (0.045 yr)',log=logy)
        data_error = np.sqrt(data_counts)
        ax1.errorbar(x_bin_centers, hist_data, yerr=data_error,fmt='o',color='black')

        hist_bg,_,_ = ax1.hist(bg, bins=x_bin_edges, weights = bg_weights, histtype='step', lw=2, color='b', normed=False, label='ICC background',log=logy)
        bg_error = np.sqrt(bg_counts)
        ax1.bar(x_bin_edges[:-1], 2*bg_error, bottom=hist_bg-bg_error, width=x_bin_width, color='b', alpha=0.25, linewidth=0)

        hist_mc,_,_ = ax1.hist(mc, weights = CMSQ_TO_MSQ*mc_weights*mc_norm, bins=x_bin_edges, histtype='step', lw=2, color='g', normed=False, label='MC neutrino',log=logy)
        mc_error = mc_event_counts/np.sqrt(mc_counts)
        ax1.bar(x_bin_edges[:-1], 2*mc_error, bottom=hist_mc-mc_error, width=x_bin_width, color='g', alpha=0.25, linewidth=0)

        hist_mc_and_bg,_,_ = ax1.hist(x_bin_centers, weights= (hist_bg+hist_mc), bins=x_bin_edges, histtype='step', lw=2, color='r', linestyle='solid', label='nu + background',log=logy)
        mc_and_bg_error = np.sqrt(bg_error**2 + mc_error**2)
        ax1.bar(x_bin_edges[:-1], 2*mc_and_bg_error, bottom=hist_mc_and_bg - mc_and_bg_error, width=x_bin_width, color='r', alpha=0.25, linewidth=0)
   
    min_hist = min(np.min(hist_bg), np.min(hist_data))
    max_hist = max(np.max(hist_bg), np.max(hist_data))
    range_hist = max_hist - min_hist
    text = r'$\nu_\tau$ appearance' + '\nPrefit'
    a_text = AnchoredText(text, loc=2, frameon=False)
    ax1.add_artist(a_text)
    ax1.legend(loc='upper right',ncol=2, frameon=False,numpoints=1,fontsize=10)
    #ax1.set_ylim(min_hist - range_hist*0.05,max_hist + 0.3*range_hist)
    ax1.grid()

    ax2 = plt.subplot2grid((3,1), (2,0),sharex=ax1)
    ratio_data_to_data = np.ones(len(hist_data)) 
    #ax2.axhline(y=1,linewidth=1, color='k')
    hist_ratio_data_to_data,_,_ = ax2.hist(x_bin_centers, weights=ratio_data_to_data , bins=x_bin_edges,histtype='step',lw=2,color='g', linestyle='solid', label='nu /burn sample')
    for i in range(0, len(hist_data)):
        if hist_data[i]==0:
            ratio_data_to_data[i] = 0
    ratio_data_data_error = np.nan_to_num(data_error/hist_data)
    ax2.errorbar(x_bin_centers, ratio_data_to_data, yerr=ratio_data_data_error,fmt='o',color='black')

    # get ratio of nu/data, if data = 0, return ratio = 0
    ratio_mc_to_data = np.nan_to_num(hist_mc/hist_data)
    error_mc_to_data = np.nan_to_num(mc_error/hist_mc)
    for i in range(0, len(hist_data)):
        if hist_data[i]==0:
            ratio_mc_to_data[i] = 0
            error_mc_to_data[i] = 0
    hist_ratio_mc_to_data,_,_ = ax2.hist(x_bin_centers, weights=ratio_mc_to_data , bins=x_bin_edges,histtype='step',lw=2,color='g', linestyle='solid', label='nu /burn sample')
    ax2.bar(x_bin_edges[:-1], 2*error_mc_to_data, bottom=hist_ratio_mc_to_data-error_mc_to_data, width=x_bin_width, color='r', alpha=0.25, linewidth=0)

    # get ratio of (nu+bg)/data, if data = 0, return ratio = 0
    ratio_mc_bg_to_data = np.nan_to_num(hist_mc_and_bg/hist_data)
    ratio_mc_bg_to_data_error = np.nan_to_num(mc_and_bg_error/hist_data)
    for i in range(0, len(hist_data)):
        if hist_data[i]==0:
            ratio_mc_bg_to_data[i] = 0
            ratio_mc_bg_to_data_error[i] = 0
    hist_ratio_mc_bg_to_data,_,_ = ax2.hist(x_bin_centers, weights= ratio_mc_bg_to_data, bins=x_bin_edges,histtype='step',lw=2,color='r', linestyle='solid', label='(nu + background)/burn sample')
    ax2.bar(x_bin_edges[:-1], 2*ratio_mc_bg_to_data_error, bottom=hist_ratio_mc_bg_to_data-ratio_mc_bg_to_data_error, width=x_bin_width, color='r', alpha=0.25, linewidth=0)
    ax2.grid()
    ax2_max = max(np.max(error_mc_to_data+hist_ratio_mc_to_data), np.max(hist_ratio_mc_bg_to_data+ratio_mc_bg_to_data_error))
    print "ax2_max = ", ax2_max
    ax2.set_ylim(0.1, ax2_max)
    fig.subplots_adjust(hspace=0)
    plt.setp(ax1.get_xticklabels(), visible=False)


    if save:
        filename = os.path.join(outdir, fig_name + '.png')
        pdf_filename = os.path.join(outdir+'/pdf/', fig_name + '.pdf')
        plt.xlabel(x_label, fontsize='large')
        #plt.legend()
        plt.savefig(filename,dpi=150)
        plt.savefig(pdf_filename,dpi=150)
        plt.clf()

def plot_one_map(map_to_plot, outdir, logE, fig_title, fig_name, save, annotate_prcs=2, max=None, min=None):
    plt.figure()
    show_map(map_to_plot, vmin= min if min!=None else np.min(map_to_plot['map']),
            vmax= max if max!=None else np.max(map_to_plot['map']), logE=logE, annotate_prcs=annotate_prcs)
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

def plot_burn_sample_MC_comparison(MC_nutau, MC_no_nutau, BS_data, MC_nutau_name, MC_no_nutau_name, BS_name, x_bin_centers, x_bin_edges, channel, x_label):
    fig = plt.figure(figsize=(8,8))
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    hist_MC_tau,_,_ = ax1.hist(x_bin_centers,weights = MC_nutau,bins=x_bin_edges,histtype='step',lw=2,color='b',label= MC_nutau_name,linestyle='solid')
    hist_MC_notau,_,_ = ax1.hist(x_bin_centers,weights = MC_no_nutau,bins=x_bin_edges,histtype='step',lw=2,color='g',label= MC_no_nutau_name,linestyle='dashed')
    hist_BS,_= np.histogram(x_bin_centers,weights =BS_data,bins=x_bin_edges)
    #ax1.errorbar(x_bin_centers,hist_BS,yerr=np.sqrt(hist_BS),fmt='o',color='black',label='data')
    upperE = .5 + np.sqrt(hist_BS + .25)
    lowerE = -.5 + np.sqrt(hist_BS + .25)
    ax1.errorbar(x_bin_centers,hist_BS,yerr=[lowerE,upperE],fmt='o',color='black',label='data')
    #if (channel == 'cscd' or channel == 'cscd+trck') and x_label == 'energy':
    ax1.legend(loc='upper right',ncol=2, frameon=False,numpoints=1,fontsize=10)
    plt.title(r'${\rm 0.045 \, yr \, %s  }$'%(channel), fontsize='large')
    min_hist = min(np.min(hist_BS), np.min(hist_MC_notau), np.min(hist_MC_tau))
    max_hist = max(np.max(hist_BS), np.max(hist_MC_notau), np.max(hist_MC_tau))
    ax1.set_ylim(min_hist - min_hist*0.4,max_hist + 0.4*max_hist)
    ax1.set_ylabel('$\#$ events')
    ax1.grid()

    #poisson_llh = np.sum( hist_BS * np.log(hist_MC_tau)- gammaln(hist_BS+1) - hist_MC_tau)
    #print poisson_llh

    x2,_ = stats.chisquare(BS_data, f_exp=MC_nutau)
    x2_nutau = x2/len(BS_data)
    x2,_ = stats.chisquare(BS_data, f_exp=MC_no_nutau)
    x2_no_nutau = x2/len(BS_data)

    ax2 = plt.subplot2grid((3,1), (2,0),sharex=ax1)

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

    hist_ratio_MC_to_BS_tau = ax2.hist(x_bin_centers, weights =ratio_MC_to_BS_tau,bins=x_bin_edges,histtype='step',lw=2,color='b', linestyle='solid', label='MC tau/data')
    hist_ratio_MC_to_BS_notau = ax2.hist(x_bin_centers, weights =ratio_MC_to_BS_notau, bins=x_bin_edges,histtype='step',lw=2,color='g', linestyle='dashed', label = 'MC notau/data')

    if x_label == 'energy':
        ax2.set_xlabel('energy [GeV]')
    if x_label == 'coszen':
        ax2.set_xlabel('coszen')
    ax2.set_ylabel('ratio (MC/data)')
    ax2.set_ylim(min(min(ratio_MC_to_BS_notau),min(ratio_MC_to_BS_tau))-0.1,max(max(ratio_MC_to_BS_notau),max(ratio_MC_to_BS_tau))+0.2)
    ax2.axhline(y=1,linewidth=1, color='r')
    #ax2.legend(loc='upper center',ncol=1, frameon=False)
    ax2.legend(loc='upper right',ncol=2, frameon=False,numpoints=1,fontsize=10)
    ax2.grid()
    #fig.subplots_adjust(hspace=0)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.savefig(args.outdir+'BurnSample_MC_%s_%s_distribution_0217.png' % (channel, x_label),dpi=150)
    plt.clf()

def plot_1D_distribution_comparison(burn_sample_maps, fit_nutau, fit_no_nutau, png_name_root):

    print "Entering plot_1D_distribution_comparison "

    # get 1D energy (coszen) distribution
    burn_sample_cscd_map = burn_sample_maps['cscd']['map']
    burn_sample_trck_map = burn_sample_maps['trck']['map']

    BurnSample_RecoEnergy_cscd = get_1D_projection(burn_sample_cscd_map, 'energy')
    BurnSample_RecoEnergy_trck = get_1D_projection(burn_sample_trck_map, 'energy')
    BurnSample_RecoCoszen_cscd = get_1D_projection(burn_sample_cscd_map, 'coszen')
    BurnSample_RecoCoszen_trck = get_1D_projection(burn_sample_trck_map, 'coszen')

    fit_nutau_cscd_map = fit_nutau['cscd']['map']
    MC_RecoEnergy_fit_nutau_cscd = get_1D_projection(fit_nutau_cscd_map, 'energy')
    MC_RecoCoszen_fit_nutau_cscd = get_1D_projection(fit_nutau_cscd_map, 'coszen')
    fit_nutau_trck_map = fit_nutau['trck']['map']
    MC_RecoEnergy_fit_nutau_trck = get_1D_projection(fit_nutau_trck_map, 'energy')
    MC_RecoCoszen_fit_nutau_trck = get_1D_projection(fit_nutau_trck_map, 'coszen')

    fit_no_nutau_cscd_map = fit_no_nutau['cscd']['map']
    MC_RecoEnergy_fit_no_nutau_cscd = get_1D_projection(fit_no_nutau_cscd_map, 'energy')
    MC_RecoCoszen_fit_no_nutau_cscd = get_1D_projection(fit_no_nutau_cscd_map, 'coszen')
    fit_no_nutau_trck_map = fit_no_nutau['trck']['map']
    MC_RecoEnergy_fit_no_nutau_trck = get_1D_projection(fit_no_nutau_trck_map, 'energy')
    MC_RecoCoszen_fit_no_nutau_trck = get_1D_projection(fit_no_nutau_trck_map, 'coszen')

    MC_RecoEnergy_fit_nutau_all_chan = MC_RecoEnergy_fit_nutau_cscd + MC_RecoEnergy_fit_nutau_trck
    MC_RecoCoszen_fit_nutau_all_chan = MC_RecoCoszen_fit_nutau_cscd + MC_RecoCoszen_fit_nutau_trck

    MC_RecoEnergy_fit_no_nutau_all_chan = MC_RecoEnergy_fit_no_nutau_cscd + MC_RecoEnergy_fit_no_nutau_trck
    MC_RecoCoszen_fit_no_nutau_all_chan = MC_RecoCoszen_fit_no_nutau_cscd + MC_RecoCoszen_fit_no_nutau_trck

    BurnSample_RecoEnergy_all_chan = BurnSample_RecoEnergy_cscd + BurnSample_RecoEnergy_trck
    BurnSample_RecoCoszen_all_chan = BurnSample_RecoCoszen_cscd + BurnSample_RecoCoszen_trck

    # plot 1D energy (coszen) distribution
    plot_burn_sample_MC_comparison( MC_RecoEnergy_fit_nutau_cscd, MC_RecoEnergy_fit_no_nutau_cscd, BurnSample_RecoEnergy_cscd, 'MC cscd (nutau)', 'MC cscd (no nutau)', 'BurnSample cscd', E_bin_centers, anlys_ebins, 'cscd', png_name_root+'energy')

    plot_burn_sample_MC_comparison( MC_RecoCoszen_fit_nutau_cscd, MC_RecoCoszen_fit_no_nutau_cscd, BurnSample_RecoCoszen_cscd, 'MC cscd (nutau)', 'MC cscd (no nutau)', 'BurnSample cscd', CZ_bin_centers, czbins, 'cscd', png_name_root+'coszen')

    plot_burn_sample_MC_comparison( MC_RecoEnergy_fit_nutau_trck, MC_RecoEnergy_fit_no_nutau_trck, BurnSample_RecoEnergy_trck, 'MC trck (nutau)', 'MC trck (no nutau)', 'BurnSample trck', E_bin_centers, anlys_ebins, 'trck', png_name_root+'energy')

    plot_burn_sample_MC_comparison( MC_RecoCoszen_fit_nutau_trck, MC_RecoCoszen_fit_no_nutau_trck, BurnSample_RecoCoszen_trck, 'MC trck (nutau)', 'MC trck (no nutau)', 'BurnSample trck', CZ_bin_centers, czbins, 'trck', png_name_root+'coszen')

    plot_burn_sample_MC_comparison( MC_RecoEnergy_fit_nutau_all_chan, MC_RecoEnergy_fit_no_nutau_all_chan, BurnSample_RecoEnergy_all_chan, 'MC cscd+trck (nutau)', 'MC cscd+trck (no nutau)', 'BurnSample cscd+trck', E_bin_centers, anlys_ebins, 'cscd+trck', png_name_root+'energy')

    plot_burn_sample_MC_comparison( MC_RecoCoszen_fit_nutau_all_chan, MC_RecoCoszen_fit_no_nutau_all_chan, BurnSample_RecoCoszen_all_chan, 'MC cscd+trck (nutau)', 'MC cscd+trck (no nutau)', 'BurnSample cscd+trck', CZ_bin_centers, czbins, 'cscd+trck', png_name_root+'coszen')

    # make 1-d slices
    burn_sample_map = burn_sample_trck_map + burn_sample_cscd_map
    fit_nutau_map = fit_nutau_trck_map + fit_nutau_cscd_map
    fit_no_nutau_map = fit_no_nutau_trck_map + fit_no_nutau_cscd_map

    for i in range(0, burn_sample_trck_map.shape[0]):
        plot_burn_sample_MC_comparison( fit_nutau_trck_map[i,:], fit_no_nutau_trck_map[i,:], burn_sample_trck_map[i,:], 'MC trck (nutau)', 'MC trck (no nutau)', 'BurnSample trck', CZ_bin_centers, czbins, 'trck', png_name_root+'coszen_E_bin%s'%i)
        plot_burn_sample_MC_comparison( fit_nutau_cscd_map[i,:], fit_no_nutau_cscd_map[i,:], burn_sample_cscd_map[i,:], 'MC cscd (nutau)', 'MC cscd (no nutau)', 'BurnSample cscd', CZ_bin_centers, czbins, 'cscd', png_name_root+'coszen_E_bin%s'%i)
        plot_burn_sample_MC_comparison( fit_nutau_map[i,:], fit_no_nutau_map[i,:], burn_sample_map[i,:], 'MC cscd+trck (nutau)', 'MC cscd+trck (no nutau)', 'BurnSample cscd+trck', CZ_bin_centers, czbins, 'cscd+trck', png_name_root+'coszen_E_bin%s'%i)




if __name__ == '__main__':

    set_verbosity(0)
    parser = ArgumentParser(description='''Quick check if all components are working reasonably well, by
making the final level hierarchy asymmetry plots from the input
settings file. ''')
    parser.add_argument('--template_settings',metavar='JSON',
                        help='Settings file to use for template generation')
    parser.add_argument('--burn_sample_file',metavar='FILE',type=str,
                        default='burn_sample/Matt_L5b_burn_sample_IC86_2_to_4.hdf5',
                        help='''HDF5 File containing burn sample.'
                        inverted corridor cut data''')
    parser.add_argument('--sim_ver',metavar='str',default='',type=str,
                        help='''name of the simulation version, can only be '4digit' or '5digit'.''')
    parser.add_argument( '--nue', metavar='H5_FILE', type=str,
                        default='',required=True, help='nue HDF5 file(s)')
    parser.add_argument( '--numu', metavar='H5_FILE', type=str,
                        default='',required=True, help='numu HDF5 file(s)')
    parser.add_argument( '--nutau', metavar='H5_FILE', type=str,
                        default='',required=True, help='nutau HDF5 file(s)')
    parser.add_argument('--background_file',metavar='FILE',type=str,
                        default='background/Matt_L5b_icc_data_IC86_2_3_4.hdf5',
                        help='''HDF5 File containing atmospheric background from 3 years'
                        inverted corridor cut data''')
    parser.add_argument('--llr', '--llr-results', default=None, dest='fit_file_llr',required=True,
                        help='use post fit parameters from LLR fit result json file')
    parser.add_argument('--profile', '--profile-results', default=None, dest='fit_file_profile',required=True,
                        help='use post fit parameters from profile fit result json file')
    parser.add_argument('-y','--y',default=0.045,type=float,required=True,
                        help='No. of livetime[ unit: Julian year] for MC, for burn sample, default is 0.045.')
    parser.add_argument('-CMSQ_TO_MSQ','--CMSQ_TO_MSQ',action='store_true',default=False,
                        help='Use conversion from cm2 to m2 in sim_weights.')
    parser.add_argument('-no_logE','--no_logE',action='store_true',default=False,
                        help='Energy in log scale.')
    parser.add_argument('--plot_sanity_checks',action='store_true',default=False,
                        help='Plot MC/burn sample sanity checks (low level quantatities, defined in fields_to_plot')
    parser.add_argument('--plot_map_checks',action='store_true',default=False,
                        help='Plot burn sample maps, pisa maps, their ratio and delta maps.')
    parser.add_argument('--use_best_fit',action='store_true',default=False,
                        help='Use best fit params to calculate neutrino weights')
    parser.add_argument('--title',metavar='str',default='',
                        help='Title of the geometry or test in plots')
    parser.add_argument('--save',action='store_true',default=False,
                        help='Save plots in outdir.')
    parser.add_argument('-o','--outdir',metavar='DIR',default='./',required=True,
                        help='Directory to save the output figures.')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='set verbosity level')
    args = parser.parse_args()
    set_verbosity(args.verbose)

    ##################### Settings Preparation  #######################

    if args.CMSQ_TO_MSQ:
        CMSQ_TO_MSQ = 1.0e-4    # If the calcuation of neutrino weight, CMSQ_TO_MSQ is not added, then need to use it here
    else:
        CMSQ_TO_MSQ = 1.0

    # fields to plot 
    fields_to_plot = ['reco_x', 'reco_y', 'reco_z', 'reco_azimuth', 'reco_trck_len', 'reco_trck_zenith', 'reco_trck_azimuth', 'reco_energy', 'reco_coszen', 
            'C2QR6', 'CausalVetoHits', 'CausalVetoPE', 'DCFiducialPE', 'ICVetoPE', 'NAbove200PE', 'NchCleaned', 'NoiseEngine', 'STW9000_DTW300Hits',
            'STW9000_DTW300PE', 'VertexGuessX', 'VertexGuessY', 'VertexGuessZ', 'rt_fid_charge', 'cog_sigma_time', 'cog_sigma_z', 'num_hit_doms',
            'dcc_veto_charge', 'vich_charge', 'cog_q1_rho', 'cog_q1_z', 'first_hlc_rho', 'first_hlc_z', 'interval', 'C2QR3', 'QR3', 'linefit_speed',
            'linefit_zenith', 'separation',   'total_charge', 'santa_direct_charge',
            'mn_start_contained', 'mn_stop_contained', 'x_prime', 'y_prime', 'z_prime', 't_prime', 'r_prime', 'rho_prime', 'theta_prime', 'phi_prime',
            'stop_x_prime', 'stop_y_prime', 'stop_z_prime', 'stop_r_prime', 'stop_rho_prime', 'stop_theta_prime', 'stop_phi_prime',
            'santa_direct_doms', 'spe11_zenith', 'pid']
            #'corridor_doms_over_threshold']

    # creat out dir
    utils.mkdir(args.outdir)
    utils.mkdir(args.outdir+'/pdf/')

    # read input settings file 
    template_settings = from_json(args.template_settings)
    template_settings['params']['icc_bg_file']['value'] = find_resource(args.background_file)
    template_settings['params']['livetime']['value'] = args.y

    # get nutau_template_settings which has systematics value from 
    # fit with nutauCC norm = 1
    nutau_template_settings = copy.deepcopy(template_settings)
    fit_file_llr = from_json(find_resource(args.fit_file_llr))
    syslist = fit_file_llr['trials'][0]['fit_results'][1].keys()
    for sys in syslist:
        if not sys == 'llh':
            val = fit_file_llr['trials'][0]['fit_results'][1][sys][0]
            if sys == 'theta23' or sys =='deltam23' or sys =='deltam31':
                sys += '_nh'
            print 'fit nutauCCnorm==1, %s at %.4f'%(sys,val)
            nutau_template_settings['params'][sys]['value'] = val

    # get nutau_template_settings which has systematics value from 
    # fit with nutauCC norm = 0
    no_nutau_template_settings = copy.deepcopy(template_settings)
    no_nutau_template_settings['params']['nutau_norm']['value'] = 0.0 
    syslist = fit_file_llr['trials'][0]['fit_results'][0].keys()
    for sys in syslist:
        if not sys == 'llh':
            val = fit_file_llr['trials'][0]['fit_results'][0][sys][0]
            if sys == 'theta23' or sys =='deltam23' or sys =='deltam31':
                sys += '_nh'
            print 'fit nutauCCnorm==0, %s at %.4f'%(sys,val)
            no_nutau_template_settings['params'][sys]['value'] = val

    # get nutau_template_settings which has systematics value from 
    # fit with nutauCC norm = free floating 
    free_nutau_template_settings = copy.deepcopy(template_settings)
    fit_file_profile = from_json(find_resource(args.fit_file_profile))
    syslist = fit_file_profile['trials'][0]['fit_results'][1].keys()
    for sys in syslist:
        if not sys == 'llh':
            val = fit_file_profile['trials'][0]['fit_results'][1][sys][0]
            if sys == 'theta23' or sys =='deltam23' or sys =='deltam31':
                sys += '_nh'
            print 'fit nutauCCnorm=free, %s at %.4f'%(sys,val)
            free_nutau_template_settings['params'][sys]['value'] = val
    if args.use_best_fit:
        # TODO
        # if use the best fit systematics (free floating nutauCC norm), right now, only uses atmos_mu_scale and
        # aeff_scale in the plot_param_distribution() function, need to add other systematics
        atmos_mu_scale = free_nutau_template_settings['params']['atmos_mu_scale']['value']
        aeff_scale = free_nutau_template_settings['params']['aeff_scale']['value']
        print "use best fit, aeff_scale = ", aeff_scale
        print "use best fit, atmos_mu_scale = ", atmos_mu_scale
    else:
        atmos_mu_scale = template_settings['params']['atmos_mu_scale']['value']
        aeff_scale = template_settings['params']['aeff_scale']['value']
        print "use nominal MC, aeff_scale = ", aeff_scale
        print "use nominal MC, atmos_mu_scale = ", atmos_mu_scale

    # get binning info
    ebins = template_settings['binning']['ebins']
    anlys_ebins = template_settings['binning']['anlys_ebins']
    czbins = template_settings['binning']['czbins']
    CZ_bin_centers = get_bin_centers(czbins)
    E_bin_centers = get_bin_centers(anlys_ebins)

    ##################### Compare Burn Sample and MC  #######################

    print "Plotting burn sample sanity checks..."
    if args.plot_sanity_checks:
        # get burn sample info
        print "     getting fields for burn sample..."
        bs_low_level = get_low_level_quantities(file_name = args.burn_sample_file, file_type='data', anlys_ebins= anlys_ebins, czbins= czbins, fields=fields_to_plot, sim_version=args.sim_ver)
        print "     getting fields for bg..."
        bg_low_level = get_low_level_quantities(file_name = args.background_file, file_type='data', anlys_ebins= anlys_ebins, czbins= czbins, fields=fields_to_plot, sim_version=args.sim_ver, cuts=[])
        print "     getting fields for MC neutrino..."
        mc_nue_low_level = get_low_level_quantities(file_name = args.nue, file_type='mc', anlys_ebins= anlys_ebins, czbins= czbins, fields=fields_to_plot, sim_version=args.sim_ver)
        mc_numu_low_level = get_low_level_quantities(file_name = args.numu, file_type='mc', anlys_ebins= anlys_ebins, czbins= czbins, fields=fields_to_plot, sim_version=args.sim_ver)
        mc_nutau_low_level = get_low_level_quantities(file_name = args.nutau, file_type='mc', anlys_ebins= anlys_ebins, czbins= czbins, fields=fields_to_plot, sim_version=args.sim_ver)

        # get neutrino weights, which is calculated via add_weight.py
        file_nue = h5py.File(find_resource(args.nue),'r')
        file_numu = h5py.File(find_resource(args.numu),'r')
        file_nutau = h5py.File(find_resource(args.nutau),'r')
        nue_cut_L6 = file_nue['IC86_Dunkman_L6']['result'] == 1 
        numu_cut_L6 = file_numu['IC86_Dunkman_L6']['result'] == 1 
        nutau_cut_L6 = file_nutau['IC86_Dunkman_L6']['result'] == 1 
        if args.use_best_fit:
            mc_nue_weight = file_nue['neutrino_weight_best_fit'][nue_cut_L6]
            mc_numu_weight = file_numu['neutrino_weight_best_fit'][numu_cut_L6]
            mc_nutau_weight = file_nutau['neutrino_weight_best_fit'][nutau_cut_L6]
        else:
            mc_nue_weight = file_nue['neutrino__weight'][nue_cut_L6]
            mc_numu_weight = file_numu['neutrino__weight'][numu_cut_L6]
            mc_nutau_weight = file_nutau['neutrino__weight'][nutau_cut_L6]
        for field_name, bs_field_content in bs_low_level.items():
            print "     field_name = ", field_name
            mc_nue_field_content = mc_nue_low_level[field_name]
            mc_numu_field_content = mc_numu_low_level[field_name]
            mc_nutau_field_content = mc_nutau_low_level[field_name]
            mc_field_content = np.concatenate([mc_nue_field_content, mc_numu_field_content, mc_nutau_field_content])
            mc_field_weight = np.concatenate([mc_nue_weight, mc_numu_weight, mc_nutau_weight])
            bg_field_content = bg_low_level[field_name]
            atmos_mu_scale = free_nutau_template_settings['params']['atmos_mu_scale']['value']
            logy = False
            if field_name in ['reco_energy', 'DCFiducialPE', 'ICVetoPE', 'linefit_speed', 'NchCleaned', 'num_hit_doms', 'pid', 'reco_trck_len', 'rt_fid_charge', 'santa_direct_charge', 'santa_direct_doms', 'STW9000_DTW300PE', 't_prime', 'total_charge']:
                # use logy for these fields
                logy = True

            print "     plotting distributions..." 
            # plot distributions without normalization
            # TODO, use livetime=0.045 yr for MC gives fewer events than burn sample, need to add systematics in to show the un-normalized histogram comparison ( i.e. event rate comparison)
            plot_param_distribution(bs_field_content, bg_field_content, mc_field_content, mc_field_weight, nbins=30, outdir=args.outdir, x_label=field_name, fig_name='burn_sample_%s_distribution'%(field_name), save=args.save, norm=False, livetime=args.y, atmos_mu_scale=atmos_mu_scale, aeff_scale=aeff_scale, logy=logy)
            # plot distributions with normalization
            plot_param_distribution(bs_field_content, bg_field_content, mc_field_content, mc_field_weight, nbins=30, outdir=args.outdir, x_label=field_name, fig_name='normalized_burn_sample_%s_distribution'%(field_name), save=args.save, norm=True, livetime=args.y, atmos_mu_scale=atmos_mu_scale, aeff_scale=aeff_scale, logy=logy)



    ##################### Plot Burn Sample maps and PISA maps #######################
    if args.plot_map_checks==True:
        # get burn sample maps
        burn_sample_maps = get_burn_sample_maps(file_name= args.burn_sample_file, anlys_ebins= anlys_ebins, czbins= czbins, output_form ='map', channel=template_settings['params']['channel']['value'])
        # plot burn sample maps
        for flav in ['cscd', 'trck']:
            plot_one_map(burn_sample_maps[flav], args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_burn_sample_%s'% (flav), fig_title=r'${\rm %s \, yr \, burn \, sample \, %s \, (Nevts: \, %.1f) }$'%(args.y, flav, np.sum(burn_sample_maps[flav]['map'])), save=args.save, max=15 if flav=='cscd' else 10, annotate_prcs=0)

        # initialize template settings
        fit_nutau_template_settings = copy.deepcopy(nutau_template_settings)
        fit_no_nutau_template_settings = copy.deepcopy(no_nutau_template_settings)
        fit_free_nutau_template_settings = copy.deepcopy(free_nutau_template_settings)
        with Timer() as t:
            fit_nutau_template_maker = TemplateMaker(get_values(fit_nutau_template_settings['params']), **fit_nutau_template_settings['binning'])
            fit_no_nutau_template_maker = TemplateMaker(get_values(fit_no_nutau_template_settings['params']), **fit_no_nutau_template_settings['binning'])
            fit_free_nutau_template_maker = TemplateMaker(get_values(fit_free_nutau_template_settings['params']), **fit_free_nutau_template_settings['binning'])
        profile.info('==> elapsed time to initialize templates: %s sec'%t.secs)

        # Make nutau templates: 1) fit_nutau, syst. = best fit; nutauCC norm = 1; 
        #                       2) fit_no_nutau, syst. = best fit; nutauCC norm = 0; 
        #                       3) fit_free_nutau, syst. = best fit; nutauCC norm = free
        fit_nutau_params = copy.deepcopy(select_hierarchy_and_nutau_norm(fit_nutau_template_settings['params'],True,1.0))
        fit_no_nutau_params = copy.deepcopy(select_hierarchy_and_nutau_norm(fit_no_nutau_template_settings['params'],True,0.0))
        fit_free_nutau_params = copy.deepcopy(select_hierarchy(fit_no_nutau_template_settings['params'],True))

        with Timer(verbose=False) as t:
            fit_nutau = fit_nutau_template_maker.get_template(get_values(fit_nutau_params),return_stages=False, no_sys_applied= False)
            fit_no_nutau = fit_no_nutau_template_maker.get_template(get_values(fit_no_nutau_params),return_stages=False, no_sys_applied= False)
            fit_free_nutau = fit_free_nutau_template_maker.get_template(get_values(fit_free_nutau_params),return_stages=False, no_sys_applied= False)
        profile.info('==> elapsed time to get NUTAU template: %s sec'%t.secs)

        #print "fit_nutau cscd = "
        #print fit_nutau['cscd']
        #print "\n"

        #print "fit_no_nutau cscd = "
        #print fit_no_nutau['cscd']
        #print "\n"

        #print "fit_free_nutau cscd = "
        #print fit_free_nutau['cscd']
        #print "\n"

        ################## PLOT MC/Data comparison ##################

        # plot 1D distribution comparison ( burn sample maps and fit_nutau maps)

        plot_1D_distribution_comparison(burn_sample_maps, fit_nutau,  fit_no_nutau, png_name_root = 'whole_region_')

        # Plot nominal PISA template (cscd and trck separately), and the ratio of burn sample to PISA template 
        for flav in ['cscd', 'trck']:
            plot_one_map(fit_nutau[flav], args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_%s_yr_NutauCCNorm_1_%s'% (args.y, flav), fig_title=r'${\rm %s \, yr \, PISA \, %s \, (\nu_{\tau} \, CC \, = \, 1 \, Nevts: \, %.1f) }$'%(args.y, flav, np.sum(fit_nutau[flav]['map'])), save=args.save, max=10+np.max(fit_nutau[flav]['map']), annotate_prcs=1)
            plot_one_map(fit_no_nutau[flav], args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_%s_yr_NutauCCNorm_0_%s'% (args.y, flav), fig_title=r'${\rm %s \, yr \, PISA \, (\nu_{\tau} \, CC \, = \, 0 \, %s \, Nevts: \, %.1f) }$'%(args.y, flav, np.sum(fit_no_nutau[flav]['map'])), save=args.save, max=10+np.max(fit_no_nutau[flav]['map']), annotate_prcs=1)
            #plot_one_map(fit_free_nutau[flav], args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_%s_yr_NutauCCNorm_0_%s'% (args.y, flav), fig_title=r'${\rm %s \, yr \, PISA \, (\nu_{\tau} \, CC \, = \, %s \, %s \, Nevts: \, %.1f) }$'%(args.y, flav, np.sum(fit_free_nutau[flav]['map'])), save=args.save, max=10+np.max(fit_free_nutau[flav]['map']), annotate_prcs=1)

            delta_pid_pisa_pid_bs = delta_map(burn_sample_maps[flav], fit_nutau[flav])
            plot_one_map(delta_pid_pisa_pid_bs, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_%s_yr_NutauCCNorm_1_PISA_BurnSample_Delta_%s'% (args.y, flav), fig_title=r'${\rm Difference \, of \, %s \, map \, (BurnSample \, - \, PISA \, , \nu_{\tau} \, CC \, = \, 1 \, Nevts: \, %.1f)}$'%(flav, np.sum(delta_pid_pisa_pid_bs['map'])), save=args.save)

            ratio_pid_pisa_pid_bs = ratio_map(burn_sample_maps[flav], fit_nutau[flav])
            plot_one_map(ratio_pid_pisa_pid_bs, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_%s_yr_NutauCCNorm_1_PISA_BurnSample_Ratio_%s'% (args.y, flav), fig_title=r'${\rm Ratio \, of \, %s \, map \, (BurnSample \, / \, PISA \, , \nu_{\tau} \, CC \, = \, 1 )}$'%(flav), save=args.save)


        # plot the cscd + trck maps and the ratio of burn sample to PISA map
        fit_nutau_no_pid = sum_map(fit_nutau['cscd'], fit_nutau['trck'])
        fit_no_nutau_no_pid = sum_map(fit_no_nutau['cscd'], fit_no_nutau['trck'])
        burn_sample_maps_no_pid = sum_map(burn_sample_maps['cscd'], burn_sample_maps['trck'])
        ratio_bs_pisa_no_pid = ratio_map( burn_sample_maps_no_pid, fit_nutau_no_pid)
        delta_bs_pisa_no_pid = delta_map( burn_sample_maps_no_pid, fit_nutau_no_pid)

        plot_one_map(fit_nutau_no_pid, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_all_channel_%s_yr_NutauCCNorm_1'% (args.y), fig_title=r'${\rm %s \, yr \, PISA \, cscd \, + \, trck \, (\nu_{\tau} \, CC \, = \, 1 \, Nevts: \, %.1f) }$'%(args.y, np.sum(fit_nutau_no_pid['map'])), save=args.save, max=10+np.max(fit_nutau_no_pid['map']), annotate_prcs=1)
        plot_one_map(fit_no_nutau_no_pid, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_all_channel_%s_yr_NutauCCNorm_0'% (args.y), fig_title=r'${\rm %s \, yr \, PISA \, cscd \, + \, trck \, (\nu_{\tau} \, CC \, = \, 0 \, Nevts: \, %.1f) }$'%(args.y, np.sum(fit_no_nutau_no_pid['map'])), save=args.save, max=10+np.max(fit_no_nutau_no_pid['map']), annotate_prcs=1)
        plot_one_map(burn_sample_maps_no_pid, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_all_channel_%s_yr_BurnSample'% (args.y), fig_title=r'${\rm %s \, yr \, Burn \, Sample \, cscd \, + \, trck \, (Nevts: \, %.1f) }$'%(args.y, np.sum(burn_sample_maps_no_pid['map'])), save=args.save, max=10+np.max(burn_sample_maps_no_pid['map']), annotate_prcs=0)
        plot_one_map(delta_bs_pisa_no_pid, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_%s_yr_NutauCCNorm_1_PISA_BurnSample_Delta_all_channel'% (args.y), fig_title=r'${\rm Difference \, of \, cscd \, + trck \, map \, (BurnSample \, - \, PISA \, , \,\nu_{\tau} \, CC \, = \, 1 )}$', save=args.save, annotate_prcs=2)
        plot_one_map(ratio_bs_pisa_no_pid, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_%s_yr_NutauCCNorm_1_PISA_BurnSample_Ratio_all_channel'% (args.y), fig_title=r'${\rm Ratio \, of \, cscd \, + trck \, map \, (BurnSample \, / \, PISA \, , \, \nu_{\tau} \, CC \, = \, 1 )}$', save=args.save, annotate_prcs=2)

        print 'no. of fit_nutau_cscd = ', np.sum(fit_nutau['cscd']['map'])
        print 'no. of fit_nutau_trck = ', np.sum(fit_nutau['trck']['map'])
        print ' total of the above two : ', np.sum(fit_nutau['cscd']['map'])+np.sum(fit_nutau['trck']['map'])
        print ' \n'
        print 'no. of fit_no_nutau_cscd = ', np.sum(fit_no_nutau['cscd']['map'])
        print 'no. of fit_no_nutau_trck = ', np.sum(fit_no_nutau['trck']['map'])
        print ' total of the above two : ', np.sum(fit_no_nutau['cscd']['map'])+np.sum(fit_no_nutau['trck']['map'])
        print ' \n'

        print "no. of burn_sample_cscd = ", np.sum(burn_sample_maps['cscd']['map'])
        print "no. of burn_sample_trck = ", np.sum(burn_sample_maps['trck']['map'])
        print ' total of the above two : ', np.sum(burn_sample_maps['cscd']['map'])+ np.sum(burn_sample_maps['trck']['map'])
        print ' \n'
   
    if not args.save: plt.show()
    else: print '\n-->>Saved all files to: ',args.outdir



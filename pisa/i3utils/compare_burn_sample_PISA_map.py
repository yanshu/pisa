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
from pisa.analysis.TemplateMaker_nutau import TemplateMaker
from pisa.utils.params import get_values, select_hierarchy_and_nutau_norm, select_hierarchy
import pisa.utils.utils as utils
import pisa.analysis.stats.Maps as Maps
from pisa.analysis.stats.Maps_nutau import get_burn_sample
from pisa.utils.log import set_verbosity,logging,profile
from pisa.resources.resources import find_resource
from pisa.utils.utils import Timer, is_linear, is_logarithmic, get_bin_centers
from pisa.utils.jsons import from_json
from pisa.utils.plot import show_map, sum_map, ratio_map, delta_map
from pisa.background.BackgroundServiceICC_nutau import BackgroundServiceICC

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
    hist_MC_tau,_,_ = ax1.hist(x_bin_centers,weights= MC_nutau,bins=x_bin_edges,histtype='step',lw=2,color='b',label= MC_nutau_name,linestyle='solid',normed=args.norm)
    hist_MC_notau,_,_ = ax1.hist(x_bin_centers,weights=MC_no_nutau,bins=x_bin_edges,histtype='step',lw=2,color='g',label= MC_no_nutau_name,linestyle='dashed',normed=args.norm)
    hist_BS,_= np.histogram(x_bin_centers,weights=BS_data,bins=x_bin_edges)
    #ax1.errorbar(x_bin_centers,hist_BS,yerr=np.sqrt(hist_BS),fmt='o',color='black',label='data')
    upperE = .5 + np.sqrt(hist_BS + .25)
    lowerE = -.5 + np.sqrt(hist_BS + .25)
    ax1.errorbar(x_bin_centers,hist_BS,yerr=[lowerE,upperE],fmt='o',color='black',label='data')
    #if (channel == 'cscd' or channel == 'cscd+trck') and x_label == 'energy':
    ax1.legend(loc='upper right',ncol=1, frameon=False,numpoints=1)
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

    hist_ratio_MC_to_BS_tau = ax2.hist(x_bin_centers, weights=ratio_MC_to_BS_tau,bins=x_bin_edges,histtype='step',lw=2,color='b', linestyle='solid', label='MC tau/data')
    hist_ratio_MC_to_BS_notau = ax2.hist(x_bin_centers, weights=ratio_MC_to_BS_notau, bins=x_bin_edges,histtype='step',lw=2,color='g', linestyle='dashed', label = 'MC notau/data')

    if x_label == 'energy':
        ax2.set_xlabel('energy [GeV]')
    if x_label == 'coszen':
        ax2.set_xlabel('coszen')
    ax2.set_ylabel('ratio (MC/data)')
    ax2.set_ylim(min(min(ratio_MC_to_BS_notau),min(ratio_MC_to_BS_tau))-0.1,max(max(ratio_MC_to_BS_notau),max(ratio_MC_to_BS_tau))+0.2)
    ax2.axhline(y=1,linewidth=1, color='r')
    #ax2.legend(loc='upper center',ncol=1, frameon=False)
    a_text = AnchoredText('nutau x2/NDF=%.2f\nno nutau x2/NDF=%.2f'%(x2_nutau,x2_no_nutau), loc=2)
    ax2.add_artist(a_text)
    ax2.grid()
    fig.subplots_adjust(hspace=0)
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
    parser.add_argument('--background_file',metavar='FILE',type=str,
                        default='background/Matt_L5b_icc_data_IC86_2_3_4.hdf5',
                        help='''HDF5 File containing atmospheric background from 3 years'
                        inverted corridor cut data''')
    parser.add_argument('-y','--y',default=0.045,type=float,
                        help='No. of livetime[ unit: Julian year]')
    parser.add_argument('-no_logE','--no_logE',action='store_true',default=False,
                        help='Energy in log scale.')
    parser.add_argument('--plot_background',action='store_true',default=False,
                        help='Plot background(from ICC data)')
    parser.add_argument('-plot_sigl_side','--plot_sigl_side',action='store_true',default=False,
                        help='''Plot signal and side region defined by'
                        sgnl_side_region_selection.json which is produced by running template_check_f_1.py''')
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
    parser.add_argument('--llr', '--llr-results', default=None, dest='fit_file_llr',required=True,
                        help='use post fit parameters from LLR fit result json file')
    parser.add_argument('--profile', '--profile-results', default=None, dest='fit_file_profile',required=True,
                        help='use post fit parameters from profile fit result json file')
    args = parser.parse_args()
    set_verbosity(args.verbose)

    # creat out dir
    utils.mkdir(args.outdir)
    utils.mkdir(args.outdir+'/pdf/')

    # get settings file for nutau norm = 1
    template_settings = from_json(args.template_settings)
    template_settings['params']['icc_bg_file']['value'] = find_resource(args.background_file)
    template_settings['params']['livetime']['value'] = args.y

    # replace with parameters determined in fit
    fit_file_llr = from_json(find_resource(args.fit_file_llr))
    syslist = fit_file_llr['trials'][0]['fit_results'][1].keys()
    for sys in syslist:
        if not sys == 'llh':
            val = fit_file_llr['trials'][0]['fit_results'][1][sys][0]
            if sys == 'theta23' or sys =='deltam23' or sys =='deltam31':
                sys += '_nh'
            print '%s at %.4f'%(sys,val)
            template_settings['params'][sys]['value'] = val

    # get settings file for nutau norm = 0
    no_nutau_template_settings = copy.deepcopy(template_settings)
    no_nutau_template_settings['params']['nutau_norm']['value'] = 0.0 

    # replace with parameters determined in fit
    syslist = fit_file_llr['trials'][0]['fit_results'][0].keys()
    for sys in syslist:
        if not sys == 'llh':
            val = fit_file_llr['trials'][0]['fit_results'][0][sys][0]
            if sys == 'theta23' or sys =='deltam23' or sys =='deltam31':
                sys += '_nh'
            print '%s at %.4f'%(sys,val)
            no_nutau_template_settings['params'][sys]['value'] = val

    # get settings file for nutau norm = free
    free_nutau_template_settings = copy.deepcopy(template_settings)

    # replace with parameters determined in fit
    #fit_file_profile = from_json(args.fit_file_profile)
    fit_file_profile = from_json(find_resource(args.fit_file_profile))
    syslist = fit_file_profile['trials'][0]['fit_results'][1].keys()
    for sys in syslist:
        if not sys == 'llh':
            val = fit_file_profile['trials'][0]['fit_results'][1][sys][0]
            if sys == 'theta23' or sys =='deltam23' or sys =='deltam31':
                sys += '_nh'
            print '%s at %.4f'%(sys,val)
            free_nutau_template_settings['params'][sys]['value'] = val

    # get binning info
    ebins = template_settings['binning']['ebins']
    anlys_ebins = template_settings['binning']['anlys_ebins']
    czbins = template_settings['binning']['czbins']

    print 'ebins = ', ebins
    print 'czbins = ', czbins
    CZ_bin_centers = get_bin_centers(czbins)
    E_bin_centers = get_bin_centers(anlys_ebins)
    #print 'E_bin_centers = ', E_bin_centers
    #print 'CZ_bin_centers = ', CZ_bin_centers

    burn_sample_maps = get_burn_sample(burn_sample_file= args.burn_sample_file, anlys_ebins= anlys_ebins, czbins= czbins, output_form ='map', cut_level='L6', channel=template_settings['params']['channel']['value'])

    if args.plot_sigl_side:
        sgnl_burn_sample_maps = get_burn_sample(burn_sample_file= args.burn_sample_file, anlys_ebins= anlys_ebins, czbins= czbins, output_form ='sgnl_map', cut_level='L6', channel=template_settings['params']['channel']['value'])
        side_burn_sample_maps = get_burn_sample(burn_sample_file= args.burn_sample_file, anlys_ebins= anlys_ebins, czbins= czbins, output_form ='side_map', cut_level='L6', channel=template_settings['params']['channel']['value'])

    # plot burn sample maps
    for flav in ['cscd', 'trck']:
        plot_one_map(burn_sample_maps[flav], args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_burn_sample_%s'% (flav), fig_title=r'${\rm %s \, yr \, burn \, sample \, %s \, (Nevts: \, %.1f) }$'%(args.y, flav, np.sum(burn_sample_maps[flav]['map'])), save=args.save, max=15 if flav=='cscd' else 10, annotate_prcs=0)


    ##################### Plot MC expectation #######################

    fit_nutau_template_settings = copy.deepcopy(template_settings)
    fit_no_nutau_template_settings = copy.deepcopy(no_nutau_template_settings)
    fit_free_nutau_template_settings = copy.deepcopy(free_nutau_template_settings)
    with Timer() as t:
        fit_nutau_template_maker = TemplateMaker(get_values(fit_nutau_template_settings['params']), **fit_nutau_template_settings['binning'])
        fit_no_nutau_template_maker = TemplateMaker(get_values(fit_no_nutau_template_settings['params']), **fit_no_nutau_template_settings['binning'])
        fit_free_nutau_template_maker = TemplateMaker(get_values(fit_free_nutau_template_settings['params']), **fit_free_nutau_template_settings['binning'])
    profile.info('==> elapsed time to initialize templates: %s sec'%t.secs)

    # Make nutau template:
    fit_nutau_params = copy.deepcopy(select_hierarchy_and_nutau_norm(fit_nutau_template_settings['params'],True,1.0))
    fit_no_nutau_params = copy.deepcopy(select_hierarchy_and_nutau_norm(fit_no_nutau_template_settings['params'],True,0.0))
    fit_free_nutau_params = copy.deepcopy(select_hierarchy(fit_no_nutau_template_settings['params'],True))

    with Timer(verbose=False) as t:
        fit_nutau = fit_nutau_template_maker.get_template(get_values(fit_nutau_params),return_stages=args.all, no_sys_applied= False)
        fit_no_nutau = fit_no_nutau_template_maker.get_template(get_values(fit_no_nutau_params),return_stages=args.all, no_sys_applied= False)
        fit_free_nutau = fit_free_nutau_template_maker.get_template(get_values(fit_free_nutau_params),return_stages=args.all, no_sys_applied= False)
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

    # plot 1D distribution comparison

    plot_1D_distribution_comparison(burn_sample_maps, fit_nutau,  fit_no_nutau, png_name_root = 'whole_region_')

    # Plot nominal PISA template (cscd and trck separately), and the ratio of burn sample to PISA template 
    for flav in ['cscd', 'trck']:
        plot_one_map(fit_nutau[flav], args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_%s_yr_NutauCCNorm_1_%s'% (args.y, flav), fig_title=r'${\rm %s \, yr \, PISA \, %s \, (\nu_{\tau} \, CC \, = \, 1 \, Nevts: \, %.1f) }$'%(args.y, flav, np.sum(fit_nutau[flav]['map'])), save=args.save, max=10+np.max(fit_nutau[flav]['map']), annotate_prcs=1)
        plot_one_map(fit_no_nutau[flav], args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_%s_yr_NutauCCNorm_0_%s'% (args.y, flav), fig_title=r'${\rm %s \, yr \, PISA \, (\nu_{\tau} \, CC \, = \, 0 \, %s \, Nevts: \, %.1f) }$'%(args.y, flav, np.sum(fit_no_nutau[flav]['map'])), save=args.save, max=10+np.max(fit_no_nutau[flav]['map']), annotate_prcs=1)

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
   

    ################ Plot background ##################
    # get background
    if args.plot_background:
        up_background_service = BackgroundServiceICC(fit_up_template_settings['binning']['anlys_ebins'],czbins[czbins<=0],**get_values(fit_nutau_up_params))
        up_background_dict = up_background_service.get_icc_bg()
        down_background_service = BackgroundServiceICC(fit_down_template_settings['binning']['anlys_ebins'],czbins[czbins>=0],**get_values(fit_nutau_down_params))
        down_background_dict = down_background_service.get_icc_bg()

        up_background_maps = {'params': fit_nutau_up['params']}
        for flav in ['trck','cscd']:
            up_background_maps[flav] = {'map':up_background_dict[flav],
                                     'ebins':fit_up_template_settings['binning']['anlys_ebins'],
                                     'czbins':czbins[czbins<=0]}
        down_background_maps = {'params': fit_nutau_down['params']}
        for flav in ['trck','cscd']:
            down_background_maps[flav] = {'map':down_background_dict[flav],
                                     'ebins':fit_down_template_settings['binning']['anlys_ebins'],
                                     'czbins':czbins[czbins>=0]}

        for flav in ['trck','cscd']:
            plt.figure()
            show_map(up_background_maps[flav],logE=not(args.no_logE),annotate_prcs=0)
            if args.save:
                filename = os.path.join(args.outdir,args.title+'_%s_yr_upgoing_background_' % (args.y) +flav+'.png')
                plt.title(r'${\rm %s \, yr \, background \, %s \, (Nevts: \, %.1f) }$'%(args.y, flav, np.sum(up_background_maps[flav]['map'])), fontsize='large')
                plt.savefig(filename,dpi=150)
                plt.clf()
            plt.figure()
            show_map(down_background_maps[flav],logE=not(args.no_logE),annotate_prcs=0)
            if args.save:
                filename = os.path.join(args.outdir,args.title+'_%s_yr_downgoing_background_' % (args.y) +flav+'.png')
                plt.title(r'${\rm %s \, yr \, background \, %s \, (Nevts: \, %.1f) }$'%(args.y, flav, np.sum(down_background_maps[flav]['map'])), fontsize='large')
                plt.savefig(filename,dpi=150)
                plt.clf()
        print 'no. of background up-going = ', np.sum(up_background_maps['cscd']['map'])+ np.sum(up_background_maps['trck']['map'])
        print 'no. of background down-going = ', np.sum(down_background_maps['cscd']['map'])+ np.sum(down_background_maps['trck']['map'])
        print 'total of the above two : ', np.sum(up_background_maps['cscd']['map'])+ np.sum(up_background_maps['trck']['map'])+np.sum(down_background_maps['cscd']['map'])+ np.sum(down_background_maps['trck']['map']) 


    ################## PLOT MC/Data comparison in signal and side regions separately ##################

    if args.plot_sigl_side:

        f_select = from_json('sgnl_side_region_selection.json')
        region_fit_nutau = {}
        region_fit_no_nutau = {}
        region_fit_nutau_minus_no_nutau = {}
        for region in ['sgnl', 'side']:
            region_fit_nutau[region] = {}
            region_fit_nutau_minus_no_nutau[region] = {}
            region_fit_no_nutau[region] = {}
            for flav in ['trck','cscd']:
                #print "region, flav" , region, " ", flav, " f_select[region][flav] = " , f_select[region][flav]
                region_fit_nutau[region][flav] = {'map':fit_nutau[flav]['map'] * f_select[region][flav],
                                                      'ebins':anlys_ebins,
                                                      'czbins':czbins}
                region_fit_no_nutau[region][flav] = {'map':fit_no_nutau[flav]['map'] * f_select[region][flav],
                                                      'ebins':anlys_ebins,
                                                      'czbins':czbins}
                delta_fit_nutau_no_nutau = delta_map(region_fit_nutau[region][flav],region_fit_no_nutau[region][flav])
                region_fit_nutau_minus_no_nutau[region][flav] = {'map':delta_fit_nutau_no_nutau['map'] * f_select[region][flav],
                                                      'ebins':anlys_ebins,
                                                      'czbins':czbins}

        sgnl_fit_nutau = {'cscd' : region_fit_nutau['sgnl']['cscd'],
                              'trck' : region_fit_nutau['sgnl']['trck']
                              }
        side_fit_nutau = {'cscd' : region_fit_nutau['side']['cscd'],
                              'trck' : region_fit_nutau['side']['trck']
                              }

        sgnl_fit_no_nutau = {'cscd' : region_fit_no_nutau['sgnl']['cscd'],
                                 'trck' : region_fit_no_nutau['sgnl']['trck']
                                 }
        side_fit_no_nutau = {'cscd' : region_fit_no_nutau['side']['cscd'],
                                 'trck' : region_fit_no_nutau['side']['trck']
                                 }
        sgnl_fit_nutau_minus_no_nutau = {'cscd' : region_fit_nutau_minus_no_nutau['sgnl']['cscd'],
                              'trck' : region_fit_nutau_minus_no_nutau['sgnl']['trck']
                              }

        plot_1D_distribution_comparison(sgnl_burn_sample_maps, sgnl_fit_nutau, sgnl_fit_no_nutau, png_name_root = 'signal_region_')
        plot_1D_distribution_comparison(side_burn_sample_maps, side_fit_nutau, side_fit_no_nutau, png_name_root = 'side_region_')

        # Plot nominal PISA template (cscd and trck separately), and the ratio of burn sample to PISA template 
        for flav in ['cscd', 'trck']:
            plt.figure()
            show_map(side_fit_nutau[flav],vmax=np.max(side_fit_nutau[flav]['map'])+10,logE=not(args.no_logE))
            if args.save:
                filename = os.path.join(args.outdir,args.title+ '_%s_yr_NutauCCNorm_1_%s_side_region.png' % (args.y, flav))
                plt.title(r'${\rm %s \, yr \, %s \, (Nevts: \, %.1f) }$'%(args.y, flav, np.sum(side_fit_nutau[flav]['map'])), fontsize='large')
                plt.savefig(filename,dpi=150)
                plt.clf()
            plt.figure()
            show_map(sgnl_fit_nutau[flav],vmax=np.max(sgnl_fit_nutau[flav]['map'])+10,logE=not(args.no_logE))
            if args.save:
                filename = os.path.join(args.outdir,args.title+ '_%s_yr_NutauCCNorm_1_%s_signal_region.png' % (args.y, flav))
                plt.title(r'${\rm %s \, yr \, %s \, (Nevts: \, %.1f) }$'%(args.y, flav, np.sum(sgnl_fit_nutau[flav]['map'])), fontsize='large')
                plt.savefig(filename,dpi=150)
                plt.clf()
            show_map(sgnl_fit_nutau_minus_no_nutau[flav],vmax=20 if flav=='cscd' else 5,logE=not(args.no_logE))
            if args.save:
                filename = os.path.join(args.outdir,args.title+ '_%s_yr_NutauCCNorm_1_minus_0_%s_signal_region.png' % (args.y, flav))
                plt.title(r'${\rm 1 yr \, %s \, (Nevts: \, %.1f) }$'%(flav, np.sum(sgnl_fit_nutau_minus_no_nutau[flav]['map'])), fontsize='large')
                plt.savefig(filename,dpi=150)
                plt.clf()


    if not args.save: plt.show()
    else: print '\n-->>Saved all files to: ',args.outdir



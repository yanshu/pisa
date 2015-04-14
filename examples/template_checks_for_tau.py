#! /usr/bin/env python
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date: 16 March 2015
#
# Quick unit test to make sure if you change something, you get a
# reasonable hierarchy asymmetry in the final result. Only input this
# script is required to take is the template settings file, and
# produces plots of the templates at each level of the analysis and
# also the hierarchy asymmetry (NO_NUTAU_i - NUTAU_i)/sqrt(NUTAU_i) in each bin i.
#


import numpy as np
import os
import copy
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.analysis.TemplateMaker import TemplateMaker
from pisa.utils.params import get_values, select_hierarchy_and_nutau_norm,change_nutau_norm_settings
from pisa.utils.log import set_verbosity,logging,profile
from pisa.utils.utils import Timer
from pisa.utils.jsons import from_json
from pisa.utils.plot import show_map
from pisa.utils.utils import get_bin_centers

def get_asymmetry(nutau,no_nutau,flavs,iType=None):
    if iType is None:
        return {flav:{
            'map': np.nan_to_num((no_nutau[flav]['map']-nutau[flav]['map'])/
                                 np.sqrt(nutau[flav]['map'])),
            'ebins':nutau[flav]['ebins'],
            'czbins': nutau[flav]['czbins'] }
                for flav in flavs}
    else:
        return {flav:{
            'map': np.nan_to_num((no_nutau[flav][iType]['map']-nutau[flav][iType]['map'])/
                                 np.sqrt(nutau[flav][iType]['map'])),
            'ebins':nutau[flav][iType]['ebins'],
            'czbins': nutau[flav][iType]['czbins'] }
                for flav in flavs}

def get_residual(map_1,map_2,flavs,iType=None):
    if iType is None:
        return {flav:{
            'map': np.nan_to_num((map_2[flav]['map']-map_1[flav]['map'])),
            'ebins':map_1[flav]['ebins'],
            'czbins': map_1[flav]['czbins'] }
                for flav in flavs}
    else:
        return {flav:{
            'map': np.nan_to_num((map_2[flav][iType]['map']-map_1[flav][iType]['map'])),
            'ebins':map_1[flav][iType]['ebins'],
            'czbins': map_1[flav][iType]['czbins'] }
                for flav in flavs}

def get_ratio_2D(nutau,no_nutau,flavs,iType=None):
    ''' Gets 2D ratio.'''
    if iType is None:
        return {flav:{
            'map': np.nan_to_num(nutau[flav]['map']/no_nutau[flav]['map']),
            'ebins':nutau[flav]['ebins'],
            'czbins': nutau[flav]['czbins'] }
                for flav in flavs}
    else:
        return {flav:{
            'map': np.nan_to_num(nutau[flav][iType]['map']/no_nutau[flav][iType]['map']),
            'ebins':nutau[flav][iType]['ebins'],
            'czbins': nutau[flav][iType]['czbins'] }
                for flav in flavs}

def get_ratio_1D(nutau,no_nutau,flavs,iType=None):
    ''' Gets 1D ratio.'''
    if iType is None:
        return {flav:{
            'map': np.nan_to_num(np.sum(nutau[flav]['map'],axis=1)/np.sum(no_nutau[flav]['map'],axis=1)),
            'ebins':nutau[flav]['ebins']}
            #'czbins': nutau[flav]['czbins'] }
                for flav in flavs}
    else:
        return {flav:{
            'map': np.nan_to_num(np.sum(nutau[flav][iType]['map'],axis=1)/np.sum(no_nutau[flav][iType]['map'],axis=1)),
            'ebins':nutau[flav][iType]['ebins']}
                for flav in flavs}

def plot_ratio(nutau,no_nutau,title='',save=False,dpi=150,outdir=""):
    '''
    Plots the 1D ratio plot between two plots.
    '''
    nutau_pid_map = nutau[4]
    no_nutau_pid_map = no_nutau[4]
    h_ratio = get_ratio_1D(nutau_pid_map,no_nutau_pid_map,['trck','cscd'])
    e_bin_edges = h_ratio['trck']['ebins']
    e_bin_centers = get_bin_centers(e_bin_edges)

    plt.figure(figsize=(5,8))
    plt.subplot(2,1,1)
    plt.plot(e_bin_centers,h_ratio['trck']['map'],color='green', linestyle='solid', marker='o',
                 markerfacecolor='blue', markersize=3)
    plt.title(title+' trck'+r' ratio', fontsize='large')
    plt.xscale('log')
    plt.subplot(2,1,2)
    plt.plot(e_bin_centers,h_ratio['cscd']['map'],color='green', linestyle='solid', marker='o',
                 markerfacecolor='blue', markersize=3)
    plt.xscale('log')
    plt.title(title+' cscd'+r' ratio', fontsize='large')
    filename = os.path.join(outdir,title+'_ratio'+'.png')
    plt.savefig(filename,dpi=dpi)
    return

def plot_residual(map_1,map_2,map_1_title='',map_2_title='',save=False,dpi=150,outdir=""):
    '''
    Plots the residual plot.
    '''
    flavors = ['trck','cscd']
    map_1_pid_map = map_1[4]
    map_2_pid_map = map_2[4]
    h_resd = get_residual(map_1_pid_map,map_2_pid_map,flavors)
    for chan in ['trck','cscd']:
        plt.figure(figsize=(5,8))
        plt.subplot(3,1,1)
        show_map(map_1_pid_map[chan])
        plt.title(map_1_title+' '+chan+' counts',fontsize='large')

        plt.subplot(3,1,2)
        show_map(map_2_pid_map[chan])
        plt.title(map_2_title+' '+chan+' counts',fontsize='large')

        plt.subplot(3,1,3)
        show_map(h_resd[chan])
        plt.title(map_1_title +' '+map_2_title+' residual, '+chan+' counts',fontsize='large')

        if save:
            print "Saving %s chan..."%chan
            filename = os.path.join(outdir,map_1_title+'_'+map_2_title+'_residual_'+chan+'.png')
            plt.savefig(filename,dpi=dpi)

def plot_pid_stage(nutau,no_nutau,title='',save=False,dpi=150,outdir=""):
    '''
    Plots templates and asymmetry for only the final level stage
    '''

    h_asym = get_asymmetry(nutau,no_nutau,['trck','cscd'])

    logging.info("  Total trck events (NUTAU): %d"%np.sum(nutau['trck']['map']))
    logging.info("  Total trck events (NO_NUTAU): %d"%np.sum(no_nutau['trck']['map']))
    logging.info("  Total cscd events (NUTAU): %d"%np.sum(nutau['cscd']['map']))
    logging.info("  Total cscd events (NO_NUTAU): %d"%np.sum(no_nutau['cscd']['map']))

    for chan in ['trck','cscd']:
        plt.figure(figsize=(16,5))

        plt.subplot(1,3,1)
        show_map(nutau[chan])
        plt.title(title+' NUTAU, '+chan+' counts',fontsize='large')

        plt.subplot(1,3,2)
        show_map(no_nutau[chan])
        plt.title(title+' NO_NUTAU, '+chan +' counts',fontsize='large')

        plt.subplot(1,3,3)
        sigma = np.sqrt(np.sum(h_asym[chan]['map']**2))
        show_map(h_asym[chan],cmap='RdBu_r')
        plt.title(title+' '+chan+r' asymmetry, $\sigma$ = %.3f'%sigma,
                  fontsize='large')

        if save:
            print "Saving %s chan..."%chan
            filename = os.path.join(outdir,title+'_asym_'+chan+'.png')
            plt.savefig(filename,dpi=dpi)

    return

def plot_flux_stage(flux_nutau,flux_no_nutau,save=False,title='',dpi=150,outdir=""):
    '''
    Plots flux maps templates for NUTAU/NO_NUTAU
    '''

    flav_title = {'nue':r'$\nu_e$',
                  'nue_bar':r'$\overline{\nu}_e$',
                  'numu':r'$\nu_\mu$',
                  'numu_bar':r'$\overline{\nu}_\mu$'}
    description = ' Flux [m$^{-2}$ s$^{-1}$]'
    for flav in ['nue','numu']:
        plt.figure(figsize=(8,8))

        flav_bar = flav+'_bar'
        plt.subplot(2,2,1)
        show_map(flux_nutau[flav])
        plt.title(r'NUTAU '+flav_title[flav]+description,fontsize='large')
        plt.subplot(2,2,2)
        show_map(flux_nutau[flav_bar])
        plt.title(r'NUTAU '+flav_title[flav_bar]+description,fontsize='large')
        plt.subplot(2,2,3)
        show_map(flux_no_nutau[flav])
        plt.title(r'NO_NUTAU '+flav_title[flav]+description,fontsize='large')
        plt.subplot(2,2,4)
        show_map(flux_no_nutau[flav_bar])
        plt.title(r'NO_NUTAU '+flav_title[flav_bar]+description,fontsize='large')

        plt.tight_layout()
        if save:
            print "Saving Stage 1: nue Flux maps..."
            filename = os.path.join(outdir,title+'_flux_maps_'+flav+'.png')
            plt.savefig(filename,dpi=dpi)

    return

def plot_osc_flux_stage(osc_flux_nutau,osc_flux_no_nutau,save=False,title='',
                        dpi=150,outdir=""):
    '''
    Plots osc flux maps templates for NUTAU/NO_NUTAU
    '''

    flav_title = {'nue':r'$\nu_e$',
                  'nue_bar':r'$\overline{\nu}_e$',
                  'numu':r'$\nu_\mu$',
                  'numu_bar':r'$\overline{\nu}_\mu$',
                  'nutau':r'$\nu_\tau$',
                  'nutau_bar':r'$\overline{\nu}_\tau$'}
    all_flavs = list(flav_title.keys())
    h_asym = get_asymmetry(osc_flux_nutau,osc_flux_no_nutau,all_flavs)
    description = r' Oscillated Flux [m$^{-2}$ s$^{-1}$]'
    for flav in ['nue','numu','nutau']:
        plt.figure(figsize=(16,8))

        plt.subplot(2,3,1)
        show_map(osc_flux_nutau[flav])
        plt.title(r'NUTAU '+flav_title[flav]+description,fontsize='large')
        plt.subplot(2,3,2)
        show_map(osc_flux_no_nutau[flav])
        plt.title(r'NO_NUTAU '+flav_title[flav]+description,fontsize='large')
        plt.subplot(2,3,3)
        show_map(h_asym[flav],cmap='RdBu_r')
        sigma = np.sqrt(np.sum(h_asym[flav]['map']**2))
        plt.title(r'Nutau Asymmetry: '+flav_title[flav]+', $\sigma$ = %.3f'%sigma,
                  fontsize='large')

        flav_bar = flav+'_bar'
        plt.subplot(2,3,4)
        show_map(osc_flux_nutau[flav_bar])
        plt.title(r'NUTAU '+flav_title[flav_bar]+description,fontsize='large')
        plt.subplot(2,3,5)
        show_map(osc_flux_no_nutau[flav_bar])
        plt.title(r'NO_NUTAU '+flav_title[flav_bar]+description,fontsize='large')
        plt.subplot(2,3,6)
        show_map(h_asym[flav_bar],cmap='RdBu_r')
        sigma = np.sqrt(np.sum(h_asym[flav_bar]['map']**2))
        plt.title(r'Nutau Asymmetry: '+flav_title[flav_bar]+
                  ', $\sigma$ = %.3f'%sigma,fontsize='large')

        plt.tight_layout()
        if save:
            print "Saving Stage 2: "+flav+" osc flux maps..."
            filename = os.path.join(outdir,title+'_osc_flux_maps_'+flav+'.png')
            plt.savefig(filename,dpi=dpi)

    return

def plot_true_event_rate(event_rate_nutau,event_rate_no_nutau,title='',save=False,
                         dpi=150,outdir=''):
    '''
    Plots true event rate maps
    '''
    flav_title = {'nue':r'$\nu_e^{cc}$',
                  'nue_bar':r'$\overline{\nu}_e^{cc}$',
                  'numu':r'$\nu_\mu^{cc}$',
                  'numu_bar':r'$\overline{\nu}_\mu^{cc}$',
                  'nutau':r'$\nu_\tau^{cc}$',
                  'nutau_bar':r'$\overline{\nu}_\tau^{cc}$'}
    all_flavs = list(flav_title.keys())
    h_asym = get_asymmetry(event_rate_nutau,event_rate_no_nutau,all_flavs,iType='cc')
    description=' True Event Rate [#/yr]'
    for flav in ['nue','numu','nutau']:
        plt.figure(figsize=(16,8))

        plt.subplot(2,3,1)
        show_map(event_rate_nutau[flav]['cc'],vmin=0.0)
        plt.title(r'NUTAU '+flav_title[flav]+description,fontsize='large')
        plt.subplot(2,3,2)
        show_map(event_rate_no_nutau[flav]['cc'],vmin=0.0)
        plt.title(r'NO_NUTAU '+flav_title[flav]+description,fontsize='large')
        plt.subplot(2,3,3)
        show_map(h_asym[flav],cmap='RdBu_r')
        sigma = np.sqrt(np.sum(h_asym[flav]['map']**2))
        plt.title(r'Nutau Asymmetry: '+flav_title[flav]+', $\sigma$ = %.3f'%sigma,
                  fontsize='large')

        flav_bar = flav+'_bar'
        plt.subplot(2,3,4)
        show_map(event_rate_nutau[flav_bar]['cc'],vmin=0.0)
        plt.title(r'NUTAU '+flav_title[flav_bar]+description,fontsize='large')
        plt.subplot(2,3,5)
        show_map(event_rate_no_nutau[flav_bar]['cc'],vmin=0.0)
        plt.title(r'NO_NUTAU '+flav_title[flav_bar]+description,fontsize='large')
        plt.subplot(2,3,6)
        show_map(h_asym[flav_bar],cmap='RdBu_r')
        sigma = np.sqrt(np.sum(h_asym[flav_bar]['map']**2))
        plt.title(r'Nutau Asymmetry: '+flav_title[flav_bar]+
                  ', $\sigma$ = %.3f'%sigma,fontsize='large')

        plt.tight_layout()
        if save:
            print "Saving Stage 3: "+flav+" true event rate  maps..."
            filename = os.path.join(outdir,title+'_true_event_rate_'+flav+'.png')
            plt.savefig(filename,dpi=dpi)

    return

def plot_reco_event_rate(reco_rate_nutau,reco_rate_no_nutau,save=False,title='',dpi=150,
                         outdir=''):
    '''
    Plots reco event rate maps for NUTAU/NO_NUTAU
    '''

    flav_title = {'nue_cc':r'$\nu_e^{cc}$',
                  'numu_cc':r'$\nu_\mu^{cc}$',
                  'nutau_cc':r'$\nu_\tau^{cc}$',
                  'nuall_nc':r'$\nu$ all NC'}
    all_flavs = list(flav_title.keys())
    h_asym = get_asymmetry(reco_rate_nutau,reco_rate_no_nutau,all_flavs)
    description=' Reco Event Rate [#/yr]'
    for flav in all_flavs:
        plt.figure(figsize=(16,5))

        plt.subplot(1,3,1)
        show_map(reco_rate_nutau[flav])
        plt.title(r'NUTAU '+flav_title[flav]+description,fontsize='large')
        plt.subplot(1,3,2)
        show_map(reco_rate_no_nutau[flav])
        plt.title(r'NO_NUTAU '+flav_title[flav]+description,fontsize='large')
        plt.subplot(1,3,3)
        show_map(h_asym[flav],cmap='RdBu_r')
        sigma = np.sqrt(np.sum(h_asym[flav]['map']**2))
        plt.title(r'Nutau Asymmetry: '+flav_title[flav]+', $\sigma$ = %.3f'%sigma,
                  fontsize='large')

        plt.tight_layout()
        if save:
            print "Saving Stage 4: "+flav+" reco event rate  maps..."
            filename = os.path.join(outdir,title+'_reco_event_rate_'+flav+'.png')
            plt.savefig(filename,dpi=dpi)

    return

def plot_stages(data_nutau, data_no_nutau,save=False,title='',dpi=150,outdir=""):
    '''
    Plots templates and asymmetry (where applicable) for all stages,
    (up to the final level PID stage)
    '''

    # Stage 1: Flux maps:
    flux_nutau = data_nutau[0]
    flux_no_nutau = data_no_nutau[0]
    plot_flux_stage(flux_nutau,flux_no_nutau,save=save,title=title,
                    outdir=outdir,dpi=dpi)

    # Stage 2: Oscillated Flux maps:
    osc_flux_nutau = data_nutau[1]
    osc_flux_no_nutau = data_no_nutau[1]
    plot_osc_flux_stage(osc_flux_nutau,osc_flux_no_nutau,title=title,outdir=outdir,
                        dpi=dpi,save=save)

    # Stage 3: True Event Rate maps:
    event_rate_nutau = data_nutau[2]
    event_rate_no_nutau = data_no_nutau[2]
    plot_true_event_rate(event_rate_nutau,event_rate_no_nutau,title=title,outdir=outdir,
                         dpi=dpi,save=save)

    # Stage 4: Reco Event Rate maps:
    reco_rate_nutau = data_nutau[3]
    reco_rate_no_nutau = data_no_nutau[3]
    plot_reco_event_rate(reco_rate_nutau,reco_rate_no_nutau,title=title,outdir=outdir,
                         dpi=dpi,save=save)

    # Stage 5: PID final level maps:
    pid_nutau = data_nutau[4]
    pid_no_nutau = data_no_nutau[4]
    plot_pid_stage(pid_nutau,pid_no_nutau,title=title,save=save,outdir=outdir)

    return


if __name__ == "__main__":

    set_verbosity(0)
    parser = ArgumentParser(description='''Quick check if all components are working reasonably well, by
making the final level hierarchy asymmetry plots from the input
settings file. ''')
    parser.add_argument('mc_template_settings',metavar='JSON',
                        help='Settings file to use for MC template generation')
    parser.add_argument('para_template_settings',metavar='JSON',
                        help='Settings file to use for parametrized template generation')
    parser.add_argument('-a','--all',action='store_true',default=False,
                        help="Plot all stages 1-5 of templates and Asymmetry")
    parser.add_argument('--title',metavar="str",default='',
                        help="Title of the geometry or test in plots")
    parser.add_argument('--save',action='store_true',default=False,
                        help="Save plots in cwd")
    parser.add_argument('-o','--outdir',metavar='DIR',default="",
                        help="Directory to save the output figures.")
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='set verbosity level')
    args = parser.parse_args()
    set_verbosity(args.verbose)

    mc_template_settings = from_json(args.mc_template_settings)
    para_template_settings = from_json(args.para_template_settings)

    with Timer() as t:
        mc_template_maker = TemplateMaker(get_values(mc_template_settings['params']),
                                       **mc_template_settings['binning'])
        para_template_maker = TemplateMaker(get_values(para_template_settings['params']),
                                       **para_template_settings['binning'])
    profile.info("==> elapsed time to initialize templates: %s sec"%t.secs)

    # Make nutau template:
    mc_no_nutau_params = copy.deepcopy(select_hierarchy_and_nutau_norm(mc_template_settings['params'],True,0.0))

    mc_nutau_params = copy.deepcopy(select_hierarchy_and_nutau_norm(mc_template_settings['params'],True,1.0))

    para_no_nutau_params = copy.deepcopy(select_hierarchy_and_nutau_norm(para_template_settings['params'],True,0.0))

    para_nutau_params = copy.deepcopy(select_hierarchy_and_nutau_norm(para_template_settings['params'],True,1.0))

    with Timer(verbose=False) as t:
        mc_nutau = mc_template_maker.get_template(get_values(mc_nutau_params),return_stages=args.all)
        para_nutau = para_template_maker.get_template(get_values(para_nutau_params),return_stages=args.all)
    profile.info("==> elapsed time to get NUTAU template: %s sec"%t.secs)
    with Timer(verbose=False) as t:
        mc_no_nutau = mc_template_maker.get_template(get_values(mc_no_nutau_params),return_stages=args.all)
        para_no_nutau = para_template_maker.get_template(get_values(para_no_nutau_params),return_stages=args.all)
    profile.info("==> elapsed time to get NO_NUTAU template: %s sec"%t.secs)

    # Or equivalently, if args.all:

    plot_residual(mc_nutau,para_nutau,map_1_title='mc_nutau',map_2_title='para_nutau',save=args.save,outdir=args.outdir+'mc_vs_para_tau/')
    plot_residual(mc_no_nutau,para_no_nutau,map_1_title='mc_no_nutau',map_2_title='para_no_nutau',save=args.save,outdir=args.outdir+'mc_vs_para_notau/')

    plot_ratio(mc_nutau,para_nutau,title='mc_to_para_nutau',save=args.save,outdir=args.outdir+'mc_vs_para_tau/')
    plot_ratio(mc_no_nutau,para_no_nutau,title='mc_to_para_no_nutau',save=args.save,outdir=args.outdir+'mc_vs_para_notau/')

    #if type(mc_nutau) is tuple:
    #    #plot_stages(mc_nutau,mc_no_nutau,title=args.title,save=args.save,outdir=args.outdir+'mc/')
    #    #plot_stages(para_nutau,para_no_nutau,title=args.title,save=args.save,outdir=args.outdir+'para/')
    #    plot_stages(mc_nutau,para_nutau,title=args.title,save=args.save,outdir=args.outdir+'mc_vs_para_tau/')
    #    plot_stages(mc_no_nutau,para_no_nutau,title=args.title,save=args.save,outdir=args.outdir+'mc_vs_para_notau/')
    #else:
    #    #plot_pid_stage(mc_nutau,mc_no_nutau,title=args.title,save=args.save,outdir=args.outdir+'mc/')
    #    #plot_pid_stage(para_nutau,para_no_nutau,title=args.title,save=args.save,outdir=args.outdir+'para/')
    #    plot_residual(mc_nutau,para_nutau,map_1_title='mc_nutau',map_2_title='para_nutau',save=args.save,outdir=args.outdir+'mc_vs_para_tau/')
    #    plot_residual(mc_no_nutau,para_no_nutau,map_1_title='mc_no_nutau',map_2_title='para_no_nutau',save=args.save,outdir=args.outdir+'mc_vs_para_notau/')

    if not args.save: plt.show()
    else: print "\n-->>Saved all files to: ",args.outdir

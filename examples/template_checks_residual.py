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

def get_residual_error(map_up,map_down,flavs):
    return {flav:{
        'map': np.nan_to_num(np.sqrt(np.square(map_up[flav]['map'])+np.square(map_down[flav]['map']))),
        'ebins':map_up[flav]['ebins'],
        'czbins': map_up[flav]['czbins'] }
            for flav in flavs}

def get_ratio_error(map_up,map_down,flavs):
    return {flav:{
        'map': np.nan_to_num((map_up[flav]['map']/map_down[flav]['map'])
            *np.sqrt(1.0/map_up[flav]['map']+1.0/map_down[flav]['map'])),
        'ebins':map_up[flav]['ebins'],
        'czbins': map_up[flav]['czbins'] }
            for flav in flavs}

def get_asymmetry_residual(residual_nutau, residual_no_nutau, residual_nutau_error, residual_no_nutau_error,flavs):
    return {flav:{
        'map': np.nan_to_num((residual_nutau[flav]['map']-residual_no_nutau[flav]['map'])/
            np.sqrt(np.square(residual_nutau_error[flav]['map'])+np.square(residual_nutau_error[flav]['map']))),
        'ebins':residual_nutau[flav]['ebins'],
        'czbins': residual_nutau[flav]['czbins'] }
        for flav in flavs}

def get_asymmetry_ratio(ratio_nutau, ratio_no_nutau, ratio_nutau_error, ratio_no_nutau_error,flavs):
    return {flav:{
        'map': np.nan_to_num((ratio_nutau[flav]['map']-ratio_no_nutau[flav]['map'])/
            np.sqrt(np.square(ratio_nutau_error[flav]['map'])+np.square(ratio_nutau_error[flav]['map']))),
        'ebins':ratio_nutau[flav]['ebins'],
        'czbins': ratio_nutau[flav]['czbins'] }
        for flav in flavs}

def get_residual(map_1,map_2,flavs,iType=None):
    if iType is None:
        return {flav:{
            'map': np.nan_to_num((map_1[flav]['map']-map_2[flav]['map'])),
            'ebins':map_1[flav]['ebins'],
            'czbins': map_1[flav]['czbins'] }
                for flav in flavs}
    else:
        return {flav:{
            'map': np.nan_to_num((map_1[flav][iType]['map']-map_2[flav][iType]['map'])),
            'ebins':map_1[flav][iType]['ebins'],
            'czbins': map_1[flav][iType]['czbins'] }
                for flav in flavs}

def get_up_map(maps,flavs,iType=None):
    map = maps[4]   #plot the last stage
    czbin_edges = len(map['cscd']['czbins'])
    czbin_mid_idx = (czbin_edges-1)/2
    print 'nczbin = ', len(map['cscd']['czbins'])
    if iType is None:
        return {flav:{
            'map': map[flav]['map'][:,0:czbin_mid_idx],
            'ebins':map[flav]['ebins'],
            'czbins': map[flav]['czbins'][0:czbin_mid_idx+1] }
                for flav in flavs}
    else:
        return {flav:{
            'map': map[flav][iType]['map'][:,0:czbin_mid_idx],
            'ebins':map[flav][iType]['ebins'],
            'czbins': map[flav][iType]['czbins'][0:czbin_mid_idx+1] }
                for flav in flavs}

def get_flipped_down_map(maps,flavs,iType=None):
    ''' Gets the downgoing map and flip it.'''
    map = maps[4]   #use the last stage
    czbin_edges = len(map['cscd']['czbins'])
    czbin_mid_idx = (czbin_edges-1)/2
    if iType is None:
        return {flav:{
            'map': np.fliplr(map[flav]['map'][:,czbin_mid_idx:]),
            'ebins':map[flav]['ebins'],
            'czbins': np.sort(-map['trck']['czbins'][czbin_mid_idx:]) }
                for flav in flavs}
    else:
        return {flav:{
            'map': np.fliplr(map[flav][iType]['map'][:,czbin_mid_idx:]), 
            'ebins':map[flav][iType]['ebins'],
            'czbins': np.sort(-map['trck'][iType]['czbins'][czbin_mid_idx:]) }
                for flav in flavs}

def get_ratio_2D(map_up,map_down,flavs,iType=None):
    ''' Gets 2D ratio.'''
    print "ebins: ",map_down['cscd']['ebins']
    print "czbins: ",map_down['cscd']['czbins']
    print "map up(cscd): ", map_up['cscd']['map']
    print "map down(cscd): ", map_down['cscd']['map']
    print "ratio of up/down: ", map_up['cscd']['map']/map_down['cscd']['map']
    for i in range(0,9):
        for j in range(0,9):
            if(map_down['cscd']['map'][i][j]==0):
                print "at bin(",i,", ", j, "), bin=0 "
    if iType is None:
        return {flav:{
            'map': np.nan_to_num(map_up[flav]['map']/map_down[flav]['map']),
            'ebins':map_up[flav]['ebins'],
            'czbins': map_up[flav]['czbins'] }
                for flav in flavs}
    else:
        return {flav:{
            'map': np.nan_to_num(map_up[flav][iType]['map']/map_down[flav][iType]['map']),
            'ebins':map_up[flav][iType]['ebins'],
            'czbins': map_up[flav][iType]['czbins'] }
                for flav in flavs}

def get_ratio_1D(map_1,map_2,flavs,iType=None):
    ''' Gets 1D ratio of two maps (ratio of map_1 to map_2 in each energy bin).'''
    if iType is None:
        return {flav:{
            'map': np.nan_to_num(np.sum(map_1[flav]['map'],axis=1)/np.sum(map_2[flav]['map'],axis=1)),
            'ebins':map_1[flav]['ebins']}
                for flav in flavs}
    else:
        return {flav:{
            'map': np.nan_to_num(np.sum(map_1[flav][iType]['map'],axis=1)/np.sum(map_2[flav][iType]['map'],axis=1)),
            'ebins':map_1[flav][iType]['ebins']}
                for flav in flavs}

def plot_ratio_1D(map_1,map_2,map_1_title='',map_2_title='',save=False,dpi=150,outdir=""):
    '''
    Plots the two maps and the 1D ratio plot between them (ratio of map_1 to map_2 as a function of energy).
    '''

    if type(map_1) is tuple:
        h_ratio = get_ratio_1D(map_1[4],map_2[4],['trck','cscd'])
    else:
        h_ratio = get_ratio_1D(map_1,map_2,['trck','cscd'])
    e_bin_edges = h_ratio['trck']['ebins']
    e_bin_centers = get_bin_centers(e_bin_edges)

    plt.figure(figsize=(5,8))
    plt.subplot(2,1,1)
    plt.plot(e_bin_centers,h_ratio['trck']['map'],color='green', linestyle='solid', marker='o',
                 markerfacecolor='blue', markersize=3)
    plt.title(map_1_title+'/'+map_2_title+' trck'+r' ratio', fontsize='large')
    plt.xscale('log')
    plt.subplot(2,1,2)
    plt.plot(e_bin_centers,h_ratio['cscd']['map'],color='green', linestyle='solid', marker='o',
                 markerfacecolor='blue', markersize=3)
    plt.xscale('log')
    plt.title(map_1_title+'/'+map_2_title+' cscd'+r' ratio', fontsize='large')
    filename = os.path.join(outdir,map_1_title+'_'+map_2_title+'_ratio'+'.png')
    plt.savefig(filename,dpi=dpi)
    return

def plot_ratio_of_1D_ratios(map_1,map_2,map_3,map_4,title='',save=False,dpi=150,outdir=""):
    '''
    Plots the ratio of ratios (ratio of map_1/map_2 to map_3/map_4 as a function of energy).
    '''

    h_ratio_1 = get_ratio_1D(map_1,map_2,['trck','cscd'])
    h_ratio_2 = get_ratio_1D(map_3,map_4,['trck','cscd'])
    ratio_of_ratios_trck = h_ratio_1['trck']['map']/h_ratio_2['trck']['map']
    ratio_of_ratios_cscd = h_ratio_1['cscd']['map']/h_ratio_2['cscd']['map']
    e_bin_edges = h_ratio_1['trck']['ebins']
    e_bin_centers = get_bin_centers(e_bin_edges)

    plt.figure(figsize=(5,8))
    plt.subplot(2,1,1)
    plt.plot(e_bin_centers,ratio_of_ratios_trck,color='green', linestyle='solid', marker='o',
                 markerfacecolor='blue', markersize=3)
    plt.title(title+r' trck ', fontsize='large')
    plt.xscale('log')
    plt.subplot(2,1,2)
    plt.plot(e_bin_centers,ratio_of_ratios_cscd,color='green', linestyle='solid', marker='o',
                 markerfacecolor='blue', markersize=3)
    plt.xscale('log')
    plt.title(title+r' cscd ', fontsize='large')
    filename = os.path.join(outdir,title+'.png')
    plt.savefig(filename,dpi=dpi)

    return

def plot_residual(map_1,map_2,map_1_title='',map_2_title='',save=False,dpi=150,outdir=""):
    '''
    Plots the residual plot.
    '''
    h_resd = get_residual(map_1,map_2,['trck','cscd'])
    for chan in ['trck','cscd']:
        plt.figure(figsize=(5,8))
        plt.subplot(1,3,1)
        show_map(map_1)
        plt.title(map_1_title+' '+chan+' counts',fontsize='large')

        plt.subplot(1,3,2)
        show_map(map_2)
        plt.title(map_2_title+' '+chan+' counts',fontsize='large')

        plt.subplot(1,3,2)
        show_map(h_resd[chan])
        plt.title(map_1_title +' '+map_2_title+' residual, '+chan+' counts',fontsize='large')

        if save:
            print "Saving %s chan..."%chan
            filename = os.path.join(outdir,map_1_title+'_'+map_2_title+'_residual_'+chan+'.png')
            plt.savefig(filename,dpi=dpi)

def plot_pid_stage(nutau,no_nutau,title='',save=False,dpi=150,outdir="",log=False):
    '''
    Plots templates and asymmetry for only the final level stage
    '''

    h_asym = get_asymmetry(nutau,no_nutau,['trck','cscd'])

    logging.info("  Total trck events (NUTAU): %f"%np.sum(nutau['trck']['map']))
    logging.info("  Total trck events (NO_NUTAU): %f"%np.sum(no_nutau['trck']['map']))
    logging.info("  Total cscd events (NUTAU): %f"%np.sum(nutau['cscd']['map']))
    logging.info("  Total cscd events (NO_NUTAU): %f"%np.sum(no_nutau['cscd']['map']))

    for chan in ['trck','cscd']:
        plt.figure(figsize=(16,5))

        plt.subplot(1,3,1)
        show_map(nutau[chan],log=log)
        plt.title(title+' NUTAU, '+chan+' counts',fontsize='large')

        plt.subplot(1,3,2)
        show_map(no_nutau[chan],log=log)
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
    parser.add_argument('template_settings',metavar='JSON',
                        help='Settings file to use for template generation')
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

    template_settings = from_json(args.template_settings)

    with Timer() as t:
        template_maker = TemplateMaker(get_values(template_settings['params']),
                                       **template_settings['binning'])
    profile.info("==> elapsed time to initialize templates: %s sec"%t.secs)

    # Make nutau template:
    no_nutau_params = copy.deepcopy(select_hierarchy_and_nutau_norm(template_settings['params'],True,0.0))

    nutau_params = copy.deepcopy(select_hierarchy_and_nutau_norm(template_settings['params'],True,1.0))

    with Timer(verbose=False) as t:
        nutau = template_maker.get_template(get_values(nutau_params),return_stages=args.all)
    profile.info("==> elapsed time to get NUTAU template: %s sec"%t.secs)
    with Timer(verbose=False) as t:
        no_nutau = template_maker.get_template(get_values(no_nutau_params),return_stages=args.all)
    profile.info("==> elapsed time to get NO_NUTAU template: %s sec"%t.secs)

    nutau_up = get_up_map(nutau,['trck','cscd'])
    nutau_down = get_flipped_down_map(nutau,['trck','cscd']) 

    no_nutau_up = get_up_map(no_nutau,['trck','cscd']) 
    no_nutau_down = get_flipped_down_map(no_nutau,['trck','cscd']) 

    # get residual
    residual_nutau = get_residual(nutau_up,nutau_down,['trck','cscd']) 
    residual_no_nutau = get_residual(no_nutau_up,no_nutau_down,['trck','cscd']) 
    residual_nutau_error = get_residual_error(nutau_up,nutau_down,['trck','cscd'])
    residual_no_nutau_error = get_residual_error(no_nutau_up,no_nutau_down,['trck','cscd'])

    residual_asym = get_asymmetry_residual(residual_nutau, residual_no_nutau, residual_nutau_error, residual_no_nutau_error,['trck','cscd'])
    for chan in ['trck','cscd']:
        plt.figure(figsize=(16,5))

        plt.subplot(1,3,1)
        show_map(residual_nutau[chan])
        plt.title('DC12 NUTAU, '+chan+' ratio of up to down',fontsize='large')

        plt.subplot(1,3,2)
        show_map(residual_no_nutau[chan])
        plt.title('DC12 NO_NUTAU, '+chan +' ratio of up to down',fontsize='large')

        plt.subplot(1,3,3)
        sigma = np.sqrt(np.sum(residual_asym[chan]['map']**2))
        show_map(residual_asym[chan],cmap='RdBu_r')
        plt.title('DC12 '+chan+r'ratio asymmetry, $\sigma$ = %.3f'%sigma,
                  fontsize='large')

    # Or equivalently, if args.all:
    if type(nutau) is tuple:
        plot_stages(nutau,no_nutau,title=args.title,save=args.save,outdir=args.outdir+'all/')
    else:
        plot_pid_stage(nutau,no_nutau,title=args.title,save=args.save,outdir=args.outdir+'all/')

    if type(nutau_up) is tuple:
        plot_stages(nutau_up,no_nutau_up,title=args.title,save=args.save,outdir=args.outdir+'up/')
    else:
        plot_pid_stage(nutau_up,no_nutau_up,title=args.title,save=args.save,outdir=args.outdir+'up/')

    if type(nutau_down) is tuple:
        plot_stages(nutau_down,no_nutau_down,title=args.title,save=args.save,outdir=args.outdir+'down/')
    else:
        plot_pid_stage(nutau_down,no_nutau_down,title=args.title,save=args.save,outdir=args.outdir+'down/')

    plot_ratio_1D(nutau_up,nutau_down,map_1_title='nutau_up',map_2_title='nutau_down',outdir=args.outdir+'ratio_1d/')
    plot_ratio_1D(no_nutau_up,no_nutau_down,map_1_title='no_nutau_up',map_2_title='no_nutau_down',outdir=args.outdir+'ratio_1d/')
    plot_ratio_of_1D_ratios(nutau_up,nutau_down,no_nutau_up,no_nutau_down,title='ratio_of_ratios',outdir=args.outdir+'ratio_1d/')

    if not args.save: plt.show()
    else: print "\n-->>Saved all files to: ",args.outdir

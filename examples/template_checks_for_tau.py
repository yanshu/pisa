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
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.analysis.TemplateMaker import TemplateMaker
from pisa.utils.params import get_values, select_hierarchy_and_nutau_norm,change_nutau_norm_settings
from pisa.utils.log import set_verbosity,logging,profile
from pisa.utils.utils import Timer
from pisa.utils.jsons import from_json
from pisa.utils.plot import show_map

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
        plt.title(r'Hierarchy Asymmetry: '+flav_title[flav]+', $\sigma$ = %.3f'%sigma,
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
        plt.title(r'Hierarchy Asymmetry: '+flav_title[flav_bar]+
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
        plt.title(r'Hierarchy Asymmetry: '+flav_title[flav]+', $\sigma$ = %.3f'%sigma,
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
        plt.title(r'Hierarchy Asymmetry: '+flav_title[flav_bar]+
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
        plt.title(r'Hierarchy Asymmetry: '+flav_title[flav]+', $\sigma$ = %.3f'%sigma,
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
    nutau_params = change_nutau_norm_settings(template_settings['params'],1.0,True)
    print "nutau_params : " ,nutau_params['nutau_norm']

    no_nutau_params = change_nutau_norm_settings(template_settings['params'],0.0,True)
    print "no nutau_params: ",  no_nutau_params['nutau_norm']
    with Timer(verbose=False) as t:
        nutau = template_maker.get_template(get_values(nutau_params),return_stages=args.all)
    profile.info("==> elapsed time to get NUTAU template: %s sec"%t.secs)
    with Timer(verbose=False) as t:
        no_nutau = template_maker.get_template(get_values(no_nutau_params),return_stages=args.all)
    profile.info("==> elapsed time to get NO_NUTAU template: %s sec"%t.secs)

    # Or equivalently, if args.all:
    if type(nutau) is tuple:
        plot_stages(nutau,no_nutau,title=args.title,save=args.save,outdir=args.outdir)
    else: plot_pid_stage(nutau,no_nutau,title=args.title,save=args.save,outdir=args.outdir)

    if not args.save: plt.show()
    else: print "\n-->>Saved all files to: ",args.outdir

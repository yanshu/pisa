#! /usr/bin/env python
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date: 16 March 2015
#
# Quick unit test to make sure if you channelge something, you get a
# reasonable hierarchy asymmetry in the final result. Only input this
# script is required to take is the template settings file, and
# produces plots of the templates at each level of the analysis and
# also the hierarchy asymmetry (NO_NUTAU_i - NUTAU_i)/sqrt(NUTAU_i) in each bin i.
#

import copy
import numpy as np
import os
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.analysis.TemplateMaker import TemplateMaker
from pisa.utils.params import get_values, select_hierarchy_and_nutau_norm
from pisa.analysis.stats.Maps_nutau_noDOMIce import get_pseudo_data_fmap, get_true_template
from pisa.analysis.stats.Maps import get_seed
from pisa.utils.log import set_verbosity,logging,profile
from pisa.utils.utils import Timer
from pisa.utils.jsons import from_json
from pisa.utils.plot import show_map, sum_map
from pisa.background.BackgroundServiceICC import BackgroundServiceICC

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


def plot_pid_stage(nutau,no_nutau,title='',save=False,dpi=150,outdir=''):
    '''
    Plots templates and asymmetry for only the final level stage
    '''

    h_asym = get_asymmetry(nutau,no_nutau,['trck','cscd'])

    logging.info('  Total trck events (NUTAU): %d'%np.sum(nutau['trck']['map']))
    logging.info('  Total trck events (NO_NUTAU): %d'%np.sum(no_nutau['trck']['map']))
    logging.info('  Total cscd events (NUTAU): %d'%np.sum(nutau['cscd']['map']))
    logging.info('  Total cscd events (NO_NUTAU): %d'%np.sum(no_nutau['cscd']['map']))

    for channel in ['trck','cscd']:
        plt.figure(figsize=(16,5))

        plt.subplot(1,3,1)
        show_map(nutau[channel])
        plt.title(title+' NUTAU, '+channel+' counts',fontsize='large')

        plt.subplot(1,3,2)
        show_map(no_nutau[channel])
        plt.title(title+' NO_NUTAU, '+channel +' counts',fontsize='large')

        plt.subplot(1,3,3)
        sigma = np.sqrt(np.sum(h_asym[channel]['map']**2))
        show_map(h_asym[channel],cmap='RdBu_r')
        plt.title(title+' '+channel+r' asymmetry, $\sigma$ = %.3f'%sigma,
                  fontsize='large')

        if save:
            print 'Saving %s channel...'%channel
            filename = os.path.join(outdir,title+'_asym_'+channel+'.png')
            plt.savefig(filename,dpi=dpi)

    return

def plot_flux_stage(flux_nutau,flux_no_nutau,save=False,title='',dpi=150,outdir=''):
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
            print 'Saving Stage 1: nue Flux maps...'
            filename = os.path.join(outdir,title+'_flux_maps_'+flav+'.png')
            plt.savefig(filename,dpi=dpi)

    return

def plot_osc_flux_stage(osc_flux_nutau,osc_flux_no_nutau,save=False,title='',
                        dpi=150,outdir=''):
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
            print 'Saving Stage 2: '+flav+' osc flux maps...'
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
            print 'Saving Stage 3: '+flav+' true event rate  maps...'
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
            print 'Saving Stage 4: '+flav+' reco event rate  maps...'
            filename = os.path.join(outdir,title+'_reco_event_rate_'+flav+'.png')
            plt.savefig(filename,dpi=dpi)

    return

def plot_stages(data_nutau, data_no_nutau,save=False,title='',dpi=150,outdir=''):
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


if __name__ == '__main__':

    set_verbosity(0)
    parser = ArgumentParser(description='''Quick check if all components are working reasonably well, by
making the final level hierarchy asymmetry plots from the input
settings file. ''')
    parser.add_argument('template_settings',metavar='JSON',
                        help='Settings file to use for template generation')
    parser.add_argument('--background_file',metavar='FILE',type=str,
                        default='background/IC86_3yr_ICC.hdf5',
                        help='''HDF5 File containing atmospheric background from 3 years'
                        inverted corridor cut data''')
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
    parser.add_argument('-e_reco_scale', metavar='float',type=float,
                        help='''The value of e_reco_scale''')
    parser.add_argument('-cz_reco_scale', metavar='float',type=float,
                        help='''The value of cz_reco_scale''')
    args = parser.parse_args()
    e_reco_scale_val = args.e_reco_scale
    cz_reco_scale_val = args.cz_reco_scale
    set_verbosity(args.verbose)

    template_settings = from_json(args.template_settings)
    czbins = template_settings['binning']['czbins']
    
    up_template_settings = copy.deepcopy(template_settings)
    #up_template_settings['binning']['czbins']=czbins[czbins<=0]
    up_template_settings['params']['reco_mc_wt_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_100_percent_up.hdf5'}
    up_template_settings['params']['reco_vbwkde_evts_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_10_percent_up.hdf5'}
    
    down_template_settings = copy.deepcopy(template_settings)
    #down_template_settings['binning']['czbins']=czbins[czbins>=0]
    down_template_settings['params']['pid_paramfile'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/pid/1X60_pid_down.json'}
    down_template_settings['params']['reco_vbwkde_evts_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_10_percent_down.hdf5'}

    down_template_settings['params']['e_reco_scale'] = {u'fixed': True, u'value': e_reco_scale_val , u'prior': { u'fiducial': 1.0, u'kind': 'gaussian', u'sigma': 0.02 }, u'range': [ 0.9, 1.1 ], u'scale': 1.0 }
    down_template_settings['params']['cz_reco_scale'] = {u'fixed': True, u'value': cz_reco_scale_val , u'prior': { u'fiducial': 1.0, u'kind': 'gaussian', u'sigma': 0.02 }, u'range': [ 0.9, 1.1 ], u'scale': 1.0 }

    with Timer() as t:
        template_maker_down = TemplateMaker(get_values(down_template_settings['params']), **down_template_settings['binning'])
        template_maker_up = TemplateMaker(get_values(up_template_settings['params']), **up_template_settings['binning'])
        template_maker = [template_maker_up,template_maker_down]

    profile.info('==> elapsed time to initialize templates: %s sec'%t.secs)

    # Make nutau template:
    #no_nutau_params = copy.deepcopy(select_hierarchy_and_nutau_norm(down_template_settings['params'],True,0.0))

    nutau_up_params = copy.deepcopy(select_hierarchy_and_nutau_norm(up_template_settings['params'],True,1.0))
    nutau_down_params = copy.deepcopy(select_hierarchy_and_nutau_norm(down_template_settings['params'],True,1.0))

    with Timer(verbose=False) as t:
        #nutau_up = template_maker_up.get_template(get_values(nutau_up_params),return_stages=args.all)
        nutau_down = template_maker_down.get_template(get_values(nutau_down_params),return_stages=args.all)
    profile.info('==> elapsed time to get NUTAU template: %s sec'%t.secs)
    #with Timer(verbose=False) as t:
    #    no_nutau = template_maker_down.get_template(get_values(no_nutau_params),return_stages=args.all)
    #profile.info('==> elapsed time to get NO_NUTAU template: %s sec'%t.secs)

    print 'nutau_up = ', nutau_up
    print 'nutau_down = ', nutau_down

    #print 'no_nutau = ', no_nutau
    #fmap_tau_cscd = get_pseudo_data_fmap(template_maker, get_values(nutau_params),
    #        seed=get_seed(),channel='cscd')
    #fmap_tau_trck = get_pseudo_data_fmap(template_maker, get_values(nutau_params),
    #        seed=get_seed(),channel='trck')
    #fmap_notau_cscd = get_pseudo_data_fmap(template_maker, get_values(no_nutau_params),
    #        seed=get_seed(),channel='cscd')
    #fmap_notau_trck = get_pseudo_data_fmap(template_maker, get_values(no_nutau_params),
    #        seed=get_seed(),channel='trck')

    # get background up-going
    #background_service = BackgroundServiceICC(up_template_settings['binning']['ebins'],czbins[czbins<=0],1.0,1.0,icc_bg_file=args.background_file)
    #background_dict = background_service.get_icc_bg()

    #background_maps = {'params': nutau['params']}
    #for flav in ['trck','cscd']:
    #    background_maps[flav] = {'map':background_dict[flav],
    #                             'ebins':up_template_settings['binning']['ebins'],
    #                             'czbins':czbins[czbins<=0]}
    #print 'background ' , background_maps

    nutau_up_and_down_cscd = sum_map(nutau_up['cscd'], nutau_down['cscd'])
    nutau_up_and_down_trck = sum_map(nutau_up['trck'], nutau_down['trck'])

    for channel in ['trck','cscd']:
        plt.figure()
        show_map(nutau_up[channel],vmax=150)
        print 'no. of upgoing ' ,channel , ' ', np.sum(nutau_up[channel]['map'])
        if args.save:
            print 'Saving %s channel...'%channel
            filename = os.path.join(args.outdir,args.title+ '_ERecoScale_%s_CzRecoScale_%s_f_1_up_'%(e_reco_scale_val,cz_reco_scale_val)+channel+'.png')
            plt.title(channel + ' (up) ERecoScale: %s CzRecoScale: %s Nevts: %.1f'%(e_reco_scale_val,cz_reco_scale_val, np.sum(nutau_up[channel]['map'])))
            plt.savefig(filename,dpi=150)
            plt.clf()
        show_map(nutau_down[channel],vmax=150)
        print 'no. of downgoing ', channel , ' ', np.sum(nutau_down[channel]['map'])
        if args.save:
            print 'Saving %s channel...'%channel
            filename = os.path.join(args.outdir,args.title+ '_ERecoScale_%s_CzRecoScale_%s_f_1_down_'%(e_reco_scale_val,cz_reco_scale_val)+channel+'.png')
            plt.title(channel + ' (down) ERecoScale: %s CzRecoScale: %s Nevts: %.1f'%(e_reco_scale_val,cz_reco_scale_val, np.sum(nutau_down[channel]['map'])))
            plt.savefig(filename,dpi=150)
            plt.clf()

    plt.figure()
    show_map(nutau_up_and_down_cscd,vmax=150)
    if args.save:
        filename = os.path.join(args.outdir,args.title+ '_ERecoScale_%s_CzRecoScale_%s_f_1_up_down_combined_'%(e_reco_scale_val,cz_reco_scale_val)+ 'cscd'+ '.png')
        plt.title('cscd ERecoScale: %s CzRecoScale: %s Nevts: %.1f'%(e_reco_scale_val,cz_reco_scale_val,np.sum(nutau_up_and_down_cscd['map'])))
        plt.savefig(filename,dpi=150)
        plt.clf()

    plt.figure()
    show_map(nutau_up_and_down_trck,vmax=150)
    if args.save:
        filename = os.path.join(args.outdir,args.title+ '_ERecoScale_%s_CzRecoScale_%s_f_1_up_down_combined_'%(e_reco_scale_val,cz_reco_scale_val)+ 'trck'+ '.png')
        plt.title('trck ERecoScale: %s CzRecoScale: %s Nevts: %.1f'%(e_reco_scale_val,cz_reco_scale_val,np.sum(nutau_up_and_down_trck['map'])))
        plt.savefig(filename,dpi=150)
        plt.clf()

    print "max no. of evts in nutau_up_cscd: ", np.amax(nutau_up['cscd']['map'])
    print "max no. of evts in nutau_up_trck: ", np.amax(nutau_up['trck']['map'])
    print "max no. of evts in nutau_down_cscd: ", np.amax(nutau_down['cscd']['map'])
    print "max no. of evts in nutau_down_trck: ", np.amax(nutau_down['trck']['map'])
    no_of_up = np.sum(nutau_up['trck']['map']) + np.sum(nutau_up['cscd']['map'])
    no_of_down = np.sum(nutau_down['trck']['map']) + np.sum(nutau_down['cscd']['map'])
    print 'no. of upgoing events= ' , no_of_up 
    print 'no. of downgoing events = ' , no_of_down 
    print 'Total no. of events = ' , no_of_up + no_of_down
    no_of_up_and_down = np.sum(nutau_up_and_down_trck['map']) + np.sum(nutau_up_and_down_cscd['map'])
    print 'From map_up_down, Total no. of events = ' , no_of_up_and_down

        #plt.figure()
        #show_map(background_maps[channel])
        #if args.save:
        #    print 'Saving %s channel...'%channel
        #    filename = os.path.join(args.outdir,args.title+'_background_'+channel+'.png')
        #    plt.title(args.title+'_background_'+channel)
        #    plt.savefig(filename,dpi=150)


    # Or equivalently, if args.all:
    #if type(fmap_tau_cscd) is tuple:
    #    plot_stages(fmap_tau_cscd,fmap_notau_trck,title='pseudo data',save=args.save,outdir=args.outdir)
    #else: plot_pid_stage(fmap_tau_cscd,fmap_notau_trck,title='pseudo data',save=args.save,outdir=args.outdir)
    #if type(nutau) is tuple:
    #    plot_stages(nutau,no_nutau,title=args.title,save=args.save,outdir=args.outdir)
    #else: plot_pid_stage(nutau,no_nutau,title=args.title,save=args.save,outdir=args.outdir)

    if not args.save: plt.show()
    else: print '\n-->>Saved all files to: ',args.outdir

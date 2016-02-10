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
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import scipy
from scipy.constants import Julian_year
from pisa.analysis.TemplateMaker_nutau import TemplateMaker
from pisa.utils.params import get_values, select_hierarchy_and_nutau_norm
import pisa.analysis.stats.Maps as Maps
from pisa.analysis.stats.Maps_nutau import get_up_map, get_flipped_down_map
from pisa.utils.log import set_verbosity,logging,profile
from pisa.resources.resources import find_resource
from pisa.utils.utils import Timer, is_linear, is_logarithmic, get_bin_centers
from pisa.utils.jsons import from_json
from pisa.utils.plot import show_map, sum_map, ratio_map, delta_map
from pisa.background.BackgroundServiceICC_nutau import BackgroundServiceICC
from pisa.analysis.stats.Maps_i3 import get_i3_maps

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
    parser.add_argument('--bg_scale',type=float,
                        help='atmos background scale value')
    parser.add_argument('-logE','--logE',action='store_true',default=False,
                        help='Energy in log scale.')
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
    set_verbosity(args.verbose)

    # get basic information
    template_settings = from_json(args.template_settings)
    template_settings['params']['icc_bg_file']['value'] = find_resource(args.background_file)
    template_settings['params']['atmos_mu_scale']['value'] = args.bg_scale

    ebins = template_settings['binning']['ebins']
    anlys_ebins = template_settings['binning']['anlys_ebins']
    czbins = template_settings['binning']['czbins']
    anlys_bins = (anlys_ebins,czbins)

    aeff_maps_from_i3 = get_i3_maps('aeff', cut_level='L6', year= 1, anlys_ebins=anlys_ebins, czbins=czbins)

    # plot aeff maps from i3 files
    total_no_i3_aeff = 0
    for flavor in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
        for int_type in ['cc','nc']:
            fig = plt.figure()
            show_map(aeff_maps_from_i3[flavor][int_type],logE=args.logE,annotate_prcs=1)
            if args.save:
                filename = os.path.join(args.outdir,args.title+ '_I3_file_aeff_event_rate_%s_%s_5.6_56GeV.png'% (flavor,int_type))
                plt.title(r'${\rm 1 \, yr \, I3 \, files \, aeff \, map \, %s \, %s \, (Nevts: \, %.1f) }$'%(flavor, int_type, np.sum(aeff_maps_from_i3[flavor][int_type]['map'])), fontsize='large')
                print ' From I3 files:   flavor ', flavor , ' ', int_type , ' , no. of evts = ', np.sum(aeff_maps_from_i3[flavor][int_type]['map'])
                total_no_i3_aeff += np.sum(aeff_maps_from_i3[flavor][int_type]['map'])
                plt.savefig(filename,dpi=150)
                plt.clf()
    print "In Effective Area stage, total_no_i3_aeff = ", total_no_i3_aeff
    print ' \n'


    ##################### Plot Maps from PISA #######################

    nominal_up_template_settings = copy.deepcopy(template_settings)
    nominal_up_template_settings['params']['reco_mc_wt_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_100_percent_up.hdf5'}
    nominal_up_template_settings['params']['reco_vbwkde_evts_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_10_percent_up.hdf5'}

    nominal_down_template_settings = copy.deepcopy(template_settings)
    nominal_down_template_settings['params']['reco_mc_wt_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_100_percent_down.hdf5'}
    nominal_down_template_settings['params']['pid_paramfile'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/pid/1X60_pid_down.json'}
    nominal_down_template_settings['params']['reco_vbwkde_evts_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_10_percent_down.hdf5'}

    with Timer() as t:
        nominal_template_maker_down = TemplateMaker(get_values(nominal_down_template_settings['params']), **nominal_down_template_settings['binning'])
        nominal_template_maker_up = TemplateMaker(get_values(nominal_up_template_settings['params']), **nominal_up_template_settings['binning'])

    profile.info('==> elapsed time to initialize templates: %s sec'%t.secs)

    # Make nutau template:
    nominal_nutau_up_params = copy.deepcopy(select_hierarchy_and_nutau_norm( nominal_up_template_settings['params'],True,1.0))
    nominal_nutau_down_params = copy.deepcopy(select_hierarchy_and_nutau_norm( nominal_down_template_settings['params'],True,1.0))
    nominal_no_nutau_up_params = copy.deepcopy(select_hierarchy_and_nutau_norm( nominal_up_template_settings['params'],True,0.0))
    nominal_no_nutau_down_params = copy.deepcopy(select_hierarchy_and_nutau_norm( nominal_down_template_settings['params'],True,0.0))

    with Timer(verbose=False) as t:
        nominal_nutau_up_all_stages = nominal_template_maker_up.get_template(get_values(nominal_nutau_up_params),return_stages=args.all)
        nominal_nutau_down_all_stages = nominal_template_maker_down.get_template(get_values(nominal_nutau_down_params),return_stages=args.all)
        nominal_no_nutau_up_all_stages = nominal_template_maker_up.get_template(get_values(nominal_no_nutau_up_params),return_stages=args.all)
        nominal_no_nutau_down_all_stages = nominal_template_maker_down.get_template(get_values(nominal_no_nutau_down_params),return_stages=args.all)
        aeff_map_nominal_nutau_up = nominal_nutau_up_all_stages[2]
        aeff_map_nominal_nutau_down = nominal_nutau_down_all_stages[2]
        aeff_map_nominal_no_nutau_up = nominal_no_nutau_up_all_stages[2]
        aeff_map_nominal_no_nutau_down = nominal_no_nutau_down_all_stages[2]
        nominal_nutau_up = nominal_nutau_up_all_stages[4]
        nominal_nutau_down = nominal_nutau_down_all_stages[4]
        nominal_no_nutau_up = nominal_no_nutau_up_all_stages[4]
        nominal_no_nutau_down = nominal_no_nutau_down_all_stages[4]
    profile.info('==> elapsed time to get NUTAU template: %s sec'%t.secs)

    # Get the analysis ebins (smaller than ebins)
    anlys_elements = []
    assert(len(anlys_ebins) <= len(ebins))
    for i in range(0,len(ebins)):
        if ebins[i] in anlys_ebins:
            anlys_elements.append(i)
    anlys_elements.pop()
    for flav in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
        for int_type in ['cc','nc']:
            reco_event_rate = aeff_map_nominal_nutau_up[flav][int_type]['map']
            reco_event_rate_anlys = reco_event_rate[:][anlys_elements]
            aeff_map_nominal_nutau_up[flav][int_type] = {'map': reco_event_rate_anlys,
                                                         'ebins': anlys_ebins,
                                                         'czbins': czbins}
            reco_event_rate = aeff_map_nominal_nutau_down[flav][int_type]['map']
            reco_event_rate_anlys = reco_event_rate[:][anlys_elements]
            aeff_map_nominal_nutau_down[flav][int_type] = {'map': reco_event_rate_anlys,
                                                         'ebins': anlys_ebins,
                                                         'czbins': czbins}

    #print 'aeff_map_nominal_nutau_up nue cc map = ', aeff_map_nominal_nutau_up['nue']['cc']['map']

    total_no_PISA_aeff = 0
    for flavor in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
        for int_type in ['cc','nc']:
            aeff_map_nominal_nutau = sum_map(aeff_map_nominal_nutau_up[flavor][int_type], aeff_map_nominal_nutau_down[flavor][int_type])
            #aeff_map_nominal_no_nutau = sum_map(aeff_map_nominal_no_nutau_up[flavor][int_type], aeff_map_nominal_no_nutau_down[flavor][int_type])
            # Plot Aeff map from PISA
            fig = plt.figure()
            show_map(aeff_map_nominal_nutau,logE=args.logE)
            if args.save:
                filename = os.path.join(args.outdir,args.title+ '_PISA_aeff_event_rate_%s_%s_5.6_56GeV.png'% (flavor,int_type))
                plt.title(r'${\rm 1 \, yr \, PISA \, aeff \, map \, %s \, %s \, (Nevts: \, %.1f) }$'%(flavor, int_type, np.sum(aeff_map_nominal_nutau['map'])), fontsize='large')
                print ' From PISA:   flavor ', flavor , ' ', int_type , ' , no. of evts = ', np.sum(aeff_map_nominal_nutau['map'])
                total_no_PISA_aeff += np.sum(aeff_map_nominal_nutau['map'])
                plt.savefig(filename,dpi=150)
                plt.clf()

            # Plot Aeff map (from PISA) to Aef map (from I3 files)
            ratio_aeff_pisa_aeff_i3 = ratio_map(aeff_map_nominal_nutau, aeff_maps_from_i3[flavor][int_type])
            plt.figure()
            show_map(ratio_aeff_pisa_aeff_i3, vmin= np.min(ratio_aeff_pisa_aeff_i3['map']), vmax= np.max(ratio_aeff_pisa_aeff_i3['map']))
            if args.save:
                filename = os.path.join(args.outdir,args.title+ '_Ratio_Aeff_PISA_Aeff_I3_%s_%s_'% (flavor, int_type)+ '.png')
                plt.title('Ratio of Aeff map (PISA/ I3 file) %s %s'% (flavor, int_type))
                plt.savefig(filename,dpi=150)
                plt.clf()

    print "In Effective Area stage, total_no_PISA_aeff = ", total_no_PISA_aeff
    print ' \n'


    #############  Compare PISA and I3Files Final Stage Template ############

    # get trck/cscd map from i3 files
    final_maps_from_i3 = get_i3_maps('final', cut_level ='L6', year= 1,anlys_ebins = anlys_ebins, czbins=czbins)

    # plot trck/cscd maps from i3 files
    total_no_i3_pid = 0
    for channel in ['cscd','trck']:
        fig = plt.figure()
        show_map(final_maps_from_i3[channel],logE=args.logE,vmax=250 if channel=='cscd' else 150)
        if args.save:
            filename = os.path.join(args.outdir,args.title+ '_I3_file_final_event_rate_%s_5.6_56GeV.png'% (channel))
            plt.title(r'${\rm 1 \, yr \, I3 \, files \, %s \, map \, (Nevts: \, %.1f) }$'%(channel, np.sum(final_maps_from_i3[channel]['map'])), fontsize='large')
            print ' From I3 files:  ', channel , ' , total no. of evts = ', np.sum(final_maps_from_i3[channel]['map'])
            total_no_i3_pid += np.sum(final_maps_from_i3[channel]['map'])
            plt.savefig(filename,dpi=150)
            plt.clf()
    print "In Final stage, total_no_i3_pid = ", total_no_i3_pid
    print ' \n'

    nominal_nutau_cscd = sum_map(nominal_nutau_up['cscd'], nominal_nutau_down['cscd'])
    nominal_nutau_trck = sum_map(nominal_nutau_up['trck'], nominal_nutau_down['trck'])
    nominal_no_nutau_cscd = sum_map(nominal_no_nutau_up['cscd'], nominal_no_nutau_down['cscd'])
    nominal_no_nutau_trck = sum_map(nominal_no_nutau_up['trck'], nominal_no_nutau_down['trck'])

    plt.figure()
    #show_map(nominal_nutau_cscd,vmax=np.max(nominal_nutau_cscd['map'])+10,logE=args.logE)
    show_map(nominal_nutau_cscd,vmax=250,logE=args.logE)
    if args.save:
        filename = os.path.join(args.outdir,args.title+ '_NutauCCNorm_1_'+ 'cscd_5.6_56GeV.png')
        plt.title(r'${\rm 1 \, yr \, PISA \, cscd \, (Nevts: \, %.1f) }$'%(np.sum(nominal_nutau_cscd['map'])), fontsize='large')
        plt.savefig(filename,dpi=150)
        plt.clf()

    plt.figure()
    #show_map(nominal_nutau_trck,vmax=np.max(nominal_nutau_trck['map'])+10,logE=args.logE)
    show_map(nominal_nutau_trck,vmax=150,logE=args.logE)
    if args.save:
        filename = os.path.join(args.outdir,args.title+ '_NutauCCNorm_1_'+ 'trck_5.6_56GeV.png')
        plt.title(r'${\rm 1 \, yr \, PISA \, trck \, (Nevts: \, %.1f) }$'%(np.sum(nominal_nutau_trck['map'])), fontsize='large')
        plt.savefig(filename,dpi=150)
        plt.clf()


    # Plot Ratio of Aeff map (from PISA) to Aef map (from I3 files)
    ratio_pid_pisa_pid_i3 = ratio_map(nominal_nutau_cscd, final_maps_from_i3['cscd'])
    plt.figure()
    show_map(ratio_pid_pisa_pid_i3, vmin= np.min(ratio_pid_pisa_pid_i3['map']), vmax= np.max(ratio_pid_pisa_pid_i3['map']),annotate_prcs=2)
    if args.save:
        filename = os.path.join(args.outdir,args.title+ '_Ratio_PID_PISA_PID_I3_cscd.png')
        plt.title('Ratio of PID map (PISA/ I3 file) cscd')
        plt.savefig(filename,dpi=150)
        plt.clf()

    ratio_pid_pisa_pid_i3 = ratio_map(nominal_nutau_trck, final_maps_from_i3['trck'])
    plt.figure()
    show_map(ratio_pid_pisa_pid_i3, vmin= np.min(ratio_pid_pisa_pid_i3['map']), vmax= np.max(ratio_pid_pisa_pid_i3['map']),annotate_prcs=2)
    if args.save:
        filename = os.path.join(args.outdir,args.title+ '_Ratio_PID_PISA_PID_I3_trck.png')
        plt.title('Ratio of PID map (PISA/ I3 file) trck')
        plt.savefig(filename,dpi=150)
        plt.clf()


    print "In Final stage, from PISA :"
    print 'no. of nominal_nutau_cscd = ', np.sum(nominal_nutau_cscd['map'])
    print 'no. of nominal_nutau_trck = ', np.sum(nominal_nutau_trck['map'])
    print ' total of the above two : ', np.sum(nominal_nutau_cscd['map'])+np.sum(nominal_nutau_trck['map'])
    print ' \n'
    print 'no. of nominal_no_nutau_cscd = ', np.sum(nominal_no_nutau_cscd['map'])
    print 'no. of nominal_no_nutau_trck = ', np.sum(nominal_no_nutau_trck['map'])
    print ' total of the above two : ', np.sum(nominal_no_nutau_cscd['map'])+np.sum(nominal_no_nutau_trck['map'])
    print ' \n'

    nominal_nutau_no_pid = sum_map(nominal_nutau_cscd, nominal_nutau_trck)
    final_maps_from_i3_no_pid = sum_map(final_maps_from_i3['cscd'], final_maps_from_i3['trck'])
    ratio_pisa_i3_no_pid = ratio_map(nominal_nutau_no_pid, final_maps_from_i3_no_pid)

    plt.figure()
    show_map(nominal_nutau_no_pid,vmax=400,logE=args.logE,annotate_prcs=1)
    if args.save:
        filename = os.path.join(args.outdir,args.title+ '_NutauCCNorm_1_'+ 'all_channel_5.6_56GeV.png')
        plt.title(r'${\rm 1 \, yr \, PISA \, cscd \, + \, trck \, (Nevts: \, %.1f) }$'%(np.sum(nominal_nutau_no_pid['map'])), fontsize='large')
        plt.savefig(filename,dpi=150)
        plt.clf()

    fig = plt.figure()
    show_map(final_maps_from_i3_no_pid,logE=args.logE,vmax=400,annotate_prcs=1)
    if args.save:
        filename = os.path.join(args.outdir,args.title+ '_I3_file_final_event_rate_all_channel_5.6_56GeV.png')
        plt.title(r'${\rm 1 \, yr \, I3 \, files \, cscd \, + \, trck \, map \, (Nevts: \, %.1f) }$'%(np.sum(final_maps_from_i3_no_pid['map'])), fontsize='large')
        print ' From I3 files:  cscd+trck, total no. of evts = ', np.sum(final_maps_from_i3_no_pid['map'])
        plt.savefig(filename,dpi=150)
        plt.clf()

    plt.figure()
    show_map(ratio_pisa_i3_no_pid, vmin= np.min(ratio_pisa_i3_no_pid['map']), vmax= np.max(ratio_pisa_i3_no_pid['map']),annotate_prcs=2)
    if args.save:
        filename = os.path.join(args.outdir,args.title+ '_Ratio_PID_PISA_PID_I3_all_channel.png')
        plt.title('Ratio of PID map (PISA/ I3 file) cscd + trck')
        plt.savefig(filename,dpi=150)
        plt.clf()

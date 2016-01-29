#! /usr/bin/env python
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#         Feifei Huang
#         fxh140@psu.edu
#
# date: 16 March 2015
#
# Plots the final cscd and trck templates (nutau CC norm = 1). Also can use other nutau CC norm values.
#

import copy
import numpy as np
import os
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.analysis.TemplateMaker_nutau import TemplateMaker
from pisa.utils.params import get_values, select_hierarchy_and_nutau_norm
from pisa.analysis.stats.Maps import get_seed
from pisa.utils.log import set_verbosity,logging,profile
from pisa.resources.resources import find_resource
from pisa.utils.utils import Timer
from pisa.utils.jsons import from_json
from pisa.utils.plot import show_map, sum_map, ratio_map, delta_map
from pisa.background.BackgroundServiceICC_nutau import BackgroundServiceICC

if __name__ == '__main__':

    set_verbosity(0)
    parser = ArgumentParser(description='''Plots the final PISA templates.''')
    parser.add_argument('template_settings',metavar='JSON',
                        help='Settings file to use for template generation')
    parser.add_argument('--background_file',metavar='FILE',type=str,
                        default='background/Matt_L5b_icc_data_IC86_2_3_4.hdf5',
                        help='''HDF5 File containing atmospheric background from 3 years'
                        inverted corridor cut data''')
    parser.add_argument('-logE','--logE',action='store_true',default=False,
                        help='Energy in log scale.')
    parser.add_argument('--bg_scale',type=float,
                        help="atmos background scale value")
    parser.add_argument('-y','--y',type=float,
                        help='No. of livetime[ unit: Julian year]')
    parser.add_argument('--plot_f_1_0_diff',action='store_true', default= False,
                        help='Plot template different between f=1 and f=0')
    parser.add_argument('--plot_other_nutau_norm',action='store_true', default= False,
                        help='Plot also templates with nutau CC norm value != 1.')
    parser.add_argument('--val',type=float,
                        help='Plot also templates with nutau CC norm value != 1.')
    parser.add_argument('-a','--all',action='store_true',default=False,
                        help='Plot all stages 1-5 of templates and Asymmetry')
    parser.add_argument('--plot_background',action='store_true',default=False,
                        help='Plot background(from ICC data)')
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

    template_settings = from_json(args.template_settings)
    template_settings = from_json(args.template_settings)
    template_settings['params']['icc_bg_file']['value'] = find_resource(args.background_file)
    template_settings['params']['atmos_mu_scale']['value'] = args.bg_scale
    template_settings['params']['livetime']['value'] = args.y
    
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

    with Timer(verbose=False) as t:
        print "getting nominal_nutau_up "
        nominal_nutau_up = nominal_template_maker_up.get_template(get_values(nominal_nutau_up_params),return_stages=args.all)
        nominal_nutau_down = nominal_template_maker_down.get_template(get_values(nominal_nutau_down_params),return_stages=args.all)
    profile.info('==> elapsed time to get NUTAU template: %s sec'%t.secs)

    #### Plot Background #####
    if args.plot_background:
        # get background
        czbins = nominal_up_template_settings['binning']['czbins']
        up_background_service = BackgroundServiceICC(nominal_up_template_settings['binning']['anlys_ebins'],czbins[czbins<=0],**get_values(nominal_nutau_up_params))
        up_background_dict = up_background_service.get_icc_bg()
        down_background_service = BackgroundServiceICC(nominal_down_template_settings['binning']['anlys_ebins'],czbins[czbins>=0],**get_values(nominal_nutau_down_params))
        down_background_dict = down_background_service.get_icc_bg()

        up_background_maps = {'params': nominal_nutau_up['params']}
        for flav in ['trck','cscd']:
            up_background_maps[flav] = {'map':up_background_dict[flav],
                                     'ebins':nominal_up_template_settings['binning']['anlys_ebins'],
                                     'czbins':czbins[czbins<=0]}
        down_background_maps = {'params': nominal_nutau_down['params']}
        for flav in ['trck','cscd']:
            down_background_maps[flav] = {'map':down_background_dict[flav],
                                     'ebins':nominal_down_template_settings['binning']['anlys_ebins'],
                                     'czbins':czbins[czbins>=0]}
        for channel in ['trck','cscd']:
            plt.figure()
            show_map(up_background_maps[channel],logE=args.logE)
            if args.save:
                filename = os.path.join(args.outdir,args.title+'_upgoing_background_'+channel+'.png')
                plt.title(args.title+'_upgoing_background_'+channel)
                plt.savefig(filename,dpi=150)
                plt.clf()
            plt.figure()
            show_map(down_background_maps[channel],logE=args.logE)
            if args.save:
                filename = os.path.join(args.outdir,args.title+'_downgoing_background_'+channel+'.png')
                plt.title(args.title+'_downgoing_background_'+channel)
                plt.savefig(filename,dpi=150)
                plt.clf()
        no_totl = np.sum(down_background_maps['trck']['map']) + np.sum(down_background_maps['cscd']['map']) + np.sum(up_background_maps['trck']['map']) + np.sum(up_background_maps['cscd']['map'])
        print "total no. of background events = ", no_totl


    ###### Plot nominal templates #####
    nominal_nutau_cscd = sum_map(nominal_nutau_up['cscd'], nominal_nutau_down['cscd'])
    nominal_nutau_trck = sum_map(nominal_nutau_up['trck'], nominal_nutau_down['trck'])

    plt.figure()
    show_map(nominal_nutau_cscd,vmax=np.max(nominal_nutau_cscd['map'])+10,logE=args.logE)
    if args.save:
        filename = os.path.join(args.outdir,args.title+ '_%s_yr_bg_scale_%s_NutauCCNorm_1_' % (args.y, args.bg_scale) + 'cscd_5.6_56GeV.png')
        plt.title(r'${\rm %s \, yr \, cscd \, (Nevts: \, %.1f, \, bg \, scale \, %s) }$'%(args.y, np.sum(nominal_nutau_cscd['map']), args.bg_scale), fontsize='large')
        plt.savefig(filename,dpi=150)
        plt.clf()

    plt.figure()
    show_map(nominal_nutau_trck,vmax=np.max(nominal_nutau_trck['map'])+10,logE=args.logE)
    if args.save:
        filename = os.path.join(args.outdir,args.title+ '_%s_yr_bg_scale_%s_NutauCCNorm_1_' % (args.y, args.bg_scale) + 'trck_5.6_56GeV.png')
        plt.title(r'${\rm %s \, yr \, trck \, (Nevts: \, %.1f, \, bg \, scale \, %s) }$'%(args.y, np.sum(nominal_nutau_trck['map']), args.bg_scale), fontsize='large')
        plt.savefig(filename,dpi=150)
        plt.clf()

    print "no. of nominal_nutau_cscd = ", np.sum(nominal_nutau_cscd['map'])
    print "no. of nominal_nutau_trck = ", np.sum(nominal_nutau_trck['map'])
    print " total of the above two : ", np.sum(nominal_nutau_cscd['map'])+np.sum(nominal_nutau_trck['map'])
    print " \n"
    
    if args.plot_other_nutau_norm:
        other_nutau_norm_val_up_params = copy.deepcopy(select_hierarchy_and_nutau_norm( nominal_up_template_settings['params'],True,args.val))
        other_nutau_norm_val_down_params = copy.deepcopy(select_hierarchy_and_nutau_norm( nominal_down_template_settings['params'],True, args.val))
        other_nutau_norm_val_up = nominal_template_maker_up.get_template(get_values(other_nutau_norm_val_up_params),return_stages=args.all)
        other_nutau_norm_val_down = nominal_template_maker_down.get_template(get_values(other_nutau_norm_val_down_params),return_stages=args.all)
        other_nutau_norm_val_cscd = sum_map(other_nutau_norm_val_up['cscd'], other_nutau_norm_val_down['cscd'])
        other_nutau_norm_val_trck = sum_map(other_nutau_norm_val_up['trck'], other_nutau_norm_val_down['trck'])
        plt.figure()
        show_map(other_nutau_norm_val_cscd,vmax=np.max(other_nutau_norm_val_cscd['map'])+10,logE=args.logE)
        if args.save:
            filename = os.path.join(args.outdir,args.title+ '_%s_yr_bg_scale_%s_NutauCCNorm_%s_' % (args.y, args.bg_scale, args.val) + 'cscd_5.6_56GeV.png')
            plt.title(r'${\rm %s \, yr \, cscd \, (\nu_{\tau} \, CC \, norm: \, %s , \, Nevts: \, %.1f, \, bg \, scale \, %s) }$'%(args.y, args.val, np.sum(other_nutau_norm_val_cscd['map']), args.bg_scale), fontsize='large')
            plt.savefig(filename,dpi=150)
            plt.clf()

        plt.figure()
        show_map(other_nutau_norm_val_trck,vmax=np.max(other_nutau_norm_val_trck['map'])+10,logE=args.logE)
        if args.save:
            filename = os.path.join(args.outdir,args.title+ '_%s_yr_bg_scale_%s_NutauCCNorm_%s_' % (args.y, args.bg_scale, args.val) + 'trck_5.6_56GeV.png')
            plt.title(r'${\rm %s \, yr \, trck \, (\nu_{\tau} \, CC \, norm: \, %s , \, Nevts: \, %.1f, \, bg \, scale \, %s) }$'%(args.y, args.val, np.sum(other_nutau_norm_val_trck['map']), args.bg_scale), fontsize='large')
            plt.savefig(filename,dpi=150)
            plt.clf()

        print "no. of other_nutau_norm_val_cscd = ", np.sum(other_nutau_norm_val_cscd['map'])
        print "no. of other_nutau_norm_val_trck = ", np.sum(other_nutau_norm_val_trck['map'])
        print " total of the above two : ", np.sum(other_nutau_norm_val_cscd['map'])+np.sum(other_nutau_norm_val_trck['map'])
        print " \n"
   
    if args.plot_f_1_0_diff:
        nominal_no_nutau_up_params = copy.deepcopy(select_hierarchy_and_nutau_norm( nominal_up_template_settings['params'],True,0.0))
        nominal_no_nutau_down_params = copy.deepcopy(select_hierarchy_and_nutau_norm( nominal_down_template_settings['params'],True,0.0))
        nominal_no_nutau_up = nominal_template_maker_up.get_template(get_values(nominal_no_nutau_up_params),return_stages=args.all)
        nominal_no_nutau_down = nominal_template_maker_down.get_template(get_values(nominal_no_nutau_down_params),return_stages=args.all)
        nominal_no_nutau_cscd = sum_map(nominal_no_nutau_up['cscd'], nominal_no_nutau_down['cscd'])
        nominal_no_nutau_trck = sum_map(nominal_no_nutau_up['trck'], nominal_no_nutau_down['trck'])
        nominal_nutau_minus_no_nutau_cscd = delta_map(nominal_nutau_cscd, nominal_no_nutau_cscd)
        nominal_nutau_minus_no_nutau_trck = delta_map(nominal_nutau_trck, nominal_no_nutau_trck)
        plt.figure()
        show_map(nominal_nutau_minus_no_nutau_cscd,vmax=20, logE=args.logE, xlabel = r'${\rm cos(zenith)}$', ylabel = r'${\rm Energy[GeV]}$')
        if args.save:
            scale_E = 'logE' if args.logE else 'linE'
            filename = os.path.join(args.outdir,args.title+ '_%s_yr_NutauCCNorm_1_minus_0_' % (args.y) + scale_E+ '_cscd_5.6_56GeV.png')
            plt.title(r'${\rm 1 \, yr \, cascade \, (Nevts: \, %.1f) }$'%(np.sum(nominal_nutau_minus_no_nutau_cscd['map'])), fontsize='large')
            plt.savefig(filename,dpi=150)
            plt.clf()

        plt.figure()
        show_map(nominal_nutau_minus_no_nutau_trck,vmax=5, logE=args.logE, xlabel = r'${\rm cos(zenith)}$', ylabel = r'${\rm Energy[GeV]}$')
        if args.save:
            scale_E = 'logE' if args.logE else 'linE'
            filename = os.path.join(args.outdir,args.title+ '_%s_yr_NutauCCNorm_1_minus_0_' % (args.y) + scale_E+ '_trck_5.6_56GeV.png')
            plt.title(r'${\rm 1 \, yr \, track \, (Nevts: \, %.1f) }$'%(np.sum(nominal_nutau_minus_no_nutau_trck['map'])), fontsize='large')
            plt.savefig(filename,dpi=150)
            plt.clf()


    if not args.save: plt.show()
    else: print '\n-->>Saved all files to: ',args.outdir

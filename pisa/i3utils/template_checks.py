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
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

#from pisa.analysis.TemplateMaker_nutau import TemplateMaker
from pisa.analysis.TemplateMaker_MC import TemplateMaker
from pisa.utils.params import get_values, select_hierarchy_and_nutau_norm
from pisa.utils.log import set_verbosity,logging,profile
from pisa.resources.resources import find_resource
from pisa.utils.utils import Timer
from pisa.utils.jsons import from_json
from pisa.utils.plot import show_map, sum_map, ratio_map, delta_map
from pisa.background.BackgroundServiceICC import BackgroundServiceICC

if __name__ == '__main__':

    set_verbosity(0)
    parser = ArgumentParser(description='''Plots the final PISA templates.''')
    parser.add_argument('template_settings',metavar='JSON',
                        help='Settings file to use for template generation')
    parser.add_argument('--background_file',metavar='FILE',type=str,
                        default='background/Matt_L5b_icc_data_IC86_2_3_4.hdf5',
                        help='''HDF5 File containing atmospheric background from 3 years'
                        inverted corridor cut data''')
    parser.add_argument('-no_logE','--no_logE',action='store_true',default=False,
                        help='Energy in log scale.')
    parser.add_argument('--bg_scale',type=float, default = 0,
                        help='atmos background scale value')
    parser.add_argument('-y','--y',type=float,
                        help='No. of livetime[ unit: Julian year]')
    parser.add_argument('--plot_aeff_maps',action='store_true', default= False,
                        help='Plot aeff maps') 
    parser.add_argument('--plot_f_1_0_diff',action='store_true', default= False,
                        help='Plot template different between f=1 and f=0')
    parser.add_argument('--plot_NuFluxIP_PISAIP',action='store_true', default= False,
                        help='Plot template different between using NuFlux_IP flux and PISA_IP flux')
    parser.add_argument('--plot_other_nutau_norm',action='store_true', default= False,
                        help='Plot also templates with nutau CC norm value != 1.')
    parser.add_argument('--val',type=float,
                        help='The value of nutau CC norm value != 1.')
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
    template_settings['params']['icc_bg_file']['value'] = find_resource(args.background_file)
    template_settings['params']['atmos_mu_scale']['value'] = args.bg_scale
    template_settings['params']['livetime']['value'] = args.y
    template_settings['params']['flux_mode']['value'] = 'nuflux_ip' 
    
    with Timer() as t:
        nominal_template_maker = TemplateMaker(get_values(template_settings['params']), **template_settings['binning'])
    profile.info('==> elapsed time to initialize templates: %s sec'%t.secs)

    # Make nutau template:
    nominal_nutau_params = copy.deepcopy(select_hierarchy_and_nutau_norm( template_settings['params'],True,1.0))

    with Timer(verbose=False) as t:
        print 'getting nominal_nutau '
        #nominal_nutau_all = nominal_template_maker.get_template(get_values(nominal_nutau_params),return_stages=True)
        #nominal_nutau = nominal_nutau_all[5]
        nominal_nutau = nominal_template_maker.get_template(get_values(nominal_nutau_params),return_stages=args.all)
    profile.info('==> elapsed time to get NUTAU template: %s sec'%t.secs)

    if args.plot_aeff_maps:
        aeff_maps = nominal_template_maker.get_template(get_values(nominal_nutau_params),return_stages=False, return_aeff_maps=True)

    #### Plot Background #####
    if args.plot_background:
        # get background
        czbins = template_settings['binning']['czbins']
        background_service = BackgroundServiceICC(template_settings['binning']['anlys_ebins'],**get_values(nominal_nutau_params))
        background_dict = background_service.get_icc_bg()

        background_maps = {'params': nominal_nutau['params']}
        for flav in ['trck','cscd']:
            background_maps[flav] = {'map':background_dict[flav],
                                     'ebins':template_settings['binning']['anlys_ebins'],
                                     'czbins':czbins}
        for channel in ['trck','cscd']:
            plt.figure()
            show_map(background_maps[channel],logE=not(args.no_logE))
            if args.save:
                filename = os.path.join(args.outdir,args.title+'_background_'+channel+'.png')
                plt.title(args.title+'_background_'+channel)
                plt.savefig(filename,dpi=150)
                plt.clf()
            plt.figure()
        no_totl = np.sum(np.sum(background_maps['trck']['map']) + np.sum(background_maps['cscd']['map']))
        print 'total no. of background events = ', no_totl


    ###### Plot nominal templates #####

    for flav in ['cscd', 'trck']:
        plt.figure()
        show_map(nominal_nutau[flav],vmax=np.max(nominal_nutau[flav]['map'])+10,logE=not(args.no_logE))
        if args.save:
            filename = os.path.join(args.outdir,args.title+ '_%s_yr_bg_scale_%s_NutauCCNorm_1_' % (args.y, args.bg_scale) + '%s.png'%flav)
            plt.title(r'${\rm %s \, yr \, %s \, (Nevts: \, %.1f, \, bg \, scale \, %s) }$'%(args.y, flav.replace('_',''), np.sum(nominal_nutau[flav]['map']), args.bg_scale), fontsize='large')
            plt.savefig(filename,dpi=150)
            plt.clf()

    print 'no. of nominal_nutau_cscd = ', np.sum(nominal_nutau['cscd']['map'])
    print 'no. of nominal_nutau_trck = ', np.sum(nominal_nutau['trck']['map'])
    print ' total of the above two : ', np.sum(nominal_nutau['cscd']['map'])+np.sum(nominal_nutau['trck']['map'])
    print ' \n'

    if args.plot_aeff_maps:
        for flav in ['numu', 'numu_bar', 'nue', 'nue_bar', 'nutau', 'nutau_bar']:
            for int_type in ['cc', 'nc']:
                plt.figure()
                show_map(aeff_maps[flav][int_type],vmax=np.max(aeff_maps[flav][int_type]['map'])+10,logE=not(args.no_logE))
                if args.save:
                    filename = os.path.join(args.outdir,args.title+ '_%s_yr_bg_scale_%s_NutauCCNorm_1_EffectiveAreaMap_' % (args.y, args.bg_scale) + '%s_%s.png'%(flav, int_type))
                    plt.title(r'${\rm %s \, yr \, %s \, %s \, Effective \, Area \, map }$'%(args.y, flav.replace('_',''), int_type), fontsize='large')
                    plt.savefig(filename,dpi=150)
                    plt.clf()

    if args.plot_other_nutau_norm:
        other_nutau_norm_val_params = copy.deepcopy(select_hierarchy_and_nutau_norm( template_settings['params'],True,args.val))
        other_nutau_norm_val = nominal_template_maker.get_template(get_values(other_nutau_norm_val_params),return_stages=args.all)
        for flav in ['cscd', 'trck']:
            plt.figure()
            show_map(other_nutau_norm_val[flav],vmax=np.max(other_nutau_norm_val[flav]['map'])+10,logE=not(args.no_logE))
            if args.save:
                filename = os.path.join(args.outdir,args.title+ '_%s_yr_bg_scale_%s_NutauCCNorm_%s_' % (args.y, args.bg_scale, args.val) + '%s.png'% flav)
                plt.title(r'${\rm %s \, yr \, %s \, (\nu_{\tau} \, CC \, norm: \, %s , \, Nevts: \, %.1f, \, bg \, scale \, %s) }$'%(args.y, flav.replace('_',''), args.val, np.sum(other_nutau_norm_val[flav]['map']), args.bg_scale), fontsize='large')
                plt.savefig(filename,dpi=150)
                plt.clf()

        print 'no. of other_nutau_norm_val_cscd = ', np.sum(other_nutau_norm_val['cscd']['map'])
        print 'no. of other_nutau_norm_val_trck = ', np.sum(other_nutau_norm_val['trck']['map'])
        print ' total of the above two : ', np.sum(other_nutau_norm_val['cscd']['map'])+np.sum(other_nutau_norm_val['trck']['map'])
        print ' \n'
   
    if args.plot_f_1_0_diff:
        nominal_no_nutau_params = copy.deepcopy(select_hierarchy_and_nutau_norm( template_settings['params'],True,0.0))
        no_nominal_template_maker = TemplateMaker(get_values(nominal_no_nutau_params), **template_settings['binning'])
        nominal_no_nutau = no_nominal_template_maker.get_template(get_values(nominal_no_nutau_params),return_stages=args.all)
        nominal_nutau_minus_no_nutau = {}
        nominal_nutau_minus_no_nutau['cscd'] = delta_map(nominal_nutau['cscd'], nominal_no_nutau['cscd'])
        nominal_nutau_minus_no_nutau['trck'] = delta_map(nominal_nutau['trck'], nominal_no_nutau['trck'])

        print 'no. of nominal_no_nutau_cscd = ', np.sum(nominal_no_nutau['cscd']['map'])
        print 'no. of nominal_no_nutau_trck = ', np.sum(nominal_no_nutau['trck']['map'])
        print ' total of the above two : ', np.sum(nominal_no_nutau['cscd']['map'])+np.sum(nominal_no_nutau['trck']['map'])
        print ' \n'

        for flav in ['cscd', 'trck']:
            plt.figure()
            show_map(nominal_nutau_minus_no_nutau[flav],vmax=20 if flav=='cscd' else 5, logE=not(args.no_logE), xlabel = r'${\rm cos(zenith)}$', ylabel = r'${\rm Energy[GeV]}$')
            if args.save:
                scale_E = 'linE' if args.no_logE else 'logE'
                filename = os.path.join(args.outdir,args.title+ '_%s_yr_NutauCCNorm_1_minus_0_' % (args.y) + scale_E+ '_%s.png'%flav)
                plt.title(r'${\rm 1 \, yr \, %s \, (Nevts: \, %.1f) }$'%(flav.replace('_',''), np.sum(nominal_nutau_minus_no_nutau[flav]['map'])), fontsize='large')
                plt.savefig(filename,dpi=150)
                plt.clf()

    if args.plot_NuFluxIP_PISAIP:
        nominal_pisa_ip_settings = copy.deepcopy(template_settings)
        nominal_pisa_ip_settings['params']['flux_mode']['value'] = 'integral-preserving' 
        nominal_pisa_ip_params = copy.deepcopy(select_hierarchy_and_nutau_norm( nominal_pisa_ip_settings['params'],True,1.0))
        nominal_pisa_ip_template_maker = TemplateMaker(get_values(nominal_pisa_ip_params), **template_settings['binning'])
        with Timer() as t:
            nominal_pisa_ip = nominal_pisa_ip_template_maker.get_template(get_values(nominal_pisa_ip_params),return_stages=args.all)
        profile.info('==> elapsed time to get NUTAU template (using PISA_IP): %s sec'%t.secs)

        nominal_nuflux_ip_minus_pisa_ip = {}
        nominal_nuflux_ip_minus_pisa_ip['cscd'] = delta_map(nominal_nutau['cscd'], nominal_pisa_ip['cscd'])
        nominal_nuflux_ip_minus_pisa_ip['trck'] = delta_map(nominal_nutau['trck'], nominal_pisa_ip['trck'])

        print 'no. of nominal_pisa_ip_cscd = ', np.sum(nominal_pisa_ip['cscd']['map'])
        print 'no. of nominal_pisa_ip_trck = ', np.sum(nominal_pisa_ip['trck']['map'])
        print ' total of the above two : ', np.sum(nominal_pisa_ip['cscd']['map'])+np.sum(nominal_pisa_ip['trck']['map'])
        print ' \n'

        for flav in ['cscd', 'trck']:
            plt.figure()
            show_map(nominal_nuflux_ip_minus_pisa_ip[flav],vmax=1 if flav=='cscd' else 1, logE=not(args.no_logE), xlabel = r'${\rm cos(zenith)}$', ylabel = r'${\rm Energy[GeV]}$')
            if args.save:
                scale_E = 'linE' if args.no_logE else 'logE'
                filename = os.path.join(args.outdir,args.title+ '_%s_yr_NutauCCNorm_NuFluxIP_minus_PISAIP_' % (args.y) + scale_E+ '_%s.png'%flav)
                plt.title(r'${\rm 1 \, yr \, %s \, (Nevts: \, %.1f) }$'%(flav.replace('_',''), np.sum(nominal_nuflux_ip_minus_pisa_ip[flav]['map'])), fontsize='large')
                plt.savefig(filename,dpi=150)
                plt.clf()


    if not args.save: plt.show()
    else: print '\n-->>Saved all files to: ',args.outdir

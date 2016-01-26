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

from pisa.analysis.TemplateMaker_nutau import TemplateMaker
from pisa.utils.params import get_values, select_hierarchy_and_nutau_norm
from pisa.analysis.stats.Maps import get_seed
from pisa.utils.log import set_verbosity,logging,profile
from pisa.utils.utils import Timer
from pisa.utils.jsons import from_json
from pisa.utils.plot import show_map, sum_map, ratio_map, delta_map
from pisa.background.BackgroundServiceICC_nutau import BackgroundServiceICC

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
    parser.add_argument('-logE','--logE',action='store_true',default=False,
                        help='Energy in log scale.')
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

    template_settings = from_json(args.template_settings)
    
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
        print "getting nominal_nutau_up "
        nominal_nutau_up = nominal_template_maker_up.get_template(get_values(nominal_nutau_up_params),return_stages=args.all)
        nominal_nutau_down = nominal_template_maker_down.get_template(get_values(nominal_nutau_down_params),return_stages=args.all)
        nominal_no_nutau_up = nominal_template_maker_up.get_template(get_values(nominal_no_nutau_up_params),return_stages=args.all)
        nominal_no_nutau_down = nominal_template_maker_down.get_template(get_values(nominal_no_nutau_down_params),return_stages=args.all)
    profile.info('==> elapsed time to get NUTAU template: %s sec'%t.secs)


    #print 'nominal nutau_up = ',   nominal_nutau_up
    #print 'nominal nutau_down = ', nominal_nutau_down

    # get background
    czbins = nominal_up_template_settings['binning']['czbins']
    print "czbins = ", czbins
    up_background_service = BackgroundServiceICC(nominal_up_template_settings['binning']['ebins'],czbins[czbins<=0],**get_values(nominal_nutau_up_params))
    up_background_dict = up_background_service.get_icc_bg()
    #print "up_background_dict = ", up_background_dict
    down_background_service = BackgroundServiceICC(nominal_down_template_settings['binning']['ebins'],czbins[czbins>=0],**get_values(nominal_nutau_down_params))
    down_background_dict = down_background_service.get_icc_bg()
    #print "down_background_dict = ", down_background_dict

    up_background_maps = {'params': nominal_nutau_up['params']}
    for flav in ['trck','cscd']:
        up_background_maps[flav] = {'map':up_background_dict[flav],
                                 'ebins':nominal_up_template_settings['binning']['ebins'],
                                 'czbins':czbins[czbins<=0]}
    down_background_maps = {'params': nominal_nutau_down['params']}
    for flav in ['trck','cscd']:
        down_background_maps[flav] = {'map':down_background_dict[flav],
                                 'ebins':nominal_down_template_settings['binning']['ebins'],
                                 'czbins':czbins[czbins>=0]}

    nominal_nutau_up_and_down_cscd = sum_map(nominal_nutau_up['cscd'], nominal_nutau_down['cscd'])
    nominal_nutau_up_and_down_trck = sum_map(nominal_nutau_up['trck'], nominal_nutau_down['trck'])
    nominal_no_nutau_up_and_down_cscd = sum_map(nominal_no_nutau_up['cscd'], nominal_no_nutau_down['cscd'])
    nominal_no_nutau_up_and_down_trck = sum_map(nominal_no_nutau_up['trck'], nominal_no_nutau_down['trck'])

    nominal_nutau_minus_no_nutau_up_and_down_cscd = delta_map(nominal_nutau_up_and_down_cscd, nominal_no_nutau_up_and_down_cscd)
    nominal_nutau_minus_no_nutau_up_and_down_trck = delta_map(nominal_nutau_up_and_down_trck, nominal_no_nutau_up_and_down_trck)
    print "nominal_nutau_minus_no_nutau_up_and_down_trck = ", nominal_nutau_minus_no_nutau_up_and_down_trck

    #### Plot nutau_up , nutau_down, etc. ####
    #for channel in ['trck','cscd']:
    #    plt.figure()
    #    show_map(nominal_nutau_up[channel],vmax=25 if channel=='cscd' else 10,logE=args.logE)
    #    print 'no. of upgoing ' ,channel , ' ', np.sum(nominal_nutau_up[channel]['map'])
    #    if args.save:
    #        print 'Saving %s channel...'%channel
    #        filename = os.path.join(args.outdir,args.title+ '_f_1_up_'+channel+'.png')
    #        plt.title(channel + ' (up) Nevts: %.1f '%(np.sum(nominal_nutau_up[channel]['map'])))
    #        plt.savefig(filename,dpi=150)
    #        plt.clf()

    #    show_map(nominal_nutau_down[channel],vmax=10 if channel=='cscd' else 10,logE=args.logE)
    #    print 'no. of downgoing ', channel , ' ', np.sum(nominal_nutau_down[channel]['map'])
    #    if args.save:
    #        print 'Saving %s channel...'%channel
    #        filename = os.path.join(args.outdir,args.title+ '_f_1_down_'+channel+'.png')
    #        plt.title(channel + ' (down) Nevts: %.1f '%(np.sum(nominal_nutau_down[channel]['map'])))
    #        plt.savefig(filename,dpi=150)
    #        plt.clf()

    #plt.figure()
    #show_map(nominal_nutau_up_and_down_cscd,vmax=25,logE=args.logE)
    #if args.save:
    #    filename = os.path.join(args.outdir,args.title+ '_f_1_up_down_combined_'+ 'cscd.png')
    #    plt.title('cscd (up+down) Nevts: %.1f '%(np.sum(nominal_nutau_up_and_down_cscd['map'])))
    #    plt.savefig(filename,dpi=150)
    #    plt.clf()

    #plt.figure()
    #show_map(nominal_nutau_up_and_down_trck,vmax=10,logE=args.logE)
    #if args.save:
    #    filename = os.path.join(args.outdir,args.title+ '_f_1_up_down_combined_'+ 'trck.png')
    #    plt.title('trck (up+down) Nevts: %.1f '%(np.sum(nominal_nutau_up_and_down_trck['map'])))
    #    plt.savefig(filename,dpi=150)
    #    plt.clf()

    plt.figure()
    show_map(nominal_nutau_minus_no_nutau_up_and_down_cscd,vmax=20, logE=args.logE, xlabel = r'${\rm cos(zenith)}$', ylabel = r'${\rm Energy[GeV]}$')
    if args.save:
        scale_E = 'logE' if args.logE else 'linE'
        filename = os.path.join(args.outdir,args.title+ '_NutauCCNorm_1_minus_0_'+ scale_E+ '_up_down_combined_'+ 'cscd_5.6_56GeV.png')
        plt.title(r'${\rm 1 \, yr \, cascade \, (Nevts: \, %.1f) }$'%(np.sum(nominal_nutau_minus_no_nutau_up_and_down_cscd['map'])), fontsize='large')
        plt.savefig(filename,dpi=150)
        plt.clf()

    plt.figure()
    show_map(nominal_nutau_minus_no_nutau_up_and_down_trck,vmax=5, logE=args.logE, xlabel = r'${\rm cos(zenith)}$', ylabel = r'${\rm Energy[GeV]}$')
    if args.save:
        scale_E = 'logE' if args.logE else 'linE'
        filename = os.path.join(args.outdir,args.title+ '_NutauCCNorm_1_minus_0_' + scale_E + '_up_down_combined_'+ 'trck_5.6_56GeV.png')
        plt.title(r'${\rm 1 \, yr \, track \, (Nevts: \, %.1f) }$'%(np.sum(nominal_nutau_minus_no_nutau_up_and_down_trck['map'])), fontsize='large')
        plt.savefig(filename,dpi=150)
        plt.clf()

    print "no. of nominal_nutau_up_and_down_cscd = ", np.sum(nominal_nutau_up_and_down_cscd['map'])
    print "no. of nominal_no_nutau_up_and_down_cscd = ", np.sum(nominal_no_nutau_up_and_down_cscd['map'])
    print "no. of nominal_nutau_minus_no_nutau_up_and_down_cscd = ", np.sum(nominal_nutau_minus_no_nutau_up_and_down_cscd['map'])
    #print "max no. of evts in nutau_up_cscd: ", np.amax(nominal_nutau_up['cscd']['map'])
    #print "max no. of evts in nutau_up_trck: ", np.amax(nominal_nutau_up['trck']['map'])
    #print "max no. of evts in nutau_down_cscd: ", np.amax(nominal_nutau_down['cscd']['map'])
    #print "max no. of evts in nutau_down_trck: ", np.amax(nominal_nutau_down['trck']['map'])
    #print "max no. of evts in nominal_nutau_up_and_down_cscd: ", np.amax(nominal_nutau_up_and_down_cscd['map'])
    #print "max no. of evts in nominal_nutau_up_and_down_trck: ", np.amax(nominal_nutau_up_and_down_trck['map'])
    #no_of_up = np.sum(nominal_nutau_up['trck']['map']) + np.sum(nominal_nutau_up['cscd']['map'])
    #no_of_down = np.sum(nominal_nutau_down['trck']['map']) + np.sum(nominal_nutau_down['cscd']['map'])
    #print 'no. of upgoing events= ' , no_of_up 
    #print 'no. of downgoing events = ' , no_of_down 
    #print 'Total no. of events = ' , no_of_up + no_of_down
    #no_of_up_and_down = np.sum(nominal_nutau_up_and_down_trck['map']) + np.sum(nominal_nutau_up_and_down_cscd['map'])
    #print 'From map_up_down, Total no. of events = ' , no_of_up_and_down
    
    #### Plot background ####
    #for channel in ['trck','cscd']:
    #    plt.figure()
    #    show_map(up_background_maps[channel],logE=args.logE)
    #    if args.save:
    #        filename = os.path.join(args.outdir,args.title+'_upgoing_background_'+channel+'.png')
    #        plt.title(args.title+'_upgoing_background_'+channel)
    #        plt.savefig(filename,dpi=150)
    #        plt.clf()
    #    plt.figure()
    #    show_map(down_background_maps[channel],logE=args.logE)
    #    if args.save:
    #        filename = os.path.join(args.outdir,args.title+'_downgoing_background_'+channel+'.png')
    #        plt.title(args.title+'_downgoing_background_'+channel)
    #        plt.savefig(filename,dpi=150)
    #        plt.clf()
    if not args.save: plt.show()
    else: print '\n-->>Saved all files to: ',args.outdir

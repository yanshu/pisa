#! /usr/bin/env python
#
# author: Timothy C. Arlen
#         Feifei Huang
#
# date: 27 Jan 2016
#
#   Compare PISA reco and final maps when using a larger energy range (from 5.6 to an energy above 56 GeV, e.g. 177 GeV) and a smaller 
#   energy range ( 5.6 to 56 GeV). 
#   python compare_PISA_map_E_range_177GeV.py DC12_mc_0syst_1yr_LLR_e8_cz16_simp_nufit_prior_new_aeff_177GeV.json --title IC86_1yr --save -o PISA_Energy_Range_cmpr_new_aeff_no_background/ -logE --bg_scale 0 --all --y 1.0
#
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
    parser.add_argument('--y',type=float,
                        help='No. of livetime[ unit: Julian year]')
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
    template_settings['params']['livetime']['value'] = args.y

    ebins = template_settings['binning']['ebins']
    czbins = template_settings['binning']['czbins']
    print 'ebins = ', ebins
    print 'czbins = ', czbins
    CZ_bin_centers = get_bin_centers(czbins)
    E_bin_centers = get_bin_centers(ebins)

    current_ebins = template_settings['binning']['ebins'][:9]
    current_E_bin_centers = get_bin_centers(current_ebins)

    #print 'E_bin_centers = ', E_bin_centers
    #print 'CZ_bin_centers = ', CZ_bin_centers

    flav_title = {}
    flav_title['nue'] = 'nue'
    flav_title['numu'] = 'numu'
    flav_title['nutau'] = 'nutau'
    flav_title['nue_bar'] = 'nue \, bar'
    flav_title['numu_bar'] = 'numu \, bar'
    flav_title['nutau_bar'] = 'nutau \, bar'
    flav_title['nue_cc'] = 'nue \, CC'
    flav_title['numu_cc'] = 'numu \, CC'
    flav_title['nutau_cc'] = 'nutau \, CC'
    flav_title['nuall_nc'] = 'nuall \, NC'
    flav_title['cscd'] = 'cascade'
    flav_title['trck'] = 'track'


    ##################### Plot Aeff Maps comparison  #######################

    ##### Get Maps from using Larger Energy range (5.6 to 177 GeV range)  #####
    up_template_settings = copy.deepcopy(template_settings)
    up_template_settings['params']['reco_mc_wt_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_100_percent_up.hdf5'}
    up_template_settings['params']['reco_vbwkde_evts_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_10_percent_up.hdf5'}

    down_template_settings = copy.deepcopy(template_settings)
    down_template_settings['params']['reco_mc_wt_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_100_percent_down.hdf5'}
    down_template_settings['params']['pid_paramfile'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/pid/1X60_pid_down.json'}
    down_template_settings['params']['reco_vbwkde_evts_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_10_percent_down.hdf5'}

    with Timer() as t:
        template_maker_down = TemplateMaker(get_values(down_template_settings['params']), **down_template_settings['binning'])
        template_maker_up = TemplateMaker(get_values(up_template_settings['params']), **up_template_settings['binning'])

    profile.info('==> elapsed time to initialize templates: %s sec'%t.secs)

    # Make nutau template:
    nutau_up_params = copy.deepcopy(select_hierarchy_and_nutau_norm( up_template_settings['params'],True,1.0))
    nutau_down_params = copy.deepcopy(select_hierarchy_and_nutau_norm( down_template_settings['params'],True,1.0))
    no_nutau_up_params = copy.deepcopy(select_hierarchy_and_nutau_norm( up_template_settings['params'],True,0.0))
    no_nutau_down_params = copy.deepcopy(select_hierarchy_and_nutau_norm( down_template_settings['params'],True,0.0))

    with Timer(verbose=False) as t:
        nutau_up_all_stages = template_maker_up.get_template(get_values(nutau_up_params),return_stages=args.all)
        nutau_down_all_stages = template_maker_down.get_template(get_values(nutau_down_params),return_stages=args.all)
        no_nutau_up_all_stages = template_maker_up.get_template(get_values(no_nutau_up_params),return_stages=args.all)
        no_nutau_down_all_stages = template_maker_down.get_template(get_values(no_nutau_down_params),return_stages=args.all)
        aeff_map_nutau_up = nutau_up_all_stages[2]
        aeff_map_nutau_down = nutau_down_all_stages[2]
        aeff_map_no_nutau_up = no_nutau_up_all_stages[2]
        aeff_map_no_nutau_down = no_nutau_down_all_stages[2]
        reco_map_nutau_up = nutau_up_all_stages[3]
        reco_map_nutau_down = nutau_down_all_stages[3]
        reco_map_no_nutau_up = no_nutau_up_all_stages[3]
        reco_map_no_nutau_down = no_nutau_down_all_stages[3]
        final_map_nutau_up = nutau_up_all_stages[4]
        final_map_nutau_down = nutau_down_all_stages[4]
        final_map_no_nutau_up = no_nutau_up_all_stages[4]
        final_map_no_nutau_down = no_nutau_down_all_stages[4]
    profile.info('==> elapsed time to get NUTAU template: %s sec'%t.secs)


    ##### Get Maps from using Smaller Energy range (current 5.6 to 56 GeV range)  #####

    smallerRange_up_template_settings = copy.deepcopy(template_settings)
    smallerRange_up_template_settings['binning']['ebins'] = template_settings['binning']['ebins'][:9]
    smallerRange_up_template_settings['params']['reco_mc_wt_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_100_percent_up.hdf5'}
    smallerRange_up_template_settings['params']['reco_vbwkde_evts_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_10_percent_up.hdf5'}

    smallerRange_down_template_settings = copy.deepcopy(template_settings)
    smallerRange_down_template_settings['binning']['ebins'] = template_settings['binning']['ebins'][:9]
    smallerRange_down_template_settings['params']['reco_mc_wt_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_100_percent_down.hdf5'}
    smallerRange_down_template_settings['params']['pid_paramfile'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/pid/1X60_pid_down.json'}
    smallerRange_down_template_settings['params']['reco_vbwkde_evts_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_10_percent_down.hdf5'}

    with Timer() as t:
        smallerRange_template_maker_down = TemplateMaker(get_values(smallerRange_down_template_settings['params']), **smallerRange_down_template_settings['binning'])
        smallerRange_template_maker_up = TemplateMaker(get_values(smallerRange_up_template_settings['params']), **smallerRange_up_template_settings['binning'])

    profile.info('==> elapsed time to initialize templates: %s sec'%t.secs)

    # Make nutau template:
    smallerRange_nutau_up_params = copy.deepcopy(select_hierarchy_and_nutau_norm( smallerRange_up_template_settings['params'],True,1.0))
    smallerRange_nutau_down_params = copy.deepcopy(select_hierarchy_and_nutau_norm( smallerRange_down_template_settings['params'],True,1.0))
    smallerRange_no_nutau_up_params = copy.deepcopy(select_hierarchy_and_nutau_norm( smallerRange_up_template_settings['params'],True,0.0))
    smallerRange_no_nutau_down_params = copy.deepcopy(select_hierarchy_and_nutau_norm( smallerRange_down_template_settings['params'],True,0.0))

    with Timer(verbose=False) as t:
        smallerRange_nutau_up_all_stages = smallerRange_template_maker_up.get_template(get_values(smallerRange_nutau_up_params),return_stages=args.all)
        smallerRange_nutau_down_all_stages = smallerRange_template_maker_down.get_template(get_values(smallerRange_nutau_down_params),return_stages=args.all)
        smallerRange_no_nutau_up_all_stages = smallerRange_template_maker_up.get_template(get_values(smallerRange_no_nutau_up_params),return_stages=args.all)
        smallerRange_no_nutau_down_all_stages = smallerRange_template_maker_down.get_template(get_values(smallerRange_no_nutau_down_params),return_stages=args.all)
        smallerRange_aeff_map_nutau_up = smallerRange_nutau_up_all_stages[2]
        smallerRange_aeff_map_nutau_down = smallerRange_nutau_down_all_stages[2]
        smallerRange_aeff_map_no_nutau_up = smallerRange_no_nutau_up_all_stages[2]
        smallerRange_aeff_map_no_nutau_down = smallerRange_no_nutau_down_all_stages[2]
        smallerRange_reco_map_nutau_up = smallerRange_nutau_up_all_stages[3]
        smallerRange_reco_map_nutau_down = smallerRange_nutau_down_all_stages[3]
        smallerRange_reco_map_no_nutau_up = smallerRange_no_nutau_up_all_stages[3]
        smallerRange_reco_map_no_nutau_down = smallerRange_no_nutau_down_all_stages[3]
        smallerRange_final_map_nutau_up = smallerRange_nutau_up_all_stages[4]
        smallerRange_final_map_nutau_down = smallerRange_nutau_down_all_stages[4]
        smallerRange_final_map_no_nutau_up = smallerRange_no_nutau_up_all_stages[4]
        smallerRange_final_map_no_nutau_down = smallerRange_no_nutau_down_all_stages[4]
    profile.info('==> elapsed time to get NUTAU template: %s sec'%t.secs)


    ###### Plot Aeff map comparison #####
    #total_no_aeff_map_largerRange = 0
    #total_no_aeff_map_smallerRange = 0
    #for flavor in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
    #    for int_type in ['cc','nc']:
    #        # select Aeff map from the larger energy range
    #        aeff_map_nutau_up_and_down = sum_map(aeff_map_nutau_up[flavor][int_type], aeff_map_nutau_down[flavor][int_type])
    #        smallerRange_aeff_map_nutau_up_and_down = sum_map(smallerRange_aeff_map_nutau_up[flavor][int_type], smallerRange_aeff_map_nutau_down[flavor][int_type])
    #        aeff_map_nutau_up_and_down_largerRange = {'map':aeff_map_nutau_up_and_down['map'][:][:8],
    #                                                                      'ebins': ebins[ebins<60],
    #                                                                      'czbins': czbins}
    #        #print '>>>> shape of aeff_map_nutau_up_and_down_largerRange ', np.shape(aeff_map_nutau_up_and_down_largerRange['map'])
    #        #print '<<<< shape of smallerRange_aeff_map_nutau_up_and_down = ', np.shape(smallerRange_aeff_map_nutau_up_and_down['map'])
    #        # Plot Aeff map taken from the larger energy range
    #        fig = plt.figure()
    #        show_map(aeff_map_nutau_up_and_down_largerRange,logE=args.logE)
    #        if args.save:
    #            filename = os.path.join(args.outdir,args.title+ '_PISA_aeff_event_rate_%s_%s_5.6_56GeV.png'% (flavor,int_type))
    #            plt.title(r'${\rm %s \, yr \, PISA \, aeff \, map \, %s \, %s \, (Nevts: \, %.1f) }$'%(args.y, flav_title[flavor], int_type, np.sum(aeff_map_nutau_up_and_down_largerRange['map'])), fontsize='large')
    #            #print ' From PISA:   flavor ', flavor , ' ', int_type , ' , no. of evts = ', np.sum(aeff_map_nutau_up_and_down_largerRange['map'])
    #            total_no_aeff_map_largerRange += np.sum(aeff_map_nutau_up_and_down_largerRange['map'])
    #            total_no_aeff_map_smallerRange += np.sum(smallerRange_aeff_map_nutau_up_and_down['map'])
    #            plt.savefig(filename,dpi=150)
    #            plt.clf()

    #        # Plot Ratio 
    #        ratio_larger_to_smaller = ratio_map(aeff_map_nutau_up_and_down_largerRange, smallerRange_aeff_map_nutau_up_and_down)
    #        plt.figure()
    #        show_map(ratio_larger_to_smaller, vmin= np.min(ratio_larger_to_smaller['map']), vmax= np.max(ratio_larger_to_smaller['map']))
    #        if args.save:
    #            filename = os.path.join(args.outdir,args.title+ '_Ratio_Aeff_PISA_Aeff_LargerRange_to_SmallerRange_%s_%s_'% (flavor, int_type)+ '.png')
    #            plt.title('Ratio of Aeff maps(larger range/smaller range) %s %s'% (flavor, int_type))
    #            plt.savefig(filename,dpi=150)
    #            plt.clf()
    #print 'total_no_aeff_map_largerRange = ', total_no_aeff_map_largerRange
    #print 'total_no_aeff_map_smallerRange = ', total_no_aeff_map_smallerRange
    #print '\n\n'


###################### Plot Reco Map comparison #######################

    total_no_reco_map_largerRange = 0
    total_no_reco_map_smallerRange = 0
    for flavor in ['nue_cc','numu_cc','nutau_cc','nuall_nc']:
        # select Reco map from the larger energy range
        reco_map_nutau_up_and_down = sum_map(reco_map_nutau_up[flavor], reco_map_nutau_down[flavor])
        smallerRange_reco_map_nutau_up_and_down = sum_map(smallerRange_reco_map_nutau_up[flavor], smallerRange_reco_map_nutau_down[flavor])
        reco_map_nutau_up_and_down_largerRange = {'map':reco_map_nutau_up_and_down['map'][:][:8],
                                                                      'ebins': ebins[ebins<60],
                                                                      'czbins': czbins}
        # Plot Reco map taken from the smaller energy range
        fig = plt.figure()
        show_map(smallerRange_reco_map_nutau_up_and_down,logE=args.logE)
        if args.save:
            filename = os.path.join(args.outdir,args.title+ '_PISA_Reco_SmallerRange_event_rate_%s_5.6_56GeV.png'% (flavor))
            plt.title(r'${\rm %s \, yr \, PISA \, reco \, map \, %s \, (Nevts: \, %.1f, \, smaller \, range) }$'%(args.y, flav_title[flavor], np.sum(smallerRange_reco_map_nutau_up_and_down['map'])), fontsize='large')
            plt.savefig(filename,dpi=150)
            plt.clf()

        fig = plt.figure()
        show_map(reco_map_nutau_up_and_down_largerRange,logE=args.logE)
        if args.save:
            filename = os.path.join(args.outdir,args.title+ '_PISA_Reco_LargerRange_event_rate_%s_5.6_56GeV.png'% (flavor))
            plt.title(r'${\rm %s \, yr \, PISA \, reco \, map \, %s \, (Nevts: \, %.1f, \, larger \, range) }$'%(args.y, flav_title[flavor], np.sum(reco_map_nutau_up_and_down_largerRange['map'])), fontsize='large')
            total_no_reco_map_largerRange += np.sum(reco_map_nutau_up_and_down_largerRange['map'])
            total_no_reco_map_smallerRange += np.sum(smallerRange_reco_map_nutau_up_and_down['map'])
            plt.savefig(filename,dpi=150)
            plt.clf()

        # Plot Ratio 
        ratio_larger_to_smaller = ratio_map(reco_map_nutau_up_and_down_largerRange, smallerRange_reco_map_nutau_up_and_down)
        plt.figure()
        show_map(ratio_larger_to_smaller, vmin= np.min(ratio_larger_to_smaller['map']), vmax= np.max(ratio_larger_to_smaller['map']))
        if args.save:
            filename = os.path.join(args.outdir,args.title+ '_Ratio_Reco_PISA_LargerRange_to_SmallerRange_%s_'% (flavor )+ '.png')
            plt.title(r'${\rm %s \, yr \, reco \, maps \, ratio \,(larger/smaller \, E \, range) \, %s}$' % (args.y, flav_title[flavor]))
            plt.savefig(filename,dpi=150)
            plt.clf()

    print 'total_no_reco_map_largerRange = ', total_no_reco_map_largerRange
    print 'total_no_reco_map_smallerRange = ', total_no_reco_map_smallerRange
    percentage_more = (100*(total_no_reco_map_largerRange-total_no_reco_map_smallerRange)/total_no_reco_map_smallerRange)
    print 'use a energy range from ', ebins[0], ' to ', ebins[-1], ' , the reco. map has %.3f percent more events' %  percentage_more



###################### Plot Final Map comparison #######################

    total_no_final_map_largerRange = 0
    total_no_final_map_smallerRange = 0
    for flavor in ['trck', 'cscd']:
        # select Final map from the larger energy range
        final_map_nutau_up_and_down = sum_map(final_map_nutau_up[flavor], final_map_nutau_down[flavor])
        smallerRange_final_map_nutau_up_and_down = sum_map(smallerRange_final_map_nutau_up[flavor], smallerRange_final_map_nutau_down[flavor])
        final_map_nutau_up_and_down_largerRange = {'map':final_map_nutau_up_and_down['map'][:][:8],
                                                                      'ebins': ebins[ebins<60],
                                                                      'czbins': czbins}
        # Plot Final map taken from the smaller energy range
        fig = plt.figure()
        show_map(smallerRange_final_map_nutau_up_and_down,logE=args.logE)
        if args.save:
            filename = os.path.join(args.outdir,args.title+ '_PISA_Final_SmallerRange_event_rate_%s_5.6_56GeV.png'% (flavor))
            plt.title(r'${\rm %s \, yr \, PISA \, final \, map \, %s \, (Nevts: \, %.1f, \, smaller \, range) }$'%(args.y, flav_title[flavor], np.sum(smallerRange_final_map_nutau_up_and_down['map'])), fontsize='large')
            plt.savefig(filename,dpi=150)
            plt.clf()

        fig = plt.figure()
        show_map(final_map_nutau_up_and_down_largerRange,logE=args.logE)
        if args.save:
            filename = os.path.join(args.outdir,args.title+ '_PISA_Final_LargerRange_event_rate_%s_5.6_56GeV.png'% (flavor))
            plt.title(r'${\rm %s \, yr \, PISA \, final \, map \, %s \, (Nevts: \, %.1f, \, larger \, range) }$'%(args.y, flav_title[flavor], np.sum(final_map_nutau_up_and_down_largerRange['map'])), fontsize='large')
            total_no_final_map_largerRange += np.sum(final_map_nutau_up_and_down_largerRange['map'])
            total_no_final_map_smallerRange += np.sum(smallerRange_final_map_nutau_up_and_down['map'])
            plt.savefig(filename,dpi=150)
            plt.clf()

        # Plot Ratio 
        ratio_larger_to_smaller = ratio_map(final_map_nutau_up_and_down_largerRange, smallerRange_final_map_nutau_up_and_down)
        plt.figure()
        show_map(ratio_larger_to_smaller, vmin= np.min(ratio_larger_to_smaller['map']), vmax= np.max(ratio_larger_to_smaller['map']))
        if args.save:
            filename = os.path.join(args.outdir,args.title+ '_Ratio_Final_PISA_LargerRange_to_SmallerRange_%s_'% (flavor )+ '.png')
            plt.title(r'${\rm %s \, yr \, final \, maps \, ratio \,(larger/smaller \, E \, range) \, %s}$' % (args.y, flav_title[flavor]))
            plt.savefig(filename,dpi=150)
            plt.clf()

    print 'total_no_final_map_largerRange = ', total_no_final_map_largerRange
    print 'total_no_final_map_smallerRange = ', total_no_final_map_smallerRange
    percentage_more = (100*(total_no_final_map_largerRange-total_no_final_map_smallerRange)/total_no_final_map_smallerRange)
    print 'use a energy range from ', ebins[0], ' to ', ebins[-1], ' , the final. map has %.3f percent more events' %  percentage_more


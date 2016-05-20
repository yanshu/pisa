#! /usr/bin/env python
#
# get_DH_slopes.py
#
# Get the slopes for DomEff and HoleIce fits.
#
# author: Feifei Huang - fxh140@psu.edu
#         Tim Arlen - tca3@psu.edu
#
# date:   02-July-2015
#
import matplotlib as mpl
mpl.use('Agg')
import copy
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pisa.utils.utils as utils
from pisa.utils.log import logging, profile, physics
from pisa.utils.jsons import from_json,to_json
from pisa.analysis.GetMCError import GetMCError
from pisa.utils.params import get_values, change_nutau_norm_settings, select_hierarchy
from pisa.utils.plot import show_map
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

parser = ArgumentParser(description='''Get the slopes for DOMeff and HoleIce fits. ''',
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-t','--template_settings',type=str,
                    metavar='JSONFILE', required = True,
                    help='''Settings related to the template generation and systematics.''')
parser.add_argument('-s','--sim',type=str,
                    metavar='simu', required = True,
                    help='''Which simulation, can only be 4digit, 5digit, dima_p1, or dima_p2''')
parser.add_argument('--name',type=str,
                    metavar='name', default='10_by_16', help ="name to be added at the end of all files,\
                    avoid overwriting previous files when doing tests.")
parser.add_argument('--use_event_PISA',action='store_true',default=False,
                    help="Use event-by-event PISA; otherwise, use histogram-based PISA") 
parser.add_argument('--no_NC_osc',action='store_true',default=False,
                    help="Use no oscillation for NC, for cmpr with oscFit.") 
parser.add_argument('--IMH',action='store_true',default=False,
                    help="Use inverted mass hiearchy.")
parser.add_argument('--templ_already_saved',action='store_true',default=False,
                    help="Read templates from already saved file; saves time when only need plotting.")
parser.add_argument('--plot',action='store_true',default=False,
                    help="Plot the fits of DOM efficiency and hole ice for each bin.")
parser.add_argument('-o','--outdir',type=str,
                    metavar='DIR', required = True, help='''Output directory''')
args = parser.parse_args()

#Read in the settings
outdir = args.outdir
use_NMH = not(args.IMH)
print "Use NMH : ", use_NMH
if args.use_event_PISA:
    from pisa.analysis.TemplateMaker_MC import TemplateMaker
    pisa_mode = 'event'
else:
    from pisa.analysis.TemplateMaker_nutau import TemplateMaker
    pisa_mode = 'hist'
utils.mkdir(outdir)
utils.mkdir(outdir+'/plots/')
utils.mkdir(outdir+'/plots/png/')
utils.mkdir(outdir+'/plots/')
# if templates already save
if args.templ_already_saved:
    output_template = from_json(outdir+ '%s_%s_DomEff_HoleIce_templates_%s.json'% (args.sim, pisa_mode, args.name)) 
    templates = output_template['templates']
    MCmaps = output_template['MCmaps']
    template_settings = output_template['template_settings']
else:
    template_settings = from_json(args.template_settings)
czbin_edges = template_settings['binning']['czbins']
ebin_edges = template_settings['binning']['anlys_ebins']
channel = template_settings['params']['channel']['value']
x_steps = 0.01

# Write run info into a dict
if args.sim == '4digit':
    run_list = [ '50', '60', '61', '64', '65', '70', '71', '72']
    nominal_run = '60'
    hole_ice_nominal = 0.02
    dom_eff_nominal = 1.0 
    dict_run = {'50': {'dom_eff': 0.91, 'hole_ice': 0.02},
                '60': {'dom_eff': 1.00, 'hole_ice': 0.02},
                '61': {'dom_eff': 0.95, 'hole_ice': 0.02},
                '64': {'dom_eff': 1.10, 'hole_ice': 0.02},
                '65': {'dom_eff': 1.05, 'hole_ice': 0.02},
                '70': {'dom_eff': 0.91, 'hole_ice': 0.00},
                '71': {'dom_eff': 0.91, 'hole_ice': 0.033},
                '72': {'dom_eff': 0.91, 'hole_ice': 0.01}}

elif args.sim == '5digit':
    run_list = [ '551', '552', '553', '554', '555', '556', '585', '560', '561', '564', '565', '572', '573']
    nominal_run = '585'
    hole_ice_nominal = 0.02
    dom_eff_nominal = 1.0 
    dict_run = {'551': {'dom_eff': 0.85, 'hole_ice': 0.02},
                '552': {'dom_eff': 0.90, 'hole_ice': 0.02},
                '553': {'dom_eff': 0.95, 'hole_ice': 0.02},
                '554': {'dom_eff': 1.05, 'hole_ice': 0.02},
                '555': {'dom_eff': 1.10, 'hole_ice': 0.02},
                '556': {'dom_eff': 1.15, 'hole_ice': 0.02},
                '585': {'dom_eff': 1.00, 'hole_ice': 0.02},
                '560': {'dom_eff': 1.00, 'hole_ice': 0.01},
                '561': {'dom_eff': 1.00, 'hole_ice': 0.033},
                '564': {'dom_eff': 1.00, 'hole_ice': 0.022},
                '565': {'dom_eff': 1.00, 'hole_ice': 0.018},
                '572': {'dom_eff': 1.00, 'hole_ice': 0.0275},
                '573': {'dom_eff': 1.00, 'hole_ice': 0.0125}}

elif args.sim == 'dima_p1':
    run_list = [ '600', '601', '603', '604', '605', '606', '608', '610', '611', '612', '613']
    nominal_run = '600'
    hole_ice_nominal = 0.25
    dom_eff_nominal = 1.0 
    dict_run = {'600': {'dom_eff': 1.00, 'hole_ice': 0.25},
                '601': {'dom_eff': 0.88, 'hole_ice': 0.25},
                '603': {'dom_eff': 0.94, 'hole_ice': 0.25},
                '604': {'dom_eff': 0.97, 'hole_ice': 0.25},
                '605': {'dom_eff': 1.03, 'hole_ice': 0.25},
                '606': {'dom_eff': 1.06, 'hole_ice': 0.25},
                '608': {'dom_eff': 1.12, 'hole_ice': 0.25},
                '610': {'dom_eff': 1.00, 'hole_ice': 0.15},
                '611': {'dom_eff': 1.00, 'hole_ice': 0.20},
                '611': {'dom_eff': 1.00, 'hole_ice': 0.20},
                '612': {'dom_eff': 1.00, 'hole_ice': 0.30},
                '613': {'dom_eff': 1.00, 'hole_ice': 0.35}}

elif args.sim == 'dima_p2':
    run_list = [ '612', '620', '621', '622', '623', '624']
    nominal_run = '612'
    hole_ice_nominal = 0.0
    dom_eff_nominal = 1.0 
    dict_run = {'612': {'hole_ice_fwd': 0.0},
                '620': {'hole_ice_fwd': 2.0},
                '621': {'hole_ice_fwd': -5.0},
                '622': {'hole_ice_fwd': -3.0},
                '623': {'hole_ice_fwd': 1.0},
                '624': {'hole_ice_fwd': -1.0}}

    #dict_run = {'612': {'dom_eff': 1.00, 'hole_ice': 0.30, 'hole_ice_fwd': 0.0},
    #            '620': {'dom_eff': 1.00, 'hole_ice': 0.30, 'hole_ice_fwd': 2.0},
    #            '621': {'dom_eff': 1.00, 'hole_ice': 0.30, 'hole_ice_fwd': -5.0},
    #            '622': {'dom_eff': 1.00, 'hole_ice': 0.30, 'hole_ice_fwd': -3.0},
    #            '623': {'dom_eff': 1.00, 'hole_ice': 0.30, 'hole_ice_fwd': 1.0},
    #            '624': {'dom_eff': 1.00, 'hole_ice': 0.30, 'hole_ice_fwd': -1.0}}
else:
    raise ValueError( "sim allowed: ['5digit', '4digit', 'dima_p1', 'dima_p2']")


if args.sim == '5digit':
    fits_DOMEff = {'trck':{'slopes':{}}, 'cscd':{'slopes':{}}, 'nominal_value': 1}
    fits_HoleIce = {'trck':{'slopes':{}}, 'cscd':{'slopes':{}}, 'nominal_value': 0.02}
elif args.sim == '4digit':
    fits_DOMEff = {'trck':{'slopes':{}, 'fixed_ratios':{}}, 'cscd':{'slopes':{}, 'fixed_ratios':{}}, 'nominal_value': 1 }
    fits_HoleIce = {'trck':{'slopes':{}, 'fixed_ratios':{}}, 'cscd':{'slopes':{}, 'fixed_ratios':{}}, 'nominal_value': 0.02 }
elif args.sim == 'dima_p1':
    fits_DOMEff = {'trck':{'slopes':{}}, 'cscd':{'slopes':{}}, 'nominal_value': 1}
    fits_HoleIce = {'trck':{'slopes':{}}, 'cscd':{'slopes':{}}, 'nominal_value': 0.25}
else:
    fits_HoleIce = {'trck':{'slopes':{}}, 'cscd':{'slopes':{}}, 'nominal_value': 0.0}
# Get templates and MC events 
if not args.templ_already_saved:
    templates = {}
    MCmaps = {}
    for run_num in run_list:
        DH_template_settings = copy.deepcopy(template_settings)
        MCmaps[str(run_num)] = {'trck':{}, 'cscd':{}}
        templates[str(run_num)] = {'trck':{}, 'cscd':{}}
        print "run_num = ", run_num
        if args.sim == '5digit':
            assert(DH_template_settings['params']['pid_mode']['value']=='mc') # right now, only use MC mode for PID for the 5-digit sets 
            aeff_mc_file = 'aeff/events__deepcore__IC86__runs_12%s-16%s:20000__proc_v5digit__unjoined.hdf5' % (run_num,run_num)
            reco_mc_file = 'aeff/events__deepcore__IC86__runs_12%s-16%s:20000__proc_v5digit__joined_G_nue_cc+nuebar_cc_G_numu_cc+numubar_cc_G_nutau_cc+nutaubar_cc_G_nuall_nc+nuallbar_nc.hdf5' % (run_num, run_num)
        elif args.sim == '4digit':
            aeff_mc_file = 'aeff/events__deepcore__ic86__runs_12%s-16%s:200__proc_v4digit__unjoined.hdf5' % (run_num,run_num)
            reco_mc_file = 'aeff/events__deepcore__ic86__runs_12%s-16%s:200__proc_v4digit__joined_G_nue_cc+nuebar_cc_G_numu_cc+numubar_cc_G_nutau_cc+nutaubar_cc_G_nuall_nc+nuallbar_nc.hdf5' % (run_num, run_num)
            # if use param mode for PID, need to use pid_param_file_up and _down:
            pid_param_file_up = 'pid/1X%s_pid.json' % run_num
            pid_param_file_down = 'pid/1X%s_pid_down.json' % run_num
            DH_template_settings['params']['pid_paramfile_up']['value'] = pid_param_file_up 
            DH_template_settings['params']['pid_paramfile_down']['value'] = pid_param_file_down
        elif args.sim == 'dima_p1':
            aeff_mc_file = 'aeff/events__deepcore__IC86__runs_12%s1-12%s3,14%s1-14%s3,16%s1-16%s3__proc_v5digit__unjoined_with_fluxes.hdf5' % (run_num,run_num,run_num,run_num,run_num,run_num)
            reco_mc_file = 'aeff/events__deepcore__IC86__runs_12%s1-12%s3,14%s1-14%s3,16%s1-16%s3__proc_v5digit__joined_G_nue_cc+nuebar_cc_G_numu_cc+numubar_cc_G_nutau_cc+nutaubar_cc_G_nuall_nc+nuallbar_nc.hdf5' % (run_num, run_num, run_num,run_num,run_num,run_num)
        else:
            aeff_mc_file = 'aeff/events__deepcore__IC86__runs_12%s1-12%s3,14%s1-14%s3,16%s1-16%s3__proc_v5digit__unjoined_with_fluxes.hdf5' % (run_num,run_num,run_num,run_num,run_num,run_num)
            reco_mc_file = 'aeff/events__deepcore__IC86__runs_12%s1-12%s3,14%s1-14%s3,16%s1-16%s3__proc_v5digit__joined_G_nue_cc+nuebar_cc_G_numu_cc+numubar_cc_G_nutau_cc+nutaubar_cc_G_nuall_nc+nuallbar_nc.hdf5' % (run_num, run_num, run_num,run_num,run_num,run_num)
        DH_template_settings['params']['aeff_weight_file']['value'] = aeff_mc_file
        DH_template_settings['params']['reco_mc_wt_file']['value'] = reco_mc_file
        DH_template_settings['params']['pid_events']['value'] = reco_mc_file    #pid file same as reco_mc file
        DH_template_settings['params']['atmos_mu_scale']['value'] = 0.0
    
        DH_template_maker = TemplateMaker(get_values(DH_template_settings['params']), no_sys_maps=True,**DH_template_settings['binning'])
        if args.use_event_PISA:
            # for comparison with oscFit: turn off NC oscillation
            template = DH_template_maker.get_template(get_values(change_nutau_norm_settings(DH_template_settings['params'], 1.0 ,nutau_norm_fix=True, normal_hierarchy=use_NMH)),no_sys_maps=True,use_cut_on_trueE=False,turn_off_osc_NC=args.no_NC_osc)
        else:
            template = DH_template_maker.get_template(get_values(change_nutau_norm_settings(DH_template_settings['params'], 1.0 ,nutau_norm_fix=True, normal_hierarchy=use_NMH)),no_sys_maps=True)

        templates[str(run_num)]['trck'] = template['trck']['map']
        templates[str(run_num)]['cscd'] = template['cscd']['map']
        if args.sim == 'dima_p2':
            templates[str(run_num)]['hole_ice_fwd'] = dict_run[str(run_num)]['hole_ice_fwd']
        else:
            templates[str(run_num)]['dom_eff'] = dict_run[str(run_num)]['dom_eff']
            templates[str(run_num)]['hole_ice'] = dict_run[str(run_num)]['hole_ice']

        # Get MC events map 
        MCMap = GetMCError(get_values(DH_template_settings['params']), DH_template_settings['binning']['anlys_ebins'], DH_template_settings['binning']['czbins'], reco_mc_file)
        tmap_MC = MCMap.get_mc_events_map(True, get_values(DH_template_settings['params']), reco_mc_file)
        MCmaps[str(run_num)]['trck'] = tmap_MC['trck']['map']
        MCmaps[str(run_num)]['cscd'] = tmap_MC['cscd']['map']
    
    #Assemble output dict
    output_template = {'templates' : templates,
                       'MCmaps': MCmaps,
                       'template_settings' : template_settings}
    to_json(output_template, outdir+'%s_%s_DomEff_HoleIce_templates_%s.json'%(args.sim, pisa_mode, args.name))


# Do fits (linear and quadratic) for each bin
for flav in ['trck','cscd']:
    templ_list = []
    templ_err_list = []
    if args.sim != 'dima_p2':
        dom_eff_list = []
    hole_ice_list = []
    k_DE_linear = np.empty(np.shape(templates[nominal_run][flav])) 
    k_DE_quad = np.empty(np.shape(templates[nominal_run][flav])) 
    p_DE_quad = np.empty(np.shape(templates[nominal_run][flav])) 
    if args.sim == '4digit':
        fixed_ratio = np.empty(np.shape(templates[nominal_run][flav])) 
    k_HI_quad = np.empty(np.shape(templates[nominal_run][flav])) 
    p_HI_quad = np.empty(np.shape(templates[nominal_run][flav])) 
    k_HI_linear = np.empty(np.shape(templates[nominal_run][flav])) 
    for run_num in run_list: 
        if args.sim != 'dima_p2':
            dom_eff_list.append(templates[str(run_num)]['dom_eff'])
            hole_ice_list.append(templates[str(run_num)]['hole_ice'])
        else:
            hole_ice_list.append(templates[str(run_num)]['hole_ice_fwd'])
        templ_list.append(templates[str(run_num)][flav])
        templ_err_list.append(templates[str(run_num)][flav]/np.sqrt(MCmaps[str(run_num)][flav]))
   
    if args.sim != 'dima_p2':
        dom_eff = np.array(dom_eff_list)
    hole_ice = np.array(hole_ice_list)      # unit: cm-1
    templ = np.array(templ_list)
    templ_err = np.array(templ_err_list)

    y_val_max = np.max(np.divide(templ, templates[nominal_run][flav]))
    y_val_min = np.min(np.divide(templ, templates[nominal_run][flav]))

    tml_shape = np.shape(templ)
    n_ebins = tml_shape[1] 
    n_czbins = tml_shape[2] 

    ############################### DOM efficiency ######################################
    if args.sim != 'dima_p2':
        cut_holeice = hole_ice==hole_ice_nominal        # select elements when hole ice = nominal value, when generating fits for dom_eff
        for i in range(0,n_ebins):
            for j in range(0,n_czbins):
        
                ########### Get Data ############
                dom_eff_values = dom_eff[cut_holeice]
                bin_counts = templ[cut_holeice,i,j]
                bin_err = templ_err[cut_holeice,i,j]
                nominal_bin_counts = bin_counts[dom_eff_values==dom_eff_nominal]
                nominal_bin_err = bin_err[dom_eff_values==dom_eff_nominal]
                bin_ratio_values = bin_counts/nominal_bin_counts  #divide by the nominal value
                bin_ratio_err_values = bin_ratio_values * np.sqrt(np.square(nominal_bin_err/nominal_bin_counts)+np.square(bin_err/bin_counts))

                if args.sim == '4digit':
                    # line goes through point (0.02, fixed_r_val), fixed_r_val is the value for dom_eff = 0.91 and hole ice = 0.02
                    fixed_r_val = bin_ratio_values[dom_eff_values==0.91]
                    fixed_ratio[i][j]= fixed_r_val
                    exec('def dom_eff_linear_through_point(x, k): return k*x + %s - k*0.91'%fixed_r_val)
                    exec('def dom_eff_quadratic_through_point(x, k, p): return k*(x- 0.91) + p*(x- 0.91)**2 + %s'%fixed_r_val)
                else:
                    # i.e. when args.sim == '5digit' or args.sim == 'dima_p1':
                    exec('def dom_eff_linear_through_point(x, k): return k* (x - %s) + 1.0'%dom_eff_nominal)
                    exec('def dom_eff_quadratic_through_point(x, k, p): return k*(x- %s) + p*(x-%s)**2 + 1.0'%(dom_eff_nominal,dom_eff_nominal))


                ########### DOM efficiency Fits #############

                popt_1, pcov_1 = curve_fit(dom_eff_linear_through_point, dom_eff_values, bin_ratio_values)
                k1 = popt_1[0]
                k_DE_linear[i][j]= k1
   
                popt_2, pcov_2 = curve_fit(dom_eff_quadratic_through_point, dom_eff_values, bin_ratio_values)
                k2 = popt_2[0]
                p2 = popt_2[1]
                k_DE_quad[i][j]= k2
                p_DE_quad[i][j]= p2

                if args.plot:
                    fig_num = i * n_czbins+ j
                    if (fig_num == 0 or fig_num == n_czbins * n_ebins):
                        fig = plt.figure(num=1, figsize=( 4*n_czbins, 4*n_ebins))
                    subplot_idx = n_czbins*(n_ebins-1-i)+ j + 1
                    #print 'subplot_idx = ', subplot_idx
                    plt.subplot(n_ebins, n_czbins, subplot_idx)
                    #plt.title("CZ:[%s, %s] E:[%.1f, %.1f]"% (czbin_edges[j], czbin_edges[j+1], ebin_edges[i], ebin_edges[i+1]))
                    plt.scatter(dom_eff_values, bin_ratio_values, color='blue')
                    plt.errorbar(dom_eff_values, bin_ratio_values, yerr=bin_ratio_err_values,fmt='none')
                    plt.xlim(0.7,1.2)
                    plt.ylim(y_val_min-0.1,y_val_max+0.1)

                    dom_func_plot_x = np.arange(0.8 - x_steps, 1.2 + x_steps, x_steps)
                    dom_func_plot_y_linear = dom_eff_linear_through_point(dom_func_plot_x, k1)
                    dom_func_plot_linear, = plt.plot(dom_func_plot_x, dom_func_plot_y_linear, 'k-')
                    dom_func_plot_y_quad = dom_eff_quadratic_through_point(dom_func_plot_x, k2,p2)
                    dom_func_plot_quad, = plt.plot(dom_func_plot_x, dom_func_plot_y_quad, 'r-')
                    if j > 0:
                        plt.setp(plt.gca().get_yticklabels(), visible=False)
                    if i > 0:
                        plt.setp(plt.gca().get_xticklabels(), visible=False)
                    if(fig_num == n_czbins * n_ebins-1):
                        plt.figtext(0.5, 0.04, 'cos(zen)',fontsize=60,ha='center') 
                        plt.figtext(0.09, 0.5, 'energy',rotation=90,fontsize=60,ha='center') 
                        plt.figtext(0.5, 0.95, 'DOM eff. slopes %s'%flav, fontsize=60,ha='center')
                        fig.subplots_adjust(hspace=0)
                        fig.subplots_adjust(wspace=0)
                        plt.savefig(outdir+ 'plots/'+'%s_%s_fits_domeff_%s_%s.pdf'%(args.sim, pisa_mode, flav, args.name))
                        plt.savefig(outdir+ 'plots/png/'+'%s_%s_fits_domeff_%s_%s.png'%(args.sim, pisa_mode, flav, args.name))
                        plt.clf()

        fits_DOMEff[flav]['slopes'] = k_DE_linear
        fits_DOMEff[flav]['linear'] = k_DE_quad
        fits_DOMEff[flav]['quadratic'] = p_DE_quad

    
    ############################### Hole Ice ######################################

    if args.sim == '4digit':
        cut_domeff = dom_eff== 0.91       # select elements when dom_eff = 1, when generating fits for hole_ice
    elif args.sim == '5digit' or args.sim == 'dima_p1':
        cut_domeff = dom_eff== 1.0        # select elements when dom_eff = 1, when generating fits for hole_ice
    else:
        cut_domeff = np.ones(len(hole_ice)).astype(bool)    # for dima_p2 sets, all DOM eff values are already nominal value, so no cut
    for i in range(0,n_ebins):
        for j in range(0,n_czbins):
    
            ########### Get Data ############
            hole_ice_values = hole_ice[cut_domeff]
            bin_counts = templ[cut_domeff,i,j]
            bin_err = templ_err[cut_domeff,i,j]
            nominal_bin_counts = bin_counts[hole_ice_values==hole_ice_nominal]
            nominal_bin_err = bin_err[hole_ice_values==hole_ice_nominal]
            #print "hole_ice_values = ", hole_ice_values
            bin_ratio_values = bin_counts/nominal_bin_counts  #divide by the nominal value
            bin_ratio_err_values = bin_ratio_values * np.sqrt(np.square(nominal_bin_err/nominal_bin_counts)+np.square(bin_err/bin_counts))

            exec('def hole_ice_linear_through_point(x, k): return k* (x - %s) + 1.0'%hole_ice_nominal)
            if args.sim == '4digit':
                fixed_r_val = bin_ratio_values[hole_ice_values==hole_ice_nominal]
                # line goes through point (hole_ice_nominal, fixed_r_val), fixed_r_val is the value for dom_eff = 0.91 and hole ice = hole_ice_nominal 
                exec('def hole_ice_quadratic_through_point(x, k, p): return k*(x-%s) + p*(x-%s)**2 + %s'%(hole_ice_nominal, hole_ice_nominal, fixed_r_val))

            else:
                # line goes through point (hole_ice_nominal, 1) 
                exec('def hole_ice_quadratic_through_point(x, k, p): return k*(x-%s) + p*(x-%s)**2 + 1.0'%(hole_ice_nominal, hole_ice_nominal))

            ########### Hole Ice Fit #############
            popt_1, pcov_1 = curve_fit(hole_ice_linear_through_point, hole_ice_values, bin_ratio_values)
            k1 = popt_1[0]
            k_HI_linear[i][j]= k1

            popt_2, pcov_2 = curve_fit(hole_ice_quadratic_through_point, hole_ice_values, bin_ratio_values)
            k2 = popt_2[0]
            p2 = popt_2[1]
            k_HI_quad[i][j]= k2
            p_HI_quad[i][j]= p2

            if args.plot:
                fig_num = i * n_czbins+ j
                if (fig_num == 0 or fig_num == n_czbins * n_ebins):
                    fig = plt.figure(num=2, figsize=( 4*n_czbins, 4*n_ebins))
                subplot_idx = n_czbins*(n_ebins-1-i)+ j + 1
                plt.subplot(n_ebins, n_czbins, subplot_idx)
                #plt.title("CZ:[%s, %s] E:[%.1f, %.1f]"% (czbin_edges[j], czbin_edges[j+1], ebin_edges[i], ebin_edges[i+1]))
                plt.scatter(hole_ice_values, bin_ratio_values, color='blue')
                plt.errorbar(hole_ice_values, bin_ratio_values, yerr=bin_ratio_err_values,fmt='none')
                plt.ylim(y_val_min-0.1,y_val_max+0.1)

                if args.sim == '4digit':
                    ice_func_plot_x = np.arange(-0.02, 0.06 + x_steps, x_steps)
                    plt.xlim(-0.02,0.06)
                elif args.sim == '5digit':
                    ice_func_plot_x = np.arange(0.005, 0.04 + x_steps, x_steps)
                    plt.xlim(0.005,0.04+x_steps)
                elif args.sim == 'dima_p1':
                    ice_func_plot_x = np.arange(0.12, 0.37 + x_steps, x_steps)
                    plt.xlim(0.12,0.37+x_steps)
                else:
                    ice_func_plot_x = np.arange(-5.5, 2.5 + x_steps, x_steps)
                    plt.xlim(-5.5,2.5+x_steps)
                ice_func_plot_y_linear = hole_ice_linear_through_point(ice_func_plot_x, k1)
                ice_func_plot_linear, = plt.plot(ice_func_plot_x, ice_func_plot_y_linear, 'k-')
                ice_func_plot_y_quad = hole_ice_quadratic_through_point(ice_func_plot_x, k2,p2)
                ice_func_plot_quad, = plt.plot(ice_func_plot_x, ice_func_plot_y_quad, 'r-')
                if j > 0:
                    plt.setp(plt.gca().get_yticklabels(), visible=False)
                if i > 0:
                    plt.setp(plt.gca().get_xticklabels(), visible=False)

                if(fig_num==n_czbins * n_ebins-1):
                    fig.subplots_adjust(hspace=0)
                    fig.subplots_adjust(wspace=0)
                    plt.figtext(0.5, 0.04, 'cos(zen)',fontsize=60,ha='center') 
                    plt.figtext(0.09, 0.5, 'energy',rotation=90,fontsize=60,ha='center') 
                    if args.sim != 'dima_p2':
                        plt.figtext(0.5, 0.95, 'Hole Ice fits %s'%flav,fontsize=60,ha='center')
                        plt.savefig(outdir+ 'plots/'+'%s_%s_fits_holeice_%s_%s.pdf'%(args.sim, pisa_mode, flav, args.name))
                        plt.savefig(outdir+ 'plots/png/'+'%s_%s_fits_holeice_%s_%s.png'%(args.sim, pisa_mode, flav, args.name))
                    else:
                        plt.figtext(0.5, 0.95, 'Hole Ice fwd fits %s'%flav,fontsize=60,ha='center')
                        plt.savefig(outdir+ 'plots/'+'%s_%s_fits_holeice_fwd_%s_%s.pdf'%(args.sim, pisa_mode, flav, args.name))
                        plt.savefig(outdir+ 'plots/png/'+'%s_%s_fits_holeice_fwd_%s_%s.png'%(args.sim, pisa_mode, flav, args.name))
                    plt.clf()

    fits_HoleIce[flav]['slopes'] = k_HI_linear
    fits_HoleIce[flav]['linear'] = k_HI_quad
    fits_HoleIce[flav]['quadratic'] = p_HI_quad

    if args.sim == '4digit':
        fits_DOMEff[flav]['fixed_ratios'] = fixed_ratio 
        fits_HoleIce[flav]['fixed_ratios'] = fixed_ratio

#And write to file
if args.sim != 'dima_p2':
    to_json(fits_DOMEff,outdir+'%s_%s_DomEff_fits_%s.json'% (args.sim, pisa_mode, args.name))
    to_json(fits_HoleIce,outdir+'%s_%s_HoleIce_fits_%s.json'% (args.sim, pisa_mode, args.name))
else:
    to_json(fits_HoleIce,outdir+'%s_%s_HoleIce_fwd_fits_%s.json'% (args.sim, pisa_mode, args.name))


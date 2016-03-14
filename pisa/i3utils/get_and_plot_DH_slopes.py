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

from pisa.utils.log import logging, profile, physics
from pisa.utils.jsons import from_json,to_json
from pisa.analysis.GetMCError import GetMCError
from pisa.analysis.TemplateMaker_nutau import TemplateMaker
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
parser.add_argument('--plot',action='store_true',default=False,
                    help="Plot the fits of DOM efficiency and hole ice for each bin.")
args = parser.parse_args()

png_name = "slopes_plots"

#Read in the settings
template_settings = from_json(args.template_settings)
czbin_edges = template_settings['binning']['czbins']
ebin_edges = template_settings['binning']['anlys_ebins']
channel = template_settings['params']['channel']['value']
x_steps = 0.0001

template_maker = TemplateMaker(get_values(template_settings['params']), **template_settings['binning'])

templates = {}
MCmaps = {}
fits_DOMEff = {}
fits_HoleIce = {}

fits_DOMEff = {'trck':{'slopes':{}, 'fixed_ratios':{}},
                 'cscd':{'slopes':{}, 'fixed_ratios':{}},
                 'nominal_value': 1
                 }
fits_HoleIce = {'trck':{'slopes':{}, 'fixed_ratios':{}},
                 'cscd':{'slopes':{}, 'fixed_ratios':{}},
                 'nominal_value': 0.02
                 }

# Get templates from 8 MC sets
for run_num in [50,60,61,64,65,70,71,72]:
    MCmaps[str(run_num)] = {'trck':{},
                             'cscd':{}
                             }
    templates[str(run_num)] = {'trck':{},
                             'cscd':{}
                             }
    print "run_num = ", run_num
    aeff_mc_file = '~/pisa/pisa/resources/aeff/1X%i_aeff_mc.hdf5' % run_num
    pid_param_file_up = '~/pisa/pisa/resources/pid/1X%i_pid.json' % run_num
    pid_param_file_down = '~/pisa/pisa/resources/pid/1X%i_pid_down.json' % run_num
    reco_mc_file = '~/pisa/pisa/resources/events/1X%i_weighted_aeff_joined_nu_nubar.hdf5' % run_num
    DH_template_settings = copy.deepcopy(template_settings)
    DH_template_settings['params']['aeff_weight_file']['value'] = aeff_mc_file
    DH_template_settings['params']['reco_mc_wt_file']['value'] = reco_mc_file
    DH_template_settings['params']['pid_paramfile_up']['value'] = pid_param_file_up 
    DH_template_settings['params']['pid_paramfile_down']['value'] = pid_param_file_down
    DH_template_settings['params']['atmos_mu_scale']['value'] = 0.0

    DH_template_maker = TemplateMaker(get_values(DH_template_settings['params']), **DH_template_settings['binning'])

    template = DH_template_maker.get_template(get_values(change_nutau_norm_settings(DH_template_settings['params'], 1.0 ,True, normal_hierarchy=True)),no_sys_applied=True)

    templates[str(run_num)]['trck'] = template['trck']['map']
    templates[str(run_num)]['cscd'] = template['cscd']['map']
    
    MCMap = GetMCError(get_values(DH_template_settings['params']), DH_template_settings['binning']['anlys_ebins'], DH_template_settings['binning']['czbins'], reco_mc_file)
    tmap_MC = MCMap.get_mc_events_map(True, get_values(DH_template_settings['params']), reco_mc_file)
    MCmaps[str(run_num)]['trck'] = tmap_MC['trck']['map']
    MCmaps[str(run_num)]['cscd'] = tmap_MC['cscd']['map']

for flav in ['trck','cscd']:
    templ_list = []
    templ_err_list = []
    k_DE = np.empty(np.shape(templates['60']['trck'])) 
    fixed_ratio = np.empty(np.shape(templates['60']['trck'])) 
    k_HI = np.empty(np.shape(templates['60']['trck'])) 
    p_HI = np.empty(np.shape(templates['60']['trck'])) 
    for run_num in [50,60,61,64,65,70,71,72]:
        # (DOM efficiency, HoleIce Scattering): (0.91,50), (1.0,50), (0.95,50), (1.1,50), (1.05,50),(0.91,no),(0.91,30),(0.91,100)
        templ_list.append(templates[str(run_num)][flav])
        templ_err_list.append(templates[str(run_num)][flav]/np.sqrt(MCmaps[str(run_num)][flav]))
    
    templ = np.array(templ_list)
    templ_err = np.array(templ_err_list)

    y_val_max = np.max(np.divide(templ, templ[1]))
    y_val_min = np.min(np.divide(templ, templ[1]))

    tml_shape = np.shape(templ)
    print "shape : " , np.shape(templ)
    n_ebins = tml_shape[1] 
    n_czbins = tml_shape[2] 
    ############################### DOM efficiency ######################################
    
    for i in range(0,n_ebins):
        for j in range(0,n_czbins):
    
            ########### Get Data ############
            dom_eff = np.array([0.91, 1.0, 0.95, 1.1, 1.05, 0.91, 0.91, 0.91])
            hole_ice = np.array([1.0/50, 1.0/50, 1.0/50, 1.0/50, 1.0/50, 0.0, 1.0/30, 1.0/100])         #unit: cm-1
            bin_counts = np.array([templ[0][i][j],templ[1][i][j],templ[2][i][j],templ[3][i][j],templ[4][i][j],templ[5][i][j],templ[6][i][j],templ[7][i][j]]) 
            bin_err = np.array([templ_err[0][i][j],templ_err[1][i][j],templ_err[2][i][j],templ_err[3][i][j],templ_err[4][i][j],templ_err[5][i][j],templ_err[6][i][j],templ_err[7][i][j]]) 
            bin_ratio_values = bin_counts/bin_counts[1]  #divide by the nominal value templ[1][i]

            bin_ratio_err = bin_ratio_values * np.sqrt(np.square(bin_err[1]/bin_counts[1])+np.square(bin_err/bin_counts))

            if templ[1][i][j] == 0:
                print "templ[1][", i , "][", j, "] == 0 !!!!!!!!!"

            fixed_r_val = bin_ratio_values[0]

            # line goes through point (0.02, fixed_r_val), fixed_r_val is the value for dom_eff = 0.91 and hole ice = 0.02
            exec('def dom_eff_linear_through_point(x, k): return k*x + %s - k*0.91'%fixed_r_val)

            ########### DOM efficiency #############

            popt_1, pcov_1 = curve_fit(dom_eff_linear_through_point,dom_eff[0:5],bin_ratio_values[0:5])
            k1 = popt_1[0]
            k_DE[i][j]= k1
            fixed_ratio[i][j]= fixed_r_val
   
            if args.plot:
                fig_num = i * n_czbins+ j
                if (fig_num == 0 or fig_num == n_czbins * n_ebins):
                    fig = plt.figure(num=1, figsize=( 4*n_czbins, 4*n_ebins))
                subplot_idx = n_czbins*(n_ebins-1-i)+ j + 1
                #print 'subplot_idx = ', subplot_idx
                plt.subplot(n_ebins, n_czbins, subplot_idx)
                #plt.title("CZ:[%s, %s] E:[%.1f, %.1f]"% (czbin_edges[j], czbin_edges[j+1], ebin_edges[i], ebin_edges[i+1]))
                plt.scatter(dom_eff[0:5], bin_ratio_values[0:5], color='blue')
                plt.errorbar(dom_eff[0:5], bin_ratio_values[0:5], yerr=bin_ratio_err[0:5],fmt='none')
                plt.xlim(0.7,1.2)
                #plt.ylim(0.6,1.5)
                plt.ylim(y_val_min,y_val_max)

                dom_func_plot_x = np.arange(0.8 - x_steps, 1.2 + x_steps, x_steps)
                best_fit = dom_eff_linear_through_point(dom_func_plot_x, k1)
                dom_func_plot, = plt.plot(dom_func_plot_x, best_fit, 'r-')
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
                    plt.savefig(png_name + '_domeff_%s.png'%flav )
                    plt.savefig(png_name + '_domeff_%s.pdf'%flav )
                    plt.clf()

    
            ########### Hole Ice #############

    for i in range(0,n_ebins):
        for j in range(0,n_czbins):
    
            ########### Get Data ############
            dom_eff = np.array([0.91, 1.0, 0.95, 1.1, 1.05, 0.91, 0.91, 0.91])
            hole_ice = np.array([1.0/50, 1.0/50, 1.0/50, 1.0/50, 1.0/50, 0.0, 1.0/30, 1.0/100])         #unit: cm-1
            bin_counts = np.array([templ[0][i][j],templ[1][i][j],templ[2][i][j],templ[3][i][j],templ[4][i][j],templ[5][i][j],templ[6][i][j],templ[7][i][j]]) 
            bin_err = np.array([templ_err[0][i][j],templ_err[1][i][j],templ_err[2][i][j],templ_err[3][i][j],templ_err[4][i][j],templ_err[5][i][j],templ_err[6][i][j],templ_err[7][i][j]]) 
            bin_ratio_values = bin_counts/bin_counts[1]  #divide by the nominal value templ[1][i]

            bin_ratio_err = bin_ratio_values * np.sqrt(np.square(bin_err[1]/bin_counts[1])+np.square(bin_err/bin_counts))

            if templ[1][i][j] == 0:
                print "templ[1][", i , "][", j, "] == 0 !!!!!!!!!"

            fixed_r_val = bin_ratio_values[0]

            # line goes through point (0.02, fixed_r_val), fixed_r_val is the value for dom_eff = 0.91 and hole ice = 0.02
            exec('def hole_ice_quadratic_through_point(x, k, p): return k*(x-0.02) + p*(x-0.02)**2 + %s'%fixed_r_val)
            
            # get x values and y values
            ice_x = np.array([hole_ice[0],hole_ice[5],hole_ice[6],hole_ice[7]])
            ice_y = np.array([bin_ratio_values[0],bin_ratio_values[5],bin_ratio_values[6],bin_ratio_values[7]])
            ice_y_err = np.array([bin_ratio_err[0],bin_ratio_err[5],bin_ratio_err[6],bin_ratio_err[7]])

            if args.plot:
                fig_num = i * n_czbins+ j
                if (fig_num == 0 or fig_num == n_czbins * n_ebins):
                    fig = plt.figure(num=2, figsize=( 4*n_czbins, 4*n_ebins))
                subplot_idx = n_czbins*(n_ebins-1-i)+ j + 1
                plt.subplot(n_ebins, n_czbins, subplot_idx)
                #plt.title("CZ:[%s, %s] E:[%.1f, %.1f]"% (czbin_edges[j], czbin_edges[j+1], ebin_edges[i], ebin_edges[i+1]))
                plt.scatter(ice_x, ice_y, color='blue')
                plt.errorbar(ice_x, ice_y, yerr=ice_y_err,fmt='none')
                plt.xlim(-0.02,0.06)
                plt.ylim(0.5,1.5)

                popt_2, pcov_2 = curve_fit(hole_ice_quadratic_through_point,ice_x,ice_y)
                k2 = popt_2[0]
                p2 = popt_2[1]
                k_HI[i][j]= k2
                p_HI[i][j]= p2

                ice_func_plot_x = np.arange(-0.02, 0.06 + x_steps, x_steps)
                best_fit = hole_ice_quadratic_through_point(ice_func_plot_x, k2,p2)
                ice_func_plot, = plt.plot(ice_func_plot_x, best_fit, 'r-')
                if j > 0:
                    plt.setp(plt.gca().get_yticklabels(), visible=False)
                if i > 0:
                    plt.setp(plt.gca().get_xticklabels(), visible=False)

                if(fig_num==n_czbins * n_ebins-1):
                    fig.subplots_adjust(hspace=0)
                    fig.subplots_adjust(wspace=0)
                    plt.figtext(0.5, 0.04, 'cos(zen)',fontsize=60,ha='center') 
                    plt.figtext(0.09, 0.5, 'energy',rotation=90,fontsize=60,ha='center') 
                    plt.figtext(0.5, 0.95, 'Hole Ice fits %s'%flav,fontsize=60,ha='center')
                    plt.savefig(png_name + '_holeice_%s.png'%flav )
                    plt.savefig(png_name + '_holeice_%s.pdf'%flav )
                    plt.clf()


    fits_DOMEff[flav]['slopes'] = k_DE 
    fits_HoleIce[flav]['linear'] = k_HI
    fits_HoleIce[flav]['quadratic'] = p_HI
    fits_DOMEff[flav]['fixed_ratios'] = fixed_ratio 
    fits_HoleIce[flav]['fixed_ratios'] = fixed_ratio

#Assemble output dict
output_template = {'templates' : templates,
                   'template_settings' : template_settings}
#And write to file
to_json(output_template,'DomEff_templates_up_down_10_by_16.json')
to_json(fits_DOMEff,'DomEff_linear_fits_10_by_16.json')
to_json(fits_HoleIce,'HoleIce_quadratic_fits_10_by_16.json')


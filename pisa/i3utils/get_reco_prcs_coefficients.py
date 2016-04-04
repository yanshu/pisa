#! /usr/bin/env python
#
# get_eight_templates.py
#
# Get eight templates from eight MC sets (using different DOM eff. and hole ice values), write all template bin values to a json file.
# Use plot_three_linear_fits_in_template.py to produce bin value ratio vs DOM eff. and hole ice, with three different kinds of linear fits.
#
# author: Feifei Huang - fxh140@psu.edu
#         Tim Arlen - tca3@psu.edu
#
# date:   22-July-2015
#

import numpy as np
import copy
import random as rnd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pisa.utils.utils as utils
from pisa.utils.log import logging, profile, physics, set_verbosity
from pisa.utils.jsons import from_json,to_json
from pisa.resources.resources import find_resource
from pisa.analysis.TemplateMaker_nutau import TemplateMaker
from pisa.analysis.GetMCError import GetMCError
from pisa.utils.params import get_values, change_nutau_norm_settings

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

parser = ArgumentParser(description='''Get eight templates from eight MC sets (using different DOM eff. and hole ice values), write all template bin values to a json file. ''', formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-t','--template_settings',type=str,
                    metavar='JSONFILE', required = True,
                    help='''Settings related to the template generation and systematics.''')
parser.add_argument('-s','--sim',type=str,
                    metavar='simu', required = True,
                    help='''Which simulation, can only be 4digit, 5digit, or dima''')
parser.add_argument('--templ_already_saved',action='store_true',default=False,
                    help="Read templates from already saved file; saves time when only need plotting.")
parser.add_argument('--reco_prcs_vals',type=str,
                    metavar='reco_prcs_vals',
                    default = 'np.linspace(0.7,2.0,14)', help = '''The reco. precision values to use.''')
parser.add_argument('--plot',action='store_true',default=False,
                    help="Plot the fits of DOM efficiency and hole ice for each bin.")
parser.add_argument('-pd','--pseudo_data_settings',type=str,
                    metavar='JSONFILE',default=None,
                    help='''Settings for pseudo data templates, if desired to be different from template_settings.''')
parser.add_argument('-o','--outdir',type=str,
                    metavar='DIR', required = True, help='''Output directory''')
parser.add_argument('-v', '--verbose', action='count', default=None,
                    help='set verbosity level')
args = parser.parse_args()

set_verbosity(args.verbose)

#Read in the settings
x_steps = 0.04
outdir = args.outdir
utils.mkdir(outdir)
utils.mkdir(outdir+'/plots')
template_settings = from_json(args.template_settings)
czbin_edges = template_settings['binning']['czbins']
ebin_edges = template_settings['binning']['anlys_ebins']
template_settings['params']['atmos_mu_scale']['value'] = 0

template_settings = copy.deepcopy(template_settings)
pseudo_data_settings = from_json(args.pseudo_data_settings) if args.pseudo_data_settings is not None else template_settings

# make sure that both pseudo data and template are using the same
# channel. Raise Exception and quit otherwise
channel = template_settings['params']['channel']['value']
if channel != pseudo_data_settings['params']['channel']['value']:
    error_msg = "Both template and pseudo data must have same channel!\n"
    error_msg += " pseudo_data_settings channel: '%s', template channel: '%s' "%(pseudo_data_settings['params']['channel']['value'],channel)
    raise ValueError(error_msg)

#template_maker = TemplateMaker(get_values(template_settings['params']), **template_settings['binning'])

#reco_mc_file = from_json(find_resource(template_settings['params']['reco_mc_wt_file']['value']))
if args.sim == '4digit':
    reco_mc_file = "~/pisa/pisa/resources/aeff/events__deepcore__ic86__runs_1260-1660:200__proc_v6__joined_G_nue_cc+nuebar_cc_G_numu_cc+numubar_cc_G_nutau_cc+nutaubar_cc_G_nuall_nc+nuallbar_nc.hdf5"
elif args.sim == '5digit':
    reco_mc_file = "~/pisa/pisa/resources/aeff/events__deepcore__IC86__runs_12585-16585:20000__proc_v5digit__joined_G_nue_cc+nuebar_cc_G_numu_cc+numubar_cc_G_nutau_cc+nutaubar_cc_G_nuall_nc+nuallbar_nc.hdf5"
elif args.sim == 'dima':
    #TODO    
    print "to do, dima sets"

def func_cubic_through_nominal(x, a, b, c):
    return a*x*x*x + b*x*x + c*x + 1.0 - a - b - c

tmaps = {}
coeffs = {}
MCmaps = {}
reco_prcs_vals = eval(args.reco_prcs_vals)
data_nutau_norm = 1.0
print "reco_prcs_vals = ", reco_prcs_vals

if not args.templ_already_saved:
    for precision_tag in ['e_reco_precision_up', 'e_reco_precision_down', 'cz_reco_precision_up', 'cz_reco_precision_down']:
        MCmaps[precision_tag] = {}
        tmaps[precision_tag] = {}
    
        for reco_prcs_val in reco_prcs_vals:
            tmaps[precision_tag][str(reco_prcs_val)] = {'trck':{},
                                                        'cscd':{}}
            MCmaps[precision_tag][str(reco_prcs_val)] = {'trck':{},
                                                         'cscd':{}}
    
            template_settings_Reco = copy.deepcopy(template_settings)
            template_settings_Reco['params'][precision_tag]['value'] = reco_prcs_val
            template_settings_Reco['params']['nutau_norm']['value'] = data_nutau_norm 
    
            RP_template_maker = TemplateMaker(get_values(template_settings_Reco['params']), **template_settings_Reco['binning'])
            tmap = RP_template_maker.get_template(get_values(change_nutau_norm_settings(template_settings_Reco['params'], data_nutau_norm ,True, normal_hierarchy=True)),no_sys_applied= True, apply_reco_prcs=True)
            tmaps[precision_tag][str(reco_prcs_val)]['trck'] = tmap['trck']['map']
            tmaps[precision_tag][str(reco_prcs_val)]['cscd'] = tmap['cscd']['map']
    
            MCMap = GetMCError(get_values(template_settings_Reco['params']), template_settings_Reco['binning']['anlys_ebins'], template_settings_Reco['binning']['czbins'], reco_mc_file)
            tmap_MC = MCMap.get_mc_events_map(True, get_values(template_settings_Reco['params']), reco_mc_file)
            MCmaps[precision_tag][str(reco_prcs_val)]['trck'] = tmap_MC['trck']['map']
            MCmaps[precision_tag][str(reco_prcs_val)]['cscd'] = tmap_MC['cscd']['map']
    
    #Assemble output dict
    output = {'tmaps' : tmaps,
              'MCmaps' : MCmaps,
              'coeffs' : coeffs,
              'template_settings' : template_settings}
    if args.pseudo_data_settings is not None:
        output['pseudo_data_settings'] = pseudo_data_settings
    to_json(output,outdir+'%s_RecoPrcs_templates_10_by_16.json'%args.sim)
else:
    # if templates already saved
    output_template = from_json(outdir+'%s_RecoPrcs_templates_10_by_16.json'%args.sim)
    tmaps = output_template['tmaps']
    MCmaps = output_template['MCmaps']


for precision_tag in ['e_reco_precision_up', 'e_reco_precision_down', 'cz_reco_precision_up', 'cz_reco_precision_down']:
    templ_list = []
    coeffs[precision_tag] = {'trck':{},
                             'cscd':{}}
    for flav in ['trck','cscd']:
        templ = []
        templ_err = []
        templ_nominal = []
        templ_nominal_err = []
        for reco_prcs_val in reco_prcs_vals:
            templ.append(tmaps[precision_tag][str(reco_prcs_val)][flav])  

            ## Get template error (either standard deviation or standard error):
            # standard deviation: sqrt(n_event_rate):
            #templ_err.append(np.sqrt(tmapsprecision_tag][str(reco_prcs_val)]))

            # standard error: sqrt(n_event_rate)/sqrt(N_mc):
            templ_err.append(np.sqrt(tmaps[precision_tag][str(reco_prcs_val)][flav])/np.sqrt(MCmaps[precision_tag][str(reco_prcs_val)][flav]))  

        templ_nominal = np.array(tmaps[precision_tag]['1.0'][flav])
        templ_nominal_err = np.array(np.sqrt(tmaps[precision_tag]['1.0'][flav])/np.sqrt(MCmaps[precision_tag]['1.0'][flav]))
        templ = np.array(templ)
        templ_err = np.array(templ_err)
        templ_nominal = np.array(templ_nominal)
        templ_nominal_err = np.array(templ_nominal_err)

        tml_shape = np.shape(templ)
        n_ebins = tml_shape[1] 
        n_czbins = tml_shape[2] 
        coeff = np.empty(np.shape(tmaps[precision_tag]['1.0'][flav]), dtype = object) 
        y_val_max = np.max(np.divide(templ, templ_nominal))
        y_val_min = np.min(np.divide(templ, templ_nominal))
        print "y_val_min = ", y_val_min
        print "y_val_max = ", y_val_max
        for i in range(0, n_ebins):
            for j in range(0, n_czbins):
                bin_counts = templ[:,i,j]

                # if use bin value ratio (to nominal bin value) as y axis
                bin_ratio_values = templ[:,i,j]/templ_nominal[i,j]  #divide by the nominal value

                # standard error of ratio n1/n2: (n1/n2)*sqrt((SE1/n1)^2 + (SE2/n2)^2) 
                bin_ratio_err_values = bin_ratio_values * np.sqrt(np.square(templ_nominal_err[i,j]/templ_nominal[i,j])+np.square(templ_err[:,i,j]/templ[:,i,j]))
    
                ## Cubic Fit  (through (1, 1) point)
                popt, pcov = curve_fit(func_cubic_through_nominal, reco_prcs_vals, bin_ratio_values)
                a = popt[0]
                b = popt[1]
                c = popt[2]
                coeff[i,j] = [a, b, c]

                if args.plot:
                    fig_num = i * n_czbins+ j
                    if (fig_num == 0 or fig_num == n_czbins * n_ebins):
                        fig = plt.figure(num=1, figsize=( 4*n_czbins, 4*n_ebins))
                    subplot_idx = n_czbins*(n_ebins-1-i)+ j + 1
                    #print 'subplot_idx = ', subplot_idx
                    plt.subplot(n_ebins, n_czbins, subplot_idx)
                    plt.title("CZ:[%s, %s] E:[%.1f, %.1f]"% (czbin_edges[j], czbin_edges[j+1], ebin_edges[i], ebin_edges[i+1]))
                    plt.scatter(reco_prcs_vals, bin_ratio_values, color='blue')
                    plt.errorbar(reco_prcs_vals, bin_ratio_values, yerr=bin_ratio_err_values,fmt='none')
                    plt.xlim(0.7,2.0)
                    plt.ylim(y_val_min-0.01,y_val_max+0.01)
                    cubic_func_plot_x = np.arange(0.7 - x_steps, 2.0 + x_steps, x_steps)
                    cubic_func_plot_y = func_cubic_through_nominal(cubic_func_plot_x, a, b, c)
                    cubic_func_plot, = plt.plot(cubic_func_plot_x, cubic_func_plot_y, 'r-')
                    #if j > 0:
                    #    plt.setp(plt.gca().get_yticklabels(), visible=False)
                    #if i > 0:
                    #    plt.setp(plt.gca().get_xticklabels(), visible=False)
                    if(fig_num == n_czbins * n_ebins-1):
                        #plt.figtext(0.5, 0.04, 'cos(zen)',fontsize=60,ha='center') 
                        #plt.figtext(0.09, 0.5, 'energy',rotation=90,fontsize=60,ha='center') 
                        #plt.figtext(0.5, 0.95, 'Reco Precision cubic fits %s'%flav, fontsize=60,ha='center')
                        #fig.subplots_adjust(hspace=0)
                        #fig.subplots_adjust(wspace=0)
                        plt.savefig(outdir+ 'plots/'+'%s_%s_fits_reco_prcs_%s.png'%(args.sim, precision_tag, flav))
                        plt.savefig(outdir+ 'plots/'+'%s_%s_fits_reco_prcs_%s.pdf'%(args.sim, precision_tag, flav))
                        plt.clf()

        coeffs[precision_tag][flav] = coeff


#And write to file
#to_json(coeffs,outdir+'%s_RecoPrcs_fits_10_by_16.json'%args.sim)
to_json(coeffs,outdir+'%s_RecoPrecisionCubicFitCoefficients_0.7_2.0_data_tau_10_by_16.json'%args.sim)


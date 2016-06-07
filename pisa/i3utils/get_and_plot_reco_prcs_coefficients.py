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
import pisa.utils.events as events
from pisa.utils.jsons import from_json,to_json
from pisa.resources.resources import find_resource
from pisa.analysis.GetMCError import GetMCError
from pisa.utils.params import get_values, change_nutau_norm_settings

import numpy as np
import numpy.ma as ma
import matplotlib as mpl
mpl.use('Agg')
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
                    default = 'np.linspace(0.7,1.3,13)', help = '''The reco. precision values to use.''')
parser.add_argument('--plot',action='store_true',default=False,
                    help="Plot the fits of DOM efficiency and hole ice for each bin.")
parser.add_argument('--use_event_PISA',action='store_true',default=False,
                    help="Use event-by-event PISA; otherwise, use histogram-based PISA") 
parser.add_argument('--no_NC_osc',action='store_true',default=False,
                    help="Use no oscillation for NC, for cmpr with oscFit.") 
parser.add_argument('--use_mask',action='store_true',default=False,
                    help="Mask the right corner when setting the y_val_max.") 
parser.add_argument('--IMH',action='store_true',default=False,
                    help="Use inverted mass hiearchy.")
parser.add_argument('--plotMC',action='store_true',default=False,
                    help="Plot the MC events number in each bin vs DOM efficiency and hole ice values.")
parser.add_argument('--plotReso',action='store_true',default=False,
                    help="Plot the Resolution.")
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
x_steps = 0.05
outdir = args.outdir
if args.use_event_PISA:
    from pisa.analysis.TemplateMaker_MC import TemplateMaker
    pisa_mode = 'event'
else:
    from pisa.analysis.TemplateMaker_nutau import TemplateMaker
    pisa_mode = 'hist'
use_NMH = not(args.IMH)
if args.no_NC_osc:
    nc_osc_mode = 'no_NC_osc'
else:
    nc_osc_mode = 'has_NC_osc'
utils.mkdir(outdir)
utils.mkdir(outdir+'/plots')
utils.mkdir(outdir+'/plots/png/')
if args.templ_already_saved:
    # if templates already saved
    output_template = from_json(outdir+'/%s_%s_RecoPrcs_templates_%s.json'% (args.sim, pisa_mode, nc_osc_mode))
    tmaps = output_template['tmaps']
    MCmaps = output_template['MCmaps']
    coeffs = output_template['coeffs']
    template_settings = output_template['template_settings']
else:
    template_settings = from_json(args.template_settings)

czbin_edges = template_settings['binning']['czbins']
ebin_edges = template_settings['binning']['anlys_ebins']
template_settings['params']['atmos_mu_scale']['value'] = 0

pseudo_data_settings = from_json(args.pseudo_data_settings) if args.pseudo_data_settings is not None else template_settings

# make sure that both pseudo data and template are using the same
# channel. Raise Exception and quit otherwise
channel = template_settings['params']['channel']['value']
if channel != pseudo_data_settings['params']['channel']['value']:
    error_msg = "Both template and pseudo data must have same channel!\n"
    error_msg += " pseudo_data_settings channel: '%s', template channel: '%s' "%(pseudo_data_settings['params']['channel']['value'],channel)
    raise ValueError(error_msg)

if args.sim == '4digit':
    reco_mc_file = "aeff/events__deepcore__ic86__runs_1260-1660:200__proc_v4digit__joined_G_nue_cc+nuebar_cc_G_numu_cc+numubar_cc_G_nutau_cc+nutaubar_cc_G_nuall_nc+nuallbar_nc.hdf5"
elif args.sim == '5digit':
    reco_mc_file = "aeff/events__deepcore__IC86__runs_12585-16585:20000__proc_v5digit__joined_G_nue_cc+nuebar_cc_G_numu_cc+numubar_cc_G_nutau_cc+nutaubar_cc_G_nuall_nc+nuallbar_nc.hdf5"
elif args.sim == 'dima_p1':
    run_num = 600
    reco_mc_file = 'aeff/events__deepcore__IC86__runs_12%s1-12%s3,14%s1-14%s3,16%s1-16%s3__proc_v5digit__joined_G_nue_cc+nuebar_cc_G_numu_cc+numubar_cc_G_nutau_cc+nutaubar_cc_G_nuall_nc+nuallbar_nc.hdf5' % (run_num, run_num, run_num,run_num,run_num,run_num)
elif args.sim == 'dima_p2':
    #TODO    
    print "to do, dima_p2 sets"
else:
    raise ValueError( "sim allowed: ['5digit', '4digit', 'dima']")

# Just make sure the reco_mc_file is correct
template_settings['params']['reco_mc_wt_file']['value'] = reco_mc_file
template_settings['params']['reco_vbwkde_evts_file']['value'] = reco_mc_file
template_settings['params']['pid_events']['value'] = reco_mc_file

if args.plotReso:
    utils.mkdir(outdir+'/plots/resolutions_withweight/')
    utils.mkdir(outdir+'/plots/resolutions_noweight/')
    utils.mkdir(outdir+'/plots/resolutions_withweight/png/')
    utils.mkdir(outdir+'/plots/resolutions_noweight/png/')
    evts = events.Events(find_resource(reco_mc_file))
    for flavor in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
        flavor_dict = {}
        logging.debug("Working on %s kernels"%flavor)
        for int_type in ['cc','nc']:
            true_energy = evts[flavor][int_type]['true_energy']
            true_coszen = evts[flavor][int_type]['true_coszen']
            reco_energy = evts[flavor][int_type]['reco_energy']
            reco_coszen = evts[flavor][int_type]['reco_coszen']
            weight = evts[flavor][int_type]['weighted_aeff']

    for flavor in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
        for int_type in ['cc','nc']:
            n_ebins = len(ebin_edges)-1
            n_czbins = len(czbin_edges)-1
            for i in range(0, n_ebins):
                for j in range(0, n_czbins):
                    E_select = np.logical_and(true_energy<ebin_edges[i+1], true_energy>=ebin_edges[i])
                    CZ_select = np.logical_and(true_coszen<czbin_edges[j+1], true_coszen>=czbin_edges[j])
                    # select values in this bin
                    select = np.logical_and(E_select, CZ_select)
                    E_true = true_energy[select]
                    CZ_true = true_coszen[select]
                    E_reco = reco_energy[select]
                    CZ_reco = reco_coszen[select]
                    weight_select = weight[select]
                    fig_num = i * n_czbins+ j
                    if (fig_num == 0 or fig_num == n_czbins * n_ebins):
                        fig = plt.figure(num=3, figsize=( 4*n_czbins, 4*n_ebins))
                        fig = plt.figure(num=4, figsize=( 4*n_czbins, 4*n_ebins))
                    plt.figure(3)
                    subplot_idx = n_czbins*(n_ebins-1-i)+ j + 1
                    plt.subplot(n_ebins, n_czbins, subplot_idx)
                    plt.title("CZ:[%s, %s] E:[%.1f, %.1f]"% (czbin_edges[j], czbin_edges[j+1], ebin_edges[i], ebin_edges[i+1]))
                    plt.hist((E_reco-E_true)/E_true, weights=weight_select, bins=50)
                    if(fig_num == n_czbins * n_ebins-1):
                        plt.savefig(outdir+ '/plots/resolutions_withweight/png/'+'%s_%s_%s_resolutions_%s_withweight.png'%(args.sim , pisa_mode, flavor, int_type))
                        plt.savefig(outdir+ '/plots/resolutions_withweight/'+'%s_%s_%s_resolutions_%s_withweight.pdf'%(args.sim , pisa_mode, flavor, int_type))
                        plt.clf()

                    plt.figure(4)
                    plt.subplot(n_ebins, n_czbins, subplot_idx)
                    plt.title("CZ:[%s, %s] E:[%.1f, %.1f]"% (czbin_edges[j], czbin_edges[j+1], ebin_edges[i], ebin_edges[i+1]))
                    plt.hist((E_reco-E_true)/E_true, bins=50)
                    if(fig_num == n_czbins * n_ebins-1):
                        plt.savefig(outdir+ '/plots/resolutions_noweight/png/'+'%s_%s_%s_%s_resolutions_noweight.png'%(args.sim , pisa_mode, flavor, int_type))
                        plt.savefig(outdir+ '/plots/resolutions_noweight/'+'%s_%s_%s_%s_resolutions_noweight.pdf'%(args.sim , pisa_mode, flavor, int_type))
                        plt.clf()


def func_cubic_through_nominal(x, a, b, c):
    return a*x*x*x + b*x*x + c*x + 1.0 - a - b - c

reco_prcs_vals = eval(args.reco_prcs_vals)
data_nutau_norm = 1.0
print "reco_prcs_vals = ", reco_prcs_vals

if not args.templ_already_saved:
    RP_template_maker = TemplateMaker(get_values(change_nutau_norm_settings(template_settings['params'], 1.0 ,nutau_norm_fix=True, normal_hierarchy=use_NMH)), no_sys_maps=True, **template_settings['binning'])
    tmaps = {}
    coeffs = {}
    MCmaps = {}
    for precision_tag in ['e_reco_precision_up', 'e_reco_precision_down', 'cz_reco_precision_up', 'cz_reco_precision_down']:
        MCmaps[precision_tag] = {}
        tmaps[precision_tag] = {}
    
        for reco_prcs_val in reco_prcs_vals:
            print "Getting maps for ", precision_tag , " = ", reco_prcs_val
            tmaps[precision_tag][str(reco_prcs_val)] = {'trck':{},
                                                        'cscd':{}}
            MCmaps[precision_tag][str(reco_prcs_val)] = {'trck':{},
                                                         'cscd':{}}
    
            template_settings_Reco = copy.deepcopy(template_settings)
            template_settings_Reco['params'][precision_tag]['value'] = reco_prcs_val
            template_settings_Reco['params']['nutau_norm']['value'] = data_nutau_norm 
            tmap = RP_template_maker.get_template(get_values(change_nutau_norm_settings(template_settings_Reco['params'], data_nutau_norm ,True, normal_hierarchy=use_NMH)), no_sys_maps= True, apply_reco_prcs=True, turn_off_osc_NC=args.no_NC_osc)
            tmaps[precision_tag][str(reco_prcs_val)]['trck']['map'] = tmap['trck']['map']
            tmaps[precision_tag][str(reco_prcs_val)]['cscd']['map'] = tmap['cscd']['map']
            tmaps[precision_tag][str(reco_prcs_val)]['trck']['sumw2_nu'] = tmap['trck']['sumw2_nu'] 
            tmaps[precision_tag][str(reco_prcs_val)]['cscd']['sumw2_nu'] = tmap['cscd']['sumw2_nu'] 
    
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
    to_json(output,outdir+'/%s_%s_RecoPrcs_templates_%s.json'% (args.sim, pisa_mode, nc_osc_mode))


#print "nominal MCmaps = ", MCmaps['e_reco_precision_up']['1.0']['cscd']
for precision_tag in ['e_reco_precision_up', 'e_reco_precision_down', 'cz_reco_precision_up', 'cz_reco_precision_down']:
    coeffs[precision_tag] = {'trck':{},
                             'cscd':{}}
    for flav in ['trck','cscd']:
        templ = []
        templ_MC = []
        templ_err = []
        for reco_prcs_val in reco_prcs_vals:
            templ.append(tmaps[precision_tag][str(reco_prcs_val)][flav]['map'])  
            templ_MC.append(MCmaps[precision_tag][str(reco_prcs_val)][flav])  
            templ_err.append(np.sqrt(tmaps[precision_tag][str(reco_prcs_val)][flav]['sumw2_nu']))

        templ = np.array(templ)
        templ_MC = np.array(templ_MC)
        templ_err = np.array(templ_err)
        templ_nominal = np.array(tmaps[precision_tag]['1.0'][flav]['map'])
        templ_nominal_err = np.array(np.sqrt(tmaps[precision_tag]['1.0'][flav]['sumw2_nu']))

        tml_shape = np.shape(templ)
        n_ebins = tml_shape[1] 
        n_czbins = tml_shape[2] 
        #print "tml_shape = ", tml_shape
        coeff = np.empty(np.shape(tmaps[precision_tag]['1.0'][flav]['map']), dtype = object) 
        all_ratios = np.divide(templ, templ_nominal)
        #print "all_ratios = ", all_ratios
        if args.use_mask:
            # mask the right corner ( usually very low statistics, thus has the largest ratio)
            mask = np.zeros(np.shape(all_ratios))
            mask[:,0,n_czbins-1] = 1
            #print "mask = ", mask[0]
            all_ratios_mask = ma.masked_array(all_ratios, mask = mask)
            #print "all_ratios_mask = ", all_ratios_mask
            y_val_max = np.max(all_ratios_mask)
            y_val_min = np.min(all_ratios_mask)
        else:
            y_val_max = np.max(all_ratios)
            y_val_min = np.min(all_ratios)
        print "y_val_min = ", y_val_min
        print "y_val_max = ", y_val_max
        for i in range(0, n_ebins):
            for j in range(0, n_czbins):
                bin_counts = templ[:,i,j]
                bin_err = templ_err[:,i,j]
                nominal_bin_counts = templ_nominal[i,j]
                nominal_bin_err = templ_nominal_err[i,j] 

                # if use bin value ratio (to nominal bin value) as y axis
                bin_ratio_values = templ[:,i,j]/templ_nominal[i,j]  #divide by the nominal value
                if templ_nominal[i,j] ==0:
                    print "templ_nominal[i,j]  == 0: "

                # standard error of ratio n1/n2: (n1/n2)*sqrt((SE1/n1)^2 + (SE2/n2)^2) 
                bin_ratio_err_values = bin_ratio_values * np.sqrt(np.square(nominal_bin_err/nominal_bin_counts)+np.square(bin_err/bin_counts))
    
                ## Cubic Fit  (through (1, 1) point)
                sigma = copy.deepcopy(bin_ratio_err_values)
                popt, pcov = curve_fit(func_cubic_through_nominal, reco_prcs_vals, bin_ratio_values, sigma=sigma)
                a = popt[0]
                b = popt[1]
                c = popt[2]
                coeff[i,j] = [a, b, c]

                if args.plotMC:
                    fig_num = i * n_czbins+ j
                    if (fig_num == 0 or fig_num == n_czbins * n_ebins):
                        fig = plt.figure(num=1, figsize=( 4*n_czbins, 4*n_ebins))
                    plt.figure(1)
                    subplot_idx = n_czbins*(n_ebins-1-i)+ j + 1
                    #print 'subplot_idx = ', subplot_idx
                    plt.subplot(n_ebins, n_czbins, subplot_idx)
                    plt.title("CZ:[%s, %s] E:[%.1f, %.1f]"% (czbin_edges[j], czbin_edges[j+1], ebin_edges[i], ebin_edges[i+1]))
                    plt.scatter(reco_prcs_vals, templ_MC[:,i,j], color='blue')
                    plt.xlim(min(reco_prcs_vals)-0.01, max(reco_prcs_vals)+0.01)
                    if(fig_num == n_czbins * n_ebins-1):
                        plt.savefig(outdir+ '/plots/png/'+'%s_%s_%s_MC_number_reco_prcs_%s.png'%(args.sim, pisa_mode, precision_tag, flav))
                        plt.savefig(outdir+ '/plots/'+'%s_%s_%s_MC_number_reco_prcs_%s.pdf'%(args.sim, pisa_mode, precision_tag, flav))
                        plt.clf()


                if args.plot:
                    fig_num = i * n_czbins+ j
                    if (fig_num == 0 or fig_num == n_czbins * n_ebins):
                        fig = plt.figure(num=2, figsize=( 4*n_czbins, 4*n_ebins))
                    plt.figure(2)
                    subplot_idx = n_czbins*(n_ebins-1-i)+ j + 1
                    #print 'subplot_idx = ', subplot_idx
                    plt.subplot(n_ebins, n_czbins, subplot_idx)
                    plt.title("CZ:[%s, %s] E:[%.1f, %.1f]"% (czbin_edges[j], czbin_edges[j+1], ebin_edges[i], ebin_edges[i+1]))
                    plt.scatter(reco_prcs_vals, bin_ratio_values, color='blue')
                    plt.errorbar(reco_prcs_vals, bin_ratio_values, yerr=bin_ratio_err_values,fmt='none')
                    plt.xlim(min(reco_prcs_vals)-0.01, max(reco_prcs_vals)+0.01)
                    plt.ylim(y_val_min-0.01,y_val_max+0.01)
                    cubic_func_plot_x = np.arange(min(reco_prcs_vals)- x_steps, max(reco_prcs_vals) + x_steps, x_steps)
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
                        plt.savefig(outdir+ '/plots/png/'+'%s_%s_%s_fits_reco_prcs_%s_%s.png'%(args.sim, pisa_mode, precision_tag, flav, nc_osc_mode))
                        plt.savefig(outdir+ '/plots/'+'%s_%s_%s_fits_reco_prcs_%s_%s.pdf'%(args.sim, pisa_mode, precision_tag, flav, nc_osc_mode))
                        #plt.savefig(outdir+ '/plots/png/'+'%s_%s_%s_fits_reco_prcs_%s.png'%(args.sim, precision_tag, flav))
                        #plt.savefig(outdir+ '/plots/'+'%s_%s_%s_fits_reco_prcs_%s.pdf'%(args.sim, precision_tag, flav))
                        plt.clf()
                        plt.clf()

        coeffs[precision_tag][flav] = coeff


#And write to file
to_json(coeffs,outdir+'/%s_%s_RecoPrecisionCubicFitCoefficients_%s_%s_data_tau_%s.json'%(args.sim, pisa_mode, min(reco_prcs_vals), max(reco_prcs_vals), nc_osc_mode))


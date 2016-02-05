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

from pisa.utils.log import logging, profile, physics, set_verbosity
from pisa.utils.jsons import from_json,to_json
from pisa.analysis.llr.LLHAnalysis_nutau import find_max_llh_bfgs
from pisa.analysis.stats.Maps import get_seed, flatten_map
from pisa.analysis.stats.Maps_nutau import get_flipped_map, get_combined_map, get_up_map, get_flipped_down_map
from pisa.analysis.TemplateMaker_nutau import TemplateMaker
from pisa.analysis.GetMCError import GetMCError
from pisa.utils.params import get_values, select_hierarchy_and_nutau_norm,change_nutau_norm_settings, select_hierarchy

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

parser = ArgumentParser(description='''Get eight templates from eight MC sets (using different DOM eff. and hole ice values), write all template bin values to a json file. ''', formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-t','--template_settings',type=str,
                    metavar='JSONFILE', required = True,
                    help='''Settings related to the template generation and systematics.''')
parser.add_argument('--reco_prcs_vals',type=str,
                    metavar='reco_prcs_vals',
                    default = 'np.linspace(0.7,2.0,14)', help = '''The reco. precision values to use.''')
parser.add_argument('-pd','--pseudo_data_settings',type=str,
                    metavar='JSONFILE',default=None,
                    help='''Settings for pseudo data templates, if desired to be different from template_settings.''')
parser.add_argument('-o','--outfile',type=str,default='llh_data.json',metavar='JSONFILE',
                    help="Output filename.")
parser.add_argument('-v', '--verbose', action='count', default=None,
                    help='set verbosity level')
args = parser.parse_args()

set_verbosity(args.verbose)

#Read in the settings
template_settings = from_json(args.template_settings)
czbins = template_settings['binning']['czbins']

up_template_settings = copy.deepcopy(template_settings)
up_template_settings['params']['reco_vbwkde_evts_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_10_percent_up.hdf5'}
up_template_settings['params']['reco_mc_wt_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_100_percent_up.hdf5'}

down_template_settings = copy.deepcopy(template_settings)
down_template_settings['params']['pid_paramfile'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/pid/1X60_pid_down.json'}
down_template_settings['params']['reco_vbwkde_evts_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_10_percent_down.hdf5'}
down_template_settings['params']['reco_mc_wt_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_100_percent_down.hdf5'}

pseudo_data_settings = from_json(args.pseudo_data_settings) if args.pseudo_data_settings is not None else template_settings

# make sure that both pseudo data and template are using the same
# channel. Raise Exception and quit otherwise
channel = template_settings['params']['channel']['value']
if channel != pseudo_data_settings['params']['channel']['value']:
    error_msg = "Both template and pseudo data must have same channel!\n"
    error_msg += " pseudo_data_settings channel: '%s', template channel: '%s' "%(pseudo_data_settings['params']['channel']['value'],channel)
    raise ValueError(error_msg)

template_maker_down = TemplateMaker(get_values(down_template_settings['params']),
                                             **down_template_settings['binning'])
template_maker_up = TemplateMaker(get_values(up_template_settings['params']),
                               **up_template_settings['binning'])
template_maker = [template_maker_up,template_maker_down]

reco_mc_file_up = '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_100_percent_up.hdf5'
reco_mc_file_down = '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_100_percent_down.hdf5'

def func_cubic_through_nominal(x, a, b, c):
    return a*x*x*x + b*x*x + c*x + 1.0 - a - b - c

tmaps = {}
coeffs = {}
MCmaps = {}

reco_prcs_vals = eval(args.reco_prcs_vals)
print "reco_prcs_vals = ", reco_prcs_vals
#for data_tag, data_nutau_norm in [('data_tau', 1.0), ('data_notau', 0.0)]:
for data_tag, data_nutau_norm in [('data_notau', 0.0)]:
#for data_tag, data_nutau_norm in [('data_tau', 1.0)]:

    tmaps[data_tag] = {}
    MCmaps[data_tag] = {}
    coeffs[data_tag] = {}

    for precision_tag in ['e_reco_precision_up', 'e_reco_precision_down', 'cz_reco_precision_up', 'cz_reco_precision_down']:
        MCmaps[data_tag][precision_tag] = {}
        tmaps[data_tag][precision_tag] = {}

        for reco_prcs_val in reco_prcs_vals:
            tmaps[data_tag][precision_tag][str(reco_prcs_val)] = {'trck':{'up':{}, 'down':{}},
                                                                  'cscd':{'up':{}, 'down':{}}}
            MCmaps[data_tag][precision_tag][str(reco_prcs_val)] = {'trck':{'up':{}, 'down':{}},
                                                                  'cscd':{'up':{}, 'down':{}}}

            template_settings_Reco = copy.deepcopy(up_template_settings)
            template_settings_Reco['params'][precision_tag]['value'] = reco_prcs_val
            template_settings_Reco['params']['nutau_norm']['value'] = data_nutau_norm 

            tmap_up = template_maker_up.get_template(get_values(change_nutau_norm_settings(template_settings_Reco['params'], data_nutau_norm ,True, normal_hierarchy=True)))
            tmap_down = template_maker_down.get_template(get_values(change_nutau_norm_settings(template_settings_Reco['params'], data_nutau_norm ,True, normal_hierarchy=True)))

            template_up_down_combined = get_combined_map(tmap_up,tmap_down, channel= channel)
            template_up = get_up_map(template_up_down_combined, channel= channel)
            reflected_template_down = get_flipped_down_map(template_up_down_combined, channel= channel)

            tmaps[data_tag][precision_tag][str(reco_prcs_val)]['trck']['up'] = template_up['trck']['map']
            tmaps[data_tag][precision_tag][str(reco_prcs_val)]['cscd']['up'] = template_up['cscd']['map']
            tmaps[data_tag][precision_tag][str(reco_prcs_val)]['trck']['down'] = reflected_template_down['trck']['map']
            tmaps[data_tag][precision_tag][str(reco_prcs_val)]['cscd']['down'] = reflected_template_down['cscd']['map']

            MCMap_up = GetMCError(get_values(template_settings_Reco['params']), up_template_settings['binning']['ebins'], up_template_settings['binning']['czbins'], reco_mc_file_up)
            MCMap_down = GetMCError(get_values(template_settings_Reco['params']), down_template_settings['binning']['ebins'], down_template_settings['binning']['czbins'], reco_mc_file_down)
            tmap_MC_up = MCMap_up.get_mc_events_map(get_values(template_settings_Reco['params']), reco_mc_file_up)
            tmap_MC_down = MCMap_down.get_mc_events_map(get_values(template_settings_Reco['params']), reco_mc_file_down)
            #print "tmap_MC_up = ", tmap_MC_up
            #print "tmap_MC_down = ", tmap_MC_down
            tmap_MC_up_down_combined = get_combined_map(tmap_MC_up, tmap_MC_down, channel= channel)
            template_MC_up = get_up_map(tmap_MC_up_down_combined, channel= channel)
            reflected_template_MC_down = get_flipped_down_map(tmap_MC_up_down_combined, channel= channel)

            MCmaps[data_tag][precision_tag][str(reco_prcs_val)]['trck']['up'] = template_MC_up['trck']['map']
            MCmaps[data_tag][precision_tag][str(reco_prcs_val)]['cscd']['up'] = template_MC_up['cscd']['map']
            MCmaps[data_tag][precision_tag][str(reco_prcs_val)]['trck']['down'] = reflected_template_MC_down['trck']['map']
            MCmaps[data_tag][precision_tag][str(reco_prcs_val)]['cscd']['down'] = reflected_template_MC_down['cscd']['map']


    for precision_tag in ['e_reco_precision_up', 'e_reco_precision_down', 'cz_reco_precision_up', 'cz_reco_precision_down']:
        templ_list = []
        coeffs[data_tag][precision_tag] = {'trck':{'up':{}, 'down':{}},
                                           'cscd':{'up':{}, 'down':{}}}
        for flav in ['trck','cscd']:
            for direction in ['up','down']:
                templ = []
                templ_err = []
                templ_nominal = []
                templ_nominal_err = []
                for reco_prcs_val in reco_prcs_vals:
                    templ.append(tmaps[data_tag][precision_tag][str(reco_prcs_val)][flav][direction])  

                    ## Get template error (either standard deviation or standard error):
                    # standard deviation: sqrt(n_event_rate):
                    #templ_err.append(np.sqrt(tmaps['data_tau'][precision_tag][str(reco_prcs_val)]))

                    # standard error: sqrt(n_event_rate)/sqrt(N_mc):
                    templ_err.append(np.sqrt(tmaps[data_tag][precision_tag][str(reco_prcs_val)][flav][direction])/np.sqrt(MCmaps[data_tag][precision_tag][str(reco_prcs_val)][flav][direction]))  

                templ_nominal = np.array(tmaps[data_tag][precision_tag]['1.0'][flav][direction])
                templ_nominal_err = np.array(np.sqrt(tmaps[data_tag][precision_tag]['1.0'][flav][direction])/np.sqrt(MCmaps[data_tag][precision_tag]['1.0'][flav][direction]))
                templ = np.array(templ)
                templ_err = np.array(templ_err)
                templ_nominal = np.array(templ_nominal)
                templ_nominal_err = np.array(templ_nominal_err)

                tml_shape = np.shape(templ)
                coeff = np.empty(np.shape(tmaps[data_tag][precision_tag]['1.0'][flav][direction]), dtype = object) 
                for i in range(0,tml_shape[1]):
                    for j in range(0,tml_shape[2]):
                        bin_counts = templ[:,i,j]

                        # if use bin value ratio (to nominal bin value) as y axis
                        y_val = templ[:,i,j]/templ_nominal[i,j]  #divide by the nominal value

                        # standard error of ratio n1/n2: (n1/n2)*sqrt((SE1/n1)^2 + (SE2/n2)^2) 
                        y_err = y_val * np.sqrt(np.square(templ_nominal_err[i,j]/templ_nominal[i,j])+np.square(templ_err[:,i,j]/templ[:,i,j]))
        
                        ## Cubic Fit  (through (1, 1) point)
                        popt, pcov = curve_fit(func_cubic_through_nominal, reco_prcs_vals, y_val)
                        a = popt[0]
                        b = popt[1]
                        c = popt[2]
                        coeff[i,j] = [a, b, c]
                coeffs[data_tag][precision_tag][flav][direction] = coeff
        

#Assemble output dict
output = {'tmaps' : tmaps,
          'MCmaps' : MCmaps,
          'coeffs' : coeffs,
          'template_settings_up' : up_template_settings,
          'template_settings_down' : down_template_settings}
if args.pseudo_data_settings is not None:
    output['pseudo_data_settings'] = pseudo_data_settings

    #And write to file
to_json(output,args.outfile)


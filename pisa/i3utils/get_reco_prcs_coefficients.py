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

template_settings = copy.deepcopy(template_settings)
pseudo_data_settings = from_json(args.pseudo_data_settings) if args.pseudo_data_settings is not None else template_settings

# make sure that both pseudo data and template are using the same
# channel. Raise Exception and quit otherwise
channel = template_settings['params']['channel']['value']
if channel != pseudo_data_settings['params']['channel']['value']:
    error_msg = "Both template and pseudo data must have same channel!\n"
    error_msg += " pseudo_data_settings channel: '%s', template channel: '%s' "%(pseudo_data_settings['params']['channel']['value'],channel)
    raise ValueError(error_msg)

template_maker = TemplateMaker(get_values(template_settings['params']),
                               **template_settings['binning'])
#reco_mc_file = from_json(find_resource(template_settings['params']['reco_mc_wt_file']['value']))
reco_mc_file = "~/pisa/pisa/resources/aeff/events__deepcore__ic86__runs_1260-1660:200__proc_v6__joined_G_nue_cc+nuebar_cc_G_numu_cc+numubar_cc_G_nutau_cc+nutaubar_cc_G_nuall_nc+nuallbar_nc.hdf5"
template_settings['params']['atmos_mu_scale']['value'] = 0

def func_cubic_through_nominal(x, a, b, c):
    return a*x*x*x + b*x*x + c*x + 1.0 - a - b - c

tmaps = {}
coeffs = {}
MCmaps = {}
reco_prcs_vals = eval(args.reco_prcs_vals)
data_nutau_norm = 1.0
print "reco_prcs_vals = ", reco_prcs_vals

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

        tmap = template_maker.get_template(get_values(change_nutau_norm_settings(template_settings_Reco['params'], data_nutau_norm ,True, normal_hierarchy=True)),no_sys_applied= True)
        tmaps[precision_tag][str(reco_prcs_val)]['trck'] = tmap['trck']['map']
        tmaps[precision_tag][str(reco_prcs_val)]['cscd'] = tmap['cscd']['map']

        MCMap = GetMCError(get_values(template_settings_Reco['params']), template_settings['binning']['anlys_ebins'], template_settings['binning']['czbins'], reco_mc_file)
        tmap_MC = MCMap.get_mc_events_map(get_values(template_settings_Reco['params']), reco_mc_file)
        MCmaps[precision_tag][str(reco_prcs_val)]['trck'] = tmap_MC['trck']['map']
        MCmaps[precision_tag][str(reco_prcs_val)]['cscd'] = tmap_MC['cscd']['map']


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
        coeff = np.empty(np.shape(tmaps[precision_tag]['1.0'][flav]), dtype = object) 
        for i in range(0,tml_shape[1]):
            for j in range(0,tml_shape[2]):
                bin_counts = templ[:,i,j]

                # if use bin value ratio (to nominal bin value) as y axis
                y_val = templ[:,i,j]/templ_nominal[i,j]  #divide by the nominal value

                # standard error of ratio n1/n2: (n1/n2)*sqrt((SE1/n1)^2 + (SE2/n2)^2) 
                #y_err = y_val * np.sqrt(np.square(templ_nominal_err[i,j]/templ_nominal[i,j])+np.square(templ_err[:,i,j]/templ[:,i,j]))
    
                ## Cubic Fit  (through (1, 1) point)
                popt, pcov = curve_fit(func_cubic_through_nominal, reco_prcs_vals, y_val)
                a = popt[0]
                b = popt[1]
                c = popt[2]
                coeff[i,j] = [a, b, c]
        coeffs[precision_tag][flav] = coeff

#Assemble output dict
output = {'tmaps' : tmaps,
          'MCmaps' : MCmaps,
          'coeffs' : coeffs,
          'template_settings' : template_settings}
if args.pseudo_data_settings is not None:
    output['pseudo_data_settings'] = pseudo_data_settings

#And write to file
#to_json(output_template,'RecoPrcs_templates_up_down_10_by_16.json')
to_json(coeffs,args.outfile)


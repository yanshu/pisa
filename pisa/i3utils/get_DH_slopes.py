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

import copy
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.utils.log import logging, profile, physics
from pisa.utils.jsons import from_json,to_json
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
args = parser.parse_args()

def func_plane(param,a,b,c,d):
    x = param[0]
    y = param[1]
    return -(a/c)*x - (b/c)*y + d/c 

#Read in the settings
template_settings = from_json(args.template_settings)
czbins = template_settings['binning']['czbins']
ebins = template_settings['binning']['ebins']
anlys_ebins = template_settings['binning']['anlys_ebins']
channel = template_settings['params']['channel']['value']

template_settings = copy.deepcopy(template_settings)
template_maker = TemplateMaker(get_values(template_settings['params']), **template_settings['binning'])

templates = {}
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
    templates[str(run_num)] = {'trck':{},
                             'cscd':{}
                             }
    print "run_num = ", run_num
    aeff_mc_file = '~/pisa/pisa/resources/aeff/1X%i_aeff_mc.hdf5' % run_num
    pid_param_file_up = '~/pisa/pisa/resources/pid/1X%i_pid.json' % run_num
    reco_mc_file = '~/pisa/pisa/resources/events/1X%i_weighted_aeff_joined_nu_nubar.hdf5' % run_num
    pid_param_file_down = '~/pisa/pisa/resources/pid/1X%i_pid_down.json' % run_num
    DH_template_settings = copy.deepcopy(template_settings)
    DH_template_settings['params']['aeff_weight_file']['value'] = aeff_mc_file
    DH_template_settings['params']['reco_mc_wt_file']['value'] = reco_mc_file
    DH_template_settings['params']['pid_paramfile_up']['value'] = pid_param_file_up 
    DH_template_settings['params']['pid_paramfile_down']['value'] = pid_param_file_down
    DH_template_settings['params']['atmos_mu_scale']['value'] = 0

    DH_template_maker = TemplateMaker(get_values(DH_template_settings['params']), **DH_template_settings['binning'])

    template = DH_template_maker.get_template(get_values(change_nutau_norm_settings(DH_template_settings['params'], 1.0 ,True, normal_hierarchy=True)),no_sys_applied=True)

    templates[str(run_num)]['trck'] = template['trck']['map']
    templates[str(run_num)]['cscd'] = template['cscd']['map']

for flav in ['trck','cscd']:
    templ_list = []
    k_DE = np.empty(np.shape(templates['60']['trck'])) 
    fixed_ratio = np.empty(np.shape(templates['60']['trck'])) 
    k_HI = np.empty(np.shape(templates['60']['trck'])) 
    for run_num in [50,60,61,64,65,70,71,72]:
        # (DOM efficiency, HoleIce Scattering): (0.91,50), (1.0,50), (0.95,50), (1.1,50), (1.05,50),(0.91,no),(0.91,30),(0.91,100)
        templ_list.append(templates[str(run_num)][flav])
    
    templ = np.array(templ_list)
    tml_shape = np.shape(templ)
    print "shape : " , np.shape(templ)
    ############################### DOM efficiency ######################################
    
    for i in range(0,tml_shape[1]):
        for j in range(0,tml_shape[2]):
    
            ########### Get Data ############
            dom_eff = np.array([0.91, 1.0, 0.95, 1.1, 1.05, 0.91, 0.91, 0.91])
            hole_ice = np.array([1.0/50, 1.0/50, 1.0/50, 1.0/50, 1.0/50, 0.0, 1.0/30, 1.0/100])         #unit: cm-1
            bin_counts = np.array([templ[0][i][j],templ[1][i][j],templ[2][i][j],templ[3][i][j],templ[4][i][j],templ[5][i][j],templ[6][i][j],templ[7][i][j]]) 
            bin_ratio_values = bin_counts/templ[1][i][j]  #divide by the nominal value templ[1][i]
            if templ[1][i][j] == 0:
                print "templ[1][", i , "][", j, "] == 0 !!!!!!!!!"

            fixed_r_val = bin_ratio_values[0]

            # line goes through point (0.02, fixed_r_val), fixed_r_val is the value for dom_eff = 0.91 and hole ice = 0.02
            exec('def hole_ice_linear_through_point(x, k): return k*x + %s - k*0.02'%fixed_r_val)
            
            # line goes through point (0.02, fixed_r_val), fixed_r_val is the value for dom_eff = 0.91 and hole ice = 0.02
            exec('def dom_eff_linear_through_point(x, k): return k*x + %s - k*0.91'%fixed_r_val)


            ########### DOM efficiency #############
    
            popt_1, pcov_1 = curve_fit(dom_eff_linear_through_point,dom_eff[0:5],bin_ratio_values[0:5])
            k1 = popt_1[0]
            k_DE[i][j]= k1
            fixed_ratio[i][j]= fixed_r_val

    
            ########### Hole Ice #############

            ice_x = np.array([hole_ice[0],hole_ice[5],hole_ice[6],hole_ice[7]])
            ice_y = np.array([fixed_r_val,bin_ratio_values[5],bin_ratio_values[6],bin_ratio_values[7]])
    
            popt_2, pcov_2 = curve_fit(hole_ice_linear_through_point,ice_x,ice_y)
            k2 = popt_2[0]
            k_HI[i][j]= k2

    fits_DOMEff[flav]['slopes'] = k_DE 
    fits_HoleIce[flav]['slopes'] = k_HI
    fits_DOMEff[flav]['fixed_ratios'] = fixed_ratio 
    fits_HoleIce[flav]['fixed_ratios'] = fixed_ratio

#Assemble output dict
output_template = {'templates' : templates,
                   'template_settings' : template_settings}
#And write to file
to_json(output_template,'DomEff_templates_up_down_10_by_16.json')
to_json(fits_DOMEff,'DomEff_linear_fits_10_by_16.json')
to_json(fits_HoleIce,'HoleIce_linear_fits_10_by_16.json')


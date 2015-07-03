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
from pisa.analysis.llr.LLHAnalysis_nutau import find_max_llh_bfgs
from pisa.analysis.stats.Maps import get_seed, flatten_map
from pisa.analysis.stats.Maps_nutau_noDOMIce import get_pseudo_data_fmap, get_true_template, get_flipped_map
from pisa.analysis.TemplateMaker import TemplateMaker
from pisa.utils.params import get_values, select_hierarchy_and_nutau_norm,change_nutau_norm_settings, select_hierarchy

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

def func_simple_linear(x,k,b):
    return k*x+b

def hole_ice_linear_through_point(x,k):
    # line goes through point (0.2,1)
    return k*x+1-k*0.2  

def dom_eff_linear_through_point(x,k):
    # line goes through point (0.91,1)
    return k*x+1-k*0.91

#Read in the settings
template_settings = from_json(args.template_settings)
czbins = template_settings['binning']['czbins']

up_template_settings = copy.deepcopy(template_settings)
up_template_settings['binning']['czbins']=czbins[czbins<=0]

down_template_settings = copy.deepcopy(template_settings)
down_template_settings['binning']['czbins']=czbins[czbins>=0]
down_template_settings['params']['pid_paramfile'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/pid/1X60_pid_down.json'}

template_maker_down = TemplateMaker(get_values(down_template_settings['params']), **down_template_settings['binning'])
template_maker_up = TemplateMaker(get_values(up_template_settings['params']), **up_template_settings['binning'])
template_maker = [template_maker_up,template_maker_down]

true_templates = []
results = {}
slopes = {}

#for data_tag, data_nutau_norm in [('data_tau',1.0),('data_notau',0.0)]:
for data_tag, data_nutau_norm in [('data_tau',1.0)]:

    results[data_tag] = {}
    slopes[data_tag] = {}
    slopes[data_tag]['k_DomEff'] = {'trck':{'up':{}, 'down':{}},
                                    'cscd':{'up':{}, 'down':{}}
                                    }
    slopes[data_tag]['k_HoleIce'] = {'trck':{'up':{}, 'down':{}},
                                    'cscd':{'up':{}, 'down':{}}
                                    }

    # Get templates from 8 MC sets
    for run_num in [50,60,61,64,65,70,71,72]:
        results[data_tag][str(run_num)] = {'trck':{'up':{}, 'down':{}},
                                          'cscd':{'up':{}, 'down':{}}
                                          }
        print "run_num = ", run_num
        aeff_mc_file = '~/pisa/pisa/resources/aeff/1X%i_aeff_mc.hdf5' % run_num
        reco_mc_file = '~/pisa/pisa/resources/events/1X%i_weighted_aeff_joined_nu_nubar.hdf5' % run_num
        pid_param_file_up = '~/pisa/pisa/resources/pid/1X%i_pid.json' % run_num
        pid_param_file_down = '~/pisa/pisa/resources/pid/1X%i_pid_down.json' % run_num
        DH_up_template_settings = copy.deepcopy(up_template_settings)
        DH_up_template_settings['params']['aeff_weight_file']['value'] = aeff_mc_file
        DH_up_template_settings['params']['reco_mc_wt_file']['value'] = reco_mc_file 
        DH_up_template_settings['params']['pid_paramfile']['value'] = pid_param_file_up 

        DH_down_template_settings = copy.deepcopy(down_template_settings)
        DH_down_template_settings['params']['aeff_weight_file']['value'] = aeff_mc_file
        DH_down_template_settings['params']['reco_mc_wt_file']['value'] = reco_mc_file 
        DH_down_template_settings['params']['pid_paramfile']['value'] = pid_param_file_down

        DH_template_maker_down = TemplateMaker(get_values(DH_down_template_settings['params']), **DH_down_template_settings['binning'])
        DH_template_maker_up = TemplateMaker(get_values(DH_up_template_settings['params']), **DH_up_template_settings['binning'])
        DH_template_maker = [DH_template_maker_up,DH_template_maker_down]
   
        tmap_up = DH_template_maker_up.get_template(get_values(change_nutau_norm_settings(DH_up_template_settings['params'], 1.0 ,True)))
        tmap_down = DH_template_maker_down.get_template(get_values(change_nutau_norm_settings(DH_up_template_settings['params'], 1.0 ,True)))
        #print "template_map_up = ", tmap_up
        #print "template_map_down = ", tmap_down
        results[data_tag][str(run_num)]['trck']['up'] = tmap_up['trck']['map']
        results[data_tag][str(run_num)]['cscd']['up'] = tmap_up['cscd']['map']
        results[data_tag][str(run_num)]['trck']['down'] = tmap_down['trck']['map']
        results[data_tag][str(run_num)]['cscd']['down'] = tmap_down['cscd']['map']

#print "tmap_up['trck']['map'] = ", tmap_up['trck']['map']

for flav in ['trck','cscd']:
    for direction in ['up','down']:
        templ_list = []
        k_DE = np.empty(np.shape(results['data_tau']['60']['trck']['up'])) 
        k_HI = np.empty(np.shape(results['data_tau']['60']['trck']['up'])) 
        for run_num in [50,60,61,64,65,70,71,72]:
            # (DOM efficiency, HoleIce Scattering): (0.91,50), (1.0,50), (0.95,50), (1.1,50), (1.05,50),(0.91,no),(0.91,30),(0.91,100)
            templ_list.append(results['data_tau'][str(run_num)][flav][direction])
        
        templ = np.array(templ_list)
        tml_shape = np.shape(templ)
        print np.shape(templ)
        ############################### DOM efficiency ######################################
        
        for i in range(0,tml_shape[1]):
            for j in range(0,tml_shape[2]):
        
                ########### Get Data ############
                dom_eff = np.array([0.91,1.0,0.95,1.1,1.05,0.91,0.91,0.91])
                hole_ice = 10*np.array([1.0/50,1.0/50,1.0/50,1.0/50,1.0/50,0.0,1.0/30,1.0/100])         #unit: mm-1
                bin_counts = np.array([templ[0][i][j],templ[1][i][j],templ[2][i][j],templ[3][i][j],templ[4][i][j],templ[5][i][j],templ[6][i][j],templ[7][i][j]]) 
                #bin_values = bin_counts
                bin_ratio_values = bin_counts/templ[0][i][j]  #divide by the nominal value templ[0][i]
        
                ########### Linear fits #############
        
                #popt_linear, pcov_linear = curve_fit(func_simple_linear,dom_eff[0:5],bin_ratio_values[0:5])
                #k_domeff = popt_linear[0]
                #b_domeff = popt_linear[1]
                #print " fit a line for bin vs domeff, k_domeff, b_domeff = ", k_domeff, " ", b_domeff
        
                popt_1, pcov_1 = curve_fit(dom_eff_linear_through_point,dom_eff[0:5],bin_ratio_values[0:5])
                k1 = popt_1[0]
                k_DE[i][j]=k1

        slopes[data_tag]['k_DomEff'][flav][direction] = k_DE 
        
        ############################### Hole Ice ######################################

        for i in range(0,tml_shape[1]):
            for j in range(0,tml_shape[2]):

                ########### Get Data ############
                dom_eff = np.array([0.91,1.0,0.95,1.1,1.05,0.91,0.91,0.91])
                hole_ice = 10*np.array([1.0/50,1.0/50,1.0/50,1.0/50,1.0/50,0.0,1.0/30,1.0/100])        # unit: mm-1
                bin_counts = np.array([templ[0][i][j],templ[1][i][j],templ[2][i][j],templ[3][i][j],templ[4][i][j],templ[5][i][j],templ[6][i][j],templ[7][i][j]]) 
                #bin_values = bin_counts
                bin_ratio_values = bin_counts/templ[0][i][j]  #divide by the nominal value templ[0][i][j]
        
                ########### Linear fits #############
                ice_x = np.array([hole_ice[0],hole_ice[5],hole_ice[6],hole_ice[7]])
                ice_y = np.array([bin_ratio_values[0],bin_ratio_values[5],bin_ratio_values[6],bin_ratio_values[7]])
        
                #popt_linear, pcov_linear = curve_fit(func_simple_linear,ice_x,ice_y)
                #k_ice = popt_linear[0]
                #b_ice = popt_linear[1]
                #print " fit a line for bin vs ice, k_ice, b_ice = ", k_ice, " ", b_ice
        
                popt_2, pcov_2 = curve_fit(hole_ice_linear_through_point,ice_x,ice_y)
                k2 = popt_2[0]
                k_HI[i][j]=k2

        slopes[data_tag]['k_HoleIce'][flav][direction] = k_HI

#Assemble output dict
output_template = {'templates' : results,
                   'template_settings_up' : up_template_settings,
                   'template_settings_down' : down_template_settings}
output_slope = { 'slopes': slopes,
                   'template_settings_up' : up_template_settings,
                   'template_settings_down' : down_template_settings}
#And write to file
to_json(output_template,'DH_templates_up_down.json')
to_json(output_slope,'DH_slopes_up_downs.json')


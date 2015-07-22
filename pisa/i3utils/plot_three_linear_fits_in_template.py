#! /usr/bin/env python
#
# plot_three_linear_fits_in_template.py 
#
# Use the json file containing all bin values generated from eight different MC sets, get the plots 
# of bin value ratio vs DOM eff. and hole ice, with three different kinds of linear fits: plane fit, simple
# linear fit, and linear fit through the point at domeff = 0.91 and hole ice scattering length= 50 cm.
#
# author: Feifei Huang - fxh140@psu.edu
#
# date:   22-July-2015
#

import os
import sys
from os.path import expandvars

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(description=''' Write the slopes from linear fit of bin value ratio vs DOM efficiency
        or Hole ice scattering to a json file.''', formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('-i','--data_file',type=str,
                         metavar='JSONFILE', required = True,
                         help='''Input json file containing the bin values information from eight MC sets.''')
parser.add_argument('-o', '--png_name', type=str,
                         metavar='NAME', required = True,
                         help='Output png file.')
parser.add_argument('-n', '--npoints', type = int,
                         metavar='INT', required = True,
                         help='No. of points for x axis')
parser.add_argument('-n_ebins', '--n_ebins', type = int,
                         metavar='INT', required = True,
                         help='No. of ebins')
parser.add_argument('-n_czbins', '--n_czbins', type = int,
                         metavar='INT', required = True,
                         help='No. of czbins in all sky map (n_czbins/2 is the no. of czbins for the up-going map).')
args = parser.parse_args()
data_file = args.data_file
png_name = args.png_name
n_ebins = args.n_ebins
n_czbins = args.n_czbins

assert(n_czbins%2 == 0)
half_n_czbins = n_czbins/2

from pisa.utils.jsons import from_json,to_json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.optimize import curve_fit
npoints = int(args.npoints)

def func_plane(param,a,b,c,d):
    x = param[0]
    y = param[1]
    return -(a/c)*x - (b/c)*y + d/c 

def func_simple_linear(x,k,b):
    return k*x+b

def hole_ice_linear_through_point(x,k):
    # line goes through point (0.02 cm-1,1), i.e. hole ice scattering len = 50 cm
    return k*x+1-k*0.02  

def dom_eff_linear_through_point(x,k):
    # line goes through point (0.91,1)
    return k*x+1-k*0.91

file = from_json(data_file)

results = file['results']
for i in range(0,1):
    templ = []
    templ_escale = []
    templ_err = []
    chi_squared_domeff = []
    k_DE = []
    k_HI = []
    chi_squared_ice = []
    for run_num in [50,60,61,64,65,70,71,72]:
        # (DOM efficiency, HoleIce Scattering): (0.91,50), (1.0,50), (0.95,50), (1.1,50), (1.05,50),(0.91,no),(0.91,30),(0.91,100)
        templ.append(results['data_tau'][str(run_num)])  
        templ_err.append(results['data_tau'][str(run_num)]/results['MCErr'][str(run_num)])  

        ############################### DOM efficiency ######################################

    for j in range(0,2*n_ebins*n_czbins):
        bin_num_x = j % half_n_czbins
        bin_num_y = (j / half_n_czbins) % n_ebins
        #print 'j = ', j , ' bin_num = ( ', bin_num_x , ', ', bin_num_y , ')'

        ########### Get Data ############
        dom_eff = np.array([0.91,1.0,0.95,1.1,1.05,0.91,0.91,0.91])
        hole_ice = np.array([1.0/50,1.0/50,1.0/50,1.0/50,1.0/50,0.0,1.0/30,1.0/100])
        bin_counts = np.array([templ[0][j],templ[1][j],templ[2][j],templ[3][j],templ[4][j],templ[5][j],templ[6][j],templ[7][j]]) 
        #z_val = bin_counts
        z_val = bin_counts/templ[0][j]  #divide by the nominal value templ[0][j]
        bin_MC_err = np.array([templ_err[0][j],templ_err[1][j],templ_err[2][j],templ_err[3][j],templ_err[4][j],templ_err[5][j],templ_err[6][j],templ_err[7][j]]) 
        y_err = z_val*np.sqrt(np.square(templ_err[0][j]/templ[0][j])+np.square(bin_MC_err/bin_counts))

        ########### Plane Fit  ##################
        
        popt, pcov = curve_fit(func_plane,np.array([dom_eff,hole_ice]),z_val)
        a = popt[0]
        b = popt[1]
        c = popt[2]
        d = popt[3]
        #print ' fit a plane, a, b, c, d = ', a, ' ', b, ' ' , c , ' ' , d

        ########### Plot #############

        if(j==0 or j== half_n_czbins * n_ebins or j== 2* half_n_czbins * n_ebins or j== 3* half_n_czbins * n_ebins):
            plt.figure(figsize=( 4*n_ebins, 4*half_n_czbins))
        if(j< n_czbins*n_ebins):
            subplot_idx = half_n_czbins*(n_ebins-1-bin_num_y)+bin_num_x+1
        else:
            subplot_idx = half_n_czbins*(n_ebins-1-bin_num_y)+half_n_czbins-1-bin_num_x+1
        #print 'subplot_idx = ', subplot_idx
        plt.subplot(n_ebins,half_n_czbins,subplot_idx)
        plt.scatter(dom_eff[0:5],z_val[0:5],color='blue')
        plt.errorbar(dom_eff[0:5],z_val[0:5],yerr=y_err[0:5],fmt='none')
        plt.xlim(0.8,1.2)
        plt.ylim(0.6,2.2)

        popt_linear, pcov_linear = curve_fit(func_simple_linear,dom_eff[0:5],z_val[0:5])
        k_domeff = popt_linear[0]
        b_domeff = popt_linear[1]
        line_linear_domeff, = plt.plot(np.linspace(0.8,1.2,npoints),k_domeff*np.linspace(0.8,1.2,npoints)+b_domeff,'k-')
        #print ' fit a line for bin vs domeff, k_domeff, b_domeff = ', k_domeff, ' ', b_domeff
        #chi2_domeff = np.sum(np.square((k_domeff*dom_eff[0:5]+b_domeff-z_val[0:5])/y_err[0:5]))
        #chi_squared_domeff.append(chi2_domeff)

        popt_1, pcov_1 = curve_fit(dom_eff_linear_through_point,dom_eff[0:5],z_val[0:5])
        k1 = popt_1[0]
        k_DE.append(k1)
        line_linear_through_domeff, = plt.plot(np.linspace(0.8,1.2,npoints),k1*np.linspace(0.8,1.2,npoints)+1-k1*0.91,'g-')
        chi2_domeff = np.sum(np.square((k1*dom_eff[0:5]+1-k1*0.91-z_val[0:5])/y_err[0:5]))

        domeff_best_fit = func_plane(np.array([np.linspace(0.8,1.2,npoints),0.02*np.ones(npoints)]),popt[0],popt[1],popt[2],popt[3])
        line_plane_proj_domeff, = plt.plot(np.linspace(0.8,1.2,npoints),domeff_best_fit,'r-')
        #plt.legend([line_linear_domeff,line_linear_through_domeff,line_plane_proj_domeff],['linear','linear(pass point)','plane projection'],loc='upper left')
        #plt.xlabel('DOM efficiency')
        #plt.ylabel('bin val/nominal val')
        #plt.title('bin: %i, plot_#: %i, chi2 = %.02f' % (j,subplot_idx,chi2_domeff))
        plt.title('bin: %i, chi2 = %.02f' % (j, chi2_domeff))
        if(j==half_n_czbins * n_ebins-1):
            plt.savefig(png_name+'_fits_cmpr_DOMEff_upgoing_cscd.png')
            plt.clf()
        if (j==2*half_n_czbins * n_ebins-1):
            plt.savefig(png_name+'_fits_cmpr_DOMEff_upgoing_trck.png')
            plt.clf()
        if(j==3*half_n_czbins * n_ebins-1):
            plt.savefig(png_name+'_fits_cmpr_DOMEff_downgoing_cscd.png')
            plt.clf()
        if(j==4*half_n_czbins * n_ebins-1):
            plt.savefig(png_name+'_fits_cmpr_DOMEff_downgoing_trck.png')
            plt.clf()




        ############################### Hole Ice ######################################


    for j in range(0,2*n_ebins*n_czbins):
        bin_num_x = j % half_n_czbins
        bin_num_y = (j / half_n_czbins) % n_ebins
        #print 'j = ', j , ' bin_num = ( ', bin_num_x , ', ', bin_num_y , ')'

        ########### Get Data ############
        dom_eff = np.array([0.91,1.0,0.95,1.1,1.05,0.91,0.91,0.91])
        hole_ice = np.array([1.0/50,1.0/50,1.0/50,1.0/50,1.0/50,0.0,1.0/30,1.0/100])
        bin_counts = np.array([templ[0][j],templ[1][j],templ[2][j],templ[3][j],templ[4][j],templ[5][j],templ[6][j],templ[7][j]]) 
        #z_val = bin_counts
        z_val = bin_counts/templ[0][j]  #divide by the nominal value templ[0][j]
        bin_MC_err = np.array([templ_err[0][j],templ_err[1][j],templ_err[2][j],templ_err[3][j],templ_err[4][j],templ_err[5][j],templ_err[6][j],templ_err[7][j]]) 
        y_err = z_val*np.sqrt(np.square(templ_err[0][j]/templ[0][j])+np.square(bin_MC_err/bin_counts))

        ########### Plane Fit  ##########
        
        popt, pcov = curve_fit(func_plane,np.array([dom_eff,hole_ice]),z_val)
        a = popt[0]
        b = popt[1]
        c = popt[2]
        d = popt[3]
        #print ' fit a plane, a, b, c, d = ', a, ' ', b, ' ' , c , ' ' , d

        ########### Plot #############
        
        if(j==0 or j== half_n_czbins * n_ebins or j== 2* half_n_czbins * n_ebins or j== 3* half_n_czbins * n_ebins):
            plt.figure(figsize=( 4*n_ebins, 4*half_n_czbins))
        if(j< n_czbins*n_ebins):
            subplot_idx = half_n_czbins*(n_ebins-1-bin_num_y)+bin_num_x+1
        else:
            subplot_idx = half_n_czbins*(n_ebins-1-bin_num_y)+half_n_czbins-1-bin_num_x+1
        #print 'subplot_idx = ', subplot_idx
        plt.subplot(n_ebins,half_n_czbins,subplot_idx)
        ice_x = np.array([hole_ice[0],hole_ice[5],hole_ice[6],hole_ice[7]])
        ice_y = np.array([z_val[0],z_val[5],z_val[6],z_val[7]])
        ice_y_err = np.array([y_err[0],y_err[5],y_err[6],y_err[7]])
        plt.xlim(-0.003,0.065)
        plt.ylim(0.2,1.6)
        plt.scatter(ice_x, ice_y,color='blue')
        plt.errorbar(ice_x,ice_y,yerr=ice_y_err,fmt='none')

        popt_linear, pcov_linear = curve_fit(func_simple_linear,ice_x,ice_y)
        k_ice = popt_linear[0]
        b_ice = popt_linear[1]
        line_linear_holeice, = plt.plot(np.linspace(-0.002,0.06,npoints),k_ice*np.linspace(-0.002,0.06,npoints)+b_ice,'k-')
        #chi2_ice = np.sum(np.square((k_ice*ice_x+b_ice-ice_y)/ice_y_err[0:4]))
        #print ' fit a line for bin vs ice, k_ice, b_ice = ', k_ice, ' ', b_ice
        popt_2, pcov_2 = curve_fit(hole_ice_linear_through_point,ice_x,ice_y)
        k2 = popt_2[0]
        k_HI.append(k2)
        line_linear_through_holeice, = plt.plot(np.linspace(-0.002,0.06,npoints),k2*np.linspace(-0.002,0.06,npoints)+1-k2*0.02,'g-')
        chi2_ice = np.sum(np.square((k2*ice_x+1-k2*0.02-ice_y)/ice_y_err[0:4]))

        ice_best_fit = func_plane(np.array([0.91*np.ones(npoints),np.linspace(-0.002,0.06,npoints)]),popt[0],popt[1],popt[2],popt[3])
        val_dom_fix_holeice_change = func_plane(np.array([0.8*np.ones(npoints),np.linspace(-0.002,0.06,npoints)]),popt[0],popt[1],popt[2],popt[3])
        if (np.any(val_dom_fix_holeice_change<= 0.0)):
            print ' j = ', j
            print 'dom_eff = 0.8, hole_ice = np.linspace(-0.002,0.06,npoints), bin/nom. = ', val_dom_fix_holeice_change
        line_plane_proj_holeice, = plt.plot(np.linspace(-0.002,0.06,npoints),ice_best_fit,'r-')
        if k_ice>=0:
            leg_loc_ice = 'upper left'
        else:
            leg_loc_ice = 'upper right'
        #plt.legend([line_linear_holeice,line_linear_through_holeice,line_plane_proj_holeice],['linear','linear(pass point)','plane projection'],loc=leg_loc_ice)
        #plt.xlabel('Hole Ice')
        #plt.ylabel('bin val/nominal val')
        plt.title('bin: %i, chi2 = %.02f' % (j,chi2_ice))
        if(j==half_n_czbins * n_ebins-1):
            plt.savefig(png_name+'_fits_cmpr_HoleIce_upgoing_cscd.png')
            plt.clf()
        if (j==2*half_n_czbins * n_ebins-1):
            plt.savefig(png_name+'_fits_cmpr_HoleIce_upgoing_trck.png')
            plt.clf()
        if(j==3*half_n_czbins * n_ebins-1):
            plt.savefig(png_name+'_fits_cmpr_HoleIce_downgoing_cscd.png')
            plt.clf()
        if(j==4*half_n_czbins * n_ebins-1):
            plt.savefig(png_name+'_fits_cmpr_HoleIce_downgoing_trck.png')
            plt.clf()

    # Writes slopes to file
    #output = {'k_DomEff' : k_DE,
    #          'k_HoleIce' : k_HI}
    #print 'k_DE = ', k_DE
    #print 'k_HI = ', k_HI
    #to_json(output,'DomEff_HoleIce_slopes.json')
    #slopes = from_json('DomEff_HoleIce_slopes.json')
    #array_k_DE = np.array(slopes['k_DomEff'])
    #print array_k_DE[0:10]
    #print type(slopes['k_HoleIce'])


#! /usr/bin/env python
#
# Plots all effective areas for a geometry.
#
#

import os
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
parser = ArgumentParser(description='Plots all nu and nubar effective areas for a given geometry.',formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('geometry',type=str,help="geometry of effective area to plot.")
parser.add_argument('--cuts_path',type=str,default='cuts_V5',help='cuts directory')
parser.add_argument('--all_cz',action='store_true',default=False,help='if use all sky MC events')
parser.add_argument('--path_to_aeff',default=None,type=str,help="Path to aeff files if not located in standard place.")
parser.add_argument('--emax',default=40.0,type=float,help="max energy to plot on x axis [GeV]")
parser.add_argument('--ebins',metavar='FUNC',type=str,
                    default='np.logspace(np.log10(1.0),np.log10(80.0),40)',
                    help='''Python text to convert to lambda function for energy
                    bin edges. i.e. np.linspace(1,40,40)''')
parser.add_argument('--ymax',default=2.0e-4,type=float,help="Max on y axis.")
parser.add_argument('--logx',action='store_true',help="log scale for energy axis.")
parser.add_argument('--cuts_name',type=str,default='MattL6',help='cuts directory')
args = parser.parse_args()


path_to_aeff = os.getenv("FISHER")+'/resources/a_eff/'+args.geometry+'/'+args.cuts_path if args.path_to_aeff is None else args.path_to_aeff


def plotEffAreaGeom(path_to_aeff,i,flavor,error_bar,lineOpt,xbins):
    dataFile = os.path.join(path_to_aeff,"a_eff_vs_e_"+flavor+".dat")

    fh = open(dataFile,'r')
    egyList = []; aeffList = []
    for line in fh.readlines():
        line = line.rstrip()
        line_split = line.split()
        egyList.append(line_split[0])
        aeffList.append(line_split[1])
    fh.close()    
    #plt.plot(egyList,aeffList,lineOpt,label=flavor,lw=2)

    if error_bar:
        dataFile = os.path.join(path_to_aeff,"a_eff_vs_e_"+flavor+"_data.dat")
        fh = open(dataFile,'r')
        egyList = []; aeffList = []; aeff_errList = []
        for line in fh.readlines():
            line = line.rstrip()
            line_split = line.split()
            egyList.append(float(line_split[0]))
            aeffList.append(float(line_split[1]))
            aeff_errList.append(float(line_split[2]))
        fh.close()
        #plt.errorbar(egyList,aeffList,color=colorList[i],yerr=aeff_errList,fmt='.',lw=2)
        is_bar = 'bar' in flavor
        linestyle = 'dashed' if is_bar else 'solid'
        plt.hist(egyList,weights= aeffList,bins=xbins,histtype='step',lw=2,color=colorList[i],linestyle=linestyle, label=flavor)
        plt.xscale('log')
    return


error_bar = True
colorList = ['r','g','b','k','c','m']
fig_nu = plt.figure(figsize=(6,5),dpi=150)
ebins = eval(args.ebins)

flavors = ['nue_cc','numu_cc','nutau_cc','nuall_nc']
for i,flavor in enumerate(flavors):
    lineOpt = colorList[i]+'-'
    plotEffAreaGeom(path_to_aeff,i,flavor,error_bar,lineOpt, ebins)
print "Saving fig nu..."

flavors = ['nuebar_cc','numubar_cc','nutaubar_cc','nuallbar_nc']
for i,flavor in enumerate(flavors):
    lineOpt = colorList[i]+'--'
    plotEffAreaGeom(path_to_aeff,i,flavor,error_bar,lineOpt, ebins)

if args.all_cz:
    plt.title(args.geometry+" "+ args.cuts_name + " effective areas (all sky)")
else:
    plt.title(args.geometry+" "+ args.cuts_name + " effective areas (only up)")
plt.xlabel(r'$\nu$ Energy [GeV]')
plt.ylabel(r'Eff. Area [$m^2$]')
plt.xlim(5.0,args.emax)
plt.ylim(2.0e-8,args.ymax)
plt.grid()
plt.yscale('log')
if args.logx: plt.xscale('log')
plt.legend(loc='lower right',fontsize=8,ncol=2)
print "Saving fig nubar..."
if args.all_cz:
    fig_nu.savefig(args.geometry+'_aeff_nu_vs_E_all_sky'+args.cuts_name+'.png',dpi=170)
else:
    fig_nu.savefig(args.geometry+'_aeff_nu_vs_E_only_up'+args.cuts_name+'.png',dpi=170)
fig_nu.show()


raw_input("PAUSED...Press <ENTER> to close plots and continue.")

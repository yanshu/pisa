#! /usr/bin/env python
#
# Plots all effective areas for a geometry.
#
#

import os
from matplotlib import pyplot as plt
import numpy as np

from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
parser = ArgumentParser(description='Plots all nu and nubar effective areas for a given geometry.',formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('geometry',type=str,help="geometry of effective area to plot.")
parser.add_argument('--cuts_path',type=str,default='cuts_V5',help='cuts directory')
parser.add_argument('--all_cz',action='store_true',default=False,help='if use all sky MC events')
parser.add_argument('--path_to_aeff',default=None,type=str,help="Path to aeff files if not located in standard place.")
parser.add_argument('--czmax',default=1.0,type=float,help="max coszen to plot on x axis [GeV]")
parser.add_argument('--czmin',default=-1.0,type=float,help="min coszen to plot on x axis [GeV]")
parser.add_argument('--ymax',default=1.0e-1,type=float,help="Max on y axis.")
parser.add_argument('--logx',action='store_true',help="log scale for x axis.")
args = parser.parse_args()


path_to_aeff = os.getenv("FISHER")+'/resources/a_eff/'+args.geometry+'/'+args.cuts_path if args.path_to_aeff is None else args.path_to_aeff


def plotEffAreaGeom(path_to_aeff,i,flavor,error_bar):
    dataFile = os.path.join(path_to_aeff,"a_eff_vs_cz_"+flavor+".dat")
    lineOpt = colorList[i]+'-'

    fh = open(dataFile,'r')
    czList = []; aeffList = []
    for line in fh.readlines():
        line = line.rstrip()
        line_split = line.split()
        czList.append(line_split[0])
        aeffList.append(line_split[1])
    fh.close()    
    plt.plot(czList,aeffList,lineOpt,label=flavor,lw=2)
    print "min aeffList = ", min(aeffList)

    if error_bar:
        dataFile = os.path.join(path_to_aeff,"a_eff_vs_cz_"+flavor+"_data.dat")
        fh = open(dataFile,'r')
        czList = []; aeffList = []; aeff_errList = []
        for line in fh.readlines():
            line = line.rstrip()
            line_split = line.split()
            czList.append(float(line_split[0]))
            aeffList.append(float(line_split[1]))
            aeff_errList.append(float(line_split[2]))
        fh.close()
        plt.errorbar(czList,aeffList,color=colorList[i],yerr=aeff_errList,fmt='.',lw=2)
    return


error_bar = True
colorList = ['k','b','g','c','r','m']
fig_nu = plt.figure(figsize=(6,5),dpi=150)

flavors = ['nue','numu','nutau','nuall_nc']
for i,flavor in enumerate(flavors):
    plotEffAreaGeom(path_to_aeff,i,flavor,error_bar)
if args.all_cz:
    plt.title(args.geometry+" nu all effective areas (all sky)")
else:
    plt.title(args.geometry+" nu all effective areas (only up)")
plt.xlabel(r'cos(zen)')
plt.ylabel(r'Eff. Area [$m^2$]')
plt.xlim(args.czmin,args.czmax)
plt.ylim(1.0e-4,args.ymax)
plt.grid()
plt.yscale('log')
if args.logx: plt.xscale('log')
plt.legend(loc='lower left',fontsize=10)
print "Saving fig nu..."
if args.all_cz:
    fig_nu.savefig(args.geometry+'_aeff_nu_vs_CZ_all_sky.png',dpi=170)
else:
    fig_nu.savefig(args.geometry+'_aeff_nu_vs_CZ_only_up.png',dpi=170)
fig_nu.show()

fig_nubar = plt.figure(figsize=(6,5),dpi=150)
flavors = ['nuebar','numubar','nutaubar','nuallbar_nc']
for i,flavor in enumerate(flavors):
    plotEffAreaGeom(path_to_aeff,i,flavor,error_bar)

if args.all_cz:
    plt.title(args.geometry+" nubar all effective areas (all sky)")
else:
    plt.title(args.geometry+" nubar all effective areas (only up)")
plt.xlabel(r'cos(zen)')
plt.ylabel(r'Eff. Area [$m^2$]')
plt.xlim(args.czmin,args.czmax)
plt.ylim(1.0e-4,args.ymax)
plt.grid()
plt.yscale('log')
if args.logx: plt.xscale('log')
plt.legend(loc='lower left',fontsize=10)
print "Saving fig nubar..."
if args.all_cz:
    fig_nubar.savefig(args.geometry+'_aeff_nubar_vs_CZ_all_sky.png',dpi=170)
else:
    fig_nubar.savefig(args.geometry+'_aeff_nubar_vs_CZ_only_up.png',dpi=170)
fig_nubar.show()


raw_input("PAUSED...Press <ENTER> to close plots and continue.")

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
parser.add_argument('--emax',default=40.0,type=float,help="max energy to plot on x axis [GeV]")
parser.add_argument('--ymax',default=1.0e-4,type=float,help="Max on y axis.")
parser.add_argument('--logx',action='store_true',help="log scale for energy axis.")
parser.add_argument('--cuts_name',type=str,default='MattL6',help='cuts directory')
args = parser.parse_args()


path_to_aeff = os.getenv("FISHER")+'/resources/a_eff/'+args.geometry+'/'+args.cuts_path if args.path_to_aeff is None else args.path_to_aeff


def plotEffAreaGeom(path_to_aeff,i,flavor,error_bar,lineOpt):
    dataFile = os.path.join(path_to_aeff,"a_eff_"+flavor+".dat")

    fh = open(dataFile,'r')
    egyList = []; aeffList = []
    for line in fh.readlines():
        line = line.rstrip()
        line_split = line.split()
        egyList.append(line_split[0])
        aeffList.append(line_split[1])
    fh.close()    
    plt.plot(egyList,aeffList,lineOpt,label=flavor,lw=2)

    if error_bar:
        dataFile = os.path.join(path_to_aeff,"a_eff_"+flavor+"_data.dat")
        fh = open(dataFile,'r')
        egyList = []; aeffList = []; aeff_errList = []
        for line in fh.readlines():
            line = line.rstrip()
            line_split = line.split()
            egyList.append(float(line_split[0]))
            aeffList.append(float(line_split[1]))
            aeff_errList.append(float(line_split[2]))
        fh.close()
        plt.errorbar(egyList,aeffList,color=colorList[i],yerr=aeff_errList,fmt='.',lw=2)
    return


error_bar = True
colorList = ['k','b','g','c','r','m']
fig_nu = plt.figure(figsize=(6,5),dpi=150)

flavors = ['nue','numu','nutau','nuall_nc']
for i,flavor in enumerate(flavors):
    lineOpt = colorList[i]+'-'
    plotEffAreaGeom(path_to_aeff,i,flavor,error_bar,lineOpt)
print "Saving fig nu..."

flavors = ['nuebar','numubar','nutaubar','nuallbar_nc']
for i,flavor in enumerate(flavors):
    lineOpt = colorList[i]+'--'
    plotEffAreaGeom(path_to_aeff,i,flavor,error_bar,lineOpt)

if args.all_cz:
    plt.title(args.geometry+" "+ args.cuts_name + " effective areas (all sky)")
else:
    plt.title(args.geometry+" "+ args.cuts_name + " effective areas (only up)")
plt.xlabel(r'$\nu$ Energy [GeV]')
plt.ylabel(r'Eff. Area [$m^2$]')
plt.xlim(1.0,args.emax)
plt.ylim(3.0e-7,args.ymax)
plt.grid()
plt.yscale('log')
if args.logx: plt.xscale('log')
plt.legend(loc='lower right',fontsize=10,ncol=2)
print "Saving fig nubar..."
if args.all_cz:
    fig_nu.savefig(args.geometry+'_aeff_nu_vs_E_all_sky'+args.cuts_name+'.png',dpi=170)
else:
    fig_nu.savefig(args.geometry+'_aeff_nu_vs_E_only_up'+args.cuts_name+'.png',dpi=170)
fig_nu.show()


raw_input("PAUSED...Press <ENTER> to close plots and continue.")

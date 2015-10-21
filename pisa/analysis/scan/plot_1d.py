#! /usr/bin/env python

import os
import sys
from os.path import expandvars

from optparse import OptionParser

parser = OptionParser()

parser.add_option("-i", dest="data_file", help="Input data file.")
parser.add_option("-p", dest="param_name", help="Parameter's name.")
parser.add_option("--pd", dest="data_tag", help="Pseudo data tag, data_tau or data_notau.")
(options, args) = parser.parse_args()
data_file = options.data_file
png_name = data_file.split('.')[0]+'.png'
param = options.param_name
data_tag = options.data_tag

from pisa.utils.jsons import from_json
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

file = from_json(data_file)

trials = file["trials"]
print "no. of trials = ", len(trials)
for i in range(0,len(trials)):
    trial = trials[i]
    fit = trial[data_tag]['hypo_free']
    llh = fit['llh']
    param_1 = fit[param]
    x = param_1
    y = llh

    #fig = plt.figure()
    #axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    #axes.plot(f1, llh, 'r')
    #axes.set_xlabel("dm31")
    #axes.set_ylabel("llh")
    #plt.show()

    #cut = (x==0.8)
    #x = param_1[cuts1 & cuts2]
    #x = param_1[cut]
    #y = llh[cut]

    plt.scatter(x,y)
    #plt.xlabel("nu_nubar_ratio")
    plt.xlabel("%s"%param)
    plt.ylabel("-llh")
    plt.grid()
    if len(trials) == 1:
        plt.savefig(png_name.split('.')[0]+'.png')
    else:
        plt.savefig(png_name.split('.')[0]+'_%i'%i+'.png')
    plt.show()

    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #z = llh
    #cuts = [z<0]
    #print z[cuts]
    #x = f1[cuts]
    #y = param_1[cuts]
    #x, y  = np.meshgrid(x,y)
    ##surf = ax.plot_surface(x[cuts],y[cuts],z[cuts])
    #surf = ax.plot_surface(x,y,z[cuts])
    #plt.show()


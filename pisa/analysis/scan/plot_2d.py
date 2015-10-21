#! /usr/bin/env python

import os
import sys
from os.path import expandvars

from optparse import OptionParser
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = OptionParser()

parser.add_option("-i", dest="data_file", help="Input data file.")
parser.add_option("--p1", dest="param_name_1", help="The first parameter's name.")
parser.add_option("--p2", dest="param_name_2", help="The second parameter's name.")
parser.add_option("--pd", dest="data_tag", help="Pseudo data tag, data_tau or data_notau.")
(options, args) = parser.parse_args()
data_file = options.data_file
png_name = data_file.split('.')[0]+'.png'
p_1 = options.param_name_1
p_2 = options.param_name_2
data_tag = options.data_tag

from pisa.utils.jsons import from_json
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt

file = from_json(data_file)

trials = file["trials"]
print "no. of trials = ", len(trials)
for i in range(0,len(trials)):
    trial = trials[i]
    fit = trial[data_tag]['hypo_free']
    llh = fit['llh']
    param_1 = fit[p_1]
    param_2 = fit[p_2]

    #cuts1 = (param_2==1.0)
    #x = param_1[cuts1]
    #y = param_2[cuts1]
    #z = llh[cuts1]
    x = param_1 
    y = param_2
    z = llh
    print "set(x) = ", set(x)
    print "set(y) = ", set(y)
    print "len(x) = ", len(x), " ,len(y) = " , len(y) , " ,len(z)=" , len(z)
    min_z = 10000000
    x_when_zmin = 0
    y_when_zmin = 0
    for i in range(0,len(z)):
        if(z[i]<min_z):
            min_z = z[i]
            x_when_zmin = x[i] 
            y_when_zmin = y[i]
    #print x_when_zmin, " " , y_when_zmin, " ", min_z

    plt.hist2d(x,y,weights = z)
    plt.colorbar()
    plt.xlabel("%s"%p_1)
    plt.ylabel("%s"%p_2)
    if len(trials) == 1:
        plt.savefig(png_name.split('.')[0]+'.png')
    else:
        plt.savefig(png_name.split('.')[0]+'_%i'%i+'.png')
    plt.show()
    plt.clf()

    #fig = plt.figure()
    #axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    #axes.plot(param_2, llh, 'r')
    #axes.set_xlabel("dm31")
    #axes.set_ylabel("llh")
    #plt.show()

    #cuts1 = (param_2==1.0)
    #cuts3 = (param_1==0.0025)
    #x = param_1[cuts1 & cuts3]
    #y = llh[cuts1 & cuts3]
    #plt.scatter(x,y)
    #plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Z = llh
    #cuts = [z<0]
    #print z[cuts]
    #x = param_2[cuts]
    #y = param_1[cuts]
    X, Y  = np.meshgrid(x,y)
    #surf = ax.plot_surface(X,Y,Z)
    #surf = ax.plot_surface(x[cuts],y[cuts],z[cuts])

    # wire plot
    #ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

    # scatter plot
    #ax.scatter(x, y, z)

    # plot_trisurf
    ax.plot_trisurf(x, y, z)

    plt.savefig(png_name.split('.')[0]+'_3d_%i.png'%i)
    plt.show()
    plt.clf()

    plt.scatter(x,z)
    plt.xlabel("%s"%p_1)
    plt.ylabel("-llh")
    plt.grid()
    if len(trials) == 1:
        plt.savefig(png_name.split('.')[0]+'_%s.png'%p_1)
    else:
        plt.savefig(png_name.split('.')[0]+'_%i'%i+'_%s.png'%p_1)
    plt.show()
    plt.clf()
    
    plt.scatter(y,z)
    plt.xlabel("%s"%p_2)
    plt.ylabel("-llh")
    plt.grid()
    if len(trials) == 1:
        plt.savefig(png_name.split('.')[0]+'_%s.png'%p_2)
    else:
        plt.savefig(png_name.split('.')[0]+'_%i'%i+'_%s.png'%p_2)
    plt.show()
    plt.clf()
    

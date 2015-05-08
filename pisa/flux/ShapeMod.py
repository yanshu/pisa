#
# UncService.py
#
# This uncertainty service provides flux uncertainty values for a energy
# bins. It loads an uncertainty table as given in Barr http://arxiv.org/pdf/astro-ph/0611266.pdf
#
#
# author: Joakim Sandroos
#         sandroos@nbi.dk
#
# date:   2015-03-07

import os
import numpy as np
import math
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.interpolate import *
from pisa.utils.log import logging
from pisa.utils.utils import get_bin_centers, get_bin_sizes
from pisa.utils.jsons import from_json, to_json, json_string
from pisa.utils.utils import *
from pisa.resources.resources import open_resource

#Global definition of primaries for which there is a neutrino flux
#primaries = ['numu'] #, 'numu_bar', 'nue', 'nue_bar']
primaries = ['numu', 'numu_bar', 'nue', 'nue_bar']

def data_spliner(filename, ebins): #take filename, return spline
    en, dat = np.loadtxt(open_resource(filename)).T
    en2 = np.concatenate([en, ebins])
    dum = [0] * len(ebins)
    dat2 = np.concatenate([dat, dum])
    up = len(en) -1
    endat = sorted(zip(en2,dat2))
    for entry in endat:
        if (entry[0] > en[0] and entry[0] < en[up] and entry[1]==0.0):
            endat.remove(entry)
    en2,dat2 = zip(*endat)
    Ret_Spline = InterpolatedUnivariateSpline(en2, dat2, k=1)
    return Ret_Spline
        
def modify_shape(ebins, czbins, *args, **params): #takes in *args as modification factors and filename strings. Must be sorted in matching way for both or wrong shape will be weighted wrong.  
    '''Get the uncertainty for the given
    bin edges in energy and the primary.'''
    FactorTable = []
    spline_dict = {}
    evals = get_bin_centers(ebins)
    zbins = [1] * len(czbins)
    czvals = get_bin_centers(zbins)
    for arg in args:
        if (type(arg)==float or type(arg)==np.float64):
            FactorTable.append(arg)
        else:
            namae = "".join(arg[9]) ##used to be 9 as it matches the position of the defining letter in the filename
            logging.trace("namae is:  %s"%namae)
            spline_dict[namae] = data_spliner(arg, ebins)

    ########## ADD ALL SPLINES TOGETHER #######################
    for i, v in enumerate(spline_dict):
        if (i==0):
            datatable = FactorTable[0]*0.01*spline_dict[v].__call__(evals)
        else:
            datatable += FactorTable[i-1]*0.01*spline_dict[v].__call__(evals) #CONTAINS SUM OF SPLINES        

    #end for loop
    return_table = []
    for i,v in enumerate(czvals):
        return_table.append(0)
        return_table[i]= datatable
        #end for loop
    return return_table

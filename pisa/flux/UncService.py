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


class UncService():
    '''overall class to read in modification maps'''

    def __init__(self, ebins, smooth=0.05, **params):
        global spline_dict
        spline_dict = {}
        global dum
        dum = [0] * len(ebins)

    def data_spliner(self, filename, ebins): #take filename, return spline
        en, dat = np.loadtxt(open_resource(filename)).T
        en2 = np.concatenate([en, ebins])
        dat2 = np.concatenate([dat, dum])
        up = len(en) -1
        endat = sorted(zip(en2,dat2))
        for entry in endat:
            if (entry[0] > en[0] and entry[0] < en[up] and entry[1]==0.0):
                endat.remove(entry)
        en2,dat2 = zip(*endat)
        Ret_Spline = InterpolatedUnivariateSpline(en2, dat2, k=1)
        return Ret_Spline
        
    def get_unc(self, unc_model, ebins, czbins, gettype, **params):
        '''Get the uncertainty for the given
           bin edges in energy and the primary.'''

        datatable = []
        global evals
        evals = get_bin_centers(ebins)
        zbins = [1] * len(czbins)
        czvals = get_bin_centers(zbins)
        spline_dict["A"] = unc_model.data_spliner("~/jsandroo/pisa/pisa/resources/flux/UNC_SUM.txt", ebins)
        #spline_dict["A"] = unc_model.data_spliner("~/UncData/UNC_A.txt")
        #spline_dict["B"] = unc_model.data_spliner("~/UncData/UNC_B.txt")
        #spline_dict["C"] = unc_model.data_spliner("~/UncData/UNC_C.txt")
        #        spline_dict["A"] = unc_model.data_spliner(UNC_files["A"])
        #        spline_dict["B"] = unc_model.data_spliner(UNC_files["B"])
        #        spline_dict["C"] = unc_model.data_spliner(UNC_files["C"])

#        spline = unc_model.data_spliner("~/UncData/UNC_A.txt")


        ########## ADD ALL SPLINES TOGETHER #######################
        for i, v in enumerate(spline_dict):
            if (v == 'A'):
                datatable = 0.01*spline_dict[v].__call__(evals)
            else:
                datatable += 0.01*spline_dict[v].__call__(evals) #CONTAINS SUM OF SPLINES
            

            #datatable = np.power(spline_dict[v].__call__(evals), 2)

        
        #return_table = evals, spline.__call__(evals)
        #return_table = zip(czvals, datatable)
        return_table = []
        for i,v in enumerate(czvals):
            return_table.append(0)
            return_table[i]= datatable
        return return_table

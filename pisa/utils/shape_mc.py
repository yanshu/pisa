#
# previously UncService.py
#
# These function provides the option to modify the shape of a map by a given shape multiplied by a constant.
# It is used for modification of the flux by energy dependent functions as given in Barr http://arxiv.org/pdf/astro-ph/0611266.pdf
#
# IMPORTANT: It returns the square of the modification but with the appropriate sign as given by the modification factor input. ie: if we set flux_hadronic_H = -2 this function will return -4 * flux_hadronic_H^2
#
# author: Joakim Sandroos
#         sandroos@nbi.dk
#
# date:   2015-03-07

import os
import numpy as np
import math
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.interpolate import InterpolatedUnivariateSpline
from pisa.utils.log import logging
from pisa.utils.utils import get_bin_centers, get_bin_sizes
from pisa.utils.jsons import from_json
from pisa.resources.resources import open_resource, find_resource
import inspect

class SplineService():
    def data_spliner(self, filename, ebins): #take filename, return spline
        en = (np.array(filename.keys())).astype(np.float)
        dat = (np.array(filename.values())).astype(np.float)
        #ensure sorting
        en, dat = zip(*sorted(zip(en,dat)))
        #Patch up with zeros over full energy range
        #as ext=1 for InterpolatedUnivariateSpline only available from scipy v0.15.0
        below = (ebins < en[0])
        above = (ebins > en[-1])
        en = np.concatenate((ebins[below],en,ebins[above]))
        dat = np.concatenate((np.zeros_like(ebins[below]),dat,np.zeros_like(ebins[above])))
        #return splined function
        spline = InterpolatedUnivariateSpline(en, dat, k=1)
        return spline

    def __init__(self, ebins, evals, dictFile=None):
        self.SplineDict = {}
        #evals = get_bin_centers(ebins)
        self.ebins = ebins
        self.evals = evals
        self.datadict = from_json(find_resource(dictFile))
        self.splines= {}
        for entry in self.datadict:
            logging.info('Splining for parameter: %s '%(entry))
            spline = self.data_spliner(self.datadict[entry], ebins)
            self.splines[entry] = spline 
        
    def add_splines(self, evals):
        print "start getting all splines for evals..."
        for entry in self.datadict:
            logging.info('Splining for parameter: %s '%(entry))
            self.SplineDict[entry] = self.splines[entry](evals) 
            print "     len self.SplineDict[", entry, "] = ", len(self.SplineDict[entry])
        print "finished getting all splines for evals"
            

    def modify_shape(self, ebins, czbins, factor, fname, event_by_event=False, pre_saved_splines=None):
        '''
        Calculate the contribution to the shape modification for a given uncertainty provided in file fname 
        with a factor (given in percent) 
        '''
        evals = get_bin_centers(ebins)
        czvals = get_bin_centers(czbins)
        if event_by_event:
            evals = ebins
            czvals = czbins

        logging.info("\n keys in spline dict: %s " %self.SplineDict.keys())
        logging.info("vs fname: %s \n" %fname)
        if pre_saved_splines == None:
            self.add_splines(evals)
        else:
            self.SplineDict = pre_saved_splines
        splines = self.SplineDict[fname]
        
        #Calculate the return table
        sign = 1 if factor >=0 else -1
        datatable = np.zeros_like(evals) + sign * factor * factor * splines * splines#np.multiply(factor, np.array(self.spline))#(evals)
        if not event_by_event:
            datatable = np.tile(datatable,(len(czvals),1))
        
        return datatable  #amap*datatable

    def get_genie_spline(self, type, flav):
        assert(type in ['MaCCQE', 'MaRES', 'AhtBY', 'BhtBY', 'CV1uBY', 'CV2uBY'])
        assert(flav in ['nue', 'numu', 'nutau', 'nue_bar', 'numu_bar', 'nutau_bar', 'nuall_nc', 'nuallbar_nc'])
        key = type+'_'+flav
        return self.splines[key](self.evals)

    def get_barr_spline(self, key):
        assert(key in [ 'flux_hadronic_A', 'flux_hadronic_B', 'flux_hadronic_C', 'flux_hadronic_D',
                 'flux_hadronic_E', 'flux_hadronic_F', 'flux_hadronic_G', 'flux_hadronic_H',
                 'flux_hadronic_I', 'flux_hadronic_W', 'flux_hadronic_X', 'flux_hadronic_Y',
                 'flux_hadronic_Z', 'flux_pion_chargeratio_Chg', 'flux_prim_norm_a',
                 'flux_prim_exp_norm_b', 'flux_prim_exp_factor_c', 'flux_spectral_index_d'])
        return self.splines[key](self.evals)

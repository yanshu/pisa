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
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.interpolate import bisplrep, bisplev, UnivariateSpline
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
    print 'test'

    def data_spliner(self, filename): #take filename, return spline
        inTable = np.loadtxt(open_resource(filename)).T
        Ret_Spline = UnivariateSpline(inTable[0], inTable[1])
        return Ret_Spline
    
    def __init__(self, smooth=0.05, **params):
        global spline_dict
        spline_dict = {}

        
    def get_unc(self, ebins, scale, gettype):
        '''Get the uncertainty for the given
           bin edges in energy and the primary.'''
        
        #Evaluate the flux at the bin centers
        global evals
        evals = get_bin_centers(ebins)

        print 'start the splining procedure'
        spline_dict["A"] = unc_model.data_spliner("~/UncData/UNC_A.txt")
        spline_dict["B"] = unc_model.data_spliner("~/UncData/UNC_B.txt")
        spline_dict["C"] = unc_model.data_spliner("~/UncData/UNC_C.txt")
        #        spline_dict["A"] = unc_model.data_spliner(UNC_files["A"])
        #        spline_dict["B"] = unc_model.data_spliner(UNC_files["B"])
        #        spline_dict["C"] = unc_model.data_spliner(UNC_files["C"])

        spline = unc_model.data_spliner("~/UncData/UNC_A.txt")
        
        for i, v in enumerate(spline_dict):
            print 'testing ', v, ". ", spline_dict[v]
            spline_dict[v].__call__(evals)
            
        
        return_table = evals, scale * spline.__call__(evals)
        return return_table


parser = ArgumentParser(description='Take a settings file '
        'as input and write out a set of flux maps',
        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--ebins', metavar='[1.0,2.0,...]', type=json_string,
                        help= '''Edges of the energy bins in units of GeV. ''',
                        default=np.logspace(np.log10(1.0),np.log10(80.0),40) )

args = parser.parse_args()

global unc_model
unc_model = UncService()
unc_map = unc_model.get_unc(args.ebins, 1, 'flux_unc')



#########################################
## old code has been parked here for now
#########################################

#        if gettype == 'flux_unc':
 #           spline = data_spliner('../resourcs/flux/UNC_SUM.txt')
  #          if prim != 'numu':
   #             "We don't have the uncertainty for other types than numu for now"
    #            return 0

#unc_dict contains a dictionary of energy values and neutrino types

#Load the data table
#        table = np.loadtxt(open_resource("../resources/flux/UNC_SUM.txt")).T
        #print 'content of UNC table: ', table

#        parameters = ['A','B','C']
        
#        tableA = np.loadtxt(open_resource("../resources/flux/UNC_B.txt")).T
 #       tableB = np.loadtxt(open_resource("../resources/flux/UNC_B.txt")).T
  #      tableC = np.loadtxt(open_resource("../resources/flux/UNC_C.txt")).T

        #columns in Honda files are in the same order
        #print 'Where does UNC enegry come from: ', ['energy']
        #        cols = ['energy']+primaries
#        cols = ['energy']+['unc']
 #       print 'content of the UNC cols table: ', cols

#        unc_dict = dict(zip(cols, table))

#        print 'unc dict keys: ', unc_dict.keys()
        #unc_mod_dict = dict(zip(parameters #### finish this clever way of doing things later
#        A_dict = dict(zip(cols,tableA))
        
        #generate table of linear combinations of uncertainty parameters
#       for energy, value in tableA:
 #          value = parA * value
            
#        unc_dict_par = tableA[1]
#
 #       : tableA,
  #                      'B': tableB,
   #                     'C': tableC,
    #                }

#        for key in unc_dict.iterkeys():
            #There are 20 lines per zenith range
            #flux_dict[key] = np.array(np.split(flux_dict[key], 20)) #no need to split when we don't have zen bins in the input file
#            if not key=='energy':
 #               unc_dict[key] = unc_dict[key].T #transpose the read in, to a column list

        #        for prim in primaries:
#        for prim in primaries in unc_dict.keys():
 #           unc_dict[prim] = unc_dict[prim]*0.01 #convert from percentage to decimal

        #Now get a spline representation of the unc table.
#        logging.debug('Make spline representation of flux Uncertainty')

#        self.spline_dict = {}
#        global spline
 #       for nutype in primaries:
  #          unc_flux = unc_dict[nutype].T             #Get the flux uncertainty
   #         spline =  UnivariateSpline(unc_dict['energy'], unc_flux) ## spline is a python spline object that can be used directly. See documentation: http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html #shoud get a new name for each nutype

    #    A_unc = A_dict['numu']
     #   splineA = UnivariateSpline(A_dict['energy'], A_unc)

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
    print 'testing dat shit'

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
    
    def __init__(self, ebins, smooth=0.05, **params):
        global spline_dict
        spline_dict = {}
        global dum
        #print 'args.ebins: ', len(args.ebins)
        dum = [0] * len(ebins)
        
    def get_unc(self, unc_model, ebins, czbins, gettype):
        '''Get the uncertainty for the given
           bin edges in energy and the primary.'''
        datatable = []
        #Evaluate the flux at the bin centers
        global evals
        evals = get_bin_centers(ebins)
        zbins = [1] * len(czbins)
        czvals = get_bin_centers(zbins)
        #print 'len zbins: ', len(czvals)
        #print 'lenth evals: ', len(evals)

        print 'start the splining procedure'
        spline_dict["A"] = unc_model.data_spliner("pisa/resources/flux/UNC_SUM.txt", ebins)
        #spline_dict["A"] = unc_model.data_spliner("~/UncData/UNC_A.txt")
        #spline_dict["B"] = unc_model.data_spliner("~/UncData/UNC_B.txt")
        #spline_dict["C"] = unc_model.data_spliner("~/UncData/UNC_C.txt")
        #        spline_dict["A"] = unc_model.data_spliner(UNC_files["A"])
        #        spline_dict["B"] = unc_model.data_spliner(UNC_files["B"])
        #        spline_dict["C"] = unc_model.data_spliner(UNC_files["C"])

#        spline = unc_model.data_spliner("~/UncData/UNC_A.txt")


        ########## ADD ALL SPLINES TOGETHER #######################
        for i, v in enumerate(spline_dict):
            #            print 'testing ', v, ". ", spline_dict[v]
            #           print 'evals: ', evals
            if (v == 'A'):
                datatable = 0.01*spline_dict[v].__call__(evals)
            else:
                datatable += 0.01*spline_dict[v].__call__(evals) #CONTAINS SUM OF SPLINES
            
            #print 'V is: ', v
            #datatable = np.power(spline_dict[v].__call__(evals), 2)
            #print args.ebins, datatable
        
        #return_table = evals, spline.__call__(evals)
        #return_table = zip(czvals, datatable)
        return_table = []
        for i,v in enumerate(czvals):
            #print 'i, v: ', i, v
            return_table.append(0)
            return_table[i]= datatable
            #print 'return_table: ', return_table
            #print 'filling return_table: ', return_table[i]
        #print 'length of datatable: ', len(datatable)
        #print 'lenth of datatable[0]: ', len(datatable[0])
        
        #print 'length of return_table: ', len(return_table)
        #print 'lenth of datatalbe[0]: ', return_table[0]
        return return_table


#parser2 = ArgumentParser(description='Take a settings file '
 #       'as input and write out a set of flux maps',
  #      formatter_class=ArgumentDefaultsHelpFormatter)

#parser2.add_argument('--ebins', metavar='[1.0,2.0,...]', type=json_string,
 #                       help= '''Edges of the energy bins in units of GeV. ''',
  #                      default=np.logspace(np.log10(1.0),np.log10(80.0),40) )

#parser2.add_argument('--czbins', metavar='[-1.0,-0.8.,...]', type=json_string,
 #                    help= '''Edges of the cos(zenith) bins.''',
  #                   default = np.linspace(-1.,0.,21))

#args = parser2.parse_args()

#print '# energy bins: ', len(args.ebins)
#global unc_model
#unc_model = UncService()
#unc_map = unc_model.get_unc(ebins, 'flux_unc')
#to_json(unc_map, 'unc_sum_out.json')



####
# test dat shit out
####

#from pisa.flux.HondaFluxService import HondaFluxService, primaries
#flux_model = HondaFluxService(args.flux_file)

#print type(flux_model)
#print dir(flux_model)
#print flux_model.keys()
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

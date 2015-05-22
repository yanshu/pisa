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
from scipy.interpolate import InterpolatedUnivariateSpline
from pisa.utils.log import logging
from pisa.utils.utils import get_bin_centers, get_bin_sizes
from pisa.resources.resources import open_resource

def data_spliner(filename, ebins): #take filename, return spline

    #Load the data
    en, dat = np.loadtxt(open_resource(filename)).T
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
        
def modify_shape(amap, ebins, czbins, factor, fname):
    '''
    Apply the shape modification for a given uncertainty provided in file fname 
    with a factor (given in percent) to the input map amap. 
    '''
    evals = get_bin_centers(ebins)
    czvals = get_bin_centers(czbins)
    
    #Get the splined uncertainty values
    spline = data_spliner(fname, ebins)

    #Calculate the retun table
    datatable = np.ones_like(evals) + factor*0.01*spline(evals)
    datatable = np.tile(datatable,(len(czvals),1))

    logging.trace('Apply shape uncertainty %s at %.2f percent'%(fname,factor))
    return amap*datatable

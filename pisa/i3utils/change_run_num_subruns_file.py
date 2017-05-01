#!/usr/bin/env python

import os
import sys
import h5py
import shutil
import numpy as np
from pisa.utils.hdf import from_hdf, to_hdf
import pisa.resources.resources as resources

from os.path import expandvars
from optparse import OptionParser

parser = OptionParser()

parser.add_option("-i", dest="infile", help="Input file.")

(options, args) = parser.parse_args()
infile = options.infile
print infile

if not os.path.exists(options.infile):
    print "The directory you gave does not exist, please try again."
else:
    if not infile.endswith('hdf5'):
        print "file not hdf5"
        pass
    data_file = os.path.basename(infile)
    print "data_file = ", data_file
    #file_name = data_file.split('.')[0]
    #run_num = file_name.split('_')[2]

    file_name = data_file.split('.')[2]
    run_num = file_name.lstrip('0')

    #run_num = '16600'
    print "run_num = ", run_num

    run_num_1 = run_num + '1'
    run_num_2 = run_num + '2'
    run_num_3 = run_num + '3'
    if run_num.startswith('12'):
        energy_1 = 4
        energy_2 = 12
    if run_num.startswith('14'):
        energy_1 = 5
        energy_2 = 80 
    if run_num.startswith('16'):
        energy_1 = 10 
        energy_2 = 30 

    hdf_file = h5py.File(infile, "r+")
    run = hdf_file['I3EventHeader']['Run']
    true_e = hdf_file['trueNeutrino']['energy']
    for i in range(0,len(true_e)):
        #print "true_e[i] = ", true_e[i]
        if true_e[i] <= energy_1:
            run[i] = run_num_1
        elif true_e[i] <= energy_2:
            run[i] = run_num_2 
        else:
            run[i] = run_num_3 
    hdf_file['I3EventHeader']['Run'] = run
    print "hdf_file['I3EventHeader']['Run'] = ", hdf_file['I3EventHeader']['Run'] 
    hdf_file.close()

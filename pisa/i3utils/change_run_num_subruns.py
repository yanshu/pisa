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

parser.add_option("-i", dest="data_dir", help="Input data dir.")

(options, args) = parser.parse_args()
data_dir = options.data_dir
print data_dir

if not os.path.isdir(options.data_dir):
    print "The directory you gave does not exist, please try again."
else:
    dirListing = os.listdir(options.data_dir)
    totFileCount = len(dirListing)

count = 0 

for aFile in dirListing:
    data_file = "%s/%s" % (data_dir, aFile)	
    print "data_file = ", data_file
    file_name = aFile.split('.')[0]
    run_num = file_name.split('_')[1]
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

    hdf_file = h5py.File(data_file, "r+")
    run = hdf_file['I3EventHeader']['Run']
    true_e = hdf_file['trueNeutrino']['energy']
    for i in range(0,len(true_e)):
        if true_e[i] <= energy_1:
            run[i] = run_num_1
        elif true_e[i] <= energy_2:
            run[i] = run_num_2 
        else:
            run[i] = run_num_3 
    hdf_file['I3EventHeader']['Run'] = run
    hdf_file.close()

    hdf_file, attrs = from_hdf(resources.find_resource(data_file), return_attrs = True)
    genie_dict = {}
    for name in ['AhtBY', 'BhtBY', 'CV1uBY', 'CV2uBY', 'MaCCQE', 'MaCCRES', 'MaCOHpi', 'MaNCEL', 'MaNCRES']:
        if name == 'vector_index':
            continue
        idx = hdf_file['GENIE_SystematicsReweight']['vector_index']
        if name in ['AhtBY', 'BhtBY', 'CV1uBY', 'CV2uBY', 'MaCCQE', 'MaCCRES', 'MaCOHpi', 'MaNCEL', 'MaNCRES']:
            for i in range(0,4):
                cut = idx%4==i
                genie_dict['rw_'+name+'_%i'%i] = hdf_file['GENIE_SystematicsReweight'][name][cut]
    hdf_file['GENIE_SystematicsReweight_Split'] = genie_dict 
    to_hdf(hdf_file, aFile, attrs=attrs, overwrite=True)


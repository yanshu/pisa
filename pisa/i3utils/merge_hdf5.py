#! /usr/bin/env python
import numpy as np
import os.path
import h5py
import copy
import pisa.resources.resources as resources
from pisa.utils.log import set_verbosity,logging,profile
from pisa.utils.hdf import from_hdf, to_hdf
import pisa.utils.utils as utils
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def merge_files(data_file_paths, outdir):
    print 'data_file_path = ', data_file_paths
    data_file_merge, attrs_merge = copy.deepcopy(from_hdf(resources.find_resource(data_file_paths[0]), return_attrs = True))
    print "attrs_merge = ", attrs_merge
    for data_file_path in data_file_paths[1:]:
        data_file, attrs = from_hdf(resources.find_resource(data_file_path), return_attrs = True)
        print "attrs = ", attrs
        print "data_file keys = ", data_file.keys()
        for key in data_file.keys():
            if key=='__I3Index__':
                for g_key in data_file[key].keys():
                    if g_key=='__I3Index__':
                        continue
                    data_file_merge[key][g_key]= np.concatenate([data_file_merge[key][g_key], data_file[key][g_key]])
            else:
                data_file_merge[key]= np.concatenate([data_file_merge[key], data_file[key]])
                print "data_file[",key,"] type = ", type(data_file[key])

    data_file_name = os.path.basename(data_file_paths[0])
    utils.mkdir(args.outdir)
    output_file_name = outdir + '/' + data_file_name.split('.hdf5')[0]+'_merged.hdf5'
    if not os.path.isfile(output_file_name):
        to_hdf(data_file_merge, output_file_name, attrs=attrs, overwrite=True)
    else:
        print 'File %s already exists, skipped. Please delete it or rename it.' % output_file_name


if __name__ == '__main__':

    parser = ArgumentParser(description='''Add neutrino fluxes (and neutrino weights(osc*flux*sim_weight) if needed) for each event. ''')
    parser_file = parser.add_mutually_exclusive_group(required=True)
    parser_file.add_argument( '-i', '--input_files', metavar='H5_FILE', type=str, help='input HDF5 files.')
    parser.add_argument('-o','--outdir',metavar='DIR',default='', help='Directory to save the output figures.')
    args = parser.parse_args()
    set_verbosity(0)

    # get file name
    hd5_file_names = eval(args.input_files)
    outdir = args.outdir
    
    merge_files(hd5_file_names, outdir=outdir)

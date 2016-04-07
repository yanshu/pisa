#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   18 September 2015
#
# A set of utility functions for dealing with the post processing of
# the LLR Analysis files
#

from __future__ import division, print_function

import h5py
import numpy as np

from pisa.utils.log import logging

def processTrial(combined, partial_run):
    """
    Works with appendTrials (see documentation good)
    """

    good_keys = ['seed','hypo_NMH','hypo_IMH']
    for key,value in partial_run.items():
        if key not in good_keys:
            raise KeyError("ERROR: key: {0:%s} must be one of {1:%s}"
                           .format(key, good_keys))

        if key == 'seed':
            combined[key].append(value)
        else:
            if key not in combined.keys():
                combined[key] = {}
            for key1 in partial_run[key].keys():
                # "opt_flags" and "opt_data"
                if key1 not in combined[key].keys(): combined[key][key1] = {}
                for key2 in partial_run[key][key1].keys():
                   # the "llh" entry is a list of dicts of values-per-channel
                   # (i.e. one per step the optimizer takes)
                   if key2 == "llh" and "llh" not in combined[key][key1].keys():
                       combined[key][key1]["llh"] = {}
                   # the other entries (parameters) are lists
                   elif key2 not in combined[key][key1].keys():
                       combined[key][key1][key2] = []
                   # opt flags describe the outcome of the minimization process,
                   # so no list
                   if key1=='opt_flags':
                       combined[key][key1][key2].append(partial_run[key][key1][key2])
                   else:
                       if key2=="llh":
                           for chan in partial_run[key][key1][key2][-1].keys():
                               if chan not in combined[key][key1][key2].keys():
                                   # make a separate list of llh values for each channel
                                   combined[key][key1][key2][chan] = []
                               combined[key][key1][key2][chan].append(
                                         partial_run[key][key1][key2][-1][chan])
                       else:
                           # separate list of best fits for each parameter
                           combined[key][key1][key2].append(
                                     partial_run[key][key1][key2][-1])

    return

def appendTrials(combined, partial_run):
    """
    Appends the data from partial_run (one of the independent
    processing runs on the cluster) into the combined data dictionary.

    Data is actually appended to combined in processTrial() fn.

    \Params:
      * combined - combined hierarchical data dictionary for the llh and log
        info that will be the final output
      * partial_run - data dictionary for the llh data that is aggregated
        into 'combined' data and saved for later processing.

    \Modifies:
      * combined - modified by adding partial_run info to it.

    \Returns:
      * None
    """

    for key in partial_run.keys():
        # true_h_fiducial or false_h_best_fit
        # Check if combined has been defined this far:
        if key not in combined.keys():
            combined[key] = {nkey: [] for nkey in partial_run[key].keys() if nkey!='trials'}
            # Seed needs to be an array:
            combined[key]['seed'] = []
            # Store false h settings and corresponding llh just once (first time they are found)
            if 'false' in key:
                try:
                    if len(combined[key]['false_h_settings'])==0:
                        combined[key]['false_h_settings'] = partial_run[key]['false_h_settings']
                    if len(combined[key]['llh_null'])==0:
                        # partial_run[key]['llh_null'] has dict in list, needs special treatment
                        combined[key]['llh_null'] = partial_run[key]['llh_null']
                        combined[key]['llh_null']['llh'] = combined[key]['llh_null']['llh'][0]
                    if len(combined[key]['opt_flags'])==0:
                        combined[key]['opt_flags'] = partial_run[key]['opt_flags']
                except: pass
        ntrials =  len(partial_run[key]['trials'])
        new_keys = partial_run[key].keys()
        # Loop over each trial, adding seed and data in
        # true_NMH/true_IMH to
        for ii in xrange(ntrials):
            processTrial(combined[key], partial_run[key]['trials'][ii])


    return


def saveDict(data,fh):
    """
    Saves a data dictionary to filehandle (or group), fh, as long as the
    group to be written to the hdf5 file is one of:
      [np.ndarray, list, float, int, bool, str, None]
    """

    for k,v in data.items():
        if type(v) == dict:
            group = fh.create_group(k)
            logging.trace("  creating group %s"%k)
            saveDict(v,group)
        else:
            saveNonDict(k, v, fh)

    return


def saveNonDict(key, data, fh):

    logging.trace("    key: %s is of type: %s"%(key,type(data)))
    if type(data) in [np.ndarray,list]:
        logging.trace("    >>saving to dataset: %s"%key)
        fh.create_dataset(key,data=data)
    elif type(data) in [float,int,bool]:
        logging.trace("   >>saving '%s' to dataset: %s"%(str(data),key))
        fh.create_dataset(key,data=data)
    elif type(data) == str:
        logging.trace("   >>saving '%s' to dataset: %s"%(data,key))
        dtype = h5py.special_dtype(vlen=str)
        fh.create_dataset(key,data=data,dtype=dtype)
    elif data is None:
        # NOTE: we convert it to the boolean false here, since
        # I can't find a good way to store the 'None' type in h5py
        data = False
        logging.trace("   >>saving '%s' to dataset: %s"%(data,key))
        fh.create_dataset(key,data=data)
    else:
        raise ValueError("Key: '%s' is unrecognized type: '%s'"%(key,type(data)))

    return

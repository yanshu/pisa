#! /usr/bin/env python
#
#
# Testing our converter for python dictionary to hdf5 using h5py
#

import h5py
import numpy as np
from datetime import datetime
from pisa.utils.jsons import from_json
from pisa.utils.log import logging, set_verbosity
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def save_dict(data,fh):
    '''
    Saves a data dictionary to filehandle (or group), fh.
    '''

    for k,v in data.items():
        if type(v) == dict:
            group = fh.create_group(k)
            logging.trace("  creating group %s"%k)
            save_dict(v,group)
        else:
            logging.trace("    key: %s is of type: %s"%(k,type(v)))
            if type(v) in [np.ndarray,list]:
                logging.trace("    >>saving to dataset: %s"%k)
                fh.create_dataset(k,data=v)
            elif type(v) in [float,int,bool]:
                logging.trace("   >>saving '%s' to dataset: %s"%(str(v),k))
                fh.create_dataset(k,data=v)
            elif type(v) == str:
                logging.trace("   >>saving '%s' to dataset: %s"%(v,k))
                dtype = h5py.special_dtype(vlen=str)
                fh.create_dataset(k,data=v,dtype=dtype)
            elif v is None:
                # NOTE: we convert it to the boolean false here, since
                # I can't find a good way to store the 'None' type in h5py
                v = False
                logging.trace("   >>saving '%s' to dataset: %s"%(v,k))
                fh.create_dataset(k,data=v)
            else:
                raise ValueError("Key: '%s' is unrecognized type: '%s'"%(k,type(v)))

    return

def process_trial(trials,trial):
    '''
    Expects a trial to be a dictionary of {'llh': [ ], 'param1': [], 'param2': [], etc.}
    where param1, etc. are the parameters varied in the trial. The content of each key is
    an array of either 1) JUST the final value found by the minimizer or 2) ALL the values
    chosen along the way to finding the final minimized value. In either case, appending
    the final value [-1] gets the minimized, final value, which is what we are after.
    '''
    if not trials:
        trials = dict(trial)
        for dkey in trials.keys():
            for hkey in trial[dkey].keys():
                if hkey == 'seed':
                    trials[dkey][hkey] = []
                    continue
                for key in trials[dkey][hkey].keys():
                    trials[dkey][hkey][key] = []
    else:
        for dkey in trial.keys():
            for hkey in trial[dkey].keys():
                if hkey == 'seed':
                    trials[dkey][hkey].append(trial[dkey][hkey])
                    continue
                for key in trial[dkey][hkey].keys():
                    trials[dkey][hkey][key].append(trial[dkey][hkey][key][-1])
    return trials

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('outfile',metavar='HDF5',type=str,help="output filename")
parser.add_argument('infiles',nargs='*',help="input llh files to combine into one output hdf5 file.")
parser.add_argument('-v', '--verbose', action='count', default=None,
                        help='''set verbosity level''')
args = parser.parse_args()
set_verbosity(args.verbose)

logging.warn("processing "+str(len(args.infiles))+" files...")

logging.warn("Saving to file: %s"%args.outfile)

mod_num = len(args.infiles)/20

start_time = datetime.now()

minimizer_settings = {}
template_settings_up = {}
pseudo_data_settings = {}
trials = {}
for i,filename in enumerate(args.infiles):
    if mod_num > 0:
        if i%mod_num == 0: print "  >> %d files done..."%i
    try:
        data = from_json(filename)
    except:
        print "Skipping file: ",filename
        continue

    if not minimizer_settings:
        minimizer_settings = data['minimizer_settings']

    if not template_settings_up:
        template_settings_up = data['template_settings_up']

    if not pseudo_data_settings:
        try: pseudo_data_settings = data['pseudo_data_settings']
        except: pass

    if len(args.infiles)==1:
        trials= data['trials'][0]
    else:
        for trial in data['trials']:
            trials = process_trial(trials,trial)

#print "trials: ", trials
fh = h5py.File(args.outfile,'w')

min_group = fh.create_group('minimizer_settings')
template_group = fh.create_group('template_settings_up')
trial_group = fh.create_group('trials')

save_dict(minimizer_settings,min_group)
save_dict(template_settings_up,template_group)
save_dict(trials,trial_group)

if pseudo_data_settings:
    pd_group = fh.create_group('pseudo_data_settings')
    save_dict(pseudo_data_settings,pd_group)

fh.close()

print "Time to process files: %s"%(datetime.now() - start_time)



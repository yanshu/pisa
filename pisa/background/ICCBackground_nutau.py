#! /usr/bin/env python
#
# ICCBackground_nutau.py
#
# This module adds the inverted corridor cut background from 3 years of
# data (data/user/jpa14/Matt/level5b/data/IC86_1, IC86-2, IC86-3). For 
# nutau analysis only, which needs different treatment of up and down-
# going template.
#
# author: Timothy C. Arlen
#         Feifei Huang
#
# date:   May 5, 2015
#

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import h5py
from pisa.utils.log import logging, physics, set_verbosity
from pisa.utils.utils import check_binning, get_binning
from pisa.utils.jsons import from_json,to_json
from pisa.utils.proc import report_params, get_params, add_params
from pisa.background.BackgroundServiceICC_nutau import BackgroundServiceICC


def add_icc_background(event_rate_pid_maps,background_service,atmos_mu_scale,livetime,**kwargs):

    """
    Primary function for this stage, returns the event map with ICC 
    background added to the event_rate_pid_map from the PID stage.
    """

    # Be verbose on input
    params = get_params()
    report_params(params, units = ['', ''])

    event_rate_maps = {'params': add_params(params,event_rate_pid_maps['params'])}

    # Get ICC background dictionary
    background_dict = background_service.get_icc_bg()
    for flav in ['trck','cscd']:
        ebins, czbins = get_binning(event_rate_pid_maps[flav])
        event_rate_pid_map = event_rate_pid_maps[flav]['map']
        event_rate = event_rate_pid_map + background_dict[flav] * atmos_mu_scale * livetime
        event_rate_maps[flav] = {'map':event_rate,
                                 'ebins':ebins,
                                 'czbins':czbins}
    return event_rate_maps

if __name__ == '__main__':

    parser = ArgumentParser(description='Use the event_rate_pid_maps, ICC background file '
                          'and the atmos_mu_scale to get the final event rate map',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('event_rate_pid_maps',metavar='JSON',type=from_json,
                     help='''JSON event rate pid input file with the following parameters:
                        {"cscd": {'czbins':[], 'ebins':[], 'map':[]},
                         "trck": {'czbins':[], 'ebins':[], 'map':[]}} ''')
    parser.add_argument('--background_file',metavar='FILE',type=str,
                        default='background/IC86_3yr_ICC.hdf5',
                        help='''HDF5 File containing atmospheric background from 3 years'
                        inverted corridor cut data''')
    parser.add_argument('--atmos_mu_scale',type=float,default=1.0,
                        help='''Overall scale on atmospheric muons for livetime = 1.0 yr''')
    parser.add_argument('--livetime',type=float,default=1.0,
                        help='''livetime in years to re-scale by.''')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE', type=str,
                        action='store',default="event_rate.json",
                        help='''file to store the output''')
    parser.add_argument('-v', '--verbose', action='count', default=None,
                        help='''set verbosity level''')
    args = parser.parse_args()

    #Set verbosity level
    set_verbosity(args.verbose)

    #Check binning
    ebins, czbins = check_binning(args.event_rate_pid_maps)

    logging.info("Defining background_service...")
    background_service = BackgroundServiceICC(ebins,czbins,icc_bg_file=args.background_file)

    event_rate_maps = add_icc_background(args.event_rate_pid_maps,background_service,args.atmos_mu_scale,args.livetime)

    logging.info("Saving output to: %s"%args.outfile)
    to_json(event_rate_maps,args.outfile)


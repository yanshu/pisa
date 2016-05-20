#! /usr/bin/env python
#
# ICCBackground.py
#
# This module adds the inverted corridor cut background from 3 years of
# data (data/user/jpa14/Matt/level5b/data/IC86_1, IC86-2, IC86-3).
#
# author: Timothy C. Arlen
#         Feifei Huang
#
# date:   May 5, 2015
#

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
from pisa.utils.log import logging, physics, set_verbosity
from pisa.utils.utils import check_binning, get_binning
from pisa.utils.jsons import from_json,to_json
from pisa.utils.proc import report_params, get_params, add_params
from pisa.background.BackgroundServiceICC import BackgroundServiceICC


def add_icc_background(event_rate_pid_maps, background_service, atmos_mu_scale,
                        livetime, atmmu_f, noise_f, use_atmmu_f, **kwargs):

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
    n_nu = np.sum(event_rate_pid_maps['cscd']['map']) + np.sum(event_rate_pid_maps['trck']['map'])
    n_mu = np.sum(background_dict['cscd']) + np.sum(background_dict['trck'])
    for flav in ['trck','cscd']:
        ebins, czbins = get_binning(event_rate_pid_maps[flav])
        event_rate_pid_map = event_rate_pid_maps[flav]['map']
        if use_atmmu_f:
            # this is the oscFit way of defining the scale factor for background
            # n_nu is the total no. of MC neutrinos
            # atmmu_f is the fraction of atmospheric muons to total no. of events (neutrinos + background + noise), default = 0.2
            # noise_f is the fraction of noise, it's set to 0 for MSU sample, since noise is only a few events (study from JP)
            # n_total = n_total * atmmu_f + n_total * noise_f + n_nu
            # thus: scale = n_total * atmmu_f/n_mu, where n_total = n_nu / (1 - atmmu_f - noise_f)
            scale = n_nu * atmmu_f / n_mu / (1 - atmmu_f - noise_f)
            bg_rate_pid_map = background_dict[flav] * scale
            sumw2 = background_dict[flav] * scale**2
        else:
            # this is another way of defining the scale factor for background
            bg_rate_pid_map = background_dict[flav] * atmos_mu_scale * livetime
            sumw2 = background_dict[flav] * atmos_mu_scale**2 * livetime**2
        #replace zero entry errors (which are 0 here), with 0.5 * the smallest absolute error....if everything is zero, replace everything by one
        #if np.count_nonzero(bg_rate_pid_map) > 0:
        #    bg_rate_pid_map[bg_rate_pid_map==0] = np.min(bg_rate_pid_map[bg_rate_pid_map>0])/2.
        #else:
        #    bg_rate_pid_map = np.ones_like(bg_rate_pid_map)
        if np.count_nonzero(sumw2) > 0:
            sumw2[sumw2==0] = np.min(sumw2[sumw2>0])/2.
        else:
            # if they are all zero, return 1 as error for every bin
            sumw2 = np.ones_like(sumw2)
        event_rate_maps[flav] = {'map':event_rate_pid_map + bg_rate_pid_map,
                                 'sumw2':sumw2,
                                 'map_nu':event_rate_pid_map,
                                 'map_mu':bg_rate_pid_map,
                                 'sumw2_nu':np.zeros(np.shape(sumw2)),
                                 'sumw2_mu':sumw2,
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
    parser.add_argument('--atmos_mu_scale',type=float,default= 0.37,
                        help='''Overall scale on atmospheric muons for livetime = 1.0 yr''')
    parser.add_argument('--atmmu_f',type=float,default= 0.2,
                        help='''Fraction of atmospheric muons to total no. of neutrinos+background+noise''')
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


# ! /usr/bin/env python
#
# GetMCError.py
#
# Class to get the MC number of events.
#
# author: Timothy C. Arlen - tca3@psu.edu
#         Sebastian Boeser - sboeser@uni-mainz.de
#         Feifei Huang - fxh140@psu.edu
#
# date:  05/14/2015 
#

import sys
import numpy as np
import h5py
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pisa.utils.log import logging, profile, set_verbosity
from pisa.resources.resources import find_resource
from pisa.utils.params import get_values, select_hierarchy
from pisa.utils.jsons import from_json, to_json, json_string
from pisa.utils.utils import Timer
from pisa.pid import PID

class GetMCError:
    '''
    GetMCError returns a map of the number of MC events for trck and cscd channel.

    '''
    def __init__(self, template_settings, ebins, czbins, reco_mc_wt_file=None, **kwargs):

        '''
        Parameters:
        * template_settings - dictionary of all template-making settings
        * reco_mc_wt_file - the hdf5 file that contains the reconstruction information
        '''
        
        self.ebins = ebins
        self.czbins = czbins
        self.simfile = reco_mc_wt_file
        logging.debug("Using %u bins in energy from %.2f to %.2f GeV"%
                      (len(self.ebins)-1, self.ebins[0], self.ebins[-1]))
        logging.debug("Using %u bins in cos(zenith) from %.2f to %.2f"%
                      (len(self.czbins)-1, self.czbins[0], self.czbins[-1]))

        # PID Service:
        pid_mode = template_settings['pid_mode']
        self.pid_service = PID.pid_service_factory(ebins= ebins, czbins=czbins, **template_settings)
        return


    def get_mc_events_map(self, apply_reco_prcs, params, simfile, **kargs):
        '''
        '''
        logging.info('Opening file: %s'%(simfile))
        try:
            fh = h5py.File(find_resource(simfile),'r')
        except IOError,e:
            logging.error("Unable to open event data file %s"%simfile)
            logging.error(e)
            sys.exit(1)
       
        mc_event_maps = {'params': params}
        e_reco_precision_up = params['e_reco_precision_up']
        e_reco_precision_down = params['e_reco_precision_down']
        cz_reco_precision_up = params['cz_reco_precision_up']
        cz_reco_precision_down = params['cz_reco_precision_down']
        all_flavors_dict = {}
        for flavor in ['nue', 'numu','nutau']:
            flavor_dict = {}
            logging.debug("Working on %s "%flavor)
            for int_type in ['cc','nc']:
                true_energy = np.array(fh[flavor+'/'+int_type+'/true_energy'])
                true_coszen = np.array(fh[flavor+'/'+int_type+'/true_coszen'])
                reco_energy = np.array(fh[flavor+'/'+int_type+'/reco_energy'])
                reco_coszen = np.array(fh[flavor+'/'+int_type+'/reco_coszen'])
                
                if apply_reco_prcs:
                    if e_reco_precision_up != 1:
                        reco_energy[true_coszen<=0] *= e_reco_precision_up
                        reco_energy[true_coszen<=0] -= (e_reco_precision_up - 1) * true_energy[true_coszen<=0]

                    if e_reco_precision_down != 1:
                        reco_energy[true_coszen>0] *= e_reco_precision_down
                        reco_energy[true_coszen>0] -= (e_reco_precision_down - 1) * true_energy[true_coszen>0]

                    if cz_reco_precision_up != 1:
                        reco_coszen[true_coszen<=0] *= cz_reco_precision_up
                        reco_coszen[true_coszen<=0] -= (cz_reco_precision_up - 1) * true_coszen[true_coszen<=0]

                    if cz_reco_precision_down != 1:
                        reco_coszen[true_coszen>0] *= cz_reco_precision_down
                        reco_coszen[true_coszen>0] -= (cz_reco_precision_down - 1) * true_coszen[true_coszen>0]

                while np.any(reco_coszen<-1) or np.any(reco_coszen>1):
                    reco_coszen[reco_coszen>1] = 2-reco_coszen[reco_coszen>1]
                    reco_coszen[reco_coszen<-1] = -2-reco_coszen[reco_coszen<-1]

                bins = (self.ebins,self.czbins)
                hist_2d,_,_ = np.histogram2d(reco_energy,reco_coszen,bins=bins)
                flavor_dict[int_type] = hist_2d
            all_flavors_dict[flavor] = flavor_dict

        numu_cc_map = all_flavors_dict['numu']['cc']
        nue_cc_map = all_flavors_dict['nue']['cc']
        nutau_cc_map = all_flavors_dict['nutau']['cc']
        nuall_nc_map = all_flavors_dict['numu']['nc']

        #print " before PID, total no. of MC events = ", sum(sum(numu_cc_map))+sum(sum(nue_cc_map))+sum(sum(nutau_cc_map))+sum(sum(nuall_nc_map))
        mc_event_maps['numu_cc'] = {u'czbins':self.czbins,u'ebins':self.ebins,u'map':numu_cc_map}
        mc_event_maps['nue_cc'] =  {u'czbins':self.czbins,u'ebins':self.ebins,u'map':nue_cc_map}
        mc_event_maps['nutau_cc'] = {u'czbins':self.czbins,u'ebins':self.ebins,u'map':nutau_cc_map}
        mc_event_maps['nuall_nc'] = {u'czbins':self.czbins,u'ebins':self.ebins,u'map':nuall_nc_map}

        final_MC_event_rate = self.pid_service.get_pid_maps(mc_event_maps)
        #print "No. of MC events (trck) : ", sum(sum(final_MC_event_rate['trck']['map']))
        #print "No. of MC events (cscd) : ", sum(sum(final_MC_event_rate['cscd']['map']))
        #print "Total no. of MC events : ", sum(sum(final_MC_event_rate['trck']['map']))+ sum(sum(final_MC_event_rate['cscd']['map']))
        return final_MC_event_rate

if __name__ == '__main__':

    # parser
    parser = ArgumentParser(
        description='''Runs the template making process.''',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--reco_mc_file', type=str,
                        metavar='HDF5', required=True,
                        help='''reco mc file that contains the reconstruction information''')
    parser.add_argument('-t', '--template_settings', type=str,
                        metavar='JSONFILE', required=True,
                        help='''settings for the template generation''')
    hselect.add_argument('--apply_reco_prcs', default=False,
                        action='store_true', help="apply reco precision systematics")
    hselect = parser.add_mutually_exclusive_group(required=False)
    hselect.add_argument('--normal', dest='normal', default=True,
                        action='store_true', help="select the normal hierarchy")
    hselect.add_argument('--inverted', dest='normal', default = False,
                        action='store_false',
                         help="select the inverted hierarchy")
    parser.add_argument('-v', '--verbose', action='count', default=None,
                        help='set verbosity level.')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE',
                        type=str, action='store',default="template.json",
                        help='file to store the output')
    args = parser.parse_args()

    set_verbosity(args.verbose)

    with Timer() as t:
        #Load all the settings
        model_settings = from_json(args.template_settings)

        #Select a hierarchy
        logging.info('Selected %s hierarchy'%
                     ('normal' if args.normal else 'inverted'))
        params =  select_hierarchy(model_settings['params'],
                                   normal_hierarchy=args.normal)
        ebins = model_settings['binning']['ebins']
        czbins = model_settings['binning']['czbins']

        MC_error = GetMCError(get_values(params),ebins,czbins,args.reco_mc_file)

    profile.info("  ==> elapsed time to initialize: %s sec"%t.secs)

    #Now get the actual template
    with Timer(verbose=False) as t:
        template_maps = MC_error.get_mc_events_map(args.apply_reco_prcs, get_values(params),args.reco_mc_file)
    profile.info("==> elapsed time to get template: %s sec"%t.secs)

    logging.info("Saving file to %s"%args.outfile)
    to_json(template_maps, args.outfile)

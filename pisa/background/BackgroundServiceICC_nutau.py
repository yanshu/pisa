#
# Read from ICC background file and create background template for nutau analysis that
# needs different treatment of up and down-going template
#
# author: Timothy C. Arlen
#         Feifei Huang
#
# date:   July 27, 2014
#

import h5py
import numpy as np
import sys
from pisa.utils.log import logging
from pisa.resources.resources import find_resource
from pisa.utils import hdf

class BackgroundServiceICC:

    def __init__(self,ebins,czbins,energy_scale,aeff_scale, reco_mode,
                 reco_vbwkde_evts_file,reco_mc_wt_file, icc_bg_file=None, **kwargs):

        self.ebins = ebins
        self.czbins = czbins
        logging.info('Initializing BackgroundServiceICC...')

        if reco_mode == "MC": 
            fpath = find_resource(reco_mc_wt_file)
        if reco_mode == "vbwkde": 
            fpath = find_resource(reco_vbwkde_evts_file)
        eventsdict = hdf.from_hdf(fpath)

        # Find out if the file is down-going or not.
        if np.all(eventsdict['numu']['cc']['true_coszen']>=0):
            self.map_direction = 'down'
        elif np.all(eventsdict['numu']['cc']['true_coszen']<=0):
            self.map_direction = 'up'
        else:
            raise ValueError("reco_vbwkde_evts_file must be either up-going or down-going!")

        logging.info('Opening file: %s'%(icc_bg_file))
        try:
            bg_file = h5py.File(find_resource(icc_bg_file),'r')
        except IOError,e:
            logging.error("Unable to open icc_bg_file %s"%icc_bg_file)
            logging.error(e)
            sys.exit(1)

        self.icc_bg_dict = {}
        logging.info("Creating a ICC background dict...")

        dLLH = np.array(bg_file['IC86_Dunkman_L6']['delta_LLH'])
        reco_energy_all = np.array(bg_file['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['energy'])
        reco_coszen_all = np.array(np.cos(bg_file['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['zenith']))

        # throw away delta LLH < -3:
        reco_energy_all_cut1 = reco_energy_all[dLLH>=-3]
        reco_coszen_all_cut1 = reco_coszen_all[dLLH>=-3]
        dLLH_cut1 = dLLH[dLLH>=-3]

        if self.map_direction == 'up':
            reco_coszen_all_cut2 = reco_coszen_all_cut1[reco_coszen_all_cut1<=0.0]
            reco_energy_all_cut2 = reco_energy_all_cut1[reco_coszen_all_cut1<=0.0]
            dLLH_cut2 = dLLH_cut1[reco_coszen_all_cut1<=0.0]
        if self.map_direction == 'down':
            reco_coszen_all_cut2 = reco_coszen_all_cut1[reco_coszen_all_cut1>0.0]
            reco_energy_all_cut2 = reco_energy_all_cut1[reco_coszen_all_cut1>0.0]
            dLLH_cut2 = dLLH_cut1[reco_coszen_all_cut1>0.0]

        # write to dictionary
        for flavor in ['cscd','trck']:
            if flavor == 'cscd':
                cut = dLLH_cut2 < 3.0 
            if flavor == 'trck':
                cut = dLLH_cut2 >= 3.0 
            reco_energy = reco_energy_all_cut2[cut]*energy_scale
            reco_coszen = reco_coszen_all_cut2[cut]

            logging.debug("Working on %s background"%flavor)

            bins = (self.ebins,self.czbins)
            icc_bg_hist,_,_ = np.histogram2d(reco_energy,reco_coszen,bins=bins)

            self.icc_bg_dict[flavor] = icc_bg_hist*aeff_scale

        return

    def get_icc_bg(self,*kwargs):
        '''
        Returns the background dictionary
        '''
        return self.icc_bg_dict

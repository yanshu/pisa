#
# Read from ICC background file and create background template 
#
# author: Timothy C. Arlen
#         Feifei Huang
#
# date:   May 5, 2014
#

import h5py
import numpy as np
import sys
from pisa.utils.log import logging
from pisa.resources.resources import find_resource

class BackgroundServiceICC:

    def __init__(self,ebins,czbins,sim_ver,icc_bg_file=None,**kwargs):
        self.ebins = ebins
        self.czbins = czbins
        logging.info('Initializing BackgroundServiceICC...')

        logging.info('Opening file: %s'%(icc_bg_file))
        try:
            bg_file = h5py.File(find_resource(icc_bg_file),'r')
        except IOError,e:
            logging.error("Unable to open icc_bg_file %s"%icc_bg_file)
            logging.error(e)
            sys.exit(1)

        self.icc_bg_dict = {}
        logging.info("Creating a ICC background dict...")

        # check all level cuts are correct
        santa_doms = bg_file['IC86_Dunkman_L6_SANTA_DirectDOMs']['value']
        l3 = bg_file['IC86_Dunkman_L3']['value']
        l4 = bg_file['IC86_Dunkman_L4']['result']
        l5 = bg_file['IC86_Dunkman_L5']['bdt_score']
        assert(np.all(santa_doms>=3) and np.all(l3 == 1) and np.all(l4 == 1) and np.all(l5 >= 0.1))
        l6 = bg_file['IC86_Dunkman_L6']
        corridor_doms_over_threshold = l6['corridor_doms_over_threshold']
        inverted_corridor_cut = corridor_doms_over_threshold > 1
        assert(np.all(inverted_corridor_cut) and np.all(l6['santa_direct_doms'] >= 3) and np.all(l6['mn_start_contained'] == 1.) and np.all(l6['mn_stop_contained'] == 1.))

        dLLH = np.array(bg_file['IC86_Dunkman_L6']['delta_LLH'])
        if sim_ver == '4digit':
            reco_energy_all = np.array(bg_file['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['energy'])
            reco_coszen_all = np.array(np.cos(bg_file['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['zenith']))
            # throw away delta LLH < -3:
            reco_energy_all = reco_energy_all[dLLH>=-3]
            reco_coszen_all = reco_coszen_all[dLLH>=-3]
            dLLH = dLLH[dLLH>=-3]
            pid_cut = 3.0       # this cut value 3.0 is from MSU, might need to be optimized
        elif sim_ver == '5digit':
            reco_energy_all = np.array(bg_file['IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC']['energy'])
            reco_coszen_all = np.array(np.cos(bg_file['IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC']['zenith']))
            # throw away delta LLH < -2:
            #reco_energy_all = reco_energy_all[dLLH>=-2]
            #reco_coszen_all = reco_coszen_all[dLLH>=-2]
            #dLLH = dLLH[dLLH>=-2]
            pid_cut = 3.0       # this cut value 3.0 is from MSU, might need to be optimized
        else:
            raise ValueError('Only allow sim_ver  4digit or 5 digit!')

        # split in half for testing:
        #the commented out section was just a test for using subsets of the MC files
        #reco_energy_all = reco_energy_all[len(reco_energy_all)/2:] 
        #reco_coszen_all = reco_coszen_all[len(reco_coszen_all)/2:]
        #dLLH = dLLH[len(dLLH)/2:]
        #reco_energy_all = reco_energy_all[1::2]
        #reco_coszen_all = reco_coszen_all[1::2]
        #dLLH = dLLH[::2]

        # write to dictionary
        for flavor in ['cscd','trck']:
            if flavor == 'cscd':
                cut = dLLH < pid_cut 
            if flavor == 'trck':
                cut = dLLH >= pid_cut 
            reco_energy = reco_energy_all[cut]
            reco_coszen = reco_coszen_all[cut]

            flavor_dict = {}
            logging.debug("Working on %s background"%flavor)

            bins = (self.ebins,self.czbins)
            icc_bg_hist,_,_ = np.histogram2d(reco_energy,reco_coszen,bins=bins)

            self.icc_bg_dict[flavor] = icc_bg_hist

        return

    def get_icc_bg(self,*kwargs):
        '''
        Returns the background dictionary
        '''
        return self.icc_bg_dict


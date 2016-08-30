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

    def __init__(self,ebins,czbins,sim_ver,pid_bound,pid_remove,use_def1=False,icc_bg_file=None,**kwargs):
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
        print "icc_bg_file = ", icc_bg_file

        # sanity check 
        santa_doms = bg_file['IC86_Dunkman_L6_SANTA_DirectDOMs']['value']
        l3 = bg_file['IC86_Dunkman_L3']['value']
        l4 = bg_file['IC86_Dunkman_L4']['result']
        l5 = bg_file['IC86_Dunkman_L5']['bdt_score']
        if use_def1==True:
            l4_pass = np.all(l4==1)
        else:
            if sim_ver == 'dima' or sim_ver =='5digit':
                l4_invVICH = bg_file['IC86_Dunkman_L4']['result_invertedVICH']
                l4_pass = np.all(np.logical_or(l4==1, l4_invVICH==1))
            else:
                print "For the old simulation, def.2 background not done yet, so still use def1 for it."
                l4_pass = np.all(l4==1)
        assert(np.all(santa_doms>=3) and np.all(l3 == 1) and l4_pass and np.all(l5 >= 0.1))
        l6 = bg_file['IC86_Dunkman_L6']
        corridor_doms_over_threshold = l6['corridor_doms_over_threshold']
        inverted_corridor_cut = corridor_doms_over_threshold > 1
        assert(np.all(inverted_corridor_cut) and np.all(l6['santa_direct_doms'] >= 3) and np.all(l6['mn_start_contained'] == 1.) and np.all(l6['mn_stop_contained'] == 1.))

        dLLH = np.array(bg_file['IC86_Dunkman_L6']['delta_LLH'])
        if sim_ver == '4digit':
            reco_energy_all = np.array(bg_file['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['energy'])
            reco_coszen_all = np.array(np.cos(bg_file['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['zenith']))
        elif sim_ver == '5digit' or 'dima':
            reco_energy_all = np.array(bg_file['IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC']['energy'])
            reco_coszen_all = np.array(np.cos(bg_file['IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC']['zenith']))
        else:
            raise ValueError('Only allow sim_ver  4digit, 5 digit or dima!')

        # Cut1: throw away delta LLH < pid_remove:
        print "Cut1, removing event with LLH < pid_remove"
        cut1 = dLLH>=pid_remove
        reco_energy_cut1 = reco_energy_all[cut1]
        reco_coszen_cut1 = reco_coszen_all[cut1]
        dLLH_cut1 = dLLH[cut1]
        l5_cut1 = l5[cut1]
        pid_cut = pid_bound 
        print "pid_remove = ", pid_remove
        print "pid_bound = ", pid_bound

        # Cut2: Only keep bdt score >= 0.2 (from MSU latest result, make data/MC agree much better)
        print "Cut2, removing events with bdt_score < 0.2, i.e. only keep bdt > 0.2"
        cut2 = l5_cut1>=0.2
        reco_energy_cut2 = reco_energy_cut1[cut2]
        reco_coszen_cut2 = reco_coszen_cut1[cut2]
        dLLH_cut2 = dLLH_cut1[cut2]

        # split in half for testing:
        #the commented out section was just a test for using subsets of the MC files
        #reco_energy_cut2 = reco_energy_cut2[len(reco_energy_cut2)/2:] 
        #reco_coszen_cut2 = reco_coszen_cut2[len(reco_coszen_cut2)/2:]
        #dLLH_cut2 = dLLH_cut2[len(dLLH_cut2)/2:]
        #reco_energy_cut2 = reco_energy_cut2[1::2]
        #reco_coszen_cut2 = reco_coszen_cut2[1::2]
        #dLLH_cut2 = dLLH_cut2[::2]

        # Cut3: pid cut. Write to dictionary
        for flavor in ['cscd','trck']:
            if flavor == 'cscd':
                cut = dLLH_cut2 < pid_cut 
            if flavor == 'trck':
                cut = dLLH_cut2 >= pid_cut 
            reco_energy = reco_energy_cut2[cut]
            reco_coszen = reco_coszen_cut2[cut]

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


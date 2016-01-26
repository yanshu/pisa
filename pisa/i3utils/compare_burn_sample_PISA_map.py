#! /usr/bin/env python
#
# author: Timothy C. Arlen
#         Feifei Huang
#
# date: 20 Jan 2016
#
#   Compare maps of the burn sample in the form of the PISA final stage output with PISA MC expectation.
#

import copy
import numpy as np
import os
import h5py
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.analysis.TemplateMaker_nutau import TemplateMaker
from pisa.utils.params import get_values, select_hierarchy_and_nutau_norm
import pisa.analysis.stats.Maps as Maps
from pisa.analysis.stats.Maps_nutau import get_up_map, get_flipped_down_map
from pisa.utils.log import set_verbosity,logging,profile
from pisa.resources.resources import find_resource
from pisa.utils.utils import Timer, is_linear, is_logarithmic, get_bin_centers
from pisa.utils.jsons import from_json
from pisa.utils.plot import show_map, sum_map, ratio_map, delta_map
from pisa.background.BackgroundServiceICC_nutau import BackgroundServiceICC

def get_1D_projection(map_2d, axis):
    if axis == 'coszen':
        output_array = np.zeros(map_2d.shape[1])
        for i in range(0, map_2d.shape[0]):
            output_array += map_2d[i,:]
    if axis == 'energy':
        output_array = np.zeros(map_2d.shape[0])
        for i in range(0, map_2d.shape[1]):
            output_array += map_2d[:,i]
    return output_array

def plot_burn_sample_MC_comparison(MC_nutau, MC_no_nutau, BS_data, MC_nutau_name, MC_no_nutau_name, BS_name, x_bin_centers, x_bin_edges, channel, x_label):
    fig = plt.figure(figsize=(8,10))
    ax1 = fig.add_subplot(2,1,1)
    hist_MC_tau,_,_ = ax1.hist(x_bin_centers,weights= MC_nutau,bins=x_bin_edges,histtype='step',lw=2,color='b',label= MC_nutau_name,linestyle='solid',normed=args.norm)
    hist_MC_notau,_,_ = ax1.hist(x_bin_centers,weights=MC_no_nutau,bins=x_bin_edges,histtype='step',lw=2,color='g',label= MC_no_nutau_name,linestyle='dashed',normed=args.norm)
    hist_BS,_,_ = ax1.hist(x_bin_centers,weights=BS_data,bins=x_bin_edges,histtype='step',lw=2,color='r',label= BS_name,normed=args.norm)
    ax1.errorbar(x_bin_centers,hist_MC_tau,yerr=np.sqrt(hist_MC_tau),fmt='.b')
    ax1.errorbar(x_bin_centers,hist_MC_notau,yerr=np.sqrt(hist_MC_notau),fmt='.g')
    ax1.errorbar(x_bin_centers,hist_BS,yerr=np.sqrt(hist_BS),fmt='.r')
    if (channel == 'cscd' or channel == 'cscd+trck') and x_label == 'energy':
        ax1.legend(loc='upper right',ncol=1)
    else:
        ax1.legend(loc='upper center',ncol=2)
    ax1.set_ylim(min(min(min(hist_BS),min(hist_MC_notau)),min(hist_MC_tau))-10,max(max(max(hist_BS),max(hist_MC_notau)),max(hist_MC_tau))+40)
    ax1.grid()

    ax2 = fig.add_subplot(2,1,2)
    hist_ratio_BS_to_MC_tau = ax2.hist(x_bin_centers, weights=hist_BS/hist_MC_tau,bins=x_bin_edges,histtype='step',lw=2,color='b', linestyle='solid', label='Burn Sample/MC tau')
    hist_ratio_BS_to_MC_notau = ax2.hist(x_bin_centers, weights=hist_BS/hist_MC_notau, bins=x_bin_edges,histtype='step',lw=2,color='g', linestyle='dashed', label = 'Burn Sample/ MC notau')
    if x_label == 'energy':
        ax2.set_xlabel('energy [GeV]')
    if x_label == 'coszen':
        ax2.set_xlabel('coszen')
    ax2.set_ylabel("Burn Sample / MC")
    ax2.set_ylim(min(hist_BS/hist_MC_notau)-0.1,max(hist_BS/hist_MC_notau)+0.1)
    ax2.axhline(y=1,linewidth=1, color='k')
    ax2.legend(loc='upper center',ncol=2)
    ax2.grid()
    plt.savefig(args.outdir+"BurnSample_MC_%s_%s_distribution.png" % (channel, x_label),dpi=150)
    #plt.show()
    plt.clf()


if __name__ == '__main__':

    set_verbosity(0)
    parser = ArgumentParser(description='''Quick check if all components are working reasonably well, by
making the final level hierarchy asymmetry plots from the input
settings file. ''')
    parser.add_argument('template_settings',metavar='JSON',
                        help='Settings file to use for template generation')
    parser.add_argument('--burn_sample_file',metavar='FILE',type=str,
                        default='burn_sample/Matt_L5b_burn_sample_IC86_2_to_4.hdf5',
                        help='''HDF5 File containing burn sample.'
                        inverted corridor cut data''')
    parser.add_argument('--background_file',metavar='FILE',type=str,
                        default='background/Matt_L5b_icc_data_IC86_2_3_4.hdf5',
                        help='''HDF5 File containing atmospheric background from 3 years'
                        inverted corridor cut data''')
    parser.add_argument('--bg_scale',type=float,
                        help="atmos background scale value")
    parser.add_argument('-logE','--logE',action='store_true',default=False,
                        help='Energy in log scale.')
    parser.add_argument('-norm','--norm',action='store_true',default=False,
                        help='Normalize the histogram.')
    parser.add_argument('-a','--all',action='store_true',default=False,
                        help='Plot all stages 1-5 of templates and Asymmetry')
    parser.add_argument('--title',metavar='str',default='',
                        help='Title of the geometry or test in plots')
    parser.add_argument('--save',action='store_true',default=False,
                        help='Save plots in outdir')
    parser.add_argument('-o','--outdir',metavar='DIR',default='',
                        help='Directory to save the output figures.')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='set verbosity level')
    args = parser.parse_args()
    set_verbosity(args.verbose)

    template_settings = from_json(args.template_settings)
    template_settings['params']['icc_bg_file']['value'] = find_resource(args.background_file)
    template_settings['params']['atmos_mu_scale']['value'] = args.bg_scale

    ebins = template_settings['binning']['ebins']
    czbins = template_settings['binning']['czbins']

    print "ebins = ", ebins
    print "czbins = ", czbins
    CZ_bin_centers = get_bin_centers(czbins)
    E_bin_centers = get_bin_centers(ebins)
    print "E_bin_centers = ", E_bin_centers
    print "CZ_bin_centers = ", CZ_bin_centers

    burn_sample_file = h5py.File(find_resource(args.burn_sample_file),'r')

    dLLH = np.array(burn_sample_file['IC86_Dunkman_L6']['delta_LLH'])
    L6_result = np.array(burn_sample_file['IC86_Dunkman_L6']['result'])
    reco_energy_all = np.array(burn_sample_file['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['energy'])
    reco_coszen_all = np.array(np.cos(burn_sample_file['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['zenith']))

    print "before L6 cut, no. of burn sample = ", len(reco_coszen_all)

    dLLH_L6 = dLLH[L6_result==1]
    reco_energy_L6 = reco_energy_all[L6_result==1]
    reco_coszen_L6 = reco_coszen_all[L6_result==1]
   
    # throw away dLLH < -3
    #reco_energy_L6_cut1 = reco_energy_L6[dLLH_L6>=-3]
    #reco_coszen_L6_cut1 = reco_coszen_L6[dLLH_L6>=-3]
    #dLLH_L6_cut1 = dLLH_L6[dLLH_L6>=-3]
    reco_energy_L6_cut1 = reco_energy_L6
    reco_coszen_L6_cut1 = reco_coszen_L6
    dLLH_L6_cut1 = dLLH_L6
    print "after L6 cut, no. of burn sample = ", len(dLLH_L6_cut1)

    #if self.map_direction == 'up':
    #    reco_coszen_all_cut2 = reco_coszen_L6[reco_coszen_L6<=0.0]
    #    reco_energy_all_cut2 = reco_energy_L6[reco_coszen_L6<=0.0]
    #    dLLH_cut2 = dLLH_cut1[reco_coszen_L6<=0.0]
    #if self.map_direction == 'down':
    #    reco_coszen_all_cut2 = reco_coszen_L6[reco_coszen_L6>0.0]
    #    reco_energy_all_cut2 = reco_energy_L6[reco_coszen_L6>0.0]
    #    dLLH_cut2 = dLLH_cut1[reco_coszen_L6>0.0]

    # write burn sample data to dictionary
    burn_sample_dict = {}
    for flavor in ['cscd','trck']:
        if flavor == 'cscd':
            cut = dLLH_L6_cut1 < 3.0 
        if flavor == 'trck':
            cut = dLLH_L6_cut1 >= 3.0 
        reco_energy_L6_final = reco_energy_L6_cut1[cut]
        reco_coszen_L6_final = reco_coszen_L6_cut1[cut]

        bins = (ebins, czbins)
        burn_sample_hist,_,_ = np.histogram2d(reco_energy_L6_final,reco_coszen_L6_final,bins=bins)
        #print flavor, " burn_sample_hist = ", burn_sample_hist
        burn_sample_dict[flavor] = burn_sample_hist

    # get the burn sample maps (cz in [-1, 1])
    burn_sample_maps={}
    for flav in ['trck','cscd']:
        burn_sample_maps[flav] = {'map':burn_sample_dict[flav],
                                 'ebins':ebins,
                                 'czbins':czbins}

    burn_sample_map_up = get_up_map(burn_sample_maps, channel=template_settings['params']['channel']['value'])
    burn_sample_map_flipped_down = get_flipped_down_map(burn_sample_maps, channel=template_settings['params']['channel']['value'])
    flattend_burn_sample_map_up = Maps.flatten_map(burn_sample_map_up, channel=template_settings['params']['channel']['value'])
    flattend_burn_sample_map_flipped_down = Maps.flatten_map(burn_sample_map_flipped_down, channel=template_settings['params']['channel']['value'])
    burn_sample_in_array = np.append(flattend_burn_sample_map_up, flattend_burn_sample_map_flipped_down)

    #print "burn_sample_map_flipped_down = ", burn_sample_map_flipped_down
    #print "flattend_burn_sample_map_up = ", flattend_burn_sample_map_up
    #print "flattend_burn_sample_map_flipped_down = ", flattend_burn_sample_map_flipped_down
    #print "burn_sample_in_array = ", burn_sample_in_array
    print "     no. of events in upgoing burn sample: ", np.sum(flattend_burn_sample_map_up)
    print "     no. of events in downgoing burn sample: ", np.sum(flattend_burn_sample_map_flipped_down)
    print "     total no. of events in burn sample :", np.sum(burn_sample_in_array) 

    plt.figure()
    show_map(burn_sample_maps['cscd'],vmax=15,logE=args.logE)
    if args.save:
        filename = os.path.join(args.outdir,args.title+ '_burn_sample_up_down_combined_'+ 'cscd_5.6_56GeV.png')
        plt.title(r'${\rm 0.05 \, yr \, burn \, sample \, cscd \, (Nevts: \, %.1f) }$'%(np.sum(burn_sample_maps['cscd']['map'])), fontsize='large')
        plt.savefig(filename,dpi=150)
        plt.clf()

    plt.figure()
    show_map(burn_sample_maps['trck'],vmax=10,logE=args.logE)
    if args.save:
        filename = os.path.join(args.outdir,args.title+ '_burn_sample_up_down_combined_'+ 'trck_5.6_56GeV.png')
        plt.title(r'${\rm 0.05 \, yr \, burn \, sample \, trck \, (Nevts: \, %.1f) }$'%(np.sum(burn_sample_maps['trck']['map'])), fontsize='large')
        plt.savefig(filename,dpi=150)
        plt.clf()



    ##################### Plot MC expectation #######################

    nominal_up_template_settings = copy.deepcopy(template_settings)
    nominal_up_template_settings['params']['reco_mc_wt_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_100_percent_up.hdf5'}
    nominal_up_template_settings['params']['reco_vbwkde_evts_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_10_percent_up.hdf5'}

    nominal_down_template_settings = copy.deepcopy(template_settings)
    nominal_down_template_settings['params']['reco_mc_wt_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_100_percent_down.hdf5'}
    nominal_down_template_settings['params']['pid_paramfile'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/pid/1X60_pid_down.json'}
    nominal_down_template_settings['params']['reco_vbwkde_evts_file'] = {u'fixed': True, u'value': '~/pisa/pisa/resources/events/1X60_weighted_aeff_joined_nu_nubar_10_percent_down.hdf5'}

    with Timer() as t:
        nominal_template_maker_down = TemplateMaker(get_values(nominal_down_template_settings['params']), **nominal_down_template_settings['binning'])
        nominal_template_maker_up = TemplateMaker(get_values(nominal_up_template_settings['params']), **nominal_up_template_settings['binning'])

    profile.info('==> elapsed time to initialize templates: %s sec'%t.secs)

    # Make nutau template:
    nominal_nutau_up_params = copy.deepcopy(select_hierarchy_and_nutau_norm( nominal_up_template_settings['params'],True,1.0))
    nominal_nutau_down_params = copy.deepcopy(select_hierarchy_and_nutau_norm( nominal_down_template_settings['params'],True,1.0))
    nominal_no_nutau_up_params = copy.deepcopy(select_hierarchy_and_nutau_norm( nominal_up_template_settings['params'],True,0.0))
    nominal_no_nutau_down_params = copy.deepcopy(select_hierarchy_and_nutau_norm( nominal_down_template_settings['params'],True,0.0))

    with Timer(verbose=False) as t:
        nominal_nutau_up = nominal_template_maker_up.get_template(get_values(nominal_nutau_up_params),return_stages=args.all)
        nominal_nutau_down = nominal_template_maker_down.get_template(get_values(nominal_nutau_down_params),return_stages=args.all)
        nominal_no_nutau_up = nominal_template_maker_up.get_template(get_values(nominal_no_nutau_up_params),return_stages=args.all)
        nominal_no_nutau_down = nominal_template_maker_down.get_template(get_values(nominal_no_nutau_down_params),return_stages=args.all)
    profile.info('==> elapsed time to get NUTAU template: %s sec'%t.secs)

    #print 'nominal nutau_up = ',   nominal_nutau_up
    #print 'nominal nutau_down = ', nominal_nutau_down

    nominal_nutau_up_and_down_cscd = sum_map(nominal_nutau_up['cscd'], nominal_nutau_down['cscd'])
    #print "nominal_nutau_up_and_down_cscd = ", nominal_nutau_up_and_down_cscd
    nominal_nutau_up_and_down_trck = sum_map(nominal_nutau_up['trck'], nominal_nutau_down['trck'])
    #print "nominal_nutau_up_and_down_trck = ", nominal_nutau_up_and_down_trck
    nominal_no_nutau_up_and_down_cscd = sum_map(nominal_no_nutau_up['cscd'], nominal_no_nutau_down['cscd'])
    nominal_no_nutau_up_and_down_trck = sum_map(nominal_no_nutau_up['trck'], nominal_no_nutau_down['trck'])



    #### Plot nutau_up , nutau_down, etc. ####
    #for channel in ['trck','cscd']:
    #    plt.figure()
    #    show_map(nominal_nutau_up[channel],vmax=25 if channel=='cscd' else 10,logE=args.logE)
    #    print 'no. of upgoing ' ,channel , ' ', np.sum(nominal_nutau_up[channel]['map'])
    #    if args.save:
    #        print 'Saving %s channel...'%channel
    #        filename = os.path.join(args.outdir,args.title+ '_f_1_up_'+channel+'.png')
    #        plt.title(channel + ' (up) Nevts: %.1f '%(np.sum(nominal_nutau_up[channel]['map'])))
    #        plt.savefig(filename,dpi=150)
    #        plt.clf()

    #    show_map(nominal_nutau_down[channel],vmax=10 if channel=='cscd' else 10,logE=args.logE)
    #    print 'no. of downgoing ', channel , ' ', np.sum(nominal_nutau_down[channel]['map'])
    #    if args.save:
    #        print 'Saving %s channel...'%channel
    #        filename = os.path.join(args.outdir,args.title+ '_f_1_down_'+channel+'.png')
    #        plt.title(channel + ' (down) Nevts: %.1f '%(np.sum(nominal_nutau_down[channel]['map'])))
    #        plt.savefig(filename,dpi=150)
    #        plt.clf()

    plt.figure()
    show_map(nominal_nutau_up_and_down_cscd,vmax=15,logE=args.logE)
    if args.save:
        filename = os.path.join(args.outdir,args.title+ '_NutauCCNorm_1_up_down_combined_'+ 'cscd_5.6_56GeV.png')
        plt.title(r'${\rm 0.05 \, yr \, cscd \, (Nevts: \, %.1f) }$'%(np.sum(nominal_nutau_up_and_down_cscd['map'])), fontsize='large')
        plt.savefig(filename,dpi=150)
        plt.clf()

    plt.figure()
    show_map(nominal_nutau_up_and_down_trck,vmax=10,logE=args.logE)
    if args.save:
        filename = os.path.join(args.outdir,args.title+ '_NutauCCNorm_1_up_down_combined_'+ 'trck_5.6_56GeV.png')
        plt.title(r'${\rm 0.05 \, yr \, trck \, (Nevts: \, %.1f) }$'%(np.sum(nominal_nutau_up_and_down_trck['map'])), fontsize='large')
        plt.savefig(filename,dpi=150)
        plt.clf()

    print "no. of nominal_nutau_up_and_down_cscd = ", np.sum(nominal_nutau_up_and_down_cscd['map'])
    print "no. of nominal_nutau_up_and_down_trck = ", np.sum(nominal_nutau_up_and_down_trck['map'])
    print " total of the above two : ", np.sum(nominal_nutau_up_and_down_cscd['map'])+np.sum(nominal_nutau_up_and_down_trck['map'])
    print " \n"
    print "no. of nominal_no_nutau_up_and_down_cscd = ", np.sum(nominal_no_nutau_up_and_down_cscd['map'])
    print "no. of nominal_no_nutau_up_and_down_trck = ", np.sum(nominal_no_nutau_up_and_down_trck['map'])
    print " total of the above two : ", np.sum(nominal_no_nutau_up_and_down_cscd['map'])+np.sum(nominal_no_nutau_up_and_down_trck['map'])
    print " \n"
    #print "max no. of evts in nutau_up_cscd: ", np.amax(nominal_nutau_up['cscd']['map'])
    #print "max no. of evts in nutau_up_trck: ", np.amax(nominal_nutau_up['trck']['map'])
    #print "max no. of evts in nutau_down_cscd: ", np.amax(nominal_nutau_down['cscd']['map'])
    #print "max no. of evts in nutau_down_trck: ", np.amax(nominal_nutau_down['trck']['map'])
    #print "max no. of evts in nominal_nutau_up_and_down_cscd: ", np.amax(nominal_nutau_up_and_down_cscd['map'])
    #print "max no. of evts in nominal_nutau_up_and_down_trck: ", np.amax(nominal_nutau_up_and_down_trck['map'])
    no_of_up = np.sum(nominal_nutau_up['trck']['map']) + np.sum(nominal_nutau_up['cscd']['map'])
    no_of_down = np.sum(nominal_nutau_down['trck']['map']) + np.sum(nominal_nutau_down['cscd']['map'])
    print 'no. of upgoing events= ' , no_of_up 
    print 'no. of downgoing events = ' , no_of_down 
    print 'Total no. of events = ' , no_of_up + no_of_down
    no_of_up_and_down = np.sum(nominal_nutau_up_and_down_trck['map']) + np.sum(nominal_nutau_up_and_down_cscd['map'])
    print 'From map_up_down, Total no. of events = ' , no_of_up_and_down
   

    ################ Plot background ##################
    # get background
    czbins = nominal_up_template_settings['binning']['czbins']
    up_background_service = BackgroundServiceICC(nominal_up_template_settings['binning']['ebins'],czbins[czbins<=0],**get_values(nominal_nutau_up_params))
    up_background_dict = up_background_service.get_icc_bg()
    #print "up_background_dict = ", up_background_dict
    down_background_service = BackgroundServiceICC(nominal_down_template_settings['binning']['ebins'],czbins[czbins>=0],**get_values(nominal_nutau_down_params))
    down_background_dict = down_background_service.get_icc_bg()
    #print "down_background_dict = ", down_background_dict

    up_background_maps = {'params': nominal_nutau_up['params']}
    for flav in ['trck','cscd']:
        up_background_maps[flav] = {'map':up_background_dict[flav],
                                 'ebins':nominal_up_template_settings['binning']['ebins'],
                                 'czbins':czbins[czbins<=0]}
    down_background_maps = {'params': nominal_nutau_down['params']}
    for flav in ['trck','cscd']:
        down_background_maps[flav] = {'map':down_background_dict[flav],
                                 'ebins':nominal_down_template_settings['binning']['ebins'],
                                 'czbins':czbins[czbins>=0]}

    #for channel in ['trck','cscd']:
    #    plt.figure()
    #    show_map(up_background_maps[channel],logE=args.logE)
    #    if args.save:
    #        filename = os.path.join(args.outdir,args.title+'_upgoing_background_'+channel+'.png')
    #        plt.title(args.title+'_upgoing_background_'+channel)
    #        plt.savefig(filename,dpi=150)
    #        plt.clf()
    #    plt.figure()
    #    show_map(down_background_maps[channel],logE=args.logE)
    #    if args.save:
    #        filename = os.path.join(args.outdir,args.title+'_downgoing_background_'+channel+'.png')
    #        plt.title(args.title+'_downgoing_background_'+channel)
    #        plt.savefig(filename,dpi=150)
    #        plt.clf()


    ################## PLOT MC/Data comparison ##################

    # reco_energy_L6_final: the burn sample reco. energy

    burn_sample_up_and_down_cscd_map = burn_sample_maps['cscd']['map']
    burn_sample_up_and_down_trck_map = burn_sample_maps['trck']['map']

    ##### Plot energy distribution #####
    BurnSample_RecoEnergy_up_and_down_cscd = get_1D_projection(burn_sample_up_and_down_cscd_map, 'energy')
    BurnSample_RecoEnergy_up_and_down_trck = get_1D_projection(burn_sample_up_and_down_trck_map, 'energy')
    BurnSample_RecoCoszen_up_and_down_cscd = get_1D_projection(burn_sample_up_and_down_cscd_map, 'coszen')
    BurnSample_RecoCoszen_up_and_down_trck = get_1D_projection(burn_sample_up_and_down_trck_map, 'coszen')

    nominal_nutau_up_and_down_cscd_map = nominal_nutau_up_and_down_cscd['map']
    MC_RecoEnergy_nominal_nutau_up_and_down_cscd = get_1D_projection(nominal_nutau_up_and_down_cscd_map, 'energy')
    MC_RecoCoszen_nominal_nutau_up_and_down_cscd = get_1D_projection(nominal_nutau_up_and_down_cscd_map, 'coszen')
    nominal_nutau_up_and_down_trck_map = nominal_nutau_up_and_down_trck['map']
    MC_RecoEnergy_nominal_nutau_up_and_down_trck = get_1D_projection(nominal_nutau_up_and_down_trck_map, 'energy')
    MC_RecoCoszen_nominal_nutau_up_and_down_trck = get_1D_projection(nominal_nutau_up_and_down_trck_map, 'coszen')

    nominal_no_nutau_up_and_down_cscd_map = nominal_no_nutau_up_and_down_cscd['map']
    MC_RecoEnergy_nominal_no_nutau_up_and_down_cscd = get_1D_projection(nominal_no_nutau_up_and_down_cscd_map, 'energy')
    MC_RecoCoszen_nominal_no_nutau_up_and_down_cscd = get_1D_projection(nominal_no_nutau_up_and_down_cscd_map, 'coszen')
    nominal_no_nutau_up_and_down_trck_map = nominal_no_nutau_up_and_down_trck['map']
    MC_RecoEnergy_nominal_no_nutau_up_and_down_trck = get_1D_projection(nominal_no_nutau_up_and_down_trck_map, 'energy')
    MC_RecoCoszen_nominal_no_nutau_up_and_down_trck = get_1D_projection(nominal_no_nutau_up_and_down_trck_map, 'coszen')

    #print "MC_RecoEnergy_nominal_nutau_up_and_down_cscd = ", MC_RecoEnergy_nominal_nutau_up_and_down_cscd
    #print "MC_RecoEnergy_nominal_nutau_up_and_down_trck = ", MC_RecoEnergy_nominal_nutau_up_and_down_trck
    MC_RecoEnergy_nominal_nutau_up_and_down_all_chan = MC_RecoEnergy_nominal_nutau_up_and_down_cscd + MC_RecoEnergy_nominal_nutau_up_and_down_trck
    MC_RecoCoszen_nominal_nutau_up_and_down_all_chan = MC_RecoCoszen_nominal_nutau_up_and_down_cscd + MC_RecoCoszen_nominal_nutau_up_and_down_trck

    MC_RecoEnergy_nominal_no_nutau_up_and_down_all_chan = MC_RecoEnergy_nominal_no_nutau_up_and_down_cscd + MC_RecoEnergy_nominal_no_nutau_up_and_down_trck
    MC_RecoCoszen_nominal_no_nutau_up_and_down_all_chan = MC_RecoCoszen_nominal_no_nutau_up_and_down_cscd + MC_RecoCoszen_nominal_no_nutau_up_and_down_trck

    BurnSample_RecoEnergy_up_and_down_all_chan = BurnSample_RecoEnergy_up_and_down_cscd + BurnSample_RecoEnergy_up_and_down_trck
    BurnSample_RecoCoszen_up_and_down_all_chan = BurnSample_RecoCoszen_up_and_down_cscd + BurnSample_RecoCoszen_up_and_down_trck

    #print "MC_RecoEnergy_nominal_no_nutau_up_and_down_cscd = ", MC_RecoEnergy_nominal_no_nutau_up_and_down_cscd
    #print "BurnSample_RecoEnergy_up_and_down_cscd : ", BurnSample_RecoEnergy_up_and_down_cscd
    #print "\n"
    #print "MC_RecoEnergy_nominal_no_nutau_up_and_down_trck = ", MC_RecoEnergy_nominal_no_nutau_up_and_down_trck
    #print "BurnSample_RecoEnergy_up_and_down_trck : ", BurnSample_RecoEnergy_up_and_down_trck
    
    plot_burn_sample_MC_comparison( MC_RecoEnergy_nominal_nutau_up_and_down_cscd, MC_RecoEnergy_nominal_no_nutau_up_and_down_cscd, BurnSample_RecoEnergy_up_and_down_cscd, 'MC cscd (nutau)', 'MC cscd (no nutau)', 'BurnSample cscd', E_bin_centers, ebins, 'cscd', 'energy')

    plot_burn_sample_MC_comparison( MC_RecoCoszen_nominal_nutau_up_and_down_cscd, MC_RecoCoszen_nominal_no_nutau_up_and_down_cscd, BurnSample_RecoCoszen_up_and_down_cscd, 'MC cscd (nutau)', 'MC cscd (no nutau)', 'BurnSample cscd', CZ_bin_centers, czbins, 'cscd', 'coszen')

    plot_burn_sample_MC_comparison( MC_RecoEnergy_nominal_nutau_up_and_down_trck, MC_RecoEnergy_nominal_no_nutau_up_and_down_trck, BurnSample_RecoEnergy_up_and_down_trck, 'MC trck (nutau)', 'MC trck (no nutau)', 'BurnSample trck', E_bin_centers, ebins, 'trck', 'energy')

    plot_burn_sample_MC_comparison( MC_RecoCoszen_nominal_nutau_up_and_down_trck, MC_RecoCoszen_nominal_no_nutau_up_and_down_trck, BurnSample_RecoCoszen_up_and_down_trck, 'MC trck (nutau)', 'MC trck (no nutau)', 'BurnSample trck', CZ_bin_centers, czbins, 'trck', 'coszen')

    plot_burn_sample_MC_comparison( MC_RecoEnergy_nominal_nutau_up_and_down_all_chan, MC_RecoEnergy_nominal_no_nutau_up_and_down_all_chan, BurnSample_RecoEnergy_up_and_down_all_chan, 'MC cscd+trck (nutau)', 'MC cscd+trck (no nutau)', 'BurnSample cscd+trck', E_bin_centers, ebins, 'cscd+trck', 'energy')

    plot_burn_sample_MC_comparison( MC_RecoCoszen_nominal_nutau_up_and_down_all_chan, MC_RecoCoszen_nominal_no_nutau_up_and_down_all_chan, BurnSample_RecoCoszen_up_and_down_all_chan, 'MC cscd+trck (nutau)', 'MC cscd+trck (no nutau)', 'BurnSample cscd+trck', CZ_bin_centers, czbins, 'cscd+trck', 'coszen')
    if not args.save: plt.show()
    else: print '\n-->>Saved all files to: ',args.outdir



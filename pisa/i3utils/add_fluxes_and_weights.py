#! /usr/bin/env python
import numpy as np
import os.path
import h5py
import copy
import pisa.resources.resources as resources
from pisa.flux.IPHondaFluxService_MC_merge import IPHondaFluxService
from pisa.oscillations.Prob3OscillationServiceMC_merge import Prob3OscillationServiceMC
from pisa.utils.log import set_verbosity,logging,profile
from pisa.utils.params import get_values, select_hierarchy
from pisa.utils.jsons import from_json
from pisa.utils.hdf import from_hdf, to_hdf
import pisa.utils.utils as utils
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pisa.utils.utils import Timer, oversample_binning
CMSQ_TO_MSQ = 1.0e-4

def add_fluxes_to_file(data_file_path, file_type, phys_params, flux_service, osc_service, neutrino_weight_name, outdir, add_weights=False):
    print 'data_file_path = ', data_file_path
    if file_type == 'pisa':
        data_file, attrs = from_hdf(resources.find_resource(data_file_path), return_attrs = True)
        for prim in data_file.keys():
            for int_type in data_file[prim].keys():
                true_e = data_file[prim][int_type]['true_energy']
                true_cz = data_file[prim][int_type]['true_coszen']
                isbar = '_bar' if 'bar' in prim else ''
                nue_flux = flux_service.get_flux(true_e, true_cz, 'nue'+isbar, event_by_event=True)
                numu_flux = flux_service.get_flux(true_e, true_cz, 'numu'+isbar, event_by_event=True)
                # the opposite flavor fluxes( used only in the nu_nubar_ratio systematic)
                oppo_isbar = '' if 'bar' else in prim '_bar'
                oppo_nue_flux = flux_service.get_flux(true_e, true_cz, 'nue'+oppo_isbar, event_by_event=True)
                oppo_numu_flux = flux_service.get_flux(true_e, true_cz, 'numu'+oppo_isbar, event_by_event=True)
                data_file[prim][int_type][neutrino_weight_name+'_nue_flux'] = nue_flux
                data_file[prim][int_type][neutrino_weight_name+'_numu_flux'] = numu_flux
                data_file[prim][int_type][neutrino_weight_name+'_oppo_nue_flux'] = oppo_nue_flux
                data_file[prim][int_type][neutrino_weight_name+'_oppo_numu_flux'] = oppo_numu_flux
                # if need to calculate neutrino weights before hand
                if add_weights:
                    osc_probs = osc_service.fill_osc_prob(true_e, true_cz, event_by_event=True, **phys_params)
                    osc_flux = nue_flux*osc_probs['nue'+isbar+'_maps'][prim]+ numu_flux*osc_probs['numu'+isbar+'_maps'][prim]
                    data_file[prim][int_type][neutrino_weight_name+'_weight'] = osc_flux * data_file[prim][int_type]['weighted_aeff'] 
        data_file_name = os.path.basename(data_file_path)
        utils.mkdir(args.outdir)
        output_file_name = outdir + '/' + data_file_name.split('.hdf5')[0]+'_with_weights.hdf5' 
        to_hdf(data_file, output_file_name, attrs=attrs, overwrite=True)

    elif file_type == 'intermediate':
        data_file = h5py.File(resources.find_resource(data_file_path), 'r+')
        true_e = data_file['trueNeutrino']['energy']
        true_cz = np.cos(data_file['trueNeutrino']['zenith'])
        one_weight = data_file['I3MCWeightDict']['OneWeight']
        pdg_encoding = data_file['trueNeutrino']['pdg_encoding']
        nuDict = {'12':'nue', '14':'numu', '16':'nutau', '-12': 'nue_bar', '-14': 'numu_bar', '-16': 'nutau_bar'}
        # 4 digit simulation nugen:
        n_files = {'nue': 2700, 'numu': 4000, 'nutau': 1400}
        ngen = {}
        for nu in ['nue', 'numu', 'nutau']:
            ngen[nu] = n_files[nu]*300003*0.5 
            ngen[nu+'_bar'] = n_files[nu]*300003*0.5 
        prim = [nuDict[str(pdg)] for pdg in pdg_encoding]
        ngen_array = [ngen[flav] for flav in prim]
        # calculate sim weights
        weighted_aeff = (one_weight/ngen_array)* CMSQ_TO_MSQ
        # calculate fluxes
        flux_name_nue = ['nue_bar' if pdg<0 else 'nue' for pdg in pdg_encoding]
        flux_name_numu = ['numu_bar' if pdg<0 else 'nue' for pdg in pdg_encoding]
        with Timer(verbose=False) as t:
            nue_flux = flux_service.get_flux(true_e, true_cz, flux_name_nue, event_by_event=True)
            numu_flux = flux_service.get_flux(true_e, true_cz,flux_name_numu, event_by_event=True)
            osc_probs = osc_service.fill_osc_prob(true_e, true_cz, prim, event_by_event=True, **phys_params)
            osc_flux = nue_flux*osc_probs['nue_maps'] + numu_flux*osc_probs['numu_maps']
        print 'time = ', t.secs
        data_file[prim][int_type][neutrino_weight_name+'_nue_flux'] = nue_flux
        data_file[prim][int_type][neutrino_weight_name+'_numu_flux'] = numu_flux
        data_file[neutrino_weight_name+'_weight'] = osc_flux * weighted_aeff
        data_file.close()
    else:
        raise ValueError('file_type only allow: pisa, intermediate')


if __name__ == '__main__':

    parser = ArgumentParser(description='''Add neutrino fluxes (and neutrino weights(osc*flux*sim_weight) if needed) for each event. ''')
    parser_file = parser.add_mutually_exclusive_group(required=True)
    parser_file.add_argument( '-fp', '--pisa_file', metavar='H5_FILE', type=str, help='input HDF5 file')
    parser_file.add_argument( '-fi', '--intermediate_file', metavar='H5_FILE', type=str, help='input HDF5 file, only works for old simulation.')
    parser.add_argument( '-t', '--template_settings',metavar='JSON',
        help='''Settings file that contains informatino of flux file, oscillation
        parameters, PREM model, etc., for calculation of neutrino weights''')
    parser.add_argument('--use_best_fit',action='store_true',default=False,
                        help='Use best fit params to calculate fluxes.')
    parser.add_argument('--add_weights',action='store_true',default=False,
                        help='Calculate and store neutrino weights.')
    parser.add_argument('--profile', '--profile-results', default=None, dest='fit_file_profile',
                        help='use post fit parameters from profile fit result json file')
    parser.add_argument('-o','--outdir',metavar='DIR',default='',
                        help='Directory to save the output figures.')
    args = parser.parse_args()
    set_verbosity(0)

    # get file name
    if args.pisa_file is not None:
        hd5_file_name = args.pisa_file
        file_type = 'pisa'
    else:
        hd5_file_name = args.intermediate_file
        file_type = 'intermediate'
        print 'The input is intermediate file, make sure it is the old simulation (4 digit).'

    outdir = args.outdir
    
    # get template settings
    template_settings = from_json(args.template_settings)
    phys_params = get_values(select_hierarchy(template_settings['params'], normal_hierarchy=True))

    if args.use_best_fit:
        free_nutau_template_settings = copy.deepcopy(template_settings)
        # replace with parameters determined in fit
        fit_file_profile = from_json(resources.find_resource(args.fit_file_profile))
        syslist = fit_file_profile['trials'][0]['fit_results'][1].keys()
        for sys in syslist:
            if not sys == 'llh':
                val = fit_file_profile['trials'][0]['fit_results'][1][sys][0]
                if sys == 'theta23' or sys =='deltam23' or sys =='deltam31':
                    sys += '_nh'
                print 'fit nutauCCnorm=free, %s at %.4f'%(sys,val)
                free_nutau_template_settings['params'][sys]['value'] = val
        phys_params = get_values(select_hierarchy(template_settings['params'], normal_hierarchy=True))

    # flux and osc service
    flux_service = IPHondaFluxService(**phys_params)
    osc_service = Prob3OscillationServiceMC([],[],**phys_params)

    if args.use_best_fit:
        add_fluxes_to_file(hd5_file_name, file_type, phys_params, flux_service, osc_service, neutrino_weight_name='neutrino_best_fit', outdir=outdir)
    else:
        add_fluxes_to_file(hd5_file_name, file_type, phys_params, flux_service, osc_service, neutrino_weight_name='neutrino', outdir=outdir)

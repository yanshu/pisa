#! /usr/bin/env python
import numpy as np
import os.path
import h5py
import pisa.resources.resources as resources
from pisa.flux.IPHondaFluxService_MC import IPHondaFluxService
from pisa.oscillations.Prob3OscillationServiceMC import Prob3OscillationServiceMC
from pisa.utils.log import set_verbosity,logging,profile
from pisa.utils.params import get_values, select_hierarchy
from pisa.utils.jsons import from_json
from pisa.utils.hdf import from_hdf, to_hdf
import pisa.utils.utils as utils

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def add_weights_to_file(data_file_path, file_type, phys_params, flux_service, osc_service, outdir):
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
                osc_probs = osc_service.fill_osc_prob(true_e, true_cz, event_by_event=True, **phys_params)
                osc_flux = nue_flux*osc_probs['nue'+isbar+'_maps'][prim]+ numu_flux*osc_probs['numu'+isbar+'_maps'][prim]
                data_file[prim][int_type]['neutrino_weight'] = osc_flux * data_file[prim][int_type]['weighted_aeff'] 
        data_file_name = os.path.basename(data_file_path)
        output_file_name = outdir + '/' + data_file_name.split('.hdf5')[0]+'_with_weights.hdf5' 
        to_hdf(data_file, output_file_name, attrs=attrs, overwrite=True)

    elif file_type == 'intermediate':
        data_file = h5py.File(resources.find_resource(data_file_path), "r+")
        true_e = data_file['trueNeutrino']['energy']
        true_cz = np.cos(data_file['trueNeutrino']['zenith'])
        one_weight = data_file['I3MCWeightDict']['OneWeight']
        pdg_encoding = data_file['trueNeutrino']['pdg_encoding']
        nuDict = {'12':'nue', '14':'numu', '16':'nutau', '-12': 'nue_bar', '-14': 'numu_bar', '-16': 'nutau_bar'}
        # 4 digit simulaiton negen:
        n_files = {'nue': 2700, 'numu': 4000, 'nutau': 1400}
        ngen = {}
        for nu in ['nue', 'numu', 'nutau']:
            ngen[nu] = n_files[nu]*300003*0.5 
            ngen[nu+'_bar'] = n_files[nu]*300003*0.5 
        prim = [nuDict[str(pdg)] for pdg in pdg_encoding]
        ngen_array = [ngen[flav] for flav in prim]
        # calculate sim weights
        weighted_aeff = one_weight/ngen_array
        # calculate fluxes
        flux_name_nue = ['nue_bar' if pdg<0 else 'nue' for pdg in pdg_encoding]
        flux_name_numu = ['numu_bar' if pdg<0 else 'nue' for pdg in pdg_encoding]
        nue_flux = flux_service.get_flux(true_e, true_cz, flux_name_nue, event_by_event=True)
        numu_flux = flux_service.get_flux(true_e, true_cz,flux_name_numu, event_by_event=True)
        osc_probs = osc_service.fill_osc_prob(true_e, true_cz, prim, event_by_event=True, **phys_params)
        osc_flux = nue_flux*osc_probs['nue_maps'] + numu_flux*osc_probs['numu_maps']
        data_file['neutrino_weight'] = osc_flux * weighted_aeff
        data_file.close()
    else:
        raise ValueError('file_type only allow: pisa, intermediate')


if __name__ == '__main__':

    parser = ArgumentParser(description='''Quick check if all components are working reasonably well, by
making the final level hierarchy asymmetry plots from the input
settings file. ''')
    parser_file = parser.add_mutually_exclusive_group(required=True)
    parser_file.add_argument( '-fp', '--pisa_file', metavar='H5_FILE', type=str, help='input HDF5 file')
    parser_file.add_argument( '-fi', '--intermediate_file', metavar='H5_FILE', type=str, help='input HDF5 file')
    parser.add_argument( '-t', '--template_settings',metavar='JSON',
        help='''Settings file that contains informatino of flux file, oscillation
        parameters, PREM model, etc., for calculation of neutrino weights''')
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

    outdir = args.outdir
    utils.mkdir(args.outdir)
    
    # get template settings
    template_settings = from_json(args.template_settings)
    phys_params = get_values(select_hierarchy(template_settings['params'], normal_hierarchy=True))

    # flux and osc service
    flux_service = IPHondaFluxService(**phys_params)
    osc_service = Prob3OscillationServiceMC([],[],**phys_params)

    add_weights_to_file(hd5_file_name, file_type, phys_params, flux_service, osc_service, outdir)

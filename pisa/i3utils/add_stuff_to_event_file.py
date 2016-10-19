#! /usr/bin/env python
import numpy as np
import os.path
import h5py
import copy
import pisa.resources.resources as resources
from pisa.flux.IPHondaFluxService_MC_merge import IPHondaFluxService
from pisa.oscillations.Prob3OscillationServiceMC_merge import Prob3OscillationServiceMC
#from pisa.utils.shape import SplineService
from pisa.utils.shape_mc import SplineService
from pisa.utils.params import construct_genie_dict
from pisa.utils.params import construct_shape_dict
from pisa.utils.log import set_verbosity,logging
from pisa.utils.params import get_values, select_hierarchy
from pisa.utils.jsons import from_json
from pisa.utils.hdf import from_hdf, to_hdf
import pisa.utils.utils as utils
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pisa.utils.utils import Timer, oversample_binning
from numpy.polynomial.polynomial import polyfit
CMSQ_TO_MSQ = 1.0e-4

def fitMa(in_yvalues_array, genie_weight_array):
    rw_xvalues  = np.array([-2,-1,0,1,2])
    output_coeff_linear_array = []
    output_coeff_quad_array = []
    for i in range(0,len(genie_weight_array)):
        in_yvalues = in_yvalues_array[i]
        genie_weight = genie_weight_array[i]
        if sum(in_yvalues) == 4.0:
            output_coeff_quad_array.append(0)
            output_coeff_linear_array.append(0)
        else:
            yvalues = np.concatenate((in_yvalues[:2]/genie_weight, [1.], in_yvalues[2:]/genie_weight))
            #print "yvalues", yvalues
            fitcoeff = polyfit(rw_xvalues, yvalues, deg = 2)[::-1]
            output_coeff_quad_array.append(fitcoeff[0])
            output_coeff_linear_array.append(fitcoeff[1])
            #print "fit values", fitcoeff
    output_coeff_linear_array = np.array(output_coeff_linear_array)
    output_coeff_quad_array = np.array(output_coeff_quad_array)
    return output_coeff_linear_array, output_coeff_quad_array 

#@profile
def add_stuff_to_file(data_file_path, file_type, ebins, phys_params, flux_service, osc_service, neutrino_weight_name, outdir, add_fluxes, add_weights, add_GENIE_Barr):
    print 'data_file_path = ', data_file_path
    if file_type == 'pisa':
        data_file, attrs = from_hdf(resources.find_resource(data_file_path), return_attrs = True)
        data_file_name = os.path.basename(data_file_path)
        utils.mkdir(args.outdir)
        file_name_end = '_with'
        if add_fluxes:
            file_name_end += '_fluxes'
        if add_weights:
            file_name_end += '_weights'
        if add_GENIE_Barr:
            file_name_end += '_GENIE_Barr'
        output_file_name = outdir + '/' + data_file_name.split('.hdf5')[0]+'%s.hdf5'%file_name_end
        print "Output file name is ", output_file_name
        if not os.path.isfile(output_file_name):
            for prim in data_file.keys():
                for int_type in data_file[prim].keys():
                    true_e = data_file[prim][int_type]['true_energy']
                    true_cz = data_file[prim][int_type]['true_coszen']
                    isbar = '_bar' if 'bar' in prim else ''
                    if add_fluxes:
                        nue_flux = flux_service.get_flux(true_e, true_cz, 'nue'+isbar, event_by_event=True)
                        numu_flux = flux_service.get_flux(true_e, true_cz, 'numu'+isbar, event_by_event=True)
                        # the opposite flavor fluxes( used only in the nu_nubar_ratio systematic)
                        oppo_isbar = '' if 'bar' in prim else '_bar'
                        oppo_nue_flux = flux_service.get_flux(true_e, true_cz, 'nue'+oppo_isbar, event_by_event=True)
                        oppo_numu_flux = flux_service.get_flux(true_e, true_cz, 'numu'+oppo_isbar, event_by_event=True)
                        data_file[prim][int_type][neutrino_weight_name+'_nue_flux'] = nue_flux
                        data_file[prim][int_type][neutrino_weight_name+'_numu_flux'] = numu_flux
                        data_file[prim][int_type][neutrino_weight_name+'_oppo_nue_flux'] = oppo_nue_flux
                        data_file[prim][int_type][neutrino_weight_name+'_oppo_numu_flux'] = oppo_numu_flux
                    if add_weights:
                    # if need to calculate neutrino weights here
                        print 'type prim  = ', type(prim)
                        nue_flux = flux_service.get_flux(true_e, true_cz, 'nue'+isbar, event_by_event=True)
                        numu_flux = flux_service.get_flux(true_e, true_cz, 'numu'+isbar, event_by_event=True)
                        osc_probs = osc_service.fill_osc_prob(true_e, true_cz, str(prim), event_by_event=True, **phys_params)
                        osc_flux = nue_flux*osc_probs['nue_maps']+ numu_flux*osc_probs['numu_maps']
                        data_file[prim][int_type][neutrino_weight_name+'_weight'] = osc_flux * data_file[prim][int_type]['weighted_aeff'] 
                    if add_GENIE_Barr:
                        # code modified from Ste's apply_shape_mod() in Aeff.py
                        print "add GENIE splines for ", prim , " ", int_type
                        ### make dict of genie parameters ###
                        GENSYS = construct_genie_dict(phys_params)
                        Flux_Mod_Dict = construct_shape_dict('flux', phys_params)

                        ### make spline service for genie parameters ###
                        with Timer(verbose=False) as t:
                            genie_spline_service = SplineService(ebins=ebins, evals=true_e, dictFile = phys_params['GENSYS_files'])
                            barr_spline_service = SplineService(ebins=ebins, evals=true_e, dictFile = phys_params['flux_uncertainty_inputs'])
                        print("==> time initialize SplineService : %s sec"%t.secs)

                        with Timer(verbose=False) as t:
                            for entry in GENSYS.keys():
                                if entry == "MaCCQE" and int_type=='nc': continue
                                if int_type=='nc':
                                    flav = 'nuallbar_nc' if 'bar' in prim else 'nuall_nc'
                                    genie_spline = genie_spline_service.get_genie_spline(entry, flav)
                                else:
                                    genie_spline = genie_spline_service.get_genie_spline(entry, prim)
                                data_file[prim][int_type]['GENSYS_splines_'+entry] = genie_spline
                            for entry in Flux_Mod_Dict.keys():
                                barr_spline = barr_spline_service.get_barr_spline(entry)
                                data_file[prim][int_type]['BARR_splines_'+entry] = barr_spline
                        print("==> time getting genie_splines : %s sec"%t.secs)

                        # add quadratic fit of gensys (from oscFit)
                        for key in ['AhtBY', 'BhtBY', 'CV1uBY', 'CV2uBY', 'MaCCQE', 'MaCCRES', 'MaCOHpi', 'MaNCEL', 'MaNCRES']:
                            in_array=np.zeros((len(data_file[prim][int_type]['true_energy']),4))
                            for i in range(0,4):
                                in_array[:,i] = data_file[prim][int_type]['genie_'+key+'_%i'%i]
                            output_coeff_linear_array, output_coeff_quad_array = fitMa(in_array,data_file[prim][int_type]['genie_weight'])
                            data_file[prim][int_type]['quad_fit_'+key] = output_coeff_quad_array
                            data_file[prim][int_type]['linear_fit_'+key] = output_coeff_linear_array


            to_hdf(data_file, output_file_name, attrs=attrs, overwrite=True)
        else:
            print 'File %s already exists, skipped. Please delete it or rename it.' % output_file_name

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

    parser = ArgumentParser(description='''Add neutrino fluxes (and neutrino weights(osc*flux*sim_weight) if needed) and/or GENIE, Barr systematics
            for each event, usually add_fluxes and add_GENIE_Barr is used for PISA event-by-event analysis. ''')
    parser_file = parser.add_mutually_exclusive_group(required=True)
    parser_file.add_argument( '-fp', '--pisa_file', metavar='H5_FILE', type=str, help='input HDF5 file')
    parser_file.add_argument( '-fi', '--intermediate_file', metavar='H5_FILE', type=str, help='input HDF5 file, only works for old simulation.')
    parser.add_argument( '-t', '--template_settings',metavar='JSON',
        help='''Settings file that contains informatino of flux file, oscillation
        parameters, PREM model, etc., for calculation of neutrino weights''')
    parser.add_argument('--use_best_fit',action='store_true',default=False,
                        help='Use best fit params to calculate fluxes.')
    parser.add_argument('--use_IMH',action='store_true',default=False,
                        help='Use IMH to calculate fluxes, default is false.')
    parser.add_argument('--add_weights',action='store_true',default=False,
                        help='Calculate and store neutrino weights.')
    parser.add_argument('--add_fluxes',action='store_true',default=False,
                        help='Calculate and store neutrino weights.')
    parser.add_argument('--add_GENIE_Barr',action='store_true',default=False,
                        help='Calculate and store splines of GENIE sys. and Barr sys..')
    parser.add_argument('--best_fit_file', '--profile-results', default=None, dest='best_fit_file',
                        help='use post fit parameters from best fit result json file')
    parser.add_argument('-o','--outdir',metavar='DIR',default='',
                        help='Directory to save the output figures.')
    args = parser.parse_args()
    set_verbosity(0)

    # make sure at least one of three add_XXX arguments is true
    if args.add_fluxes==False and args.add_weights==False and args.add_GENIE_Barr==False:
        parser.error('No action requested, use at least one of add_fluxes, add_weights and add_GENIE_Barr!') 

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
    ebins = template_settings['binning']['ebins']
    print "ebins = ", ebins
    phys_params = get_values(select_hierarchy(template_settings['params'], normal_hierarchy=not(args.use_IMH)))

    if args.use_best_fit:
        free_nutau_template_settings = copy.deepcopy(template_settings)
        # replace with parameters determined in fit
        fit_file = from_json(resources.find_resource(args.best_fit_file))
        syslist = fit_file['trials'][0]['fit_results'][1].keys()
        for sys in syslist:
            if not sys == 'llh':
                val = fit_file['trials'][0]['fit_results'][1][sys][0]
                if sys == 'theta23' or sys =='deltam23' or sys =='deltam31':
                    sys += '_nh'
                print 'fit nutauCCnorm=free, %s at %.4f'%(sys,val)
                free_nutau_template_settings['params'][sys]['value'] = val
        phys_params = get_values(select_hierarchy(template_settings['params'], normal_hierarchy=not(args.use_IMH)))

    # flux and osc service
    flux_service = IPHondaFluxService(**phys_params)
    osc_service = Prob3OscillationServiceMC([],[],**phys_params)

    if args.use_best_fit:
        add_stuff_to_file(hd5_file_name, file_type, ebins, phys_params, flux_service, osc_service, neutrino_weight_name='neutrino_best_fit', outdir=outdir, add_fluxes=args.add_fluxes, add_weights=args.add_weights, add_GENIE_Barr=args.add_GENIE_Barr)
    else:
        add_stuff_to_file(hd5_file_name, file_type, ebins, phys_params, flux_service, osc_service, neutrino_weight_name='neutrino', outdir=outdir, add_fluxes=args.add_fluxes, add_weights=args.add_weights, add_GENIE_Barr=args.add_GENIE_Barr)



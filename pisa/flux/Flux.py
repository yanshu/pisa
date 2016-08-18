#! /usr/bin/env python
#
# Flux.py
#
# Sample the atmospheric neutrino flux from a FluxService with a given
# binning in cos(zenith) and energy.
#
# If desired, this will create a .json output file with the results of
# the current stage of processing.
#
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   2014-01-27

import os
import sys
import numpy as np
import copy
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.analysis.stats.Maps import apply_ratio_scale
from pisa.flux.HondaFluxService import HondaFluxService, Honda3DFluxService, primaries
from pisa.flux.IPHondaFluxService import IPHondaFluxService, IPHonda3DFluxService
from pisa.utils.jsons import from_json, to_json, json_string
from pisa.utils.utils import Timer, oversample_binning

from pisa.utils.log import logging, physics, set_verbosity
from pisa.utils.proc import report_params, get_params, add_params
from pisa.utils.utils import get_bin_centers
from pisa.utils.shape import SplineService
from pisa.utils.params import construct_shape_dict

def apply_shape_mod(flux_spline_service, flux_maps, ebins, czbins, **params):
    '''
    Taking Joakim's shape mod functionality and applying it generally
    to the flux_maps, regardless of the flux_service used
    '''
    #make flux_mod_dict and add it to the list of params.
    Flux_Mod_Dict = construct_shape_dict('flux', params)

    ### FORM A TABLE FROM THE UNCERTAINTY WEIGHTS AND THE SPLINED MAPS CORRESPONDING TO THEM - WE DISCUSSED THIS SHOUD BE DONE EXPLICITLY FOR EASIER UNDERSTANDING###
    #Now apply all the shape modification for each of the flux uncertainties
    #Modellling of the uncertainties follows the discussion
    #in Barr et al. (2006)
    return_dict = {}
    return_dict['params'] = flux_maps['params']
    for prim in primaries:
        prim_dict = {}
        prim_dict['ebins'] = flux_maps[prim]['ebins']
        prim_dict['czbins'] = flux_maps[prim]['czbins']
        mod_table = np.zeros_like(flux_maps[prim]['map'])
        # here I want a dictionary named Flux_Mod_Dict containing the mod factors as keys and UNCF_X files as entries, then I can modify the flux by:
        logging.info("now reaching the flux_mod_dict stage: \n %s"%Flux_Mod_Dict)
        for entry in Flux_Mod_Dict:
            if (params[entry]==0):
                continue
            logging.info("testing for: %s" %entry)
            Flux_Mod_Dict[entry] += 1.1
            mod_table += flux_spline_service.modify_shape(ebins, czbins, Flux_Mod_Dict[entry], entry).T

        if mod_table[mod_table<0].any():
            #remember: mod_table contains the 1 sigma modification of the flux squared and multiplied by the modification factor -
            # - this can of course be negative, but is not unphysical. It just represents the minus part of a +/- uncertainty. As such we should still assign it as an uncertainty, and just need to take the sqrt of the absolute value
            # - and then remember that this is to be applied in the negative direction, if it was negative.
            mod_table = -1 * np.sqrt(-1 * mod_table)
            
        else:
            mod_table = np.sqrt(mod_table) # just take the sqrt of the + side of the +/- uncertainty.

        prim_dict['map'] = flux_maps[prim]['map'] * (1 + mod_table) #this is where the actual modification happens
        return_dict[prim] = prim_dict

    return return_dict

def apply_nue_numu_ratio(flux_maps, nue_numu_ratio, flux_sys_renorm):
    """
    Applies the nue_numu_ratio systematic to the flux maps
    and returns the scaled maps. The actual calculation is
    done by apply_ratio_scale.
    """
    # keep both nu and nubar flux constant
    scaled_nue_flux, scaled_numu_flux = apply_ratio_scale(
        orig_maps = flux_maps,
        key1 = 'nue', key2 = 'numu',
        ratio_scale = nue_numu_ratio,
        is_flux_scale = True,
        flux_sys_renorm = flux_sys_renorm
    )

    scaled_nue_bar_flux, scaled_numu_bar_flux = apply_ratio_scale(
        orig_maps = flux_maps,
        key1 = 'nue_bar', key2 = 'numu_bar',
        ratio_scale = nue_numu_ratio,
        is_flux_scale = True,
        flux_sys_renorm = flux_sys_renorm
    )

    flux_maps['nue']['map'] = scaled_nue_flux
    flux_maps['nue_bar']['map']  =  scaled_nue_bar_flux
    flux_maps['numu']['map'] = scaled_numu_flux
    flux_maps['numu_bar']['map']  = scaled_numu_bar_flux

    return flux_maps

def apply_nu_nubar_ratio(flux_maps, nu_nubar_ratio, flux_sys_renorm):
    """
    Applies the nu_nubar_ratio systematic to the event rate
    maps and returns the scaled maps. The actual calculation is
    done by apply_ratio_scale.
    """
    flavours = flux_maps.keys()
    if 'params' in flavours: flavours.remove('params')

    for flavour in flavours:
        # process nu and nubar in one go
        if not 'bar' in flavour:
            scaled_nu_rates, scaled_nubar_rates = apply_ratio_scale(
                orig_maps = flux_maps,
                key1 = flavour, key2 = flavour+'_bar',
                ratio_scale = nu_nubar_ratio,
                is_flux_scale = True,
                flux_sys_renorm = flux_sys_renorm
            )
            flux_maps[flavour]['map'] = scaled_nu_rates
            flux_maps[flavour+'_bar']['map'] = scaled_nubar_rates

    return flux_maps


def apply_delta_index(flux_maps, delta_index, egy_med):
    """
    Applies the spectral index systematic to the flux maps by scaling
    each bin with (egy_cen/egy_med)^(-delta_index), preserving the total
    integral flux  Note that only the numu/numu_bar are scaled, because
    the nue_numu_ratio will handle the systematic on the nue flux.
    """

    for flav in ['numu','numu_bar']:
        ecen = get_bin_centers(flux_maps[flav]['ebins'])
        scale = np.power((ecen/egy_med),delta_index)
        flux_map = flux_maps[flav]['map']
        total_flux = flux_map.sum()
        logging.trace("flav: %s, total counts before scale: %f"%(flav,total_flux))
        scaled_flux = (flux_map.T*scale).T
        scaled_flux *= (total_flux/scaled_flux.sum())
        flux_maps[flav]['map'] = scaled_flux
        logging.trace("flav: %s, total counts after scale: %f"%
                      (flav,flux_maps[flav]['map'].sum()))

    return flux_maps

def get_median_energy(flux_map):
    """Returns the median energy of the flux_map-expected to be a dict
    with keys 'map', 'ebins', 'czbins'
    """

    ecen = get_bin_centers(flux_map['ebins'])
    energy = ecen[len(ecen)/2]

    return energy

def get_flux_maps(flux_service, barr_service, ebins, czbins, nue_numu_ratio,
                  nu_nubar_ratio, flux_sys_renorm, energy_scale, atm_delta_index,
                  flux_hadronic_A, flux_hadronic_B, flux_hadronic_C,
                  flux_hadronic_D, flux_hadronic_E, flux_hadronic_F,
                  flux_hadronic_G, flux_hadronic_H, flux_hadronic_I,
                  flux_hadronic_W, flux_hadronic_X, flux_hadronic_Y,
                  flux_hadronic_Z, flux_prim_norm_a, flux_prim_exp_norm_b,
                  flux_prim_exp_factor_c, flux_spectral_index_d,
                  flux_pion_chargeratio_Chg, flux_uncertainty_inputs,
                  **kwargs):
    """
    Get a set of flux maps for the different primaries.

    \params:
      * flux_service -
      * ebins/czbins - energy/coszenith bins to calculate flux
      * nue_numu_ratio - systematic to be a proxy for the realistic
        Flux_nue/Flux_numu and Flux_nuebar/Flux_numubar ratios,
        keeping both the total flux from neutrinos and antineutrinos
        constant. The adjusted ratios are given by
        "nue_numu_ratio * original ratio".
      * nu_nubar_ratio - systematic to be a proxy for the
        neutrino/anti-neutrino production/cross section ratio.
      * energy_scale - factor to scale energy bin centers by
      * atm_delta_index  - change in spectral index from fiducial
    """

    if (atm_delta_index != 0.0 and flux_spectral_index_d != 0.0): 
        raise ValueError('PISA should not be run both with "flux_spectral_index_d" and "atm_delta_index" != 0.0. Doing so will modify the flux twice. Either run with "atm_delta_index" and disable the other flux parameters, or run with "flux_spectral_index_d" and disable "atm_delta_index". This is done in the settings file by setting the default value to "0.0" and "fixed=true". Do this now and re-run. :)')

    # Be verbose on input
    params = get_params()
    report_params(params, units = [''])

    # Initialize return dict
    maps = {'params': params}

    for prim in primaries:

        # Get the flux for this primary
        maps[prim] = {'ebins': ebins,
                      'czbins': czbins,
                      'map': flux_service.get_flux(ebins*energy_scale,czbins,prim)}

        # be a bit verbose
        logging.trace("Total flux of %s is %u [s^-1 m^-2]"%
                      (prim,maps[prim]['map'].sum()))

    # now scale the nue(bar) / numu(bar) flux ratios, keeping the total
    # Flux (nue + numu, nue_bar + numu_bar) constant, or return unscaled maps:
    scaled_maps = apply_nue_numu_ratio(maps, nue_numu_ratio, flux_sys_renorm) if nue_numu_ratio != 1.0 else maps

    # now scale the nu(e/mu) / nu(e/mu)bar event count ratios, keeping the total
    # (nue + nuebar etc.) constant
    if nu_nubar_ratio != 1.:
        scaled_maps = apply_nu_nubar_ratio(scaled_maps, nu_nubar_ratio, flux_sys_renorm)

    median_energy = get_median_energy(maps['numu'])
    if atm_delta_index != 0.0:
        scaled_maps = apply_delta_index(scaled_maps, atm_delta_index, median_energy)

    #Apply Barr uncertainties (18 syst.)
    flux_sys = {'flux_hadronic_A':flux_hadronic_A, 'flux_hadronic_B':flux_hadronic_B, 'flux_hadronic_C':flux_hadronic_C,
                'flux_hadronic_D':flux_hadronic_D, 'flux_hadronic_E':flux_hadronic_E, 'flux_hadronic_F':flux_hadronic_F,
                'flux_hadronic_G':flux_hadronic_G, 'flux_hadronic_H':flux_hadronic_H, 'flux_hadronic_I':flux_hadronic_I,
                'flux_hadronic_W':flux_hadronic_W, 'flux_hadronic_X':flux_hadronic_X, 'flux_hadronic_Y':flux_hadronic_Y,
                'flux_hadronic_Z':flux_hadronic_Z, 'flux_prim_norm_a':flux_prim_norm_a,
                'flux_prim_exp_norm_b':flux_prim_exp_norm_b, 'flux_prim_exp_factor_c':flux_prim_exp_factor_c,
                'flux_spectral_index_d':flux_spectral_index_d, 'flux_pion_chargeratio_Chg':flux_pion_chargeratio_Chg}
    #print "flux_prim_exp_factor_c = ", flux_prim_exp_factor_c
    for sys in flux_sys.keys():
        params[sys] = flux_sys[sys]
    scaled_maps = apply_shape_mod(barr_service, scaled_maps, ebins, czbins, **params)
    
    return scaled_maps


if __name__ == '__main__':

    # parser
    parser = ArgumentParser(description='Take a settings file '
        'as input and write out a set of flux maps',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ebins', metavar='[1.0,2.0,...]', type=json_string,
                        help= '''Edges of the energy bins in units of GeV. ''',
                        default=np.logspace(np.log10(1.0),np.log10(80.0),20) )
    parser.add_argument('--czbins', metavar='[-1.0,-0.8.,...]', type=json_string,
                        help= '''Edges of the cos(zenith) bins.''',
                        default = np.linspace(-1.,0.,11))
    parser.add_argument('--flux_file', metavar='FILE', type=str,
                        help= '''Input flux file in Honda format. ''',
                        default = 'flux/spl-solmax-aa.d')
    parser.add_argument('--flux_mode', metavar='STRING', type=str,
                        help='''Type of flux interpolation to perform''',
                        default='bisplrep')
    parser.add_argument('--nue_numu_ratio',metavar='FLOAT',type=float,
                        help='''Factor to scale nue_flux by''',default=1.0)
    hselect.add_argument('--flux_sys_renorm', default=True,
                        action='store_true', help="Use renormalization for flux-related syst.")
    parser.add_argument('--nu_nubar_ratio',metavar='FLOAT',type=float,
                        help='''Factor to scale nu_nubar_flux by''',default=1.0)
    parser.add_argument('--delta_index',metavar='FLOAT',type=float,
                        default=0.0,help='''Shift in spectral index of numu''')
    parser.add_argument('--energy_scale',metavar='FLOAT',type=float,
                        help='''Factor to scale TRUE energy by''',default=1.0)
    parser.add_argument('--flux_hadronic_A',metavar='FLOAT',type=float,
                        help='''Factor to scale flux shape by''',default=0)
    parser.add_argument('--flux_hadronic_B',metavar='FLOAT',type=float,
                        help='''Factor to scale flux shape by''',default=0)
    parser.add_argument('--flux_hadronic_C',metavar='FLOAT',type=float,
                        help='''Factor to scale flux shape by''',default=0)
    parser.add_argument('--flux_hadronic_D',metavar='FLOAT',type=float,
                        help='''Factor to scale flux shape by''',default=0)
    parser.add_argument('--flux_hadronic_E',metavar='FLOAT',type=float,
                        help='''Factor to scale flux shape by''',default=0)
    parser.add_argument('--flux_hadronic_F',metavar='FLOAT',type=float,
                        help='''Factor to scale flux shape by''',default=0)
    parser.add_argument('--flux_hadronic_G',metavar='FLOAT',type=float,
                        help='''Factor to scale flux shape by''',default=0)   
    parser.add_argument('--flux_hadronic_H',metavar='FLOAT',type=float,
                        help='''Factor to scale flux shape by''',default=0)
    parser.add_argument('--flux_hadronic_I',metavar='FLOAT',type=float,
                        help='''Factor to scale flux shape by''',default=0)

    parser.add_argument('--flux_hadronic_W',metavar='FLOAT',type=float,
                        help='''Factor to scale flux shape by''',default=0)
    parser.add_argument('--flux_hadronic_X',metavar='FLOAT',type=float,
                        help='''Factor to scale flux shape by''',default=0)
    parser.add_argument('--flux_hadronic_Y',metavar='FLOAT',type=float,
                        help='''Factor to scale flux shape by''',default=0)
    parser.add_argument('--flux_hadronic_Z',metavar='FLOAT',type=float,
                        help='''Factor to scale flux shape by''',default=0)
    
    parser.add_argument('--flux_prim_norm_a',metavar='FLOAT',type=float,
                        help='''Factor to scale flux shape by''',default=0)
    parser.add_argument('--flux_prim_exp_norm_b',metavar='FLOAT',type=float,
                        help='''Factor to scale flux shape by''',default=0)
    parser.add_argument('--flux_prim_exp_factor_c',metavar='FLOAT',type=float,
                        help='''Factor to scale flux shape by''',default=0)   
    parser.add_argument('--flux_spectral_index_d',metavar='FLOAT',type=float,
                        help='''Factor to scale flux shape by''',default=0)
    parser.add_argument('--flux_pion_chargeratio_Chg',metavar='FLOAT',type=float,
                        help='''Factor to scale flux shape by''',default=0)
    parser.add_argument('--flux_uncertainty_inputs', metavar='DICT', type=str,
                        help='''Dictionary of files containing the shapes of uncertainties corresponding to the above parameters''')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE',
                        type=str, action='store', default='flux.json',
                        help='file to store the output')
    parser.add_argument('-v', '--verbose', action='count', default=None,
                        help='set verbosity level')
    args = parser.parse_args()

    #Set verbosity level
    set_verbosity(args.verbose)

    logging.debug("Using %u bins in energy from %.2f to %.2f GeV"%
                                (len(args.ebins)-1,args.ebins[0],args.ebins[-1]))
    logging.debug("Using %u bins in cos(zenith) from %.2f to %.2f"%
                                (len(args.czbins)-1,args.czbins[0],args.czbins[-1]))

    #Instantiate a flux model
    logging.info("Defining flux service...")

    if 'aa' in args.flux_file:
    
        if args.flux_mode.lower() == 'integral-preserving':
            logging.info("  Using Honda tables with integral-preserving interpolation...")
            flux_model = IPHondaFluxService(args.flux_file)
        else:
            logging.info("  Using Honda tables with simple cubic interpolation...")
            flux_model = HondaFluxService(args.flux_file)

        logging.info("  Tables are provided azimuth-averaged.")

    else:

        if args.flux_mode.lower() == 'integral-preserving':
            logging.info("  Using Honda tables with integral-preserving interpolation...")
            flux_model = IPHonda3DFluxService(args.flux_file)
        else:
            logging.info("  Using Honda tables with simple cubic interpolation...")
            flux_model = Honda3DFluxService(args.flux_file)

        logging.info("  Tables are provided with azimuth-dependency, so average is calculated from this.")

    #get the flux
    flux_maps = get_flux_maps(flux_model,
                              args.ebins,
                              args.czbins,
                              args.nue_numu_ratio,
                              args.nu_nubar_ratio,
                              args.flux_sys_renorm,
                              args.energy_scale,
                              args.delta_index,
                              args.flux_hadronic_A,
                              args.flux_hadronic_B,
                              args.flux_hadronic_C,
                              args.flux_hadronic_D,
                              args.flux_hadronic_E,
                              args.flux_hadronic_F,
                              args.flux_hadronic_G,
                              args.flux_hadronic_H,
                              args.flux_hadronic_I,
                              args.flux_hadronic_W,
                              args.flux_hadronic_X,
                              args.flux_hadronic_Y,
                              args.flux_hadronic_Z,
                              args.flux_prim_norm_a,
                              args.flux_prim_exp_norm_b,
                              args.flux_prim_exp_factor_c,
                              args.flux_spectral_index_d,
                              args.flux_pion_chargeratio_Chg,
                              args.flux_uncertainty_inputs,
                              **kwargs)

    #write out to a file
    logging.info("Saving output to: %s"%args.outfile)
    to_json(flux_maps, args.outfile)

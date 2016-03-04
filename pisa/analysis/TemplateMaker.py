#! /usr/bin/env python
#
# TemplateMaker.py
#
# Class to implement template-making procedure and to store as much data
# as possible to avoid re-running stages when not needed.
#
# author: Timothy C. Arlen - tca3@psu.edu
#         Sebastian Boeser - sboeser@uni-mainz.de
#
# date:   7 Oct 2014
#

import sys
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.constants import Julian_year

from pisa.utils.log import logging, tprofile, set_verbosity
from pisa.resources.resources import find_resource
from pisa.utils.params import get_fixed_params, get_free_params, get_values, select_hierarchy
from pisa.utils.fileio import from_file, to_file
from pisa.utils.utils import Timer

from pisa.flux.HondaFluxService import HondaFluxService
from pisa.flux import Flux

from pisa.oscillations.Prob3OscillationService import Prob3OscillationService
from pisa.oscillations.NucraftOscillationService import NucraftOscillationService
try:
    logging.info('Trying to import Prob3GPUOscillationService...')
    from pisa.oscillations.Prob3GPUOscillationService import Prob3GPUOscillationService
except:
    logging.warn('CAN NOT import Prob3GPUOscillationService!')
from pisa.oscillations.Oscillation import get_osc_flux

from pisa.aeff.AeffServiceMC import AeffServiceMC
from pisa.aeff.AeffServicePar import AeffServicePar
from pisa.aeff import Aeff

from pisa.reco.RecoServiceMC import RecoServiceMC
from pisa.reco.RecoServiceParam import RecoServiceParam
from pisa.reco.RecoServiceKernelFile import RecoServiceKernelFile
from pisa.reco.RecoServiceVBWKDE import RecoServiceVBWKDE
from pisa.reco import Reco

from pisa.pid import PID

class TemplateMaker:
    '''
    This class handles all steps needed to produce a template with a
    constant binning.

    The strategy employed will be to define all 'services' in the
    initialization process, make them members of the class, then use
    them later when needed.
    '''
    def __init__(self, template_params_values, ebins, czbins,
                 oversample_e, oversample_cz, **kwargs):
        """TemplateMaker class handles all of the setup and calculation of the
        templates for a given binning.

        Parameters
        ----------
        template_params_values
        ebins, czbins
            energy and coszen bin edges
        oversample_e
        oversample_cz
        """


        self.ebins = ebins
        self.czbins = czbins
        self.oversample_e = oversample_e
        self.oversample_cz = oversample_cz
        logging.debug('Using %u bins in energy from %.2f to %.2f GeV' %
                      (len(self.ebins)-1, self.ebins[0], self.ebins[-1]))
        logging.debug('Using %u bins in cos(zenith) from %.2f to %.2f' %
                      (len(self.czbins)-1, self.czbins[0], self.czbins[-1]))

        # Instantiate a flux model service
        self.flux_service = HondaFluxService(**template_params_values)

        # Oscillated Flux Service:
        osc_code = template_params_values['osc_code']
        if osc_code == 'prob3':
            self.osc_service = Prob3OscillationService(
                ebins=self.ebins, czbins=self.czbins,
                **template_params_values
            )
        elif osc_code == 'gpu':
            self.osc_service = Prob3GPUOscillationService(
                ebins=self.ebins, czbins=self.czbins,
                oversample_e=self.oversample_e,
                oversample_cz=self.oversample_cz,
                **template_params_values
            )
        elif osc_code == 'nucraft':
            self.osc_service = NucraftOscillationService(
                ebins=self.ebins, czbins=self.czbins,
                **template_params_values
            )
        else:
            error_msg = 'OscillationService NOT implemented for ' + \
                    'osc_code = %s' % osc_code
            raise NotImplementedError(error_msg)

        # Aeff/True Event Rate Service:
        aeff_mode = template_params_values['aeff_mode']
        if aeff_mode == 'param':
            logging.info(' Using effective area from PARAMETRIZATION...')
            self.aeff_service = AeffServicePar(
                ebins=self.ebins, czbins=self.czbins,
                **template_params_values
            )
        elif aeff_mode == 'MC':
            logging.info(' Using effective area from MC EVENT DATA...')
            self.aeff_service = AeffServiceMC(
                ebins=self.ebins, czbins=self.czbins,
                **template_params_values
            )
        else:
            error_msg = "aeff_mode: '%s' is not implemented! " % aeff_mode
            error_msg += " Please choose among: ['MC', 'param']"
            raise NotImplementedError(error_msg)

        # Reco Event Rate Service:
        reco_mode = template_params_values['reco_mode']
        if reco_mode == 'MC':
            self.reco_service = RecoServiceMC(
                ebins=self.ebins, czbins=self.czbins,
                **template_params_values
            )
        elif reco_mode == 'param':
            self.reco_service = RecoServiceParam(
                ebins=self.ebins, czbins=self.czbins,
                **template_params_values
            )
        elif reco_mode == 'stored':
            self.reco_service = RecoServiceKernelFile(
                ebins=self.ebins, czbins=self.czbins,
                **template_params_values
            )
        elif reco_mode == 'vbwkde':
            self.reco_service = RecoServiceVBWKDE(
                ebins=self.ebins, czbins=self.czbins,
                **template_params_values
            )
        else:
            error_msg = "reco_mode: %s is not implemented! " % reco_mode
            error_msg+=" Please choose among: ['MC', 'param', 'stored']"
            raise NotImplementedError(error_msg)

        print template_params_values
        print template_params_values.keys()
        print 'pid_constructor_params:', template_params_values['pid_constructor_params']
        print 'type(pid_constructor_params):', type(template_params_values['pid_constructor_params'])
        # Instantiate a PID service
        self.pid_service = PID.pid_service_factory(
            ebins=self.ebins, czbins=self.czbins,
            **template_params_values['pid_constructor_params']
        )


    def get_template(self, template_params_values, return_stages=False):
        '''
        Runs entire template-making chain, using parameters found in
        'template_params_values' dict. If 'return_stages' is set to True,
        returns output from each stage as a simple tuple.
        '''

        logging.info("STAGE 1: Getting Atm Flux maps...")
        with Timer() as t:
            flux_maps = Flux.get_flux_maps(
                self.flux_service, ebins=self.ebins, czbins=self.czbins,
                **template_params_values
            )
        tprofile.debug("==> elapsed time for flux stage: %s sec" % t.secs)

        logging.info("STAGE 2: Getting osc prob maps...")
        with Timer() as t:
            osc_flux_maps = get_osc_flux(flux_maps=flux_maps,
                                         osc_service=self.osc_service,
                                         oversample_e=self.oversample_e,
                                         oversample_cz=self.oversample_cz,
                                         **template_params_values)
        tprofile.debug("==> elapsed time for oscillations stage: %s sec"
                       % t.secs)

        logging.info("STAGE 3: Getting event rate true maps...")
        with Timer() as t:
            event_rate_maps = Aeff.get_event_rates(
                osc_flux_maps=osc_flux_maps, aeff_service=self.aeff_service,
                **template_params_values
            )
        tprofile.debug("==> elapsed time for aeff stage: %s sec" % t.secs)

        logging.info("STAGE 4: Getting event rate reco maps...")
        with Timer() as t:
            event_rate_reco_maps = Reco.get_reco_maps(
                event_rate_maps=event_rate_maps,
                reco_service=self.reco_service,
                **template_params_values
            )
        tprofile.debug("==> elapsed time for reco stage: %s sec" % t.secs)

        logging.info("STAGE 5: Getting pid maps...")
        with Timer(verbose=False) as t:
            final_event_rate = self.pid_service.get_pid_maps(
                event_rate_reco_maps
            )
        tprofile.debug("==> elapsed time for pid stage: %s sec" % t.secs)

        if not return_stages:
            return final_event_rate

        # Otherwise, return all stages as a simple tuple
        return (flux_maps, osc_flux_maps, event_rate_maps,
                event_rate_reco_maps, final_event_rate)

    def get_template_no_osc(self, template_params_values):
        """Runs template making chain, but without oscillations"""
        logging.info("STAGE 1: Getting Atm Flux maps...")
        with Timer() as t:
            flux_maps = Flux.get_flux_maps(
                self.flux_service, ebins=self.ebins, czbins=self.czbins,
                **template_params_values
            )
        tprofile.debug("==> elapsed time for flux stage: %s sec" % t.secs)

        # Skipping oscillation stage...
        logging.info("  >>Skipping Stage 2 in no oscillations case...")
        flavours = ['nutau', 'nutau_bar']
        # Create the empty nutau maps:
        test_map = flux_maps['nue']
        for flav in flavours:
            flux_maps[flav] = {'map': np.zeros_like(test_map['map']),
                               'ebins': np.zeros_like(test_map['ebins']),
                               'czbins': np.zeros_like(test_map['czbins'])}

        logging.info("STAGE 3: Getting event rate true maps...")
        with Timer() as t:
            event_rate_maps = Aeff.get_event_rates(
                flux_maps=flux_maps, aeff_service=self.aeff_service,
                **template_params_values
            )
        tprofile.debug("==> elapsed time for aeff stage: %s sec" % t.secs)

        logging.info("STAGE 4: Getting event rate reco maps...")
        with Timer() as t:
            event_rate_reco_maps = Reco.get_reco_maps(
                event_rate_maps, self.reco_service,
                **template_params_values
            )
        tprofile.debug("==> elapsed time for reco stage: %s sec" % t.secs)

        logging.info("STAGE 5: Getting pid maps...")
        with Timer(verbose=False) as t:
            final_event_rate = self.pid_service.get_pid_maps(
                event_rate_reco_maps
            )
        tprofile.debug("==> elapsed time for pid stage: %s sec" % t.secs)

        return final_event_rate


if __name__ == '__main__':
    # parser
    parser = ArgumentParser(
        description='''Runs the template making process.''',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-t', '--template_settings', type=str,
                        metavar='JSONFILE', required=True,
                        help='''settings for the template generation''')

    hselect = parser.add_mutually_exclusive_group(required=False)
    hselect.add_argument('--normal', dest='normal', default=True,
                         action='store_true',
                         help="select the normal hierarchy")
    hselect.add_argument('--inverted', dest='normal', default = False,
                         action='store_false',
                         help="select the inverted hierarchy")

    parser.add_argument('-v', '--verbose', action='count', default=None,
                        help='set verbosity level.')
    parser.add_argument('-s', '--save_all', action='store_true', default=False,
                        help="Save all stages.")
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE',
                        type=str, action='store',default="template.json",
                        help='file to store the output')
    args = parser.parse_args()

    set_verbosity(args.verbose)

    with Timer() as t:
        # Load all the settings
        model_settings = from_file(args.template_settings)

        #Select a hierarchy
        logging.info('Selected %s hierarchy' %
                     ('normal' if args.normal else 'inverted'))
        template_params = select_hierarchy(model_settings['params'],
                                           normal_hierarchy=args.normal)

        # Intialize template maker
        template_params_values = get_values(template_params)
        #ebins = model_settings['binning']['ebins']
        #czbins = model_settings['binning']['czbins']
        #oversample_e = model_settings['binning']['oversample_e']
        #oversample_cz = model_settings['binning']['oversample_cz']
        template_maker = TemplateMaker(template_params_values,
                                       **model_settings['binning'])
    tprofile.info('  ==> elapsed time to initialize templates: %s sec'
                  % t.secs)

    # Now get the actual template
    with Timer(verbose=False) as t:
        template_maps = template_maker.get_template(
            get_values(template_params),
            return_stages=args.save_all
        )
    tprofile.info('==> elapsed time to get template: %s sec' % t.secs)

    logging.info('Saving file to %s' % args.outfile)
    to_file(template_maps, args.outfile)

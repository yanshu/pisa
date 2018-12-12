#! /usr/bin/env python
#
# author: Justin L. Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   October 24, 2015
"""
Generate a PISA-standard-format events HDF5 file from HDF5 file(s) generated
from I3 files by the icecube.hdfwriter.I3HDFTableService
"""

import os
import numpy as np
from copy import deepcopy

from pisa.utils.log import logging, set_verbosity
import pisa.utils.utils as utils
import pisa.utils.flavInt as flavInt
import pisa.utils.events as events
import pisa.utils.mcSimRunSettings as MCSRS
import pisa.utils.dataProcParams as DPP
import pisa.resources.resources as resources
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


CMSQ_TO_MSQ = 1.0e-4

# Default fields to extract from source HDF5 files during processing
# Note that *_coszen is generated from *_zenith
EXTRACT_FIELDS = (
    'genie_weight',
    'run',
    'event',
    'true_x',
    'true_y',
    'true_z',
    'true_azimuth',
    'true_energy',
    'true_coszen',
    'reco_x',
    'reco_y',
    'reco_z',
    'reco_trck_zenith',
    'reco_azimuth',
    'reco_trck_len',
    'reco_trck_azimuth',
    'reco_energy',
    'reco_coszen',
    'one_weight',
    'pid',
    'dunkman_L3',
    'dunkman_L4',
    'dunkman_L5',
    'genie_AhtBY_0',
    'genie_AhtBY_1',
    'genie_AhtBY_2',
    'genie_AhtBY_3',
    'genie_BhtBY_0',
    'genie_BhtBY_1',
    'genie_BhtBY_2',
    'genie_BhtBY_3',
    'genie_CV1uBY_0',
    'genie_CV1uBY_1',
    'genie_CV1uBY_2',
    'genie_CV1uBY_3',
    'genie_CV2uBY_0',
    'genie_CV2uBY_1',
    'genie_CV2uBY_2',
    'genie_CV2uBY_3',
    'genie_MaCCQE_0',
    'genie_MaCCQE_1',
    'genie_MaCCQE_2',
    'genie_MaCCQE_3',
    'genie_MaCCRES_0',
    'genie_MaCCRES_1',
    'genie_MaCCRES_2',
    'genie_MaCCRES_3',
    'genie_MaCOHpi_0',
    'genie_MaCOHpi_1',
    'genie_MaCOHpi_2',
    'genie_MaCOHpi_3',
    'genie_MaNCEL_0',
    'genie_MaNCEL_1',
    'genie_MaNCEL_2',
    'genie_MaNCEL_3',
    'genie_MaNCRES_0',
    'genie_MaNCRES_1',
    'genie_MaNCRES_2',
    'genie_MaNCRES_3',
    'C2QR6', 'CausalVetoHits', 'CausalVetoPE', 'DCFiducialPE', 'ICVetoPE', 'NAbove200PE', 'NchCleaned', 'NoiseEngine', 'STW9000_DTW300Hits',
    'STW9000_DTW300PE', 'VertexGuessX', 'VertexGuessY', 'VertexGuessZ', 'rt_fid_charge', 'cog_sigma_time', 'cog_sigma_z', 'num_hit_doms',
    'dcc_veto_charge', 'vich_charge', 'cog_q1_rho', 'cog_q1_z', 'first_hlc_rho', 'first_hlc_z', 'interval', 'C2QR3', 'QR3', 'linefit_speed',
    'linefit_zenith', 'separation',   'total_charge', 'santa_direct_charge', 'mn_start_contained', 'mn_stop_contained', 'x_prime', 'y_prime',
    'z_prime', 't_prime', 'r_prime', 'rho_prime', 'theta_prime', 'phi_prime', 'stop_x_prime', 'stop_y_prime', 'stop_z_prime', 'stop_r_prime',
    'stop_rho_prime', 'stop_theta_prime', 'stop_phi_prime', 'santa_direct_doms', 'spe11_zenith', 'corridor_doms_over_threshold'
)

# Default fields to output to destination PISA events HDF5 file
OUTPUT_FIELDS = (
    'run',
    'event',
    'true_x',
    'true_y',
    'true_z',
    'true_azimuth',
    'true_energy',
    'true_coszen',
    'reco_x',
    'reco_y',
    'reco_z',
    'reco_trck_zenith',
    'reco_azimuth',
    'reco_trck_len',
    'reco_trck_azimuth',
    'reco_energy',
    'reco_coszen',
    'weighted_aeff',
    'mc_weight',
    'pid',
    'dunkman_L3',
    'dunkman_L4',
    'dunkman_L5',
    'genie_weight',
    'genie_AhtBY_0',
    'genie_AhtBY_1',
    'genie_AhtBY_2',
    'genie_AhtBY_3',
    'genie_BhtBY_0',
    'genie_BhtBY_1',
    'genie_BhtBY_2',
    'genie_BhtBY_3',
    'genie_CV1uBY_0',
    'genie_CV1uBY_1',
    'genie_CV1uBY_2',
    'genie_CV1uBY_3',
    'genie_CV2uBY_0',
    'genie_CV2uBY_1',
    'genie_CV2uBY_2',
    'genie_CV2uBY_3',
    'genie_MaCCQE_0',
    'genie_MaCCQE_1',
    'genie_MaCCQE_2',
    'genie_MaCCQE_3',
    'genie_MaCCRES_0',
    'genie_MaCCRES_1',
    'genie_MaCCRES_2',
    'genie_MaCCRES_3',
    'genie_MaCOHpi_0',
    'genie_MaCOHpi_1',
    'genie_MaCOHpi_2',
    'genie_MaCOHpi_3',
    'genie_MaNCEL_0',
    'genie_MaNCEL_1',
    'genie_MaNCEL_2',
    'genie_MaNCEL_3',
    'genie_MaNCRES_0',
    'genie_MaNCRES_1',
    'genie_MaNCRES_2',
    'genie_MaNCRES_3',
    'C2QR6', 'CausalVetoHits', 'CausalVetoPE', 'DCFiducialPE', 'ICVetoPE', 'NAbove200PE', 'NchCleaned', 'NoiseEngine', 'STW9000_DTW300Hits',
    'STW9000_DTW300PE', 'VertexGuessX', 'VertexGuessY', 'VertexGuessZ', 'rt_fid_charge', 'cog_sigma_time', 'cog_sigma_z', 'num_hit_doms',
    'dcc_veto_charge', 'vich_charge', 'cog_q1_rho', 'cog_q1_z', 'first_hlc_rho', 'first_hlc_z', 'interval', 'C2QR3', 'QR3', 'linefit_speed',
    'linefit_zenith', 'separation',   'total_charge', 'santa_direct_charge', 'mn_start_contained', 'mn_stop_contained', 'x_prime', 'y_prime',
    'z_prime', 't_prime', 'r_prime', 'rho_prime', 'theta_prime', 'phi_prime', 'stop_x_prime', 'stop_y_prime', 'stop_z_prime', 'stop_r_prime',
    'stop_rho_prime', 'stop_theta_prime', 'stop_phi_prime', 'santa_direct_doms', 'spe11_zenith', 'corridor_doms_over_threshold'
)


def makeEventsFile(data_files, outdir, run_settings, data_proc_params, cut,
                   join=None, cust_cuts=None,
                   extract_fields=EXTRACT_FIELDS, output_fields=OUTPUT_FIELDS):
    """
    Takes the simulated and reconstructed HDF5 file(s) (as converted from I3
    by icecube.hdfwriter.I3HDFTableService) as input and writes out a
    simplified PISA-standard-format HDF5 file for use in aeff, reco, and/or PID
    stages of the template maker.

    Parameters
    ----------
    data_files : dict
        File paths for finding data files for each flavor of neutrino,
        formatted as:
          {
            'nue':   [<list of nue+nuebar file paths>],
            'numu':  [<list of numu+numubar file paths>],
            'nutau': [<list of nutau+nutaubar file paths>]
          }
    outdir
        Directory path in which to store resulting files; will be generated if
        it does not already exist (including any parent directories that do not
        exist)
    run_settings
        An instantiated MCSimRunSettings object, instantiated e.g. from the
        PISA-standard resources/events/mc_sim_run_settings.json file
    data_proc_params
        An instantiated DataProcParams object, instantiated e.g. from the
        PISA-standard resources/events/data_proc_params.json file
    cut
        Name of a standard cut to use; must be specified in the relevant
        detector/processing version node of the data processing parameters
        (file from which the data_proc_params object was instantiated)
    join
        String specifying any flavor/interaction types (flavInts) to join
        together. Separate flavInts with commas (',') and separate groups
        with semicolons (';')
    cust_cuts
        dict with a single DataProcParams cut specification or list of same
        (see help for DataProcParams for detailed description of cut spec)
    extract_fields : iterable of strings
        Field names to extract from source HDF5 file
    output_fields : iterable of strings
        Fields to include in the generated PISA-standard-format events HDF5
        file; note that if 'weighted_aeff' is not preent, effective area will
        not be computed
    """

    # Create Events object to store data
    evts = events.Events()
    evts.metadata.update({
        'detector': run_settings.detector,
        'proc_ver': data_proc_params.proc_ver,
    })
    cuts = []
    if isinstance(cust_cuts, dict):
        cust_cuts = [cust_cuts]
    if cut is not None:
        evts.metadata['cuts'].append(cut)
        cuts.append(cut)
    if cust_cuts is not None:
        for ccut in cust_cuts:
            evts.metadata['cuts'].append('custom: ' + ccut['pass_if'])
            cuts.append(ccut)

    orig_outdir = outdir
    outdir = utils.expandPath(outdir)
    logging.info('Output dir spec\'d: %s', orig_outdir)
    if outdir != orig_outdir:
        logging.info('Output dir expands to: %s', outdir)
    utils.mkdir(outdir)

    detector_label = str(data_proc_params.detector)
    proc_label = 'proc_v' + str(data_proc_params.proc_ver)

    # What flavints to group together
    if join is None or join == '':
        grouped = []
        ungrouped = [flavInt.NuFlavIntGroup(k) for k in flavInt.ALL_NUFLAVINTS]
        groups_label = 'unjoined'
        logging.info('Events in the following groups will be joined together:'
                     ' (none)')
    else:
        grouped, ungrouped = flavInt.CombinedFlavIntData.xlateGroupsStr(join)
        evts.metadata['flavints_joined'] = [str(g) for g in grouped]
        groups_label = 'joined_G_' + '_G_'.join([str(g) for g in grouped])
        logging.info('Events in the following groups will be joined together: '
                     + '; '.join([str(g) for g in grouped]))

    # Find any flavints not included in the above groupings
    flavint_groupings = grouped + ungrouped
    if len(ungrouped) == 0:
        ungrouped = ['(none)']
    logging.info('Events of the following flavints will NOT be joined'
                 'together: ' + '; '.join([str(k) for k in ungrouped]))

    # Enforce that flavints composing groups are mutually exclusive
    for grp_n, flavintgrp0 in enumerate(flavint_groupings[:-1]):
        for flavintgrp1 in flavint_groupings[grp_n+1:]:
            assert len(set(flavintgrp0).intersection(set(flavintgrp1))) == 0

    flavintgrp_names = [str(flavintgrp) for flavintgrp in flavint_groupings]

    # Instantiate storage for all intermediate destination fields;
    # The data structure looks like:
    #   extracted_data[group #][interaction type][field name] = list of data
    extracted_data = [
        {
            inttype: {field:[] for field in extract_fields}
            for inttype in flavInt.ALL_NUINT_TYPES
        }
        for _ in flavintgrp_names
    ]

    # Instantiate generated-event counts for destination fields; count
    # nc separately from nc because aeff's for cc & nc add, whereas
    # aeffs intra-CC should be weighted-averaged (as for intra-NC)
    ngen = [
        {inttype:{} for inttype in flavInt.ALL_NUINT_TYPES}
        for _ in flavintgrp_names
    ]

    # Loop through all of the files, retrieving the events, filtering,
    # and recording the number of generated events pertinent to
    # calculating aeff
    filecount = {}
    all_runs = set()
    detector_geom = None
    for userspec_baseflav, fnames in data_files.iteritems():
        userspec_baseflav = flavInt.NuFlav(userspec_baseflav)
        for fname in fnames:
            # Retrieve data from all nodes specified in the processing
            # settings file
            data = data_proc_params.getData(fname, run_settings=run_settings)
            runs_in_data = set(data['run'])
            flavints_in_runs = None
            for run in runs_in_data:
                all_runs.add(run)
                if not run in filecount:
                    filecount[run] = 0
                filecount[run] += 1
                rs_run = run_settings[run]

                # Record geom; check that geom is consistent with other runs
                if detector_geom is None:
                    detector_geom = rs_run['geom']
                assert rs_run['geom'] == detector_geom, \
                        'All runs\' geometries must match!'

                # Check that all the runs in runs_in_data give the same flavints,
                # i.e. they must be sub runs for the same neutrino flavor.
                # (e.g. run 126001, 126002, 126003 are the 3 sub runs in nue file 12600)
                if flavints_in_runs is None:
                    flavints_in_runs = rs_run['flavints']
                assert rs_run['flavints'] == flavints_in_runs, \
                        'All sub runs\' flavints must match!'

            # Loop through all flavints spec'd for run
            for run_flavint in flavints_in_runs:
                barnobar = run_flavint.barNoBar()
                int_type = run_flavint.intType()

                # Retrieve this-interaction-type- & this-barnobar-only events
                # that also pass cuts. (note that cut names are strings)
                intonly_cut_data = data_proc_params.applyCuts(
                    data,
                    cuts=cuts+[str(int_type), str(barnobar)],
                    return_fields=extract_fields
                )

                # Record the generated count and data for this run/flavor for
                # each group to which it's applicable
                for grp_n, flavint_group in enumerate(flavint_groupings):
                    if not run_flavint in flavint_group:
                        continue
                    for run in runs_in_data:
                        # Instantiate a field for particles and anti-particles,
                        # keyed by the output of the barNoBar() method for each
                        if not run in ngen[grp_n][int_type]:
                            ngen[grp_n][int_type][run] = {
                                flavInt.NuFlav(12).barNoBar(): 0,
                                flavInt.NuFlav(-12).barNoBar(): 0,
                            }

                        # Record count only if it hasn't already been recorded
                        if ngen[grp_n][int_type][run][barnobar] == 0:
                            # Note that one_weight includes cc/nc:total fraction,
                            # so DO NOT specify the full flavint here, only flav
                            # (since one_weight does NOT take bar/nobar fraction,
                            # it must be included here in the ngen computation)
                            flav_ngen = run_settings.totGen(run=run,
                                                            barnobar=barnobar)
                            ngen[grp_n][int_type][run][barnobar] = flav_ngen

                    # Append the data. Note that extracted_data is:
                    # extracted_data[group n][int_type][extract field name] =
                    #   list
                    [extracted_data[grp_n][int_type][f].extend(
                        intonly_cut_data[f])
                     for f in extract_fields]


    # Compute "weighted_aeff" field:
    #
    # Within each int type (CC or NC), ngen should be added together;
    # events recorded of that int type then get their one_weight divided by the
    # total **for that int type only** to obtain the "weighted_aeff" for that
    # event (even if int types are being grouped/joined together).
    #
    # This has the effect that within a group, ...
    #   ... and within an interaction type, effective area is a weighted
    #   average of that of the flavors being combined. E.g. for CC,
    #
    #                  \sum_{run x}\sum_{flav y} (Aeff_{x,y} * ngen_{x,y})
    #       Aeff_CC = ----------------------------------------------------- ,
    #                       \sum_{run x}\sum_{flav y} (ngen_{x,y})
    #
    #   ... and then across interaction types, the results of the above for
    #   each int type need to be summed together, i.e.:
    #
    #       Aeff_total = Aeff_CC + Aeff_NC
    #
    # Note that each grouping of flavors is calculated with the above math
    # completely # independently from other flavor groupings specified.
    #
    # See Justin Lanfranchi's presentation on the PINGU Analysis call,
    # 2015-10-21, for more details.
    if 'weighted_aeff' in output_fields or 'mc_weight' in output_fields:
        fmtfields = (' '*12+'flavint_group',
                     'int type',
                     '     run',
                     'part/anti',
                     'part/anti count',
                     'aggregate count')
        fmt_n = [len(f) for f in fmtfields]
        fmt = '  '.join([r'%'+str(n)+r's' for n in fmt_n])
        lines = '  '.join(['-'*n for n in fmt_n])
        logging.info(fmt % fmtfields)
        logging.info(lines)
        for grp_n, flavint_group in enumerate(flavint_groupings):
            for int_type in flavInt.ALL_NUINT_TYPES:
                ngen_it_tot = {} 
                ngen_it_tot_tot = 0
                for run, run_counts in ngen[grp_n][int_type].iteritems():
                    ngen_it_tot[run] = 0
                    for barnobar, barnobar_counts in run_counts.iteritems():
                        ngen_it_tot_tot += barnobar_counts
                        ngen_it_tot[run] += barnobar_counts
                        logging.info(
                            fmt %
                            (flavint_group.simpleStr(), int_type, str(run),
                             barnobar, int(barnobar_counts), int(ngen_it_tot[run]))
                        )
                # Convert data to numpy array
                for field in extract_fields:
                    extracted_data[grp_n][int_type][field] = \
                            np.array(extracted_data[grp_n][int_type][field])

                # Generate weighted_aeff field for this group / int type's data
                if 'weighted_aeff' in output_fields:
                    # Get an array of ngen (different ngen for different sub run) in the data
                    array_ngen_it_tot = np.array([ngen_it_tot[run] for run in extracted_data[grp_n][int_type]['run']])
                    extracted_data[grp_n][int_type]['weighted_aeff'] = \
                            extracted_data[grp_n][int_type]['one_weight'] \
                            / array_ngen_it_tot * CMSQ_TO_MSQ
                if 'mc_weight' in output_fields:
                    array_ngen_it_tot = np.array([ngen_it_tot[run] for run in extracted_data[grp_n][int_type]['run']])
                    E = extracted_data[grp_n][int_type]['true_energy']
                    gamma = np.array([run_settings[run]['sim_spectral_index'] for run in extracted_data[grp_n][int_type]['run']])
                    Emax =  np.array([run_settings[run]['energy_max'] for run in extracted_data[grp_n][int_type]['run']])
                    Emin =  np.array([run_settings[run]['energy_min'] for run in extracted_data[grp_n][int_type]['run']])
                    extracted_data[grp_n][int_type]['mc_weight'] = \
                            np.power(E, gamma)/ array_ngen_it_tot / (Emax - Emin)

    # Report file count per run
    for run, count in filecount.items():
        logging.info('Files read, run %s: %d' % (run, count))
        ref_num_i3_files = run_settings[run]['num_i3_files']
        if count != ref_num_i3_files:
            logging.warn('Run %d, Number of files read (%d) != number of '
                         'source I3 files (%d), which may indicate an error.' %
                         (run, count, ref_num_i3_files))

    # Generate output data
    for flavint in flavInt.ALL_NUFLAVINTS:
        int_type = flavint.intType()
        for grp_n, flavint_group in enumerate(flavint_groupings):
            if not flavint in flavint_group:
                logging.trace('flavint %s not in flavint_group %s, passing.' %
                              (flavint, flavint_group))
                continue
            else:
                logging.trace(
                    'flavint %s **IS** in flavint_group %s, storing.' %
                    (flavint, flavint_group)
                )
            evts.set(
                flavint,
                {f: extracted_data[grp_n][int_type][f] for f in output_fields}
            )

    # Update metadata
    evts.metadata['geom'] = detector_geom
    evts.metadata['runs'] = sorted(all_runs)

    # Generate file name
    run_label = 'runs_' + utils.list2hrlist(all_runs)
    geom_label = '' + detector_geom
    fname = 'events__' + '__'.join([
        detector_label,
        geom_label,
        run_label,
        proc_label,
        groups_label,
    ]) + '.hdf5'

    outfpath = os.path.join(outdir, fname)
    logging.info('Writing events to ' + outfpath)

    # Save data to output file
    evts.save(outfpath)


def main():
    """Get command line arguments and call makeEventsFile() function."""

    parser = ArgumentParser(
        description='''Takes the simulated (and reconstructed) HDF5 file(s) (as
        converted from I3 by icecube.hdfwriter.I3HDFTableService) as input and
        writes out a simplified HDF5 file for use in the aeff and reco stages
        of the template maker.

        Example:
            $PISA/pisa/i3utils/make_events_file.py \
                --det "pingu" \
                --proc "5" \
                --nue ~/data/390/source_hdf5/*.hdf5 \
                --numu ~/data/389/source_hdf5/*.hdf5 \
                --nutau ~/data/388/source_hdf5/*.hdf5 \
                -vv \
                --outdir /tmp/events/''',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--det',
        metavar='DETECTOR',
        type=str,
        required=True,
        help='Detector, e.g. "PINGU" or "DeepCore". This'
             ' is used as the top-most key in run_settings.json and'
             ' data_proc_params.json files.'
    )
    parser.add_argument(
        '--proc',
        metavar='PROC_VER',
        type=str,
        required=True,
        help='''Processing version applied to simulation; processing versions
        are defined with respect to each geometry version. See
        data_proc_params.json file for definitions (or edit that file to add
        more).'''
    )

    parser.add_argument(
        '--nue',
        metavar='H5_FILE',
        type=str,
        nargs="+",
        required=True,
        help='nue HDF5 file(s)'
    )
    parser.add_argument(
        '--numu',
        metavar='H5_FILE',
        type=str,
        nargs="+",
        required=True,
        help='numu HDF5 file(s)'
    )
    parser.add_argument(
        '--nutau',
        metavar='H5_FILE',
        type=str,
        nargs="+",
        required=True,
        help='nutau HDF5 file(s)'
    )

    parser.add_argument(
        '--outdir',
        metavar='DIR',
        type=str,
        default='$PISA/pisa/resources/events',
        help='directory into which to store resulting HDF5 file'
    )

    parser.add_argument(
        '--run-settings',
        metavar='JSON_FILE',
        type=str,
        default='events/mc_sim_run_settings.json',
        help='JSON file with reference run settings'
    )

    parser.add_argument(
        '--data-proc-params',
        metavar='JSON_FILE',
        type=str,
        default='events/data_proc_params.json',
        help='JSON file with reference processing settings'
    )

    # NOTE:
    # Removed --join in favor of forcing standard events groupings to be output
    # all at once, to ensure all files get generated all the time. Also
    # need to implement validation for consistent events file usage in PISA for
    # template settings file (i.e., if different files are specified for
    # different PISA stages, ensure they all come from the same detector,
    # geometry, and processing versions and have events groupings that do not
    # lead to erroneous conclusions for the stages they're specified for)

    #parser.add_argument(
    #    '--join',
    #    const='nuecc,nuebarcc;numucc,numubarcc;nutaucc,nutaubarcc;'
    #          'nuallnc,nuallbarnc',
    #    default='',
    #    action='store',
    #    nargs='?',
    #    type=str,
    #    help= \
    #    '''Optionally join flavors together to increase statistics for Aeff
    #    and/or resolutions (aeff and reco stages, respectively). Specifying the
    #    --join option without an argument joins together: nu_x &
    #    nu_x_bar CC events together (one set for each of x=e, x=mu, and x=tau),
    #    and joins nuall NC & nuallbar NC events tegether. If a string
    #    argument is supplied, this specifies custom groups to join together
    #    instead. The string must be a semicolon-separated list each field of
    #    which itself a comma-separated list of event "flavints" (flavor and
    #    interaction type) to grup together. Any event flavint not included in
    #    that string will be found individually, i.e., not joined together with
    #    any other flavors'''
    #)

    parser.add_argument(
        '--cut',
        metavar='CUT_NAME',
        type=str,
        default='analysis',
        help='''Name of pre-defined cut to apply. See
        resources/events/data_proc_params.json for definitions for the detector
        and processing version you're working with (note that the names of cuts
        and what these entail varies by detector and processing version)'''
    )

    parser.add_argument(
        '--ccut-pass-if',
        metavar='CRITERIA',
        type=str,
        default='',
        help= \
        '''Custom cut: String containing criteria for passing a cut, using
        field names specified by the --ccut-fields argument. Standard Python-
        and numpy-namespace expressions are allowed as well, since this string
        is passed to 'eval'. E.g.:
        --ccut-fields="z:MCNeutrino/zenith,l6:IC86_Dunkman_L6/value" \
        --ccut-pass-if="(l6 == 1) & (z > pi/2)" '''
    )
    parser.add_argument(
        '--ccut-fields',
        metavar='FIELDS',
        type=str,
        default='',
        help='''Custom cut: String of comma-separated fields, each containing
        colon-separated (variable name : HDF5 address) tuples. For example,
        specifying:
        --ccut-fields="l5:IC86_Dunkman_L5/value,l6:IC86_Dunkman_L6/value"
        allows for a custom cut to be defined via --ccut-pass-if="(l5 == 1) &
        (l6 == 1)"'''
    )

    parser.add_argument(
        '--no-aeff',
        action='store_true',
        help='''Do not compute or include the 'weighted_aeff' field in the
        generated PISA events HDF5 file, disallowing use of the file for
        effective area parameterizations or the Monte Carlo aeff stage'''
    )

    parser.add_argument(
        '--no-pid',
        action='store_true',
        help='''Do not include the 'pid' field in the generated PISA events
        HDF5 file, disallowing use of the file for PID parameterizations or the
        Monte Carlo PID stage'''
    )

    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='set verbosity level'
    )

    args = parser.parse_args()

    set_verbosity(args.verbose)

    det = args.det.lower()
    proc = args.proc.lower()
    run_settings = MCSRS.DetMCSimRunsSettings(
        resources.find_resource(args.run_settings),
        detector=det
    )
    data_proc_params = DPP.DataProcParams(
        detector=det,
        proc_ver=proc,
        data_proc_params=resources.find_resource(args.data_proc_params),
    )

    logging.info('Using detector %s, processing version %s.' % (det, proc))

    extract_fields = deepcopy(EXTRACT_FIELDS)
    output_fields = deepcopy(OUTPUT_FIELDS)

    if args.no_pid:
        extract_fields = [f for f in extract_fields if f != 'pid']
        output_fields = [f for f in output_fields if f != 'pid']

    if args.no_aeff:
        output_fields = [f for f in output_fields if f != 'weighted_aeff']

    # Add any custom cuts specified on command line
    ccut = None
    if args.ccut_pass_if:
        ccut = {
            'pass_if': args.ccut_pass_if,
            'fields': args.ccut_fields.split(',')
        }

    # One events file will be produced for each of The following flavint
    # groupings
    groupings = [
        # No events joined together
        None,
        # CC events unjoined; join nuall NC and nuallbar NC separately (used
        # for generating aeff param service's aeff parameterizations)
        'nuallnc;nuallbarnc',
        # CC events paried by flav--anti-flav; nuallNC+nuallbarNC all joined
        # together; used for reco services (MC and vbwkde)
        'nuecc+nuebarcc;numucc+numubarcc;nutaucc+nutaubarcc;nuallnc+nuallbarnc',
    ]
    groupings = [None]    

    # Create the events files
    for grouping in groupings:
        makeEventsFile(
            data_files={'nue':args.nue, 'numu':args.numu, 'nutau':args.nutau},
            outdir=args.outdir,
            run_settings=run_settings,
            data_proc_params=data_proc_params,
            join=grouping,
            cut=args.cut,
            cust_cuts=ccut,
            extract_fields=extract_fields,
            output_fields=output_fields,
        )


if __name__ == "__main__":
    main()
import os
import sys

import h5py
import numpy as np

from pisa import ureg, Q_
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.map import Map, MapSet
from pisa.core.stage import Stage
from pisa.utils.log import logging
from pisa.utils.comparisons import normQuant
from pisa.utils.resources import find_resource
import copy
import pisa.utils.mcSimRunSettings as MCSRS
import pisa.utils.dataProcParams as DPP


class data(Stage):
    """Data loader stage

    Paramaters
    ----------

    params : ParamSet
        data_file : string
            path pointing to the hdf5 file containing the events
        proc_ver: string
            indicateing the proc version, for example msu_5digit
        bdt_cut : float
            futher cut apllied to events for the atm. muon rejections BDT

    Notes
    -----

    The curent versio of this code is a port from pisa v2 nutau branch.
    It clearly needs to be cleand up properly at some point.

    """

    def __init__(self, params, output_binning, disk_cache=None,
                memcache_deepcopy=True, error_method=None,
                outputs_cache_depth=20, debug_mode=None):

        expected_params = (
            'data_file',
            'proc_ver',
            'bdt_cut'
        )

        output_names = ('evts')

        super(self.__class__, self).__init__(
            use_transforms=False,
            params=params,
            expected_params=expected_params,
            output_names=output_names,
            error_method=error_method,
            disk_cache=disk_cache,
            memcache_deepcopy=memcache_deepcopy,
            outputs_cache_depth=outputs_cache_depth,
            output_binning=output_binning,
            debug_mode=debug_mode
        )

    def _compute_nominal_outputs(self):
        """load the evnts from file, perform sanity checks and histogram them
        (into final MapSet)

        """
        # get params
        data_file_name = self.params.data_file.value
        proc_version = self.params.proc_ver.value
        bdt_cut = self.params.bdt_cut.value.m_as('dimensionless')

        self.bin_names = self.output_binning.names

        # TODO: convert units using e.g. `comp_units` in stages/reco/hist.py
        self.bin_edges = []
        for name in self.bin_names:
            if 'energy' in  name:
                bin_edges = self.output_binning[name].bin_edges.to('GeV').magnitude
            else:
                bin_edges = self.output_binning[name].bin_edges.magnitude
            self.bin_edges.append(bin_edges)

        # get data with cuts defined as 'analysis' in data_proc_params.json
        fields = ['reco_energy', 'pid', 'reco_coszen']
        cut_events = self.get_fields(fields, cuts='analysis',
                        run_setting_file='events/mc_sim_run_settings.json',
                        data_proc_file='events/data_proc_params.json')
        hist, _ = np.histogramdd(sample = np.array(
            [cut_events[bin_name] for bin_name in self.bin_names]
        ).T, bins=self.bin_edges)

        maps = [Map(name=self.output_names[0], hist=hist,
                    binning=self.output_binning)]
        self.template = MapSet(maps, name='data')

    def _compute_outputs(self, inputs=None):
        """return the precomputed MpSets, since this is data, the distributions
        don't change

        """
        return self.template

    def get_fields(self, fields, cuts='analysis', run_setting_file='events/mc_sim_run_settings.json',
                        data_proc_file='events/data_proc_params.json'):
        """ Return data events' fields
        
        Paramaters
        ----------
        fields: list of strings
            the quantities to return, for example: ['reco_energy', 'pid', 'reco_coszen']

        """
        # get param
        data_file_name = self.params.data_file.value
        proc_version = self.params.proc_ver.value
        bdt_cut = self.params.bdt_cut.value.m_as('dimensionless')
        data_proc_params = DPP.DataProcParams(
                detector='deepcore',
                proc_ver=proc_version,
                data_proc_params=find_resource(data_proc_file))
        run_settings = MCSRS.DetMCSimRunsSettings(find_resource(run_setting_file), detector='deepcore')
        data = data_proc_params.getData(find_resource(data_file_name), run_settings=run_settings, file_type='data')
        fields_for_cuts = copy.deepcopy(fields)
        for param in ['pid', 'dunkman_L3', 'dunkman_L4', 'dunkman_L5', 'dunkman_L6', 'reco_energy', 'reco_zenith']:
            if param not in fields:
                fields_for_cuts.append(param)

        # get data after cuts
        cut_data = data_proc_params.applyCuts(data, cuts=cuts, return_fields=fields_for_cuts)
        cut_data['reco_coszen'] = np.cos(cut_data['reco_zenith'])

        dunkman_L4 = cut_data['dunkman_L4']
        bdt_score = cut_data['dunkman_L5']
        dunkman_L6 = cut_data['dunkman_L6']
        reco_energy = cut_data['reco_energy']
        reco_coszen = cut_data['reco_coszen']
        pid = cut_data['pid']
        all_cuts = np.logical_and(dunkman_L4==1, dunkman_L6==1)
        all_cuts = np.logical_and(all_cuts, bdt_score>=bdt_cut)
        logging.info(
            "Cut2, removing events with bdt_score < %s i.e. only keep bdt > %s"
            %(bdt_cut, bdt_cut)
        )
        for bin_name, bin_edge in zip(self.bin_names, self.bin_edges):
            bin_cut = np.logical_and(cut_data[bin_name]<= bin_edge[-1], cut_data[bin_name]>= bin_edge[0])
            all_cuts = np.logical_and(all_cuts, bin_cut)
        return_data = {} 
        for key in fields:
            return_data[key] = cut_data[key][all_cuts]
        return return_data

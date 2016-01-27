#!/usr/bin/env python
#
# author: Justin L. Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   October 24, 2015
"""Handle Monte Carlo simulation run settings"""


import pisa.utils.fileio as fileio
import pisa.utils.flavInt as flavInt
from pisa.resources import resources as resources

# Following "import *" is intentionally done so that `eval` called in
# translateSourceDict will execute with direct access to numpy namespace
from numpy import *


# TODO: make ability to serialize instantiated MCSimRunSettings and
# DetMCSimRunsSettings objects back to JSON format from which they were
# generated

# TODO: put more thought into the MCSimRunSettings class

# TODO: make sure user is forced to choose run & detector here

class MCSimRunSettings(dict):
    """Handle Monte Carlo run settings

    Parameters
    ----------
    run_settings : string or dict
    run
    detector : string or None

    Notes
    -----
    run_settings dictionary format (and same for corresponding JSON file, e.g.
    resources/events/mc_sim_run_settings.json); example is for PINGU but should
    generalize to DeepCore, etc. Note that a JSON file will use null and not
    None.

    (Also note that expressions for numerical values utilizing the Python/numpy
    namespace are valid, e.g. "2*pi" will be evaluated within the code to its
    decimal value.)

    {

      # Specify the detector name, lower case
      "pingu": {

        # Monte Carlo run number for this detector
        "388": {

          # Version of geometry, lower-case. E.g., ic86 for IceCube/DeepCore
          "geom": "v36",

          # A straightforward way of computing aeff/veff/meff is to keep all
          # simulated events and compare the #analysis/#simulated. If all
          # simulated events are kept, then the filename containing these is
          # recorded here.
          "all_gen_events_file": None,

          # Max and min azimuth angle simulated (rad)
          "azimuth_max": "2*pi",
          "azimuth_min": 0,

          # Max and min energy simulated (GeV)
          "energy_max": 80,
          "energy_min": 1,

          # GENIE simulates some un-physica events (interactions that will not
          # occur in nature). The number below was arrived at by Ken Clark, so
          # ask him for more info.
          "genie_physical_factor": 0.8095,

          # GENIE has a prescale factor (TODO: generalize or eliminate for
          # other xsec?)
          "genie_prescale_factor": 1.2,

          # Neutrino flavors simulated
          "flavints": "nutau,nutaubar",

          # #nu / (#nu + #nubar) simulated
          "nu_to_total_fract": 0.5,

          # Number of events simulated per I3 file
          "num_events_per_file": 250000,

          # Number of I3 files used
          "num_i3_files": 195,

          # Simulated spectral inde gamma; value of 1 => E*{-1}
          "sim_spectral_index": 1,

          # Version of neutrion/ice cross sections used for the simulation
          "xsec": "genie_2.6.4",

          # Max and min zenith angle simulated (rad)
          "zenith_max": "pi",
          "zenith_min": 0
        }
      }
    }


    """
    def __init__(self, run_settings, run=None, detector=None):
        # TODO: clean up this constructor!
        #if isinstance(run_settings, basestring):
        #    rsd = jsons.from_json(resources.find_resource(run_settings))
        if isinstance(run_settings, dict):
            rsd = run_settings
        else:
            raise TypeError('Unhandled run_settings type passed in arg: ' +
                            type(run_settings))
        #if detector is not None:
        #    try:
        #        rsd = rsd[detector]
        #    except:
        #        pass
        rsd = self.translateSourceDict(rsd) #{run:self.translateSourceDict(rs) for run,rs in rsd.iteritems()}
        if not detector is None:
            detector = str(detector).lower()
        self.detector = detector
        self.run = run
        self.update(rsd)

    @staticmethod
    def translateSourceDict(d):
        d['tot_gen'] = d['num_events_per_file'] * d['num_i3_files']
        d['flavints'] = flavInt.NuFlavIntGroup(d['flavints'])

        # Numeric fields are allowed to be expressions that get evaluated
        numeric_fields = ['azimuth_max',
                          'azimuth_min',
                          'energy_max',
                          'energy_min',
                          'genie_physical_factor',
                          'genie_prescale_factor',
                          'nu_to_total_fract',
                          'num_events_per_file',
                          'num_i3_files',
                          'sim_spectral_index',
                          'zenith_max',
                          'zenith_min',]
        for f in numeric_fields:
            if isinstance(d[f], basestring):
                d[f] = eval(d[f])

        return d

    def consistencyChecks(self, data, flav=None):
        # TODO: implement!
        pass

    def barnobarfract(self, barnobar=None, is_particle=None,
                      flav_or_flavint=None):
        nargs = sum([(not barnobar is None),
                     (not is_particle is None),
                     (not flav_or_flavint is None)])
        if nargs != 1:
            raise ValueError('One and only one of barnobar, is_particle, and'
                             ' flav_or_flavint must be specified. Got ' +
                             str(nargs) + ' args instead.')

        if not flav_or_flavint is None:
            is_particle = flavInt.NuFlavInt(flav_or_flavint).isParticle()
        if is_particle:
            return self['nu_to_total_fract']
        return 1 - self['nu_to_total_fract']

    def totGen(self, barnobar=None, is_particle=None, flav_or_flavint=None):
        nargs = sum([(not barnobar is None),
                     (not is_particle is None),
                     (not flav_or_flavint is None)])
        if nargs == 0:
            fract = 1.0
        else:
            fract = self.barnobarfract(barnobar=barnobar,
                                       is_particle=is_particle,
                                       flav_or_flavint=flav_or_flavint)

        return fract * self['tot_gen']


class DetMCSimRunsSettings(dict):
    """Handle Monte Carlo run settings for a detector (i.e., without specifying
    which run as is required for the MCSimRunSettings object)

    Since run is not specified at instantiation, method calls require the user
    to specify a run ID.

    Parameters
    ----------
    run_settings : string or dict
    detector : string or None

    See Also
    --------
    MCSimRunSettings : Same, but specifies a specific run at instantiation; see
                       class docstring for specification of run_settings dict /
                       JSON file
    """
    def __init__(self, run_settings, detector=None):
        if isinstance(run_settings, basestring):
            rsd = fileio.from_file(resources.find_resource(run_settings))
        elif isinstance(run_settings, dict):
            rsd = run_settings
        else:
            raise TypeError('Unhandled run_settings type passed in arg: ' +
                            type(run_settings))

        if detector:
            detector = str(detector).lower()
        self.detector = detector

        # Determine how deeply nested runs are in the dict to allow for
        # user to specify a dict that has multiple detectors in it OR
        # a dict with just a single detector in it
        if rsd.values()[0].has_key('flavints'):
            runs_d = rsd
        elif rsd.values()[0].values()[0].has_key('flavints'):
            if self.detector is None:
                if len(rsd) == 1:
                    runs_d = rsd.values()[0]
                else:
                    raise ValueError('Must specify which detector; detectors '
                                     'found: ' + str(rsd.keys()))
            else:
                runs_d = rsd[self.detector.lower()]
        else:
            raise Exception('dict must either be 3 levels: '
                            '{DET:{RUN:{...}}}; or 2 levels: {RUN:{...}}')

        # Convert the run numbers to integers (JSON files cannot have an int as
        # a key, so it is a string upon import) and convert actual run settings
        # dict to MCSimRunSettings instances
        runs_d = {int(k): MCSimRunSettings(v) for k,v in runs_d.iteritems()}

        # Save the runs_d to this object instance, which behaves like a dict
        self.update(runs_d)

    def consistencyChecks(self, data, run, flav=None):
        pass

    def barnobarfract(self, run, *args, **kwargs):
        return self[run].barnobarfract(*args, **kwargs)

    def totGen(self, run, *args, **kwargs):
        return self[run].totGen(*args, **kwargs)

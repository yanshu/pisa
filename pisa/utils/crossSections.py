#!/usr/bin/env python
#
# author: J.L. Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   October 24, 2015
"""
Define CrossSections class for importing, working with, and storing neutrino
cross sections
"""

from copy import deepcopy
import os, sys

import numpy as np
from scipy.interpolate import interp1d

from pisa.utils.resources import find_resource
from pisa.utils.fileio import expandPath, from_file, to_file
from pisa.utils import flavInt
from pisa.utils.log import logging, set_verbosity


__all__ = ['CrossSections']


# TODO: make class for groups of CX or just a function for finding eisting
# versions in a json file; eliminate the optional ver parameters in
# CrossSections class methods


class CrossSections(flavInt.FlavIntData):
    """Cross sections for each neutrino flavor & interaction type ("flavint").
    What is stored are *PER-H20-MOLECULE* cross sections, in units of [m^2].

    Be careful when using these; cross sections are often plotted in the
    literature as per-nucleon and per-energy with units [10^-38 cm^2 / GeV].

    ver : string
        Version of cross sections to load from file. E.g. 'genie_2.6.4'

    energy : None or array of float
        Energies at which cross sections are defined.

    xsec : string, dictonary, or another CrossSections object
        If str: provides PISA resource name from which to load cross sections
        If dict or CrossSections object: construct cross sections from a
            deepcopy of that object
    """
    def __init__(self, ver=None, energy=None,
                 xsec='cross_sections/cross_sections.json'):
        super(CrossSections, self).__init__()
        self.energy = energy
        self.__ver = ver
        self.__interpolants = {}
        if xsec is None:
            pass
        elif isinstance(xsec, dict):
            xsec = deepcopy(xsec)
        elif isinstance(xsec, basestring):
            assert self.energy is None
            self.energy, xsec = self.load(fpath=xsec, ver=ver)
        else:
            raise TypeError('Unhandled xsec type passed in arg: ' +
                            str(type(xsec)))
        if xsec is not None:
            super(CrossSections, self).validate(xsec)
            self.validate_xsec(self.energy, xsec)
            self.update(xsec)
            self.__define_interpolant()

    @staticmethod
    def load(fpath, ver=None, **kwargs):
        """Load cross sections from a file locatable and readable by the PISA
        from_file command. If `ver` is provided, it is used to index into the
        top level of the loaded dictionary"""
        all_xsec = from_file(fpath, **kwargs)
        if ver not in all_xsec:
            raise ValueError('Version "%s" not found. Valid versions in file'
                             '"%s" are: %s' % (ver, fpath, all_xsec.keys()))
        return all_xsec[ver]['energy'], all_xsec[ver]['xsec']

    @staticmethod
    def loadROOTFile(fpath, ver, tot_sfx='_tot', o_sfx='_o16', h_sfx='_h1',
                     plt_sfx='_plot'):
        """Load cross sections from root file, where graphs are first-level in
        hierarchy. This is yet crude and not very flexible, but at least it's
        recorded here for posterity.

        Requires ROOT and ROOT python module be installed

        Parameters
        ----------
        fpath : string
            Path to ROOT file
        ver : string
            Necessary to differentaite among different file formats that Ken
            has sent out
        tot_sfx : string (default = '_tot')
            Suffix for finding total cross sections in ROOT file (if these
            fields are found, the oxygen/hydrogen fields are skipped)
        o_sfx : string (default = '_o16')
            Suffix for finding oxygen-16 cross sections in ROOT file
        h_sfx : string (default = '_h1')
            Suffix for finding hydrogen-1 cross sections in ROOT file
        plt_sfx : string (default = '_plt')
            Suffix for plots containing cross sections per GeV in ROOT file

        Returns
        -------
        xsec : flavInt.FlavIntData
            Object containing the loaded cross sections
        """
        import ROOT

        def extractData(f, key):
            try:
                g = ROOT.gDirectory.Get(key)
                x = np.array([g.GetX()[i] for i in xrange(g.GetN())])
                y = np.array([g.GetY()[i] for i in xrange(g.GetN())])
            except AttributeError:
                raise ValueError('Possibly missing file "%s" or missing key'
                                 '"%s" within that file?' % (f, key))
            return x, y

        rfile = ROOT.TFile(fpath)
        try:
            energy = None
            xsec = flavInt.FlavIntData()
            for flavint in flavInt.ALL_NUFLAVINTS:
                fi_str = str(flavint)
                if ver == 'genie_2.6.4':
                    # Expected to contain xsect per atom; summing 2*Hydrogen
                    # and 1*Oxygen yields total cross section for water
                    # molecule.
                    O16_e, O16_xs = extractData(rfile, fi_str + o_sfx)
                    H1_e, H1_xs = extractData(rfile, fi_str + h_sfx)
                    tot_xs = (H1_xs*2 + O16_xs*1)
                    assert np.alltrue(H1_e == O16_e)
                    ext_e = O16_e
                elif ver == 'genie_2.8.6':
                    # Expected to contain xsect-per-nucleon-per-energy, so
                    # multiplying by energy and by # of nucleons (18) yields
                    # cross sections per molecule.
                    ext_e, fract_xs = extractData(rfile, fi_str + plt_sfx)
                    tot_xs = fract_xs * ext_e * 18
                else:
                    raise ValueError('Invalid or not implemented `ver`: "%s"'
                                     % ver)
                if energy is None:
                    energy = ext_e
                assert np.alltrue(ext_e == energy)
                # Note that units in the ROOT files are [1e-38 cm^2] but PISA
                # requires units of [m^2], so this conversion is made here.
                xsec[flavint] = tot_xs * 1e-38 * 1e-4
        finally:
            rfile.Close()

        CrossSections.validate_xsec(energy, xsec)

        return energy, xsec

    @classmethod
    def newFromROOT(cls, fpath, ver, **kwargs):
        """Instantiate a new CrossSections object from ROOT file.

        Parameters
        ----------
        fpath : string
            PISA resource specification for location of ROOT file
        ver : string
            Specify the version name of the cross sections loaded
        **kwargs
            Passed to method loadROOTFile()

        Returns
        -------
        Instantiated CrossSections object
        """
        energy, xsec = CrossSections.loadROOTFile(fpath, ver=ver, **kwargs)
        return cls(energy=energy, xsec=xsec, ver=ver)

    @staticmethod
    def validate_xsec(energy, xsec):
        """Validate cross sections"""
        # TODO: different validation based on cross sections version string

        # Make sure the basics are present
        xsec = flavInt.FlavIntData(xsec)

        ## No NaN's
        assert not np.any(np.isnan(energy))
        # Energy spans at least 1-100 GeV
        assert np.min(energy) <= 1
        assert np.max(energy) >= 100

        # All event flavints need to be present
        for k in flavInt.ALL_NUFLAVINTS:
            # Uses "standard" PISA indexing scheme
            x = xsec[k]
            # Arrays are same lengths
            assert len(x) == len(energy)
            # No NaN's
            assert np.sum(np.isnan(x)) == 0
            # Max xsec/energy value is in range for units of [m^2/GeV]
            assert np.max(x/energy) < 40e-42, np.max(x/energy)

    def set_version(self, ver):
        """Set the cross sections version to the string `ver`."""
        self.__ver = ver

    def get_version(self):
        """Return the cross sections version string"""
        return self.__ver

    def save(self, fpath, ver=None, **kwargs):
        """Save cross sections (and the energy specification) to a file at
        `fpath`."""
        if ver is None:
            if self.__ver is None:
                raise ValueError(
                    'Either a ver must be specified in call to `save` or it '
                    'must have been set prior to the invocation of `save`.'
                )
            ver = self.__ver
        else:
            assert ver == self.__ver

        try:
            fpath = find_resource(fpath)
        except IOError:
            pass
        fpath = os.path.expandvars(os.path.expanduser(fpath))
        all_xs = {}
        # Get any existing data from file
        if os.path.exists(fpath):
            all_xs = from_file(fpath)
        # Validate existing data by instantiating objects from each
        for v, d in all_xs.iteritems():
            CrossSections(ver=v, energy=d['energy'], xsec=d['xsec'])
        if ver in all_xs:
            logging.warning('Overwriting existing version "' + ver +
                            '" in file ' + fpath)
        all_xs[ver] = {'xsec':self, 'energy':self.energy}
        to_file(all_xs, fpath, **kwargs)

    def get_xs_value(self, flavintgroup, energy):
        """Get (combined) cross section value (in units of m^2) for
        `flavintgroup` at `energy` (in units of GeV).

        Parameters
        ----------
        flavintgroup : NuFlavIntGroup or convertible thereto
        energy : numeric or sequence thereof
            Energy (or energies) at which to evaluate total cross section, in
            units of GeV

        Returns
        -------
        Combined cross section for flavor/interaction types in units of
        m^2, evaluated at each energy. Shape of returned value matches that of
        passed `energy` parameter.
        """
        flavintgroup = flavInt.NuFlavIntGroup(flavintgroup)
        if flavintgroup not in self.__interpolants:
            self.__define_interpolant(flavintgroup=flavintgroup)
        return self.__interpolants[flavintgroup](energy)

    def get_xs_ratio_value(self, flavintgroup0, flavintgroup1, energy,
                           gamma=0):
        """Get ratio of combined cross sections for `flavintgroup0` to combined
        cross sections for `flavintgroup1`, weighted by E^{-`gamma`}.

        Parameters
        ----------
        flavintgroup0, flavintgroup1 : NuFlavIntGroup or convertible thereto
        energy : numeric or sequence thereof
            Energy (or energies) at which to evaluate total cross section, in
            units of GeV

        Returns
        -------
        Ratio of combined cross sections flavintgroup0 / flavintgroup1
        evaluated at each energy. Shape of returned value matches that of
        passed `energy` parameter.
        """
        flavintgroup0 = flavInt.NuFlavIntGroup(flavintgroup0)
        flavintgroup1 = flavInt.NuFlavIntGroup(flavintgroup1)

        self.__define_interpolant(flavintgroup=flavintgroup0)
        self.__define_interpolant(flavintgroup=flavintgroup1)

        xs_ratio_vals = self.__interpolants[flavintgroup0](energy) / \
                self.__interpolants[flavintgroup1](energy)
        # Special case to avoid multiplying by array of ones
        if gamma == 0:
            return xs_ratio_vals
        return xs_ratio_vals * energy**(-gamma)

    def __define_interpolant(self, flavintgroup=None):
        """If `flavintgroup` is None, compute all (separate) flavint
        interpolants; otherwise, compute interpolant for specified
        `flavintgroup`. Do not re-compute if already present.
        """
        if flavintgroup is None:
            flavintgroups = [flavInt.NuFlavIntGroup(fi)
                             for fi in self.flavints()]
        else:
            flavintgroups = [flavInt.NuFlavIntGroup(flavintgroup)]

        for flavintgroup in flavintgroups:
            if flavintgroup in self.__interpolants:
                continue
            combined_xs = self.__combineXS(flavintgroup)
            self.__interpolants[flavintgroup] = \
                    interp1d(x=self.energy, y=combined_xs, kind='linear',
                             copy=False, bounds_error=True, fill_value=0)

    def __combineXS(self, flavintgroup):
        """Combine all cross sections specified by the flavints in
        `flavintgroup`. All CC and NC interactions are separately grouped
        together and averaged, then the average of each interaction type
        is added to the other.

        If CC and NC interactions are present, they *must* be from the same
        flavor(s). I.e., it doesn't make sense (and so causes an exception) if
        you combine numu CC with numubar NC. It does make sense if you combine
        numu and numubar CC with numu and numubar NC, though, and this is
        allowed.

        Notes
        -----
        Does not yet implement *Ngen/spectrum-weighted* averages, which are
        necessary when combining cross sections of disparate flavor/interaction
        types from different Monte Carlo simulation runs.
        """
        flavintgroup = flavInt.NuFlavIntGroup(flavintgroup)
        # Trivial case: nothing to combine
        if len(flavintgroup.flavints()) == 1:
            return self[flavintgroup.flavints()[0]]

        cc_flavints = flavintgroup.ccFlavInts()
        nc_flavints = flavintgroup.ncFlavInts()
        if cc_flavints and nc_flavints:
            assert flavintgroup.ccFlavs() == flavintgroup.ncFlavs(), \
                    'Combining CC and NC but CC flavors do not match NC flavors'
        cc_avg_xs = 0
        if cc_flavints:
            logging.trace('cc_flavints = %s' % (cc_flavints,))
            cc_avg_xs = np.sum([self[k] for k in cc_flavints], axis=0) \
                    / len(cc_flavints)
        nc_avg_xs = 0
        if nc_flavints:
            logging.trace('nc_flavints = %s' % (nc_flavints,))
            nc_avg_xs = np.sum([self[k] for k in nc_flavints], axis=0) \
                    / len(nc_flavints)
        tot_xs = cc_avg_xs + nc_avg_xs
        logging.trace('mean(tot_xs) = %s' % (np.mean(tot_xs),))
        return tot_xs

    def get_xs_ratio_integral(self, flavintgroup0, flavintgroup1, e_range,
                    gamma=0, average=False):
        """Energy-spectrum-weighted integral of (possibly a ratio of)
        (possibly-combined) flavor/interaction type cross sections.

        Parameters
        ----------
        flavintgroup0 : NuFlavIntGroup or convertible thereto
            Flavor(s)/interaction type(s) for which to combine cross sections
            for numerator of ratio
        flavintgroup1 : None, NuFlavIntGroup or convertible thereto
            Flavor(s)/interaction type(s) for which to combine cross sections
            for denominator of ratio. If None is passed, the denominator of
            the "ratio" is effectively 1.
        e_range
            Range of energy over which to integrate (GeV)
        gamma : float >= 0
            Power law spectral index used for weighting the integral,
            E^{-`gamma`}. Note that `gamma` should be >= 0.
        average : bool
            If True, return the average of the cross section (ratio)
            If False, return the integral of the cross section (ratio)

        See also
        --------
        See __combineXS for detals on how flavints are combined.
        """
        e_min = min(e_range)
        e_max = max(e_range)

        assert e_min > 0, '`e_range` must lie strictly above 0'
        assert e_max > e_min, \
                'max(`e_range`) must be strictly larger than min(`e_range`)'
        assert gamma >= 0, '`gamma` must be >= 0'

        if flavintgroup1 is None:
            flavintgroups = [flavInt.NuFlavIntGroup(flavintgroup0)]
        else:
            flavintgroups = [flavInt.NuFlavIntGroup(flavintgroup0),
                             flavInt.NuFlavIntGroup(flavintgroup1)]

        # Create interpolant(s) (to get xs at  energy range's endpoints)
        [self.__define_interpolant(flavintgroup=fg) for fg in flavintgroups]

        all_energy = self.__interpolants[flavintgroups[0]].x
        xs_data = [self.__interpolants[fg].y for fg in flavintgroups]

        for xd in xs_data:
            logging.trace('mean(xs_data) = %e' % np.mean(xd))

        # Get indices of data points within the specified energy range
        idx = (all_energy > e_min) & (all_energy < e_max)

        # Get xsec at endpoints
        xs_endpoints = [self.__interpolants[fg]((e_min, e_max))
                        for fg in flavintgroups]

        for ep in xs_endpoints:
            logging.trace('xs_emin = %e, xs_emax = %e' % (ep[0], ep[1]))

        # Attach endpoints
        energy = np.concatenate([[e_min], all_energy[idx], [e_max]])
        xs = [np.concatenate([[ep[0]], xsd[idx], [ep[1]]])
              for ep, xsd in zip(xs_endpoints, xs_data)]

        if len(xs) == 1:
            xs = xs[0]
        else:
            xs = xs[0] / xs[1]

        # Weight xsec (or ratio) by energy spectrum
        if gamma == 0:
            wtd_xs = xs
        else:
            wtd_xs = xs*energy**(-gamma)

        logging.trace('mean(wtd_xs) = %e' % np.mean(wtd_xs))

        # Integrate via trapezoidal rule
        wtd_xs_integral = np.trapz(y=wtd_xs, x=energy)

        logging.trace('wtd_xs_integral = %e' % wtd_xs_integral)

        # Need to divide by integral of the weight function (over the same
        # energy interval as wtd_xs integral was computed) to get the average
        if average:
            if gamma == 0:
                # Trivial case
                xs_average = wtd_xs_integral / (e_max - e_min)
            else:
                # Otherwise use trapezoidal rule to approximate integral
                xs_average = wtd_xs_integral / \
                        np.trapz(y=energy**(-gamma), x=energy) #* (e_max-e_min)
            logging.trace('xs_average = %e' %(xs_average))
            return xs_average

        return wtd_xs_integral

    #def get_xs_integral(self, flavintgroup, e_range, gamma=0, average=False):
    #    """Energy-spectrum-weighted integral or average of (possibly-combined)
    #    flavor/interaction type cross section.

    #    Parameters
    #    ----------
    #    flavintgroup : NuFlavIntGroup or convertible thereto
    #        Flavor(s)/interaction type(s) for which to combine cross sections
    #    e_range
    #        Range of energy over which to integrate (GeV)
    #    gamma : float >= 0
    #        Power law spectral index used for weighting the integral,
    #        E^{-`gamma`}. Note that `gamma` should be >= 0.
    #    average : bool
    #        True: return the average of the cross section over `e_range`
    #        False: return the integral of the cross section over `e_range`

    #    of flavints specfied in `flavintgroup`; if `gamma` specified, weight
    #    integral by simulated spectrum with that power-spectral index
    #    (i.e.: E^{-gamma}).
    #    
    #    Specifying `average`=True yields the weighted-average of cross section.

    #    See also
    #    --------
    #    See __combineXS for detals on how flavints are combined.
    #    """
    #    return self.get_xs_ratio_integral(
    #        flavintgroup0=flavintgroup, flavintgroup1=None,
    #        e_range=e_range, gamma=gamma, average=average
    #    )

    def plot(self, save=None):
        """Plot cross sections per GeV; optionally, save plot to file. Requires
        matplotlib and optionally uses seaborn to set "pretty" defaults.

        save : None, str, or dict
            If None, figure is not saved (default)
            If a string, saves figure as that name
            If a dict, calls pyplot.savefig(**save) (i.e., save contains
            keyword args to savefig)
        """
        import matplotlib
        matplotlib.use('pdf')
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns
        except ImportError:
            pass

        if self.__ver is None:
            leg_ver = ''
        else:
            leg_ver = self.__ver + '\n'
        leg_fs = 11
        figsize = (9, 6)
        alpha = 1.0
        f = plt.figure(figsize=figsize)
        f.clf()
        ax1 = f.add_subplot(121)
        ax2 = f.add_subplot(122)
        ls = [dict(lw=5, ls='-'),
              dict(lw=2, ls='-'),
              dict(lw=1, ls='--'),
              dict(lw=5, ls='-'),
              dict(lw=3, ls='-.'),
              dict(lw=1, ls='-')]

        energy = self.energy
        nc_n = cc_n = 0
        for flavint in list(flavInt.ALL_NUFLAVINTS.particles()) + \
                list(flavInt.ALL_NUFLAVINTS.antiParticles()):
            # Convert from [m^2] to [1e-38 cm^2]
            xs = self[flavint] * 1e38 * 1e4
            if flavint.isCC():
                ax1.plot(energy, xs/energy,
                         alpha=alpha,
                         label=flavInt.tex(flavint.flav(), d=1),
                         **ls[cc_n%len(ls)])
                cc_n += 1
            else:
                ax2.plot(energy, xs/energy,
                         alpha=alpha,
                         label=flavInt.tex(flavint.flav(), d=1),
                         **ls[nc_n%len(ls)])
                nc_n += 1

        l1 = ax1.legend(title=leg_ver +
                        r'$\nu+{\rm H_2O}$, total CC',
                        fontsize=leg_fs, frameon=False, ncol=2)
        l2 = ax2.legend(title=leg_ver +
                        r'$\nu+{\rm H_2O}$, total NC',
                        fontsize=leg_fs, frameon=False, ncol=2)

        for (ax, leg) in [(ax1, l1), (ax2, l2)]:
            ax.set_xlim(0.1, 100)
            ax.set_xscale('log')
            ax.set_xlabel(r'$E_\nu\,{\rm [GeV]}$')
            ax.set_ylabel(
                r'$\sigma(E_\nu)/E_\nu\quad ' +
                r'\left[10^{-38} \mathrm{cm^2} \mathrm{GeV^{-1}}\right]$'
            )
            leg.get_title().set_fontsize(leg_fs)
        plt.tight_layout()

        if isinstance(save, basestring):
            logging.info('Saving cross sections plots to file ' + save)
            f.savefig(save)
        elif isinstance(save, dict):
            logging.info('Saving cross sections plots using figure.save'
                         ' params %s' % (save,))
            f.savefig(**save)


def test_CrossSections(outdir=None):
    import tempfile
    set_verbosity(2)
    if outdir is None:
        outdir = tempfile.mkdtemp()

    # "Standard" location of cross sections file in PISA; retrieve 2.6.4 for
    # testing purposes
    pisa_xs_file = 'cross_sections/cross_sections.json'
    xs = CrossSections(ver='genie_2.6.4', xsec=pisa_xs_file)

    # Location of the root file to use (not included in PISA at the moment)
    test_dir = expandPath(os.path.join('$PISA', 'tests', 'cross_sections'))
    #ROOT_xs_file = os.path.join(test_dir, 'genie_2.6.4_simplified.root')
    ROOT_xs_file = find_resource(os.path.join(
        'tests', 'data', 'xsec', 'genie_2.6.4_simplified.root'
    ))

    # Make sure that the XS newly-imported from ROOT match those stored in PISA
    if os.path.isfile(ROOT_xs_file):
        xs_from_root = CrossSections.newFromROOT(ROOT_xs_file,
                                                 ver='genie_2.6.4')
        logging.info('Found and loaded ROOT source cross sections file %s'
                     % ROOT_xs_file)
        #assert xs_from_root.allclose(xs, rtol=1e-7)

    # Check XS ratio for numu_cc to numu_cc + numu_nc (user must inspect)
    kg0 = flavInt.NuFlavIntGroup('numu_cc')
    kg1 = flavInt.NuFlavIntGroup('numu_nc')
    logging.info('\\int_1^80 xs(numu_cc) E^{-1} dE = %e' %
                 xs.get_xs_ratio_integral(kg0, None, e_range=[1, 80], gamma=1))
    logging.info('(int E^{-gamma} * (sigma_numu_cc)/'
                 'int(sigma_(numu_cc+numu_nc)) dE) /'
                 ' (int E^{-gamma} dE) = %e' %
                 xs.get_xs_ratio_integral(kg0, kg0+kg1, e_range=[1, 80],
                                          gamma=1, average=True))
    # Check that XS ratio for numu_cc+numu_nc to the same is 1.0
    assert xs.get_xs_ratio_integral(kg0+kg1, kg0+kg1, e_range=[1, 80], gamma=1,
                                    average=True) == 1.0

    # Check via plot that the

    # Plot all cross sections stored in PISA xs file
    try:
        alldata = from_file(pisa_xs_file)
        xs_versions = alldata.keys()
        for ver in xs_versions:
            xs = CrossSections(ver=ver, xsec=pisa_xs_file)
            xs.plot(save=os.path.join(
                outdir, 'pisa_' + ver + '_nuxCCNC_H2O_cross_sections.pdf'
            ))
    except ImportError as exc:
        logging.debug('Could not plot; possible that matplotlib not'
                      'installed. ImportError: %s' % exc)


if __name__ == "__main__":
    test_CrossSections()

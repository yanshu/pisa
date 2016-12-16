"""
Plotter class for doing plots easily

"""


import itertools

import numpy as np
from uncertainties import unumpy as unp

import matplotlib as mpl
# Headless mode; must set prior to pyplot import
mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText

from pisa.core.map import Map, MapSet
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils.format import dollars, text2tex, tex_join
from pisa.utils.log import logging


__all__ = ['Plotter']


# set fonts
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'


class Plotter(object):
    """
    Plotting library for PISA `Map`s, `MapSet`s, and `BinnedTensorTransform`s

    Parameters
    ----------
    outdir : str
        output directory path

    stamp : str
        stamp to be put on every subplot, e.g. 'Preliminary',
        'DeepCore nutau', etc.

    fmt : str or iterable of str
        formats to be plotted, e.g. ['pdf', 'png']

    size : (int, int)
        canvas size (inches)

    log : bool
        logarithmic z-axis

    label : str
        z-axis label

    grid : bool
        plot grid

    ratio : bool
        add ratio plots in 1-d histos

    annotate : bool
        annotate counts per bin in 2-d histos

    symmetric : bool
        force symmetric extent of z-axis

    loc : str
        either 'inside' or 'outside', defining where to put axis titles

    """
    def __init__(self, outdir='.', stamp=None, size=(8, 8), fmt='pdf',
                 log=True, label='# events', grid=True, ratio=False,
                 annotate=False, symmetric=False, loc='outside'):
        self.fig = None
        self.axes = None

        self.outdir = outdir
        self.stamp = stamp
        if isinstance(fmt, basestring):
            fmt = [fmt]
        self.fmt = fmt
        self.size = size
        self.fig = None
        self.log = log
        self.label = label
        self.grid = grid
        self.ratio = ratio
        self.annotate = annotate
        if symmetric:
            assert(self.log == False), 'cannot do log and symmetric at the same time'
        self.symmetric = symmetric
        self.reset_colors()
        self.color = 'b'
        self.loc = loc

    def reset_colors(self):
        self.colors = itertools.cycle(["r", "b", "g"])

    def next_color(self):
        self.color = next(self.colors)

    # --- helper functions ---

    def init_fig(self, figsize=None):
        """clear/initialize figure"""
        if figsize is not None:
            size = figsize
        else:
            size = self.size
        if self.fig is not None:
            plt.clf()
            plt.close(self.fig)
        self.fig, self.axes = plt.subplots(1, 1, figsize=size)
        self.fig.patch.set_facecolor('none')

    def add_stamp(self, text=None, **kwargs):
        """Add common stamp with text.

        NOTE add_stamp cannot be used on a subplot that has been de-selected
        and then re-selected. It will write over existing text.

        """
        stamp = tex_join('\n', self.stamp, text)
        if self.loc == 'inside':
            a_text = AnchoredText(dollars(stamp), loc=2, frameon=False,
                                  **kwargs)
            plt.gca().add_artist(a_text)
        elif self.loc == 'outside':
            plt.gca().set_title(dollars(stamp))

    def add_leg(self):
        """initialize legend """
        plt.gca().legend(loc='upper right', ncol=2, frameon=False, numpoints=1)

    def dump(self, fname):
        """dump figure to file"""
        for fmt in self.fmt:
            plt.savefig(self.outdir+'/'+fname+'.'+fmt, dpi=150,
                        edgecolor='none', facecolor=self.fig.get_facecolor())

    # --- 2d plots ---

    def plot_2d_single(self, map_set, **kwargs):
        """plot all maps in individual plots"""
        if isinstance(map_set, Map):
            map_set = [map_set]
        for map in map_set:
            self.init_fig()
            self.plot_2d_map(map, **kwargs)
            self.add_stamp(map.tex)
            self.dump(map.name)

    def plot_2d_array(self, map_set, n_rows=None, n_cols=None, fname=None,
                      **kwargs):
        """plot all maps or transforms in a single plot"""
        if fname is None:
            fname = 'test2d'
        self.plot_array(map_set, 'plot_2d_map', n_rows=n_rows, n_cols=n_cols,
                        **kwargs)
        self.dump(fname)

    # --- 1d plots ---

    def plot_1d_single(self, map_set, plot_axis, **kwargs):
        """plot all maps in individual plots"""
        for map in map_set:
            self.init_fig()
            self.plot_1d_projection(map, plot_axis, **kwargs)
            self.add_stamp(map.tex)
            self.dump(map.name)

    def plot_1d_array(self, map_set, plot_axis, n_rows=None, n_cols=None,
                      fname=None, **kwargs):
        """plot 1d projections as an array"""
        self.plot_array(map_set, 'plot_1d_projection', plot_axis, n_rows=n_rows,
                        n_cols=n_cols, **kwargs)
        self.dump(fname)

    def plot_1d_slices_array(self, map_sets, plot_axis, fname=None, **kwargs):
        """plot 1d projections as an array"""
        self.slices_array(map_sets, plot_axis, **kwargs)
        self.dump(fname)

    def plot_1d_all(self, map_set, plot_axis, **kwargs):
        """all one one canvas"""
        self.init_fig()
        for map in map_set:
            self.plot_1d_projection(map, plot_axis, **kwargs)
        self.add_stamp()
        self.add_leg()
        self.dump('all')

    def plot_1d_stack(self, map_set, plot_axis, **kwargs):
        """all maps stacked on top of each other"""
        self.init_fig()
        for i, map in enumerate(map_set):
            for j in range(i):
                map += map_set[j]
            self.plot_1d_projection(map, plot_axis, **kwargs)
        self.add_stamp()
        self.add_leg()
        self.dump('stack')

    def plot_1d_cmp(self, map_sets, plot_axis, fname=None, **kwargs):
        """1d comparisons for two map_sets as projections"""
        for i in range(len(map_sets[0])):
            maps = [map_set[i] for map_set in map_sets]
            self.init_fig()
            if self.ratio:
                ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
                plt.setp(ax1.get_xticklabels(), visible=False)
            self.reset_colors()
            for map in maps:
                self.next_color()
                self.plot_1d_projection(map, plot_axis, **kwargs)
            self.add_stamp()
            self.add_leg()
            if self.ratio:
                plt.subplot2grid((4, 1), (3, 0), sharex=ax1)
                self.plot_1d_ratio(maps, plot_axis, **kwargs)
            self.dump('%s_%s'%(fname, maps[0].name))


    # --- plotting core functions ---

    def plot_array(self, map_set, fun, *args, **kwargs):
        """wrapper funtion to exccute plotting function fun for every map in a
        set distributed over a grid"""
        n_rows = kwargs.pop('n_rows', None)
        n_cols = kwargs.pop('n_cols', None)
        split_axis = kwargs.pop('split_axis', None)
        if isinstance(map_set, Map):
            map_set = MapSet([map_set])

        # if dimensionality is 3, then still define a spli_axis automatically
        new_maps = []
        for map in map_set:
            if map.binning.num_dims == 3:
                if split_axis is None:
                    # Find shortest dimension
                    l = map.binning.num_bins
                    idx = l.index(min(l))
                    split_axis_ = map.binning.names[idx]
                    logging.warning(
                        'Plotter automatically splitting map %s along %s axis'
                        % (map.name, split_axis_)
                    )
                new_maps.extend(map.split(split_axis_))
            elif len(map.binning) == 2:
                new_maps.append(map)
            else:
                raise Exception('Cannot plot %i dimensional map in 2d'
                                %len(map))
        map_set = MapSet(new_maps)

        if isinstance(map_set, MapSet):
            n = len(map_set)
        elif isinstance(map_set, TransformSet):
            n = len([x for x in map_set])
        else:
            raise TypeError('Expecting to plot a MapSet or TransformSet but '
                            'got %s'%type(map_set))
        if n_rows is None and n_cols is None:
            # TODO: auto row/cols
            n_rows = np.floor(np.sqrt(n))
            while n % n_rows != 0:
                n_rows -= 1
            n_cols = n / n_rows
        assert (n <= n_cols * n_rows), 'trying to plot %s subplots on a grid with %s x %s cells'%(n, n_cols, n_rows)
        size = (n_cols*self.size[0], n_rows*self.size[1])
        self.init_fig(size)
        plt.tight_layout()
        h_margin = 1. / size[0]
        v_margin = 1. / size[1]
        self.fig.subplots_adjust(hspace=0.3, wspace=0.3, top=1-v_margin,
                                 bottom=v_margin, left=h_margin,
                                 right=1-h_margin)
        for i, map in enumerate(map_set):
            plt.subplot(n_rows, n_cols, i+1)
            getattr(self, fun)(map, *args, **kwargs)
            self.add_stamp(map.tex)

    def slices_array(self, map_sets, plot_axis, *args, **kwargs):
        """plot map_set in array using a function fun"""
        n_cols = len(map_sets[0])
        plt_binning = map_sets[0][0].binning[plot_axis]
        plt_axis_n = map_sets[0][0].binning.names.index(plot_axis)
        # determine how many slices we need accoring to map_set[0]
        n_rows = 0
        assert(len(map_sets[0][0].binning) == 2), 'only supported for 2d maps right now'
        slice_axis_n = int(not plt_axis_n)
        slice_axis = map_sets[0][0].binning.names[slice_axis_n]
        n_rows = map_sets[0][0].binning[slice_axis].num_bins
        size = (n_cols*self.size[0], self.size[1])
        self.init_fig(size)
        plt.tight_layout()
        h_margin = 1. / size[0]
        v_margin = 1. / size[1]
        # big one
        self.fig.subplots_adjust(hspace=0., wspace=0.3, top=1-v_margin,
                                 bottom=v_margin, left=h_margin,
                                 right=1-h_margin)
        stamp = self.stamp
        for i in range(len(map_sets[0])):
            for j in range(n_rows):
                plt.subplot(n_rows, n_cols, i + (n_rows - j -1)*n_cols + 1)
                self.reset_colors()
                for map_set in map_sets:
                    self.next_color()
                    map = map_set[i]
                    if slice_axis_n == 0:
                        map_slice = map[j, :]
                    else:
                        map_slice = map[:, j]
                    a_text = (
                        map_slice.binning[slice_axis].label
                        + ' [%.2f, %.2f]'
                        %(map_slice.binning[slice_axis].bin_edges[0].m,
                          map_slice.binning[slice_axis].bin_edges[-1].m)
                    )
                    self.plot_1d_projection(map_slice, plot_axis,
                                            *args, **kwargs)
                    if j != 0:
                        plt.gca().get_xaxis().set_visible(False)
                    if j == n_rows - 1:
                        if map.name == 'cscd':
                            title = 'cascades channel'
                        if map.name == 'trck':
                            title = 'tracks channel'
                        plt.gca().set_title(title)
                    plt.gca().set_ylabel('')
                    plt.gca().yaxis.set_tick_params(labelsize=8)
                    self.stamp = a_text
                    self.add_stamp(prop=dict(size=8))
        self.stamp = stamp

    def plot_2d_map(self, map, cmap='rainbow', **kwargs):
        """plot map or transform on current axis in 2d"""
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        axis = plt.gca()

        if isinstance(map, BinnedTensorTransform):
            binning = map.input_binning
        elif isinstance(map, Map):
            binning = map.binning
        else:
            raise TypeError('Unhandled `map` type %s' % map.__class__.__name__)

        dims = binning.dims
        bin_centers = binning.weighted_centers
        bin_edges = binning.bin_edges
        linlog = all([(d.is_log or d.is_lin) for d in binning])

        zmap = map.nominal_values
        if self.log:
            zmap = np.log10(zmap)

        if self.symmetric:
            vmax = max(np.max(np.ma.masked_invalid(zmap)),
                       - np.min(np.ma.masked_invalid(zmap)))
            vmin = -vmax
        else:
            if vmax == None:
                vmax = np.max(zmap[np.isfinite(zmap)])
            if vmin == None:
                vmin = np.min(zmap[np.isfinite(zmap)])
        extent = [np.min(bin_edges[0].m), np.max(bin_edges[0].m),
                  np.min(bin_edges[1].m), np.max(bin_edges[1].m)]

        # Only lin or log can be handled by imshow...otherise use colormesh
        if linlog:
            # Needs to be transposed for imshow
            img = plt.imshow(
                zmap.T, origin='lower', interpolation='nearest', extent=extent,
                aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, **kwargs
            )
        else:
            x, y = np.meshgrid(bin_edges[0], bin_edges[1])
            img = plt.pcolormesh(x, y, zmap.T, vmin=vmin, vmax=vmax, cmap=cmap,
                                 **kwargs)
        if self.annotate:
            for i in range(len(bin_centers[0])):
                for j in range(len(bin_centers[1])):
                    bin_x = bin_centers[0][i].m
                    bin_y = bin_centers[1][j].m
                    plt.annotate(
                        '%.1f'%(zmap[i, j]),
                        xy=(bin_x, bin_y),
                        xycoords=('data', 'data'),
                        xytext=(bin_x, bin_y),
                        textcoords='data',
                        va='top',
                        ha='center',
                        size=7
                    )

        axis.set_xlabel(dollars(text2tex(dims[0].label)))
        axis.set_ylabel(dollars(text2tex(dims[1].label)))
        axis.set_xlim(extent[0:2])
        axis.set_ylim(extent[2:4])

        # TODO: use log2 scale & integer tick labels if too few major gridlines
        # result from default log10 scale
        if dims[0].is_log:
            axis.set_xscale('log')
        if dims[1].is_log:
            axis.set_yscale('log')

        if self.log:
            col_bar = plt.colorbar(format=r'$10^{%.1f}$')
        else:
            col_bar = plt.colorbar()

        if self.label:
            col_bar.set_label(dollars(text2tex(self.label)))

    def plot_1d_projection(self, map, plot_axis, **kwargs):
        """plot map projected on plot_axis"""
        axis = plt.gca()
        plt_binning = map.binning[plot_axis]
        hist = self.project_1d(map, plot_axis)
        if map.tex == 'data':
            axis.errorbar(
                plt_binning.weighted_centers.m, unp.nominal_values(hist),
                yerr=unp.std_devs(hist),
                fmt='o', markersize='4', label=dollars(text2tex('data')),
                color='k', ecolor='k', mec='k', **kwargs
            )
        else:
            axis.hist(
                plt_binning.weighted_centers, weights=unp.nominal_values(hist),
                bins=plt_binning.bin_edges, histtype='step', lw=1.5,
                label=dollars(text2tex(map.tex)), color=self.color, **kwargs
            )
            axis.bar(
                plt_binning.bin_edges.m[:-1], 2*unp.std_devs(hist),
                bottom=unp.nominal_values(hist)-unp.std_devs(hist),
                width=plt_binning.bin_widths, alpha=0.25, linewidth=0,
                color=self.color, **kwargs
            )
        axis.set_xlabel(dollars(text2tex(plt_binning.label)))
        if self.label:
            axis.set_ylabel(dollars(text2tex(self.label)))
        if plt_binning.is_log:
            axis.set_xscale('log')
        if self.log:
            axis.set_yscale('log')
        else:
            axis.set_ylim(0, np.max(unp.nominal_values(hist))*1.4)
        axis.set_xlim(plt_binning.bin_edges.m[0], plt_binning.bin_edges.m[-1])
        if self.grid:
            plt.grid(True, which="both", ls='-', alpha=0.2)

    def project_1d(self, map, plot_axis):
        """sum up a map along all axes except plot_axis"""
        hist = map.hist
        plt_axis_n = map.binning.names.index(plot_axis)
        for i in range(len(map.binning)):
            if i == plt_axis_n:
                continue
            hist = np.sum(map.hist, i)
        return hist

    def plot_1d_ratio(self, maps, plot_axis):
        """make a ratio plot for a 1d projection"""
        axis = plt.gca()
        map0 = maps[0]
        plt_binning = map0.binning[plot_axis]
        hist = self.project_1d(map0, plot_axis)
        hist0 = unp.nominal_values(hist)
        err0 = unp.std_devs(hist)

        axis.set_xlim(plt_binning.bin_edges.m[0], plt_binning.bin_edges.m[-1])
        maximum = 1.0
        minimum = 1.0
        self.reset_colors()
        for j, map in enumerate(maps):
            self.next_color()
            hist = self.project_1d(map, plot_axis)
            hist1 = unp.nominal_values(hist)
            err1 = unp.std_devs(hist)
            ratio = np.zeros_like(hist0)
            ratio_error = np.zeros_like(hist0)
            for i in range(len(hist0)):
                if hist1[i]==0 and hist0[i]==0:
                    ratio[i] = 1.
                    ratio_error[i] = 1.
                elif hist1[i]!=0 and hist0[i]==0:
                    logging.warning('deviding non 0 by 0 for ratio')
                    ratio[i] = 0.
                    ratio_error[i] = 1.
                else:
                    ratio[i] = hist1[i]/hist0[i]
                    ratio_error[i] = err1[i]/hist0[i]
                    minimum = min(minimum, ratio[i])
                    maximum = max(maximum, ratio[i])

            if map.tex == 'data':
                axis.errorbar(
                    plt_binning.weighted_centers.m, ratio, yerr=ratio_error,
                    fmt='o', markersize='4', label=dollars(text2tex('data')),
                    color='k', ecolor='k', mec='k'
                )
            else:
                h, b, p = axis.hist(
                    plt_binning.weighted_centers, weights=ratio,
                    bins=plt_binning.bin_edges, histtype='step', lw=1.5,
                    label=dollars(text2tex(map.tex)), color=self.color)
                axis.bar(
                    plt_binning.bin_edges.m[:-1], 2*ratio_error,
                    bottom=ratio-ratio_error, width=plt_binning.bin_widths,
                    alpha=0.25, linewidth=0, color=self.color
                )

        if self.grid:
            plt.grid(True, which="both", ls='-', alpha=0.2)
        self.fig.subplots_adjust(hspace=0)
        axis.set_ylabel(dollars(text2tex('ratio')))
        axis.set_xlabel(dollars(text2tex(plt_binning.label)))
        # Calculate nice scale:
        off = max(maximum-1, 1-minimum)
        axis.set_ylim(1 - 1.2 * off, 1 + 1.2 * off )

    def plot_xsec(self, map_set, ylim=None, logx=True):
        from pisa.utils import fileio

        zero_np_element = np.array([0])
        for map in map_set:
            binning = map.binning
            if 'true_energy' in binning.names:
                energy_binning = binning.true_energy
            elif 'reco_energy' in binning.names:
                energy_binning = binning.reco_energy
            else:
                dim_idx = binning.index('energy', use_basenames=True)
                energy_binning = binning.dims[dim_idx]

            fig = plt.figure(figsize=self.size)
            fig.suptitle(map.name, y=0.95)
            ax = fig.add_subplot(111)
            ax.grid(b=True, which='major')
            ax.grid(b=True, which='minor', linestyle=':')
            plt.xlabel(dollars(text2tex(energy_binning.label)), size=18)
            plt.ylabel(dollars(text2tex(self.label)), size=18)
            if self.log:
                ax.set_yscale('log')
            if logx:
                ax.set_xscale('log')
            if ylim:
                ax.set_ylim(ylim)
            ax.set_xlim(np.min(energy_binning.bin_edges.m),
                        np.max(energy_binning.bin_edges.m))

            hist = map.hist
            array_element = np.hstack((hist, zero_np_element))
            ax.step(energy_binning.bin_edges.m, array_element, where='post')

            fileio.mkdir(self.outdir)
            fig.savefig(self.outdir+'/'+map.name+'.png', bbox_inches='tight',
                        dpi=150)


#
# plots.py
#
# Utility function for plotting maps
#
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#
# date:   2014-01-27

from utils import is_linear, is_logarithmic, get_bin_centers
import os
import numpy as np
import matplotlib.pyplot as plt

def show_map(pmap, title=None, cbar = True,
             vmin=None, vmax=None,
             emin=None, emax=None,
             czmin=None, czmax=None,
             invalid=False, logE=None,
             log=False, fontsize=12,
             #xlabel=r'cos $\vartheta_\mathrm{zenith}$',
             xlabel='cos(zenith)',
             ylabel='Energy (GeV)',
             zlabel=None,
             zlabel_size='large',
             annotate_no_evts=True,
             annotate_prcs=1,
             **kwargs):
    '''Plot the given map with proper axis labels using matplotlib.
       The axis orientation follows the PINGU convention:
          - energy to the right
          - cos(zenith) to the top

    Keyword arguments:

      * title: Show this title above the map

      * cbar: show a colorbar for the z-axis (True or False)

      * vmin/vmax: set the minimum/maximum color (z-axis) values.
                   If no value is provided, vmin and vmax are choosen
                   symmetrically to comprehend the full range.

      * emin/emax: set the minimum maximum energy range to show.
                   If not value is provided use the full range covered
                   by the axis.

      * czmin/czmax: same as above for cos(zenith)

      * log: use a logarithmic (log10) colour (z-axis) scale

      * logE: show the x-axis on a logarithmic scale (True or False)
              Default is "guessed" from the bins size.

      * invalid: if True show color values for NaN, None, -inf and +inf,
                 otherwise nothing (white) is plotted for these values.


    Uses pyplot.pcolormesh to do the plot. Any additional keyword arguments, in
    particular

      * cmap: defines the colormap

    are just passed on to this function.
    '''
    plt.tick_params(axis='both', which='major', labelsize=fontsize)

    #Extract the map to plot, take the log if called for
    cmap = np.log10(pmap['map']) if log else pmap['map']

    #Mask invalid values
    cmap = np.ma.masked_invalid(cmap) if not invalid else cmap

    #Get the vertical range
    if not log and vmax is None:
        vmax = np.max(np.abs(np.array(cmap)[np.isfinite(cmap)]))
    if not log and vmin is None:
        vmin = -vmax if (cmap.min() < 0) else 0.

    #Get the energy range
    if emin is None:
        emin = pmap['ebins'][0]
    if emax is None:
        emax = pmap['ebins'][-1]

    #... and for zenith range
    if czmin is None:
        czmin = pmap['czbins'][0]
    if emax is None:
        czmax = pmap['czbins'][-1]

    #Use pcolormesh to be able to show nonlinear spaces
    x,y = np.meshgrid(pmap['czbins'],pmap['ebins'])
    #colormesh = plt.pcolormesh(x,y,cmap,vmin=vmin, vmax=vmax, **kwargs)
    z=cmap
    img = plt.imshow(z, vmin=vmin, vmax=vmax,
                extent=[x.min(), x.max(), y.min(), y.max()],
                origin='lower',interpolation='none',**kwargs)
    #counts = colormesh.get_array()
    counts = img.get_array()

    cz_bin_centers = get_bin_centers(pmap['czbins'])
    e_bin_centers = get_bin_centers(pmap['ebins'])
    if annotate_no_evts:
        for i in range(0,len(e_bin_centers)):
            for j in range(0,len(cz_bin_centers)):
                if annotate_prcs == 0:
                    plt.annotate('%i'%(counts[i,j]), xy=(cz_bin_centers[j], e_bin_centers[i]), xycoords=('data', 'data'), xytext=(cz_bin_centers[j], e_bin_centers[i]), textcoords='data', va='top', ha='center', size=6)
                if annotate_prcs == 1:
                    plt.annotate('%.1f'%(counts[i,j]), xy=(cz_bin_centers[j], e_bin_centers[i]), xycoords=('data', 'data'), xytext=(cz_bin_centers[j], e_bin_centers[i]), textcoords='data', va='top', ha='center', size=6)
                if annotate_prcs == 2:
                    plt.annotate('%.2f'%(counts[i,j]), xy=(cz_bin_centers[j], e_bin_centers[i]), xycoords=('data', 'data'), xytext=(cz_bin_centers[j], e_bin_centers[i]), textcoords='data', va='top', ha='center', size=6)

    #Add nice labels
    #if xlabel == None:
    #    plt.xlabel(r'cos(zenith)',fontsize=16)
    #else:
    plt.xlabel(xlabel,fontsize=fontsize)
    #if yabel == None:
    #    plt.ylabel('Energy [GeV]',fontsize=16)
    #else:
    plt.ylabel(ylabel,fontsize=fontsize)

    #And a title
    if title is not None:
        plt.gca().set_title(title,fontsize=fontsize)

    axis = plt.gca()
    #Check wether energy axis is linear or log-scale
    if logE is None:
        logE = is_logarithmic(pmap['ebins'])

    if logE:
        axis.semilogy()
    else:
        if not is_linear(pmap['ebins']):
            print "bin edges not log or linear."
           #raise NotImplementedError('Bin edges appear to be neither logarithmically '
           #              'nor linearly distributed!')


    #Make sure that the visible range does not extend beyond the provided range
    axis.set_ylim(emin,emax)
    axis.set_xlim(czmin,czmax)

    #Show the colorbar
    if cbar:
        col_bar = plt.colorbar(format=r'$10^{%.1f}$') if log else plt.colorbar()
        if zlabel:
            col_bar.set_label(zlabel,fontsize=fontsize)
        col_bar.ax.tick_params(labelsize=fontsize)
    #Return axes for further modifications
    return axis

def show_map_swap(pmap, title=None, cbar = True,
             vmin=None, vmax=None,
             emin=None, emax=None,
             czmin=None, czmax=None,
             invalid=False, logE=None,
             log=False, fontsize=12,
             #xlabel=r'cos $\vartheta_\mathrm{zenith}$',
             ylabel='cos(zenith)',
             xlabel='Energy (GeV)',
             zlabel=None,
             zlabel_size='large',
             annotate_no_evts=True,
             annotate_prcs=1,
             **kwargs):
    '''Plot the given map with proper axis labels using matplotlib.
       The axis orientation follows the PINGU convention:
          - energy to the right
          - cos(zenith) to the top

    Keyword arguments:

      * title: Show this title above the map

      * cbar: show a colorbar for the z-axis (True or False)

      * vmin/vmax: set the minimum/maximum color (z-axis) values.
                   If no value is provided, vmin and vmax are choosen
                   symmetrically to comprehend the full range.

      * emin/emax: set the minimum maximum energy range to show.
                   If not value is provided use the full range covered
                   by the axis.

      * czmin/czmax: same as above for cos(zenith)

      * log: use a logarithmic (log10) colour (z-axis) scale

      * logE: show the x-axis on a logarithmic scale (True or False)
              Default is "guessed" from the bins size.

      * invalid: if True show color values for NaN, None, -inf and +inf,
                 otherwise nothing (white) is plotted for these values.


    Uses pyplot.pcolormesh to do the plot. Any additional keyword arguments, in
    particular

      * cmap: defines the colormap

    are just passed on to this function.
    '''
    plt.tick_params(axis='both', which='major', labelsize=fontsize)

    #Extract the map to plot, take the log if called for
    cmap = np.log10(pmap['map']) if log else pmap['map']
    cmap = np.transpose(cmap)

    #Mask invalid values
    cmap = np.ma.masked_invalid(cmap) if not invalid else cmap

    #Get the vertical range
    if not log and vmax is None:
        vmax = np.max(np.abs(np.array(cmap)[np.isfinite(cmap)]))
    if not log and vmin is None:
        vmin = -vmax if (cmap.min() < 0) else 0.

    #Get the energy range
    if emin is None:
        emin = pmap['ebins'][0]
    if emax is None:
        emax = pmap['ebins'][-1]

    #... and for zenith range
    if czmin is None:
        czmin = pmap['czbins'][0]
    if emax is None:
        czmax = pmap['czbins'][-1]

    #Use pcolormesh to be able to show nonlinear spaces
    x,y = np.meshgrid(pmap['ebins'],pmap['czbins'])
    #colormesh = plt.pcolormesh(x,y,cmap,vmin=vmin, vmax=vmax, **kwargs)
    z=cmap
    img = plt.imshow(z, vmin=vmin, vmax=vmax,
                extent=[x.min(), x.max(), y.min(), y.max()],
                origin='lower',interpolation='none',**kwargs)
    #counts = colormesh.get_array()
    counts = img.get_array()

    cz_bin_centers = get_bin_centers(pmap['czbins'])
    e_bin_centers = get_bin_centers(pmap['ebins'])
    if annotate_no_evts:
        for i in range(0,len(e_bin_centers)):
            for j in range(0,len(cz_bin_centers)):
                if annotate_prcs == 0:
                    plt.annotate('%i'%(counts[j,i]), xy=(e_bin_centers[i], cz_bin_centers[j]), xycoords=('data', 'data'), xytext=(e_bin_centers[i], cz_bin_centers[j]), textcoords='data', va='top', ha='center', size=7)
                if annotate_prcs == 1:
                    plt.annotate('%.1f'%(counts[j,i]), xy=(e_bin_centers[i], cz_bin_centers[j]), xycoords=('data', 'data'), xytext=(e_bin_centers[i], cz_bin_centers[j]), textcoords='data', va='top', ha='center', size=7)
                if annotate_prcs == 2:
                    plt.annotate('%.2f'%(counts[j,i]), xy=(e_bin_centers[i], cz_bin_centers[j]), xycoords=('data', 'data'), xytext=(e_bin_centers[i], cz_bin_centers[j]), textcoords='data', va='top', ha='center', size=7)

    #Add nice labels
    #if xlabel == None:
    #    plt.xlabel(r'cos(zenith)',fontsize=16)
    #else:
    plt.xlabel(xlabel,fontsize=fontsize)
    #if yabel == None:
    #    plt.ylabel('Energy [GeV]',fontsize=16)
    #else:
    plt.ylabel(ylabel,fontsize=fontsize)

    #And a title
    if title is not None:
        plt.gca().set_title(title,fontsize=fontsize)

    axis = plt.gca()
    #Check wether energy axis is linear or log-scale
    if logE is None:
        logE = is_logarithmic(pmap['ebins'])

    if logE:
        axis.semilogx()
    else:
        if not is_linear(pmap['ebins']):
            print "bin edges not log or linear."
           #raise NotImplementedError('Bin edges appear to be neither logarithmically '
           #              'nor linearly distributed!')


    #Make sure that the visible range does not extend beyond the provided range
    axis.set_xlim(emin,emax)
    axis.set_ylim(czmin,czmax)

    #Show the colorbar
    if cbar:
        col_bar = plt.colorbar(format=r'$10^{%.1f}$') if log else plt.colorbar()
        if zlabel:
            col_bar.set_label(zlabel,fontsize=fontsize)
        col_bar.ax.tick_params(labelsize=fontsize)
    #Return axes for further modifications
    return axis

def plot_one_map(map_to_plot, outdir, logE, fig_title, fig_name, save, max=None, min=None, annotate_prcs=1, cmap='cool'):
    plt.figure()
    show_map_swap(map_to_plot, vmin= min if min!=None else np.min(map_to_plot['map']), vmax= max if max!=None else np.max(map_to_plot['map']),annotate_prcs=annotate_prcs, logE=logE, cmap=cmap)
    if save:
        filename = os.path.join(outdir, fig_name + '.png')
        pdf_filename = os.path.join(outdir+'/pdf/', fig_name + '.pdf')
        plt.title(fig_title)
        plt.savefig(filename,dpi=150)
        plt.savefig(pdf_filename,dpi=150)
        plt.clf()

def delta_map(amap, bmap):
    '''
    Calculate the differerence between the two maps (amap - bmap), and return as
    a map dictionary.
    '''
    if not np.allclose(amap['ebins'],bmap['ebins']) or \
       not np.allclose(amap['czbins'],bmap['czbins']):
       raise ValueError('Map range does not match!')

    return { 'ebins': amap['ebins'],
             'czbins': amap['czbins'],
             'map' : amap['map'] - bmap['map'] }

def sum_map(amap, bmap):
    '''
    Calculate the sum of two maps (amap + bmap), and return as
    a map dictionary.
    '''
    if not np.allclose(amap['ebins'],bmap['ebins']) or \
       not np.allclose(amap['czbins'],bmap['czbins']):
       raise ValueError('Map range does not match!')

    return { 'ebins': amap['ebins'],
             'czbins': amap['czbins'],
             'map' : amap['map'] + bmap['map'] }

def ratio_map(amap,bmap):
    '''
    Get the ratio of two maps (amap/bmap) and return as a map
    dictionary.
    '''
    if (not np.allclose(amap['ebins'],bmap['ebins']) or
        not np.allclose(amap['czbins'],bmap['czbins'])):
        raise ValueError('Map range does not match!')

    return { 'ebins': amap['ebins'],
             'czbins': amap['czbins'],
             'map' : amap['map']/bmap['map']}

def s_over_sqrt_b(smap,bmap):
    sqrt_map = {'ebins': bmap['ebins'],
                'czbins': bmap['czbins'],
                'map':np.sqrt(bmap['map'])}
    return ratio_map(smap,sqrt_map)

def distinguishability_map(amap,bmap):
    '''
    Calculate the Akhmedov-Style distinguishability map of two maps,
    defined as amap-bmap/sqrt(amap)
    '''
    sqrt_map = {'ebins': amap['ebins'],
                'czbins': amap['czbins'],
                'map':np.sqrt(amap['map'])}
    return ratio_map(delta_map(amap,bmap),sqrt_map)

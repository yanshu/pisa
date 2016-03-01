#! /usr/bin/env python
import copy
import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy import stats
from matplotlib.offsetbox import AnchoredText
from pisa.analysis.TemplateMaker_nutau import TemplateMaker
from pisa.utils.params import get_values, select_hierarchy_and_nutau_norm
from pisa.utils.log import set_verbosity,logging,profile
from pisa.utils.jsons import from_json
from pisa.utils.plot import show_map, sum_map, ratio_map, delta_map
from pisa.analysis.stats.Maps import flatten_map
from pisa.analysis.stats.Maps_nutau import get_true_template, get_burn_sample
from pisa.utils.utils import Timer, is_linear, is_logarithmic, get_bin_centers, get_bin_sizes

class plotter(object):

    def __init__(self,livetime, outdir, logy=True, fmt='pdf'):
        self.logy= logy
        self.outdir = outdir
        self.livetime=livetime
        self.fmt = fmt

    def get_1D_projection(self,map_2d, axis):
        if axis == 'coszen':
            output_array = np.zeros(map_2d.shape[1])
            for i in range(0, map_2d.shape[0]):
                output_array += map_2d[i,:]
        if axis == 'energy':
            output_array = np.zeros(map_2d.shape[0])
            for i in range(0, map_2d.shape[1]):
                output_array += map_2d[:,i]
        return output_array

    def plot_1d(self,maps, errors, colors, names, axis, x_bin_edges, outname):
        x_bin_centers = get_bin_centers(x_bin_edges)
        x_bin_width = get_bin_sizes(x_bin_edges)
        fig = plt.figure(figsize=(8,8))
        if len(maps) == 1:
            ax1 = fig.add_subplot(111)
        else:
            ax1 = plt.subplot2grid((4,1), (0,0), rowspan=3)
        for map, error, color, name in zip(maps,errors,colors,names):
            if name == 'data':
                ax1.errorbar(x_bin_centers,map,yerr=error,fmt='o',color='black', markersize='4',label=name)
            else:
                hist,_,_ = ax1.hist(x_bin_centers,weights= map,bins=x_bin_edges,histtype='step',lw=1,color=color,linestyle='solid', label=name)
                ax1.bar(x_bin_edges[:-1],2*error, bottom=map-error, width=x_bin_width, color=color, alpha=0.25, linewidth=0)
        ax1.grid()
        ax1.set_ylabel('# entries')
        minimum = min([min(map) for map in maps])
        maximum = max([max(map) for map in maps])
        if self.logy:
            if minimum == 0:
                minimum = 1
            ax1.set_yscale('log')
            ax1.set_ylim(np.power(10,int(np.log10(minimum))),np.power(10,int(np.log10(maximum)+1)))
        else:
            ax1.set_ylim(minimum, 1.5*maximum)
        ax1.legend(loc='upper right',ncol=1, frameon=False,numpoints=1,fontsize=10)
        a_text = AnchoredText(r'$\nu_\tau$ appearance'+'\n%s years\nPreliminary'%self.livetime, loc=2, frameon=False)
        ax1.add_artist(a_text)
        if axis == 'energy':
            ax1.set_xlabel('Energy (GeV)')
            ax1.set_xscale('log')
            ax1.set_xlim(x_bin_edges[0],x_bin_edges[-1])
        if axis == 'coszen':
            ax1.set_xlabel('cos(zen)')

        if len(maps) >1:
            ax2 = plt.subplot2grid((4,1), (3,0),sharex=ax1)
            for map, error, color in zip(maps, errors, colors):
                ratio = np.zeros_like(map)
                ratio_error = np.zeros_like(map)
                for i in range(len(map)):
                    if map[i]==0 and maps[0][i]==0:
                        ratio[i] = 1
                        ratio_error[i] = 1
                    elif map[i]!=0 and maps[0][i]==0:
                        print " non zero divided by 0 !!!"
                    else:
                        ratio[i] = map[i]/maps[0][i]
                        ratio_error[i] = error[i]/maps[0][i]
                hist,_,_ = ax2.hist(x_bin_centers,weights= ratio,bins=x_bin_edges,histtype='step',lw=1,color=color,linestyle='solid')
                ax2.bar(x_bin_edges[:-1],2*ratio_error, bottom=ratio-ratio_error, width=x_bin_width, color=color, alpha=0.25, linewidth=0)
            
            if axis == 'energy':
                ax2.set_xlabel('Energy (GeV)')
                ax2.set_xscale('log')
                ax2.set_xlim(x_bin_edges[0],x_bin_edges[-1])
            if axis == 'coszen':
                ax2.set_xlabel('cos(zen)')
            ax2.grid()
            ax2.set_ylim(0.5,1.5)
            ax2.set_ylabel('ratio over %s'%names[0])
            fig.subplots_adjust(hspace=0)
            plt.setp(ax1.get_xticklabels(), visible=False)

        plt.savefig(self.outdir+'/'+outname+'.'+self.fmt, dpi=150)
        plt.clf()
        plt.close(fig)


    def plot_1d_projection(self,maps, errors, colors, names, axis, x_bin_edges, channel):
        # iterate
        outname = '%s_%s'%(channel, axis)
        pmaps = []
        perrors = []
        for map, error in zip(maps,errors):
            pmaps.append(self.get_1D_projection(map,axis))
            perrors.append(np.sqrt(self.get_1D_projection(error,axis)))
        self.plot_1d(pmaps, perrors, colors, names, axis, x_bin_edges, outname)

    def plot_1d_slices(self,maps, errors, colors, names, axis, x_bin_edges, channel):
        # iterate
        if axis == 'coszen':
                idx = range(0, maps[0].shape[0])
        if axis == 'energy':
                idx = range(0, maps[0].shape[1])
        for i in idx:
            pmaps = []
            perrors = []
            outname = '%s_%s_%s'%(channel, axis, i)
            for map, error, color, name in zip(maps,errors,colors,names):
                if axis == 'coszen':
                    pmaps.append(map[i,:])
                    perrors.append(np.sqrt(error[i,:]))
                else:
                    pmaps.append(map[:,i])
                    perrors.append(np.sqrt(error[:,i]))
            self.plot_1d(pmaps, perrors, colors, names, axis, x_bin_edges, outname)


if __name__ == '__main__':
    set_verbosity(0)
    parser = ArgumentParser(description='''Quick check if all components are working reasonably well, by
                                            making the final level hierarchy asymmetry plots from the input settings file. ''')
    parser.add_argument('-t','--template_settings',metavar='JSON',
                        help='Settings file to use for template generation')
    parser.add_argument('-no_logE','--no_logE',action='store_true',default=False,
                        help='Energy in log scale.')
    parser.add_argument('-o','--outdir',metavar='DIR',default='',
                        help='Directory to save the output figures.')
    parser.add_argument('-f', '--fit-results', default=None, dest='fit_file',
                        help='use post fit parameters from fit result json file (nutau_norm = 1)')
    parser.add_argument('-bs','--burn_sample_file',metavar='FILE',type=str,
                        default='',
                        help='''HDF5 File containing burn sample.'
                        inverted corridor cut data''')
    args = parser.parse_args()

    # get settings file for nutau norm = 1
    template_settings = from_json(args.template_settings)

    if args.fit_file:
        # replace with parameters determ,ined in fit
        fit_file_tau = from_json(args.fit_file)
        syslist = fit_file['trials'][0]['fit_results'][0].keys()
        for sys in syslist:
            if not sys == 'llh':
                val = fit_file['trials'][0]['fit_results'][0][sys][0]
                if sys == 'theta23' or sys =='deltam23' or sys =='deltam31':
                    sys += '_nh'
                print '%s at %.4f'%(sys,val)
                template_settings['params'][sys]['value'] = val

    # get binning info
    anlys_ebins = template_settings['binning']['anlys_ebins']
    czbins = template_settings['binning']['czbins']
    livetime = template_settings['params']['livetime']['value']

    # get template
    template_maker = TemplateMaker(get_values(template_settings['params']),
                                        **template_settings['binning'])
    true_template = template_maker.get_template(get_values(select_hierarchy_and_nutau_norm(template_settings['params'],normal_hierarchy=True,nutau_norm_value=1.0)))

    if args.burn_sample_file:
        burn_sample_maps = get_burn_sample(burn_sample_file= args.burn_sample_file, anlys_ebins= anlys_ebins, czbins= czbins, output_form ='map', cut_level='L6', channel=template_settings['params']['channel']['value'])

    myPlotter = plotter(livetime,args.outdir,logy=False) 

    for axis, bins in [('energy',anlys_ebins),('coszen',czbins)]:
        for channel in ['cscd','trck']:
            if args.burn_sample_file:
                plot_maps = [burn_sample_maps[channel]['map']] 
                plot_sumw2 = [burn_sample_maps[channel]['map']]
                plot_colors = ['k']
                plot_names = ['data']
            else:
                plot_maps = []
                plot_sumw2 = []
                plot_colors = []
                plot_names = []

            plot_maps.extend([true_template[channel]['map'],true_template[channel]['map_nu'],true_template[channel]['map_mu']])
            plot_sumw2.extend([true_template[channel]['sumw2'],true_template[channel]['sumw2_nu'],true_template[channel]['sumw2_mu']])
            plot_colors.extend(['b','g','r'])
            plot_names.extend(['total','nutrinos','atmospheric muons'])
            myPlotter.plot_1d_projection(plot_maps,plot_sumw2,plot_colors,plot_names ,axis, bins, channel)
            #myPlotter.plot_1d_slices(plot_maps,plot_sumw2,plot_colors,plot_names ,axis, bins, channel)

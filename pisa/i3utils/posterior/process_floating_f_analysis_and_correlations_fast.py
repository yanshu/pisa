#! /usr/bin/env python
#
# process_LLR_analysis.py - Process an analysis result of an LLR
# Analysis run, and plot all posterior parameter information if
# desired.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   31 March 2015
#

from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
from matplotlib import pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import h5py
import itertools as iter

from scipy.optimize import curve_fit
from scipy.special import erfinv
from scipy.stats import norm
from scipy.stats.stats import pearsonr

from pisa.utils.log import logging, set_verbosity
from pisa.utils.jsons import from_json
from pisa.utils.utils import get_bin_centers, Timer
from pisa.utils.params import select_hierarchy
from pisa.utils.hdf import from_hdf

# Functions for gaussian fit:
gauss = lambda x, amp, loc, width: amp*np.exp(-(x-loc)**2/(2.*width**2))

def do_gauss(xvals, yvals, **kwargs):
    f, c = curve_fit(gauss, xvals, yvals, **kwargs)
    return f, np.sqrt(np.diag(c))

def plot_gauss(xvals, fit, **kwargs):
    plt.plot(xvals, gauss(xvals, *fit), **kwargs)
    return

def get_data_frames(llh_file):
    """
    Loads data from stored hdf5 file into a data frame for each
    combination of 'pseudo_data | hypo'
    """

    fh = h5py.File(llh_file,'r')
    data_frames = []
    for dFlag in ['data_tau','data_notau']:
        #for hFlag in ['hypo_free','hypo_notau']:
        for hFlag in ['hypo_free']:

            keys = fh['trials'][dFlag][hFlag].keys()
            entries = len(fh['trials'][dFlag][hFlag][keys[0]])

            data = {key: np.array(fh['trials'][dFlag][hFlag][key]) for key in keys }
            data['pseudo_data'] = np.empty_like(data[keys[0]],dtype='|S16')
            data['pseudo_data'][:] = dFlag
            data['hypo'] = np.empty_like(data[keys[0]],dtype='|S16')
            data['hypo'][:] = hFlag

            df = DataFrame(data)
            data_frames.append(df)

    fh.close()

    return data_frames

def show_frame(df):
    pd.set_option('display.max_columns', len(df))
    pd.set_option('expand_frame_repr', False)
    pd.set_option('max_rows',20)
    logging.debug("df:\n%s"%df)

    return

def get_template_settings(llh_file):
    datafile = from_hdf(llh_file)
    return datafile['template_settings_up']['params']


def plot_llr_distributions(llr_tau,llr_notau,nbins,plot_gauss=True,
                           notau_true=False):
    """Plots LLR distributions-expects llr_tau and llr_notau to be type
    Series. Also plots vertical line signifying the mean of the
    hierarchy assumed to be given in nature, and the percentage of
    trials from the opposite hierarchy with LLR beyond the mean.
    """

    label_text = r'$ 2 * \log ( \mathcal{L}(data: NOTAU|NOTAU) / \mathcal{L}( data: NOTAU|FREE) )$'
    llr_notau.hist(bins=nbins,histtype='step',lw=2,color='r',label=label_text)
    hist_vals_notau,bincen_notau = plot_error(llr_notau,nbins,fmt='.r',lw=2)
    if plot_gauss: plot_gauss_fit(llr_notau,hist_vals_notau,bincen_notau,color='r',lw=2)

    label_text = r'$ 2 * \log ( \mathcal{L}(data: TAU|NOTAU) / \mathcal{L}(data: TAU|FREE) )$'
    llr_tau.hist(bins=nbins,histtype='step',lw=2,color='b',label=label_text)
    hist_vals_tau,bincen_tau = plot_error(llr_tau,nbins,fmt='.b',lw=2)
    if plot_gauss: plot_gauss_fit(llr_tau,hist_vals_tau,bincen_tau,color='b',lw=2)

    if notau_true:
        mean_val = llr_tau.mean()
        pvalue = 1.0 - float(np.sum(llr_notau > mean_val))/len(llr_notau)
    else:
        mean_val = llr_notau.mean()
        pvalue = float(np.sum(llr_tau > mean_val))/len(llr_tau)

    ymax = max(hist_vals_tau) if notau_true else max(hist_vals_notau)
    bincen = bincen_notau if notau_true else bincen_tau
    #vline = plt.vlines(mean_val,1,ymax,colors='k',linewidth=2,
    #                   label=("pval = %.4f"%pvalue))

    sigma_1side = np.sqrt(2.0)*erfinv(1.0 - pvalue)
    sigma_2side = norm.isf(pvalue)
    print "  pvalue: %.4f"%pvalue
    print "  sigma 1 sided (erfinv): %.4f"%sigma_1side
    print "  sigma 2 sided (isf)   : %.4f"%sigma_2side

    if notau_true:
        plt.fill_betweenx(hist_vals_notau,bincen,x2=mean_val,where=bincen < mean_val,
                          alpha=0.5,hatch='xx')
    else:
        plt.fill_betweenx(hist_vals_tau,bincen,x2=mean_val,where=bincen>mean_val,
                          alpha=0.5,hatch='xx')

    plt.ylabel('# Trials',fontsize='x-large')
    plt.xlabel('LLR value',fontsize='x-large')

    return

def plot_error(llr,nbins,**kwargs):
    """Given llr distribution Series, calculates the error bars and plots
    them """
    hist_vals,xbins = np.histogram(llr,bins=nbins)
    bincen = get_bin_centers(xbins)
    plt.errorbar(bincen,hist_vals,yerr=np.sqrt(hist_vals),**kwargs)
    return hist_vals,bincen

def plot_gauss_fit(llr,hist_vals,bincen,**kwargs):
    """Plots gaussian fit over the llr distributions."""

    fit_notau, cov = do_gauss(bincen,hist_vals,
                            p0=[np.max(hist_vals),llr.mean(),llr.std()])
    plot_gauss(bincen,fit_notau,**kwargs)

    return

def plot_mean_std(mean_val, std_val, ymax,ax):
    """Plot the mean value as a vertical line """

    vline = plt.vlines(mean_val,1,ymax,colors='b',linewidth=3,label="mean")

    xfill = np.linspace(mean_val-std_val,mean_val+std_val,10)
    ax.fill_between(xfill,0.0,ymax*0.15,alpha=0.5,hatch='x',
                    facecolor='g')
    plt.plot(xfill,np.zeros_like(xfill),lw=3,color='g',alpha=0.8,label="st dev")

    return

def plot_injected_val(injected_val,ymax):

    vline = plt.vlines(injected_val,1,ymax,colors='r',linewidth=2,
                       alpha=1.0,label="injected")
    return

def plot_prior(prior,value,ymax,ax):

    if prior is None: return
    else:
        xfill = np.linspace(value-prior,value+prior,10)
        ax.fill_between(xfill,0.0,ymax*0.1,alpha=0.4,facecolor='k')
        plt.plot(xfill,np.zeros_like(xfill),lw=3,color='k',alpha=0.4,label='prior')

    return

def plot_bound(range_bound,ymax,ax):

    xfill = np.linspace(range_bound[0],range_bound[1],10)
    ax.fill_between(xfill,0.0,ymax*0.05,alpha=0.8,hatch='xx',facecolor='y')
    plt.plot(xfill,np.zeros_like(xfill),lw=3,color='y',alpha=0.8,label='bound')

    return

def plot_correlations(frames, template_settings, pbins =20, save_fig=False, **kwargs):
    good_columns = [col for col in frames[0].columns if col not in ['hypo','pseudo_data','llh']]
    print "good_columns = ", good_columns
    #del good_columns[2]
    nplots = len(good_columns)*(len(good_columns)-1)/2
    print "len(good_columns) = ", len(good_columns)
    print "nplots = ", nplots
    max_plots_per_fig = 4
    nfigs = (nplots-1) / max_plots_per_fig + 1
    print "nfigs = ", nfigs
    logging.info("len(good_cols): %d, nfigs: %d"%(len(good_columns),nfigs))

    figs = []
    fig_names = []
    for frame in frames:
        data = frame['pseudo_data'][0]
        hypo = frame['hypo'][0]
        column_comb_tuple = tuple(iter.combinations(range(0,len(good_columns)),2))
        idx = 0
        for i in xrange(nfigs):
            #print "i = ", i

            fig = plt.figure(figsize=(12,12))
            figs.append(fig)
            fig.suptitle('Scatter plots for %s, %s'%(data,hypo),fontsize='large')

            # determine how many columns to put on each figure:
            nsubplots = max_plots_per_fig
            if (i == nfigs):
                nsubplots = nplots % nfig
            for j in range(nsubplots):
                if idx > len(column_comb_tuple)-1:
                    break
                ax = plt.subplot(2,2,j+1)
                icol_2, icol_1 = column_comb_tuple[idx]
                col_name_1 = good_columns[icol_1]
                column_1 = frame[col_name_1]
                col_name_2 = good_columns[icol_2]
                column_2 = frame[col_name_2]
                injected_vals = select_hierarchy(template_settings,normal_hierarchy=True)
                scale_1 = 1.0
                #if 'llh' not in col_name_1:
                if col_name_1 not in ['e_reco_precision_up', 'e_reco_precision_down', 'cz_reco_precision_up', 'cz_reco_precision_down', 'llh', '1st_grad', 'funcalls', 'nit', 'injected_f']:
                    scale_1 = injected_vals[col_name_1]['scale']
                if col_name_1 == 'nutau_norm':
                    scale_1 = 1
                column_1 = scale_1*column_1
                if col_name_1 == "theta23":
                    column_1 = np.sin(column_1)*np.sin(column_1)
                    col_name_1 = "[sin(theta23)]^2"

                scale_2 = 1.0
                #if 'llh' not in col_name_2:
                if col_name_2 not in ['e_reco_precision_up', 'e_reco_precision_down', 'cz_reco_precision_up', 'cz_reco_precision_down', 'llh', '1st_grad', 'funcalls', 'nit', 'injected_f']:
                    scale_2 = injected_vals[col_name_2]['scale']
                if col_name_2 == 'nutau_norm':
                    scale_2 = 1
                if col_name_2 == "theta23":
                    column_2 = np.sin(column_2)*np.sin(column_2)
                    col_name_2 = "[sin(theta23)]^2"
                column_2 = scale_2*column_2

                xmin = min(column_1)
                ymin = min(column_2)
                xmax = max(column_1)
                ymax = max(column_2)
                ax.set_xlim([xmin-abs(xmax-xmin)*0.1,xmax+0.1*abs(xmax-xmin)])
                ax.set_ylim([ymin-abs(ymax-ymin)*0.1,ymax+0.1*abs(ymax-ymin)])
                corr_result = pearsonr(column_1,column_2)
                plt.scatter(column_1,column_2)
                plt.xlabel(col_name_1)
                plt.ylabel(col_name_2)
                plt.title('corr: %.3f, p_val: %.6f'%(corr_result[0],corr_result[1]),fontsize='large')
                plt.grid(True)
                idx+=1
            if save_fig:
                filestem=args.llh_file.split('/')[-1]
                filename=(filestem.split('.hdf5')[0]+'_correlations_'+data+'_'+hypo+
                          '_'+str(i)+'.png')
                fig_names.append(filename)

    return figs,fig_names


def plot_posterior_params(frames, template_settings, plot_param_info=True,pbins=20,
                          save_fig=False,**kwargs):
    """Plot posterior parameter distributions, and related data"""

    good_columns = [col for col in frames[0].columns if col not in ['hypo','pseudo_data', 'llh', 'nutau_norm']]
    print "good_columns = ", good_columns
    #del good_columns[2]

    max_plots_per_fig = 4
    nfigs = (len(good_columns)-1) / max_plots_per_fig + 1
    print "nfigs = ", nfigs
    logging.info("len(good_cols): %d, nfigs: %d"%(len(good_columns),nfigs))

    figs = []
    fig_names = []
    colors = ['b','r','g','k','c','m']
    for frame in frames:
        data = frame['pseudo_data'][0]
        hypo = frame['hypo'][0]
        icol = 0
        for i in xrange(nfigs):
            #print "i = ", i

            fig = plt.figure(figsize=(10,10))
            figs.append(fig)
            fig.suptitle('Posteriors for %s, %s'%(data,hypo),fontsize='large')

            # determine how many columns to put on each figure:
            nsubplots = max_plots_per_fig
            if (i == nfigs):
                nsubplots = nplots % nfig
            for j in range(nsubplots):
                ax = plt.subplot(2,2,j+1)
                color = 'k' if plot_param_info else colors[icol%len(colors)]

                # Rescale the column data by 'scale'-so that these are
                # the values the optimizer actually sees, and makes it
                # easier to display on plots
                if icol >= len(good_columns):
                    break
                col_name = good_columns[icol]
                column = frame[col_name]
                #print "column = ", col_name
                #print "column[1] = ", column[1]
                #print "type column[1] = ", type(column[1])

                injected_vals = select_hierarchy(template_settings,normal_hierarchy=True)
                #if data == 'data_tau': injected_vals = select_hierarchy(template_settings,normal_hierarchy=True)
                #else: injected_vals = select_hierarchy(template_settings,normal_hierarchy=False)

                scale = 1
                if col_name not in ['e_reco_precision_up', 'e_reco_precision_down', 'cz_reco_precision_up', 'cz_reco_precision_down', 'llh', '1st_grad', 'funcalls', 'nit', 'injected_f']:
                    scale = injected_vals[col_name]['scale']
                column = scale*column

                hist,xbins,patches = plt.hist(column,histtype='step',
                                              lw=2,color=color,bins=pbins)
                plt.title(good_columns[icol],fontsize='large')
                plt.grid(True)

                # Plot extra info about priors, injected val, mean, range, etc.
                if plot_param_info:
                    ylim = ax.get_ylim()
                    ymax = ylim[1]
                    std = column.std()
                    mean = column.mean()
                    key = column.name
                    logging.debug("Processing column: %s"%key)
                    print data, " ", hypo , " ",key, " mean : ", mean, " std : ", std

                    # First, plot mean and std dev:
                    plot_mean_std(mean,std,ymax,ax)

                    # Next plot: injected val
                    #if key != 'llh':
                    if key not in ['e_reco_precision_up', 'e_reco_precision_down', 'cz_reco_precision_up', 'cz_reco_precision_down', '1st_grad','llh', 'funcalls', 'nit', 'injected_f']:
                        scale = injected_vals[key]['scale']
                        plot_injected_val(scale*injected_vals[key]['value'],ymax)

                        #if key == 'theta23':
                        #    print "injected: ",scale*injected_vals[key]['value']
                        #    print "mean: ",mean

                        fit_vals = select_hierarchy(template_settings,normal_hierarchy=True)
                        #if hypo == 'hypo_free': fit_vals = select_hierarchy(template_settings,normal_hierarchy=True)
                        #else: fit_vals = select_hierarchy(template_settings,normal_hierarchy=False)
                        # Now plot prior:
                        if injected_vals[key]['prior']['kind']=='gaussian':
                            plot_prior(scale*fit_vals[key]['prior']['sigma'],
                                       scale*fit_vals[key]['value'], ymax,ax)

                        # Finally, plot bound:
                        plot_bound(scale*fit_vals[key]['range'],ymax,ax)

                    if key in ['e_reco_precision_up', 'e_reco_precision_down', 'cz_reco_precision_up', 'cz_reco_precision_down']:
                        plot_injected_val(injected_vals[key]['value'],ymax)
                        fit_vals = select_hierarchy(template_settings,normal_hierarchy=True)
                        if injected_vals[key]['prior']['kind']=='gaussian':
                            plot_prior(scale*fit_vals[key]['prior']['sigma'],
                                       scale*fit_vals[key]['value'], ymax,ax)
                        plot_bound(scale*fit_vals[key]['range'],ymax,ax)

                    ax.set_xlim([mean-5.0*std,mean+5.0*std])
                    ax.set_ylim([ylim[0],ymax*1.2])

                    plt.legend(loc='best',fontsize='large',framealpha=0.5)
                icol+=1

            if save_fig:
                filestem=args.llh_file.split('/')[-1]
                filename=(filestem.split('.hdf5')[0]+'_'+data+'_'+hypo+
                          '_'+str(i)+'.png')
                fig_names.append(filename)

    return figs,fig_names

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('llh_file',type=str,help="Processed LLH files to analyze")
parser.add_argument('--nbins',type=int,default=50,help="Number of bins in x axis.")
parser.add_argument('--yscale',type=float,default=1.2,help='Factor to scale ymax by.')
parser.add_argument('--no_gauss',action='store_true',default=False,
                    help="Do not plot gaussian fit overlaid on distributions")
parser.add_argument('--notau_true',action='store_true',default=False,
                    help="Assumes NOTAU is the case nature gives us (rather than TAU).")

# Parameter posterior arguments
parser.add_argument('--no_params',action='store_true',default=False,
                    help="Do not plot any posterior parameter information.")
parser.add_argument('--pbins',type=int,default=20,
                    help="Number of bins in x axis for posteriors.")
parser.add_argument('--plot_param_extra',action='store_true',default=False,
                    help='Plot extra parameter information on posteriors')

parser.add_argument('-s','--save_fig',action='store_true',default=False,
                    help='Save all figures')
parser.add_argument('-v', '--verbose', action='count', default=0,
                    help='set verbosity level')
args = parser.parse_args()
set_verbosity(args.verbose)

#df_dTAU_hFREE, df_dTAU_hNOTAU, df_dNOTAU_hFREE, df_dNOTAU_hNOTAU = get_data_frames(args.llh_file)
df_dTAU_hFREE, df_dNOTAU_hFREE = get_data_frames(args.llh_file)
template_settings = get_template_settings(args.llh_file)

if args.verbose > 1: show_frame(df_dTAU_hFREE)

# Apply this function to columns of data frame to get LLR:
#get_llh_ratio = lambda hNOTAU,hFREE: 2 *(hNOTAU['llh'] - hFREE['llh'])
#llr_dTAU = get_llh_ratio(df_dTAU_hNOTAU,df_dTAU_hFREE)
#llr_dNOTAU = get_llh_ratio(df_dNOTAU_hNOTAU,df_dNOTAU_hFREE)
#logging.info("tau mean: %f"%llr_dTAU.mean())
#logging.info("notau mean: %f"%llr_dNOTAU.mean())

#plot_llr_distributions(
#    llr_dTAU,llr_dNOTAU,args.nbins,plot_gauss=(not args.no_gauss),
#    notau_true=args.notau_true)
## Configure plotting:
#ylim = plt.gca().get_ylim()
#ylim = [ylim[0],ylim[1]*args.yscale]
#plt.gca().set_ylim(ylim)
#plt.legend(loc='best',fontsize='medium')
#if args.save_fig:
#    filestem=args.llh_file.split('/')[-1]
#    filename=(filestem.split('.')[0]+'_LLR.png')
#    logging.info('Saving to file: %s'%filename)
#    plt.savefig(filename,dpi=150)


# Next plot posterior parameter distributions and related plots:
#frames = [df_dTAU_hFREE,df_dTAU_hNOTAU,df_dNOTAU_hFREE,df_dNOTAU_hNOTAU]
frames = [df_dTAU_hFREE,df_dNOTAU_hFREE]
figs,fignames = plot_posterior_params(
    frames, template_settings, plot_param_info=args.plot_param_extra,
    save_fig=args.save_fig,pbins=args.pbins)
if args.save_fig:
    for i,fig in enumerate(figs):
        fig.savefig(fignames[i],dpi=160)

corr_figs, corr_fignames = plot_correlations([df_dTAU_hFREE,df_dNOTAU_hFREE],template_settings,
    save_fig=args.save_fig,pbins=args.pbins)
if args.save_fig:
    for i,corr_fig in enumerate(corr_figs):
        corr_fig.savefig(corr_fignames[i],dpi=160)
else: plt.show()

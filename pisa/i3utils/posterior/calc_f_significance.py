#! /usr/bin/env python
#
# calc_f_significance.py
#
# Takes the processed hdf5 file of the combined data from all trial
# pseudo data sets and plots the nutau norm factor f distributions and the parameter
# distributions.
#

from pisa.utils.log import logging, set_verbosity
from pisa.utils.jsons import from_json
from pisa.utils.utils import get_bin_centers
from pisa.utils import kde

from argparse import ArgumentParser
import numpy as np
import h5py
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erfinv
from scipy.stats import norm, kstest
from math import erf, sqrt, pi, isinf

def phi(x, mu, sigma):
    return 0.5*(1 + erf( (x-mu)/sigma/sqrt(2)))

def sum_phi(x0, mu, sigma):
    assert(len(mu) == len(sigma))
    sum = 0
    for i in range(0, len(mu)):
        sum += phi(x0, mu[i], sigma[i])
    return sum

def sum_1_minus_phi(x0, mu, sigma):
    assert(len(mu) == len(sigma))
    sum = 0
    for i in range(0, len(mu)):
        sum += 1 - phi(x0, mu[i], sigma[i])
    return sum

def sum_gauss(x, amp, loc, width):
    assert(len(amp)==len(loc))
    assert(len(amp)==len(width))
    return np.array([sum(gauss(x_val, amp, loc, width)) for x_val in x])

######### For Gaussian Fit #########
gauss = lambda x, amp, loc, width: amp*np.exp(-(x-loc)**2/(2.*width**2))

def do_gauss(xvals, yvals, **kwargs):
    f, c = curve_fit(gauss, xvals, yvals, **kwargs)
    return f, np.sqrt(np.diag(c))

def plot_gauss(xvals, fit, **kwargs):
    return plt.plot(xvals, gauss(xvals, *fit), lw=2, **kwargs)

######### For VBW KDE Fit #########

def do_vbw_kde(data, **kwargs):
    #bw, mesh, pdf = kde.vbw_kde(data = data, MIN=-1.0, MAX=2.1)
    bw, mesh, pdf = kde.vbw_kde(data = data)
    return bw, mesh, pdf

####################################


def plot_f_distributions(f_data_tau,f_data_notau,nbins=100,xmax=None,
                           run_gauss=True, run_vbw_kde=True, outfilename=None,notau_true=False):
    '''
    Plots the f distributions, by forming a histogram and plotting
    it.  Will later be used to calculate area under curve of overlap,
    and significance.
    '''

    min_bin = 0.0; max_bin = 0.0
    max_bin = max(max(f_data_notau),max(f_data_tau))
    min_bin = min(min(f_data_notau),min(f_data_tau))

    print "  in_bin_NOTAU: ",np.sum(in_bin_NOTAU)
    print "  in_bin_TAU: ",np.sum(in_bin_TAU)
    f_data_notau = f_data_notau[in_bin_NOTAU]
    f_data_tau = f_data_tau[in_bin_TAU]

    xbins = np.linspace(min_bin,max_bin,nbins)
    label_text = r'$data: \/ \nu_{\tau} \/ {\rm CC \/ norm=0}$'
    #plt.figure(figsize=(8,6))
    hist_notau,bins_notau,_ = plt.hist(f_data_notau,bins=xbins,histtype='step', lw=2, color='r',label=label_text)
    bincen = get_bin_centers(xbins)
    plt.errorbar(bincen,hist_notau,yerr=np.sqrt(hist_notau),fmt='.r')

    if run_gauss:
        avg = np.mean(f_data_notau)
        print "  notau avg: ",avg
        fit_notau, cov = do_gauss(bincen,hist_notau,p0=[max(hist_notau),avg,2])
        print "  fit_notau: ", fit_notau
        print "  fit_notau_cov: ", cov
        SS_tol = np.sum(np.square(hist_notau - np.mean(hist_notau)))
        SS_res = np.sum(np.square(hist_notau - gauss(bincen, fit_notau[0], fit_notau[1], fit_notau[2])))
        R2 = 1 - SS_res/SS_tol
        print "  R2 = ", R2
        ks_notau,_ = kstest((f_data_notau - fit_notau[1])/fit_notau[2],'norm')
        gaus_notau = plot_gauss(bincen,fit_notau,color='r', label=r'${\rm Gaus\,fit}, R^2 = %.3f, {\rm KS} = %.3f $'% (R2, ks_notau))
        if args.logy: plt.yscale('log')

    label_text = r'$data: \/ \nu_{\tau} \/ {\rm CC \/ norm=1}$'
    hist_tau,bins_tau,_ = plt.hist(f_data_tau,bins=xbins,histtype='step',lw=2, color='b',label=label_text)
    plt.errorbar(bincen,hist_tau,yerr=np.sqrt(hist_tau),fmt='.b')

    if run_gauss:
        avg = np.mean(f_data_tau)
        print "  tau avg: ",avg
        fit_tau, cov = do_gauss(bincen,hist_tau,p0=[max(hist_tau),avg,2])
        print "  fit_tau: ", fit_tau
        print "  fit_tau_cov: ", cov
        SS_tol = np.sum(np.square(hist_tau - np.mean(hist_tau)))
        SS_res = np.sum(np.square(hist_tau - gauss(bincen, fit_tau[0], fit_tau[1], fit_tau[2])))
        R2 = 1 - SS_res/SS_tol
        print "  R2 = ", R2

        #f_data_tau = f_data_tau[f_data_tau<1.25]
        #f_data_tau = f_data_tau[f_data_tau>0.75]
        #hist_tau,bins,_ = plt.hist(f_data_tau,bins=xbins,histtype='step',lw=2, color='b',label=label_text)
        #avg = np.mean(f_data_tau)

        fit_tau, cov = do_gauss(bincen,hist_tau,p0=[max(hist_tau),avg,2])

        #print "use only 0.75 to 1.24, tau avg: ",avg
        #print "use only 0.75 to 1.24, fit_tau: ", fit_tau
        #print "use only 0.75 to 1.24, fit_tau_cov: ", cov

        ks_tau,_ = kstest((f_data_tau - fit_tau[1])/fit_tau[2],'norm')
        gaus_tau = plot_gauss(bincen,fit_tau,color='b', label=r'${\rm Gaus\,fit}, R^2 = %.3f, {\rm KS} = %.3f $'% (R2, ks_tau))
        plt.plot(np.ones(20)*avg, np.arange(0, max(hist_tau), max(hist_tau)/20), 'k--', lw=2)
        if args.logy: plt.yscale('log')
    sigma  = np.abs((fit_tau[1]-fit_notau[1])/fit_notau[2])
    print "  Use the distribution of f, significance = ", np.abs((fit_tau[1]-fit_notau[1])/fit_notau[2])
    print "                           one sigma + = ", np.abs(fit_tau[1]+fit_tau[2]-fit_notau[1])/fit_notau[2]
    print "                         , one sigma - = ", np.abs(fit_tau[1]-fit_tau[2]-fit_notau[1])/fit_notau[2]
    print " "
    plt.grid()

    plt.ylabel(r'${\rm Number \, of \, Trials}$',fontsize='large')
    plt.xlabel(r'$\nu_{\tau} \/ {\rm CC \, norm}$',fontsize='large')
    plt.title(r'$%s \/ {\rm yr \, significance} = \/ %.2f \/ \sigma \/ (%d \/ {\rm trials})$'%(livetime, sigma, len(f_data_tau)), fontsize=20)

    plt.xlim(min_bin - 5*((max_bin-min_bin)/nbins),max_bin+ 5*((max_bin-min_bin)/nbins))
    if args.logy:
        plt.ylim(0.5,max(max(hist_tau),max(hist_notau))*8.0)
    else:
        plt.ylim(0,max(max(hist_tau),max(hist_notau))*1.3)

    print "\nUsing histogram to calculate significance :" 
    calc_significance(f_data_notau,f_data_tau,hist_notau,hist_tau,xbins,notau_true=notau_true)

    #if args.logy:
    #    leg = plt.legend(loc='lower left',fontsize='medium')
    #else:
    #    leg = plt.legend(loc='upper right',fontsize='medium') 
    leg = plt.legend(loc='upper right',fontsize='medium', ncol = 2) 
    leg.draw_frame(False)

    if outfilename is not None:
        logging.info('Saving nutau norm f distributions to file: %s'%outfilename)
        if args.logy:
            plt.savefig(outfilename.split('.png')[0]+'_logy_gaus.png',dpi=150)
        else:
            plt.savefig(outfilename.split('.png')[0]+'_gaus.png',dpi=150)
        plt.show()

    if run_vbw_kde:
        print "\nRunning VBW KDE to estimate significance..."
        tau_bw, tau_mesh, tau_pdf = do_vbw_kde(f_data_tau)
        notau_bw, notau_mesh, notau_pdf = do_vbw_kde(f_data_notau)
        label_text = r'$data: \/ \nu_{\tau} \/ {\rm CC \/ norm=0}$'
        #plt.figure(figsize=(8,6))
        hist_tau, bins_tau, _ = plt.hist(f_data_tau,bins=xbins,histtype='step',lw=2, color='b',label=label_text)
        plt.errorbar(bincen,hist_tau,yerr=np.sqrt(hist_tau),fmt='.b')

        label_text = r'$data: \/ \nu_{\tau} \/ {\rm CC \/ norm=1}$'
        hist_notau, bins_notau, _ = plt.hist(f_data_notau,bins=xbins,histtype='step', lw=2, color='r',label=label_text)
        plt.errorbar(bincen,hist_notau,yerr=np.sqrt(hist_notau),fmt='.r')

        area_tau = sum(np.diff(bins_tau)*hist_tau)
        SS_tol = np.sum(np.square(hist_tau - np.mean(hist_tau)))
        sum_gauss_at_bincen = sum_gauss(bincen, 1/sqrt(2.0*pi)/tau_bw, f_data_tau, tau_bw)*area_tau/len(f_data_tau)
        SS_res = np.sum(np.square(hist_tau - sum_gauss_at_bincen))
        R2 = 1 - SS_res/SS_tol
        print "  data_tau, R2 = ", R2
        plt.plot(bincen, sum_gauss_at_bincen, lw=2, color='b', label = r'${\rm VBW \/ KDE} \/ R^2 = %.3f$'%R2)

        area_notau = sum(np.diff(bins_notau)*hist_notau)
        SS_tol = np.sum(np.square(hist_notau - np.mean(hist_notau)))
        sum_gauss_at_bincen = sum_gauss(bincen, 1/sqrt(2.0*pi)/notau_bw, f_data_notau, notau_bw)*area_notau/len(f_data_notau)
        SS_res = np.sum(np.square(hist_notau - sum_gauss_at_bincen))
        R2 = 1 - SS_res/SS_tol
        print "  data_notau, R2 = ", R2
        plt.plot(bincen, sum_gauss_at_bincen, lw=2, color='r', label = r'${\rm VBW \/ KDE} \/ R^2 = %.3f$'%R2)


        # calculate the p value using notau_pdf above avg of f_data_tau 
        avg_f_data_tau = np.mean(f_data_tau)
        print "  avg_f_data_tau = ", avg_f_data_tau

        plt.plot(np.ones(20)*avg_f_data_tau, np.arange(0, max(hist_tau), max(hist_tau)/20), 'k--', lw=2)

        pval_vbwkde = sum_1_minus_phi(avg_f_data_tau, f_data_notau, notau_bw)/len(f_data_notau)
        sigma_vbwkde = norm.isf(pval_vbwkde) 
        print "  pval_vbwkde = ", pval_vbwkde
        print "  sigma_vbwkde (erfinv): %.4f"% np.sqrt(2.0)*erfinv(1.0 - pval_vbwkde)
        print "  sigma_vbwkde (isf): %.4f"%(norm.isf(pval_vbwkde))

        if args.logy: plt.yscale('log')
        plt.xlim(min_bin - 5*((max_bin-min_bin)/nbins),max_bin+ 5*((max_bin-min_bin)/nbins))
        if args.logy:
            plt.ylim(0.5,max(max(hist_tau),max(hist_notau))*8.0)
        else:
            plt.ylim(0,max(max(hist_tau),max(hist_notau))*1.3)
        plt.grid()
        plt.ylabel(r'${\rm Number \, of \, Trials}$',fontsize='large')
        plt.xlabel(r'$\nu_{\tau} \/ {\rm CC \/ norm}$', fontsize='large')
        plt.title(r'$%s \/ yr \/ {\rm significance} = \/ %.2f \/ \sigma \/ ( \/ {\rm pval}: %.2E, \/ %d \/ {\rm trials})$'%(livetime, sigma_vbwkde, pval_vbwkde, len(f_data_tau)), fontsize=20)
        leg = plt.legend(loc='upper right',fontsize='medium', ncol = 2) 
        leg.draw_frame(False)
        if args.logy:
            plt.savefig(outfilename.split('.png')[0]+'_logy_vbw_kde.png',dpi=150)
        else:
            plt.savefig(outfilename.split('.png')[0]+'_vbw_kde.png',dpi=150)
        plt.show()

    return fit_tau[1], fit_notau[1], fit_tau[2], fit_notau[2], len(f_data_tau)

def calc_significance(data_notau,data_tau,hist_notau,hist_tau,xbins,notau_true=False):
    '''
    calculates significance and draws line (and cross-hatching) at
    mean of distribution
    '''

    line_label = "Avg Val, obs f=0"
    #avg_val = np.mean(data_notau)
    avg_val = np.mean(data_tau) if notau_true else np.mean(data_notau)
    logging.info("Average value: %f"%avg_val)
    #pvalue = float(sum(data_tau>avg_val))/len(data_tau)
    pvalue = 1.0 - float(sum(data_notau>avg_val))/len(data_notau) if notau_true else float(sum(data_tau>avg_val))/len(data_tau)
    print "  pvalue: %.4f"%pvalue
    sigma = np.sqrt(2.0)*erfinv(1.0 - pvalue)
    print "  sigma (erfinv): %.4f"%sigma
    print "  sigma (isf): %.4f"%(norm.isf(pvalue))
    if isinf(sigma):
        print "  No bins above the average of the nutau hypothesis."
    else:
        print "  So use the histogram, significance = ", (norm.isf(pvalue))

    #ymax = max(hist_notau)
    ymax = max(hist_tau) if notau_true else max(hist_notau)
    #vline = plt.vlines(avg_val,1,ymax,colors='k',linewidth=2,
    #                   label=("pval = %.4f"%pvalue))

    bincen = get_bin_centers(xbins)
    #if args.notau_true:
    #    plt.fill_betweenx(hist_notau,bincen,x2=avg_val,where=bincen < avg_val,
    #                      alpha=0.5,hatch='xx')
    #else:
    #    plt.fill_betweenx(hist_tau,bincen,x2=avg_val,where=bincen > avg_val,
    #                      alpha=0.5,hatch='xx')

    return pvalue

def plot_param_dist(fh):
    '''
    Plots the param distributions for each case of the LLR analysis.
    I.e. Makes a 1D plot for each of <data_tau|true_TAU>,
    <data_tau|true_NOTAU>,etc. one 4-subplot figure for each systematic
    parameter.
    '''


    nedges = 31
    colors = ['r','b','g','m','c']
    colors = np.append(colors,colors)

    nsyst = len(fh['trials']['data_tau']['hypo_free'].keys()) - 1
    for i,key in enumerate(fh['trials']['data_tau']['hypo_free'].keys()):
        if key == 'llh': continue
        fig = plt.figure(figsize=(6,6),dpi=160)
        logging.info("Plotting: %s"%key)

        if key == 'deltam31': plt.suptitle(r'$\Delta$ m$_{31}$$^{2}$ (scaled)')
        else: plt.suptitle(key)
        plt.subplot(2,2,1)
        iplt = 1
        for dkey in ['data_tau','data_notau']:
            for hkey in ['hypo_free','hypo_notau']:
                plot = plt.subplot(2,2,iplt)
                plt.title('<'+dkey+' | '+hkey+'>',fontsize='small')
                #best_fit = np.array([fh['trials'][j][dkey][hkey][key][-1] for j in range(len(data['trials']))])
                best_fit = np.array(fh['trials'][dkey][hkey][key])

                best_fit = best_fit[in_bin_NOTAU] if dkey == 'data_notau' else best_fit[in_bin_TAU]

                xmax = max(best_fit); xmin = min(best_fit)
                if key == 'theta23':
                    xmin = 0.64; xmax=0.72
                elif key == 'deltam31':
                    best_fit*=100.0;
                    xmax = 0.26 if hkey=="hypo_free" else -0.22
                    xmin = 0.23 if hkey=="hypo_free" else -0.25
                elif key == 'nu_nubar_ratio':
                    xmax = 1.3; xmin = 0.7
                delta = (xmax - xmin)/float(nedges)
                xmin = np.mean(best_fit) - 4.0*np.std(best_fit)
                xmax = np.mean(best_fit) + 4.0*np.std(best_fit)
                #logging.debug("  delta: %f"%delta)
                bin_edges = np.linspace(xmin,xmax,41)
                #if 'nu_nubar' in key or 'energy' in key: nedges=41
                #bin_edges = np.linspace(xmin-delta,xmax+delta,nedges)
                hist,_,_ = plt.hist(best_fit,bins=bin_edges,histtype='step',
                                    color=colors[i])
                ymax = max(hist)
                plt.ylim(1,ymax*1.1)
                plt.grid()
                if iplt%2 == 1: plt.ylabel('# Trials')

                plt.xlim(xmin,xmax)
                if args.logy: plt.yscale('log')

                avg_val = np.mean(best_fit)
                #vline = plt.vlines(avg_val,0,ymax,colors='k',linewidth=2)
                plot.tick_params(axis='both',labelsize=7)

                if args.fit_params:
                    bincen = get_bin_centers(bin_edges)
                    fit_param, cov = do_gauss(bincen,hist,p0=[max(hist),avg_val,0.1])
                    plot_gauss(bincen,fit_param,color=colors[i])

                iplt+=1
        filename = key+'_'+str(nsyst)+'syst.png'
        logging.info('Saving file: %s'%filename)
        plt.savefig(filename,dpi=150)
    plt.tight_layout()

    return


def plot_hist(mass_array,theta_array,bins_dm,bins_th,extent,ylabel=True,xlabel=True):
    '''
    Define a histogram from the mass,theta array data and create a
    plot. Also modify the axes to display vertical
    '''

    hist1,_,_ = np.histogram2d(theta_array,mass_array,bins=[bins_th,bins_dm])
    plt.imshow(hist1,extent=extent,aspect='auto',interpolation='nearest',
               origin='lower')#,cmap="RdBu_r")
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)

    plt.colorbar()

    if xlabel: plt.xlabel(r'$\Delta m^2_{31}$ [10$^{-2}$ eV$^2$]')
    if ylabel: plt.ylabel(r'$\theta_{23}$')
    return

# Fit a gaussian to the function:
def gauss_fn(x,*p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def get_bin_centers(edges):
    return (np.array(edges[:-1]) + np.array(edges[1:]))/2

parser = ArgumentParser()
parser.add_argument('llh_file',type=str,help="Processed LLH files to analyze")
parser.add_argument('--nbins',type=int,default=50,help="Number of bins in x axis.")
parser.add_argument('--no_gauss',action='store_true',default=False,
                    help="Do not plot gaussian fit for the nutau CC norm distribution.")
parser.add_argument('--no_vbw_kde',action='store_true',default=False,
                    help="Do not plot VBW KDE result for the nutau CC norm distribution.")
parser.add_argument('--plot_params',action='store_true',default=False,
                    help="Plots best fit param distributions.")
parser.add_argument('--fit_params',action='store_true',default=False,
                    help="Fits best fit params to gaussian.")
parser.add_argument('--xmax',default=None,type=float,
                    help="Max value on x axis if defined [default: None]")
parser.add_argument('--notau_true',action='store_true',default=False,
                    help="Assumes data is drawn from NOTAU is the true case.")
parser.add_argument('--logy',action='store_true', default=False,
                    help="Plot posterior parameter distributions on log scale.")
parser.add_argument('-o','--outfile',type=str,default=None,
                    help="Saves the f distributions to this filename, if defined.")
parser.add_argument('-y','--year',type=float,help="Number of live time years.")
parser.add_argument("--no_sys",type=int, help='''Number of systematics''')
parser.add_argument("--n_repeat",type=int,default=30000,
                    help='''Number of times to repeat the estimation at the number of''')
parser.add_argument('-v', '--verbose', action='count', default=0,
                    help='set verbosity level')
args = parser.parse_args()
print "  nbins: ",args.nbins
set_verbosity(args.verbose)
args.no_gauss = not args.no_gauss
args.no_vbw_kde = not args.no_vbw_kde
livetime = args.year


fh = h5py.File(args.llh_file,'r')
llh_file_name = args.llh_file
f_data_tau = np.array(fh['trials']['data_tau']['hypo_free']['nutau_norm'])
f_data_notau = np.array(fh['trials']['data_notau']['hypo_free']['nutau_norm'])

# Get rid of trials with warnflag == 2
#warnflag_data_tau = np.array(fh['trials']['data_tau']['hypo_free']['warnflag'])
#warnflag_data_notau = np.array(fh['trials']['data_notau']['hypo_free']['warnflag'])
#f_data_tau = f_data_tau[warnflag_data_tau==0]
#f_data_notau = f_data_notau[warnflag_data_notau==0]

# This defaults to all indices:
in_bin_TAU = np.alltrue(np.array([np.fabs(f_data_tau) >= 0.0]),axis=0)
in_bin_NOTAU = np.alltrue(np.array([np.fabs(f_data_notau) >= 0.0]),axis=0)

logging.info("Processing number of trials: %d"%len(f_data_tau))

mu_tau, mu_notau, sigma_tau, sigma_notau, ntrials = plot_f_distributions(f_data_tau,f_data_notau,nbins=args.nbins,xmax=args.xmax,
                       outfilename=args.outfile, run_gauss=args.no_gauss, run_vbw_kde = args.no_vbw_kde, notau_true=args.notau_true)

if args.plot_params: plot_param_dist(fh)

plt.show()

true_sigma = np.fabs(mu_notau - mu_tau)/sigma_notau
print "\n\n>>>>> Getting MC error on the significance (from Gaus fit)<<<<<"
print("True significance for these parameters: ",true_sigma)

sigma_error = np.zeros(args.n_repeat)
for trial in xrange(1,args.n_repeat+1):
    dist1 = np.random.normal(mu_tau, abs(sigma_tau), ntrials)
    dist2 = np.random.normal(mu_notau, abs(sigma_notau), ntrials)

    calc_sigma = np.fabs(dist2.mean() - dist1.mean())/dist2.std()

    sigma_error[trial-1] = (true_sigma - calc_sigma)


print "\n\n>>>>> Results <<<<<"
print "For %d trials, for two gaussians with parameters: "%ntrials
print "  (mu_tau, sigma1): = ("+str(mu_tau)+", "+str(sigma_tau)+")"
print "  (mu_notau, sigma2): = ("+str(mu_notau)+", "+str(sigma_notau)+")"
print "The error on the significance is: ",sigma_error.std()
print "The fractional error on the significance is: ",(sigma_error.std()/true_sigma)



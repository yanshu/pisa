#! /usr/bin/env python
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os, sys
from scipy.stats import chi2
from scipy import optimize
from matplotlib.offsetbox import AnchoredText
import matplotlib.cm as cm
from cycler import cycler
from pisa.utils.jsons import from_json
import collections
from pisa.utils.params_MH import select_hierarchy

def plot(name,data, asimov, hypos, asimov_hypos, params,trials):
    
    median = np.array([])
    sigmam = np.array([])
    sigmap = np.array([])
    sigmam2 = np.array([])
    sigmap2 = np.array([])
    for hypo in hypos:
        median = np.append(median,np.percentile(data['%.1f'%hypo],50))
        sigmam = np.append(sigmam,np.percentile(data['%.1f'%hypo],16))
        sigmam2 = np.append(sigmam2,np.percentile(data['%.1f'%hypo],5))
        sigmap = np.append(sigmap,np.percentile(data['%.1f'%hypo],84))
        sigmap2 = np.append(sigmap2,np.percentile(data['%.1f'%hypo],95))
    
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y','c','m','k']*2) +
                           cycler('linestyle', ['-']*7+['--']*7)))


    fig = plt.figure()
    fig.patch.set_facecolor('none')
    if name == 'llh':
        ax = plt.subplot2grid((6,1), (0,0), rowspan=5)
    else:
        ax = fig.add_subplot(111)
    if hypos:
        ax.fill_between(hypos,sigmam2,sigmap2,facecolor='b', linewidth=0, alpha=0.15, label='90% range')
        ax.fill_between(hypos,sigmam,sigmap,facecolor='b', linewidth=0, alpha=0.3, label='68% range')
        ax.plot(hypos,median, color='k', label='median')
    #colors = {'asimov_hole_ice_no_prior':'red','asimov_no_holeice':'k','asimov_hole_ice_tight_prior':'mediumvioletred','asimov_hole_ice_prior':'peru','asimov_scan_nutau_norm':'r'}
    #labels = {'asimov_hole_ice_no_prior':'HI no prior','asimov_no_holeice':'no HI','asimov_hole_ice_tight_prior':'HI tight prior','asimov_hole_ice_prior':'HI normal prior','asimov_scan_nutau_norm':'asimov'}
    for key in asimov.keys():
        #print key
        if asimov[key].has_key(name):
            text = key.split('_')
            #if len(text) == 2:
            #    label = 'nominal'
            #else:
            #    label = '_'.join(text[:-2]) + ' fixed'
            label = key
            ax.plot(asimov_hypos[key],asimov[key][name], label=label)
    ax.legend(loc='upper center',ncol=2, frameon=False,numpoints=1,fontsize=10)
    ax.set_xlabel(r'$\nu_{\tau}$ normalization')
    #ax.patch.set_facecolor('white')
    #ax.set_axis_bgcolor('white') 
    #ax.set_frame_on(False)
    if name == 'llh':
        ax.set_title('profile likelihood, 4 years, %s trials'%trials)
        ax2 = plt.subplot2grid((6,1), (5,0),sharex=ax)
        ax2.errorbar(np.array([1.42]),np.array([1.]),xerr=np.array([[0.47],[0.49]]),fmt='--o')
        ax2.text(0.1,0.75,r'Super-K 2013 (68%)',size=10)
        ax2.errorbar(np.array([1.8]),np.array([2.]),xerr=np.array([[1.1],[1.8]]),fmt='--o')
        ax2.text(0.1,1.75,r'Opera 2015 (90%)',size=10)
        ax2.set_ylim(0,3)
        ax2.set_xlim(0,2)
        fig.subplots_adjust(hspace=0)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.set_xlabel(r'$\nu_{\tau}$ normalization')
        for i in [0.5,1,1.5]:
            ax2.axvline(i,color='k', linestyle=':',alpha=0.5)

    else:
        ax.set_title('nuisance pulls, 4 years, %s trials'%trials)
    if name == 'llh':
        ax.set_ylabel(r'$-2\Delta LLH$')
        ax.set_ylim([0,40])
        for i in [1,4,9,16,25,36]:
            ax.axhline(i,color='k', linestyle='-',alpha=0.25)
            ax.text(1.85,i,r'$%s \sigma$'%np.sqrt(i), alpha=0.5)
    else:
        ax.set_ylabel(name)
        if params.has_key(name):
            params_value = params[name]['value']
            ax.axhline(params_value, color='g')
            ax.text(1.80,params_value,r'nominal',color='g',size=10)
            if params[name]['prior'].has_key('sigma'):
                params_sigma = params[name]['prior']['sigma']
                params_sigma = params[name]['prior']['sigma']
                ymin = params_value - params_sigma
                ymax = params_value + params_sigma
                ax.axhline(ymin, color='g', linestyle='--')
                ax.axhline(ymax, color='g', linestyle='--')
                ax.text(1.88,ymin,r'$-1\sigma$',color='g')
                ax.text(1.88,ymax,r'$+1\sigma$',color='g')
                ax.set_ylim(ymin-0.4*(params_value-ymin), ymax+0.4*(ymax-params_value))
            else:
                delta = ax.get_ylim()[1]-ax.get_ylim()[0]
                ax.set_ylim(ax.get_ylim()[0]-0.4*delta, ax.get_ylim()[1]+0.4*delta)
    for i in [0.5,1,1.5]:
        ax.axvline(i,color='k', linestyle=':',alpha=0.5)
    plt.show()
    plt.savefig('q1_%s.png'%name, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig('q1_%s.pdf'%name, facecolor=fig.get_facecolor(), edgecolor='none')

def dist(data,name,hypos, asimov_hypos, params,trials):

    for hypo in hypos:

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(data['%.1f'%hypo],20,alpha=0.5)
        ax.axvline(np.median(data['%.1f'%hypo]),color='r',linewidth=2)
        ax.patch.set_facecolor('white')
        if params.has_key(name):
            if params[name].has_key('value'):
                params_value = params[name]['value']
                ax.axvline(params_value, color='g',linewidth=2)
        ax.set_xlabel(name)
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.2)
        ax.set_title('profile likelihood, 4 years, %s trials'%trials)
        plt.show()
        plt.savefig('q%.1f_%s.png'%(hypo,name),transparent=True)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-d','--dir',metavar='dir',help='directory containg output json files', default='.') 
    parser.add_argument('--dist',action='store_true') 
    args = parser.parse_args()


    results = {}
    livetime = 0
    length = 0

    # get llh denominators for q for each seed
    for filename in os.listdir(args.dir):
        if filename.endswith('.json'):
            file = from_json(args.dir +'/'+filename)
            params = file['template_settings']['params']
            livetime = params['livetime']['value']
            for trial in file['trials']:
                mu = trial['mu_data']
                if not results.has_key(mu):
                    results[mu] = []
                results[mu].append([trial['fit_results'][1]['nutau_norm'][0],max(0,trial['q'][0])])

    mu_in = []
    mu_out_median = []
    mu_out_up = []
    mu_out_up2 = []
    mu_out_down = []
    mu_out_down2 = []
    fc_low = []
    fc_low2 = []
    fc_high = []
    fc_high2 = []
    fc_1s = []
    fc_2s = []
    fc_90 = []
    ML = []

    for mu in sorted(results.keys()):
        results[mu] = np.array(results[mu])
        data = results[mu][:,0]
        mu_in.append(mu)
        mu_out_median.append(np.percentile(data,50))
        mu_out_up.append(np.percentile(data,84))
        mu_out_down.append(np.percentile(data,16))
        mu_out_up2.append(np.percentile(data,97.5))
        mu_out_down2.append(np.percentile(data,2.5))
        # FC:
        data = results[mu]
        sorted = data[data[:,1].argsort()] 
        length = len(sorted[:,0])
        part = int(0.68*length)
        fc_1s.append(sorted[part,1])
        fc_high.append(max(sorted[:part,0]))
        fc_low.append(min(sorted[:part,0]))
        fc_90.append(sorted[int(0.9*length),1])
        fc_2s.append(sorted[int(0.95*length),1])
        part = int(0.95*length)
        fc_high2.append(max(sorted[:part,0]))
        fc_low2.append(min(sorted[:part,0]))
        ML.append(sorted[0,0])


        fig = plt.figure()
        ax = fig.add_subplot(111)
        n, bins, patches = ax.hist(data,30,alpha=0.5,normed=1)
        #n, bins, patches = ax.hist(data[:,0],30,alpha=0.5,normed=1)

        (muf, sigma) = norm.fit(data[:,0])
        y = mlab.normpdf( bins, muf, sigma)
        l = plt.plot(bins, y, 'r--', linewidth=2)

        a_text = AnchoredText('input: %.2f\nMedian: %.2f + %.2f -%.2f\nFit: %.2f +/-  %.2f'%(mu,mu_out_median[-1],mu_out_up[-1]-mu_out_median[-1],mu_out_median[-1]-mu_out_down[-1],muf, sigma), loc=2, frameon=False)
        ax.add_artist(a_text)

        fig.patch.set_facecolor('none')
        ax.set_xlabel(r'$\mu$')
        plt.show()
        plt.savefig('%s.pdf'%mu,edgecolor='none',facecolor=fig.get_facecolor())
        plt.close(fig)

    fig = plt.figure(figsize=(8,8))
    fig.patch.set_facecolor('none')
    ax = fig.add_subplot(111)
    ax.plot(mu_in, mu_out_median, color='k',linewidth=1.5 ,label='median')
    #ax.fill_between(mu_in,mu_out_down,mu_out_up,facecolor='b', linewidth=0, alpha=0.3, label='68% C.I. Neyman')
    ax.plot(mu_in,mu_out_up,color='k', linewidth=0.5, linestyle='-',label='68% C.I. Neyman')
    ax.plot(mu_in,mu_out_down,color='k', linewidth=0.5, linestyle='-')
    #ax.fill_between(mu_in,mu_out_down2,mu_out_up2,facecolor='b', linewidth=0, alpha=0.15, label='95% C.I. Neyman')
    ax.plot(mu_in,mu_out_up2,color='k', linewidth=0.5, linestyle=':', label='95% C.I. Neyman')
    ax.plot(mu_in,mu_out_down2,color='k', linewidth=0.5, linestyle=':')
    ax.fill_between(mu_in,fc_low,fc_high,facecolor='g', linewidth=0, alpha=0.3, label='68% C.I. Feldman-Cousins')
    ax.fill_between(mu_in,fc_low2,fc_high2,facecolor='g', linewidth=0, alpha=0.15, label='95% C.I. Feldman-Cousins')
    #ax.plot(mu_in, ML, color='r', label='maximum LLH')
    ax.plot(mu_in, mu_in, color='k',linestyle='-',alpha=0.2)
    ax.set_xlabel(r'$\mu_{data}$')
    ax.set_ylabel(r'$\mu_{measured}$')
    ax.set_xlim(min(mu_in),max(mu_in))
    ax.set_ylim(min(mu_in),max(mu_in))
    ax.legend(loc='lower right',ncol=1, frameon=False,numpoints=1,fontsize=10)
    a_text = AnchoredText(r'$\nu_\tau$ appearance'+'\n%s years, %s trials\nPreliminary'%(livetime,length), loc=2, frameon=False)
    ax.add_artist(a_text)
    ax.grid()
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('-')
        line.set_alpha(0.2)
    plt.show()
    plt.savefig('data.pdf',edgecolor='none',facecolor=fig.get_facecolor())
    plt.savefig('data.png',edgecolor='none',facecolor=fig.get_facecolor())

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor('none')
    ax.plot(mu_in, fc_1s, color='r',label='Feldman-Cousins')
    ax.plot(mu_in, fc_2s, color='r')
    ax.plot(mu_in, fc_90, color='r')
    ax.axhline(1,color='k',label='Asymptotic')
    ax.text(2.03,0.98,r'$1\sigma$')
    ax.axhline(4,color='k')
    ax.text(2.03,3.98,r'$2\sigma$')
    ax.axhline(2.71,color='k')
    ax.text(2.03,2.69,'$90 \%$')
    ax.axvline(0.5,color='k',alpha=0.2,linestyle='-', linewidth=0.5)
    ax.axvline(1.0,color='k',alpha=0.2,linestyle='-', linewidth=0.5)
    ax.axvline(1.5,color='k',alpha=0.2,linestyle='-', linewidth=0.5)
    ax.set_xlabel(r'$\mu_{data}$')
    ax.set_ylabel(r'q')
    ax.set_xlim(min(mu_in),max(mu_in))
    ax.set_ylim(0,5)
    ax.legend(loc='upper right',ncol=1, frameon=False,numpoints=1,fontsize=10)
    a_text = AnchoredText(r'$\nu_\tau$ appearance'+'\n%s years, %s trials\nPreliminary'%(livetime,length), loc=2, frameon=False)
    ax.add_artist(a_text)
    plt.show()
    plt.savefig('fc.pdf',edgecolor='none',facecolor=fig.get_facecolor())
    plt.savefig('fc.png',edgecolor='none',facecolor=fig.get_facecolor())

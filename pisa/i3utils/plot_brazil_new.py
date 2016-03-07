#! /usr/bin/env python
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
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
        print key
        if asimov[key].has_key(name):
            text = key.split('_')
            if len(text) == 2:
                label = 'nominal'
            else:
                label = '_'.join(text[:-2]) + ' fixed'
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


    total = 0
    asimov_hypos = collections.OrderedDict()
    asimov_results = collections.OrderedDict()
    results = []
    hypos = []
    sys = None

    # get llh denominators for q for each seed
    for filename in os.listdir(args.dir):
        if filename.endswith('.json'):
            file = from_json(args.dir +'/'+filename)
            name,_ = filename.split('.') 
            params = file['template_settings']['params']
            for trial in file['trials']:
                ts = trial['test_statistics']
                if ts == 'asimov':
                    asimov_hypos[name] = trial['nutau_norm']
                    asimov_results[name] = trial['fit_results'][0]
                    asimov_results[name]['llh'] = trial['q']
                    if name == 'nufit_prior' or name =='no_prior':
                        sys = asimov_results[name].keys()
                elif ts == 'profile':
                    hypos = trial['nutau_norm']
                    results = trial['fit_results'][0]
                    results['llh'] = trial['q']
                    total += 1


    if args.dist:
        for s in syslist:
            dist(results, asimov_results,hypos, asimov_hypos, params, total)
    else:
        for s in sys:
            plot(s,results, asimov_results,hypos, asimov_hypos, params, total)


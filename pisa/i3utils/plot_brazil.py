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

from pisa.utils.jsons import from_json

def plot(data,name,hypos,params,trials):
    
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

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.fill_between(hypos,sigmam2,sigmap2,facecolor='b', linewidth=0, alpha=0.15, label='90% range')
    ax.fill_between(hypos,sigmam,sigmap,facecolor='b', linewidth=0, alpha=0.3, label='68% range')
    ax.plot(hypos,median, color='k', label='median')
    ax.legend(loc='upper center',ncol=3, frameon=False,numpoints=1)
    ax.set_xlabel(r'$\mu$')
    ax.set_title('profile likelihood, 4 years, %s trials'%trials)
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
            ax.axhline(params_value, color='r')
            if params[name]['prior'].has_key('sigma'):
                params_sigma = params[name]['prior']['sigma']
                ymin = params_value - params_sigma
                ymax = params_value + params_sigma
                ax.axhline(ymin, color='r', linestyle='--')
                ax.axhline(ymax, color='r', linestyle='--')
                ax.text(1.88,ymin,r'$-1\sigma$',color='r')
                ax.text(1.88,ymax,r'$+1\sigma$',color='r')
                ax.set_ylim(ymin-0.4*(params_value-ymin), ymax+0.4*(ymax-params_value))
        else:
            delta = ax.get_ylim()[1]-ax.get_ylim()[0]
            ax.set_ylim(ax.get_ylim()[0]-0.4*delta, ax.get_ylim()[1]+0.4*delta)
    for i in [0.5,1,1.5]:
        ax.axvline(i,color='k', linestyle=':',alpha=0.5)
    plt.show()
    plt.savefig('q1_%s.png'%name)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-d','--dir',metavar='dir',help='directory containg output json files', default='.') 
    args = parser.parse_args()


    hypos = np.arange(0.,2.1,0.1)
    qs = {}
    sys = {}

    params = None

    syslist = ["aeff_scale","atm_delta_index","atmos_mu_scale","cz_reco_precision_down","cz_reco_precision_up","deltam31","dom_eff","e_reco_precision_down","e_reco_precision_up","energy_scale","hole_ice","nu_nubar_ratio","nue_numu_ratio","theta13","theta23"]
    for s in syslist: 
        sys[s] = {}
        for hypo in hypos:
            qs['%.1f'%hypo] = np.array([])
            sys[s]['%.1f'%hypo] = np.array([])

    total = 0
    for filename in os.listdir(args.dir):
        if filename.endswith('.json'):
            file = from_json(args.dir +'/'+filename)
            mu = filename[-9:-6]
            total += 1
            qs[mu] = np.append(qs[mu], float(file['trials'][0]['q'])) 
            for s in syslist:
                sys[s][mu] = np.append(sys[s][mu],float(file['trials'][0]['fit_results'][0][s][0]))
            if not params:
                params = file['template_settings']['params']

    total = int(total/len(hypos))
    plot(qs,'llh',hypos,params, total)
    for s in syslist:
        plot(sys[s],s,hypos,params, total)


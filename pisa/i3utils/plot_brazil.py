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
from pisa.utils.params_MH import select_hierarchy

def plot(data,name,hypos, asimov_hypos, params,trials):
    
    median = np.array([])
    sigmam = np.array([])
    sigmap = np.array([])
    sigmam2 = np.array([])
    sigmap2 = np.array([])
    asimov = np.array([])
    for hypo in hypos:
        median = np.append(median,np.percentile(data['%.1f'%hypo],50))
        sigmam = np.append(sigmam,np.percentile(data['%.1f'%hypo],16))
        sigmam2 = np.append(sigmam2,np.percentile(data['%.1f'%hypo],5))
        sigmap = np.append(sigmap,np.percentile(data['%.1f'%hypo],84))
        sigmap2 = np.append(sigmap2,np.percentile(data['%.1f'%hypo],95))
    for hypo in asimov_hypos:
        asimov = np.append(asimov,data['%.1fasimov'%hypo])

    fig = plt.figure()
    fig.patch.set_facecolor('none')
    ax = fig.add_subplot(111)
    if len(hypos)>0:
        ax.fill_between(hypos,sigmam2,sigmap2,facecolor='b', linewidth=0, alpha=0.15, label='90% range')
        ax.fill_between(hypos,sigmam,sigmap,facecolor='b', linewidth=0, alpha=0.3, label='68% range')
        ax.plot(hypos,median, color='k', label='median')
    if len(asimov)>0:
        ax.plot(asimov_hypos,asimov, color='r', label='asimov')
    ax.legend(loc='upper center',ncol=3, frameon=False,numpoints=1)
    ax.set_xlabel(r'$\mu$')
    #ax.patch.set_facecolor('white')
    #ax.set_axis_bgcolor('white') 
    #ax.set_frame_on(False)
    if name == 'llh':
        ax.set_title('profile likelihood, 4 years, %s trials'%trials)
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

    sys = {}

    params = None
    syslist = None
    hypos = np.array([])
    asimov_hypos = np.array([])
    sys = {}
    denoms = {}

    # get llh denominators for q for each seed
    for filename in os.listdir(args.dir):
        if filename.endswith('.json'):
            if 'denom' in filename:
                file = from_json(args.dir +'/'+filename)
                seed = filename[:4] 
                
                denoms[seed] = float(file['trials'][0]['llh'][0])

    total = 0
    for filename in os.listdir(args.dir):
        if filename.endswith('.json'):
            file = from_json(args.dir +'/'+filename)
            if not params:
                params = file['template_settings']['params']
                params = select_hierarchy(params, True)
            if not syslist:
                syslist = file['trials'][0]['fit_results'][0].keys()
                for s in syslist: 
                    sys[s] = {}
            if 'denom' in filename:
                continue
            elif 'asimov' in filename:
                mu = filename[-14:-11]
                asimov_hypos = np.append(asimov_hypos,float(mu))
                for s in syslist:
                    if s == 'llh':
                        sys[s]['%sasimov'%mu] = float(file['trials'][0]['q'])
                    else:
                        sys[s]['%sasimov'%mu] = float(file['trials'][0]['fit_results'][0][s][0])
            else:
                mu = filename[-9:-6]
                seed = filename[:4] 
                if not sys[syslist[0]].has_key(mu):
                    hypos = np.append(hypos,float(mu))
                    for s in syslist:
                        sys[s][mu] = np.array([])

                total += 1
                for s in syslist:
                    if s == 'llh':
                        if denoms.has_key(seed):
                            sys[s][mu] = np.append(sys[s][mu],2*(float(file['trials'][0]['llh'][0]) - denoms[seed])) 
                    else:
                        sys[s][mu] = np.append(sys[s][mu],float(file['trials'][0]['fit_results'][0][s][0]))

    if total:
        total = int(round(float(total)/len(hypos)))
    hypos.sort()
    asimov_hypos.sort()
    if args.dist:
        for s in syslist:
            dist(sys[s],s,hypos, asimov_hypos, params, total)
    else:
        for s in syslist:
            plot(sys[s],s,hypos, asimov_hypos, params, total)


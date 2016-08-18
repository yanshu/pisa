#! /usr/bin/env python
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
#mpl.rcParams['mathtext.fontset'] = 'stix'
#mpl.rcParams['font.family'] = 'STIXGeneral'
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os, sys
from scipy.stats import chi2
from scipy import optimize
from matplotlib.offsetbox import AnchoredText
from matplotlib.font_manager import FontProperties 
import matplotlib.cm as cm
from scipy.ndimage import zoom
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
    for datum in data:
        median = np.append(median,np.percentile(datum,50))
        sigmam = np.append(sigmam,np.percentile(datum,16))
        sigmam2 = np.append(sigmam2,np.percentile(datum,5))
        sigmap = np.append(sigmap,np.percentile(datum,84))
        sigmap2 = np.append(sigmap2,np.percentile(datum,95))
    if False:#name == 'llh':
        sigmap = zoom(sigmap,10)
        sigmam2 = zoom(sigmam2,10)
        sigmam = zoom(sigmam,10)
        median = zoom(median,10)
        sigmap2 = zoom(sigmap2,10)
        hypos = zoom(hypos,10)
    if name == 'llh':
        h0 = '%.2f'%(np.sqrt(median[0]))
        h0_up = '%.2f'%(np.sqrt(sigmap[0])-np.sqrt(median[0]))
        h0_down = '%.2f'%(np.sqrt(median[0])-np.sqrt(sigmam[0]))
        print 'Significance for excluding H0: %s + %s - %s'%(h0, h0_up, h0_down) 
    
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y','c','m','k']*2) +
                           cycler('linestyle', ['-']*7+['--']*7)))


    fig = plt.figure()
    fig.patch.set_facecolor('none')
    if name == 'llh':
        ax = plt.subplot2grid((6,1), (0,0), rowspan=5)
    else:
        ax = fig.add_subplot(111)
    if len(hypos):
        ax.fill_between(hypos,sigmam2,sigmap2,facecolor='b', linewidth=0.0, alpha=0.15, label='90%')
        ax.fill_between(hypos,sigmam,sigmap,facecolor='b', linewidth=0.0, alpha=0.3, label='68%')
        ax.plot(hypos,median,'k--', label='median',linewidth=2)
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
    ax.legend(loc='upper right',ncol=1, frameon=False,numpoints=1,fontsize=9)
    ax.set_xlabel(r'$\nu_{\tau}$ CC Normalization', size=14)
    ax.set_xlim(min(hypos),max(hypos))
    if name == 'llh':
        tex= r'no appearance: %s $\sigma ^{+%s}_{-%s}$'%(h0, h0_up, h0_down)
        print tex
        prop = dict(size=14)
        #a_text = AnchoredText(r'$\nu_\tau$ Appearance'+'\n%s years, %s trials\n'%(params['livetime']['value'],trials)+'Rejection of '+ tex, loc=9, frameon=False, prop=prop)
        #a_text = AnchoredText(r'$\nu_\tau$ Appearance'+'\n%s years\n'%(params['livetime']['value'])+'Rejection of '+ tex, loc=9, frameon=False,fontsize=12, prop=prop)
        #a_text = AnchoredText(r'IceCube/DeepCore $\nu_\tau$ Appearance'+'\n3 Year Sensitivity (Preliminary)' + '\nRejection of '+ tex, loc=9, frameon=True, prop=prop)
        a_text = AnchoredText(r'IceCube/DeepCore $\nu_\tau$ Appearance'+'\n3 Year Sensitivity (Preliminary)', loc=9, frameon=True, prop=prop)
    else:
        prop = dict(size=14)
        a_text = AnchoredText(r'$\nu_\tau$ Appearance'+'\n%s years, %s trials\n'%(params['livetime']['value'],trials)+'nuisnace pulls', loc=9, frameon=False, prop=prop)
        #a_text = AnchoredText(r'$\nu_\tau$ Appearance'+'\n%s years \n'%(params['livetime']['value'])+'nuisnace pulls', loc=9, frameon=False, prop=prop)
    ax.add_artist(a_text)
    #ax.patch.set_facecolor('white')
    #ax.set_axis_bgcolor('white') 
    #ax.set_frame_on(False)
    if name == 'llh':
        best_idx = np.argmin(median)
        best = hypos[best_idx]
        sig_90_per = 1.644854  # significance level at 90% range
        best_ms = np.interp(1,median[best_idx::-1],hypos[best_idx::-1])
        #best_m2s = np.interp(4,median[best_idx::-1],hypos[best_idx::-1])
        best_m2s = np.interp(sig_90_per**2,median[best_idx::-1],hypos[best_idx::-1])
        best_ps = np.interp(1,median[best_idx:],hypos[best_idx:])
        #best_p2s = np.interp(4,median[best_idx:],hypos[best_idx:])
        best_p2s = np.interp(sig_90_per**2,median[best_idx:],hypos[best_idx:])
        print best_m2s,best_ms,best,best_ps,best_p2s

        ax2 = plt.subplot2grid((6,1), (5,0),sharex=ax)
        superk_90_per=0.32*1.644854
        if args.option_1:
            ax2.errorbar(np.array([1.47]),np.array([1.]),xerr=np.array([[0.32],[0.32]]),fmt='.',color='forestgreen', elinewidth=2,capthick=2,markersize='12')
            eb2 = ax2.errorbar(np.array([1.47]),np.array([1.]),xerr=np.array([[superk_90_per],[superk_90_per]]),fmt='.',color='forestgreen', elinewidth=2,capthick=2,markersize='12')
            eb2[-1][0].set_linestyle('--')
            ax2.text(0.05,0.75,r'SuperK 2016' +' (68%, 90% extrapolated from 68%)',size=8)
        if args.option_2:
            ax2.errorbar(np.array([1.47]),np.array([1.]),xerr=np.array([[0.32],[0.32]]),fmt='.',color='forestgreen', elinewidth=2,capthick=2,markersize='12')
            ax2.text(0.05,0.75,r'SuperK 2016 (68%)',size=8)
        if args.option_3:
            eb2 = ax2.errorbar(np.array([1.47]),np.array([1.]),xerr=np.array([[superk_90_per],[superk_90_per]]),fmt='.',color='forestgreen', elinewidth=2,capthick=2,markersize='12')
            ax2.text(0.05,0.75,r'SuperK 2016' +' (90%)',size=8)

        eb3=ax2.errorbar(np.array([1.8]),np.array([2]),xerr=np.array([[1.1],[1.8]]),fmt='.',color='sienna', elinewidth=2,capthick=2,markersize='12')
        if args.option_1 or args.option_2:
            eb3[-1][0].set_linestyle('--')
        ax2.text(0.05,1.75,r'OPERA 2015 (90%)',size=8)

        if args.option_1 or args.option_2:
            ax2.errorbar(np.array([best]),np.array([3]),xerr=np.array([[best-best_ms],[best_ps-best]]),fmt='.',color='mediumblue',elinewidth=2,capthick=2,markersize='12')
            eb4=ax2.errorbar(np.array([best]),np.array([3]),xerr=np.array([[best-best_m2s],[best_p2s-best]]),fmt='.',color='mediumblue',elinewidth=2,capthick=2,markersize='12')
            eb4[-1][0].set_linestyle('--')
            ax2.text(0.05,2.75,r'IceCube/DeepCore Expected'+'\n(68%, 90%)',size=8)
        if args.option_3:
            eb4=ax2.errorbar(np.array([best]),np.array([3]),xerr=np.array([[best-best_m2s],[best_p2s-best]]),fmt='.',color='mediumblue',elinewidth=2,capthick=2,markersize='12')
            ax2.text(0.05,2.75,r'IceCube/DeepCore'+'\n Expected(90%)',size=8)
        ax2.set_ylim(0,4)
        ax2.set_xlim(0,2)
        ax2.get_yaxis().set_visible(False)
        fig.subplots_adjust(hspace=0)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.set_xlabel(r'$\nu_{\tau}$ CC Normalization', size=14)
        for i in [0.5,1,1.5]:
            ax2.axvline(i,color='k', linestyle='-',alpha=0.2)
    if name == 'llh':
        ax.set_ylabel(r'$-2\Delta {\mathrm{LLH}}$',size=14)
        ax.set_ylim([0,30])
        for i in [1,4,9,16,25,36]:
            ax.axhline(i,color='k', linestyle='-',alpha=0.2)
            ax.text(2.02,i-0.2,r'$%i\ \sigma$'%np.sqrt(i))
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
        ax.axvline(i,color='k', linestyle='-',alpha=0.2)
        ax.set_axisbelow(True)
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
        #ax.set_title('profile likelihood, %s years, %s trials'%(params['livetime']['value'],trials))
        ax.set_title('profile likelihood, 3 years')    # use 3 years instead of 2.5 years for the ICHEP poster
        plt.show()
        plt.savefig('q%.1f_%s.png'%(hypo,name),transparent=True)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-d','--dir',metavar='dir',help='directory containg output json files', default='.') 
    parser.add_argument('--dist',action='store_true') 
    parser.add_argument('--option_1',action='store_true') 
    parser.add_argument('--option_2',action='store_true') 
    parser.add_argument('--option_3',action='store_true') 
    args = parser.parse_args()


    total = 0
    asimov_hypos = collections.OrderedDict()
    asimov_results = collections.OrderedDict()
    results = []
    hypos = []
    sys = None
    data = {}

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
                    if name == 'nufit_prior' or name =='no_prior' or name == 'nominal':
                        sys = asimov_results[name].keys()
                elif ts == 'profile':
                    data['hypos'] = trial['nutau_norm']
                    for key in trial['fit_results'][0]:
                        if key == 'llh':
                            val = trial['q']
                        else:
                            val = trial['fit_results'][0][key]
                        if isinstance(val, float) or isinstance(val, int):
                            val = [val]
                        if data.has_key(key):
                            [x.append(y) for x,y in zip(data[key],val)]
                        else:
                            data[key] = [[x] for x in val]
                    total += 1

    if args.dist:
        #params = file['template_settings']['params'].keys()
        syslist = file['template_settings']['params'].keys()
        #for s in syslist:
        for param in syslist:
            dist(results, asimov_results, hypos, asimov_hypos, param, total)
    else:
        for s in data.keys():
            print "s = ", s
            if s != 'hypos':
                plot(s,data[s], asimov_results,data['hypos'], asimov_hypos, params, total)


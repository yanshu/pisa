#! /usr/bin/env python
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
from scipy.stats import chi2
from scipy import optimize
from matplotlib.offsetbox import AnchoredText

from pisa.utils.jsons import from_json


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-d','--dir',metavar='dir',help='directory containg output json files', default='.') 
    args = parser.parse_args()

    x_range = 10

    all_results = {}
    q00s = np.array([])
    q01s = np.array([])
    total=0

    #for n in xrange(1000,5000):
    for filename in os.listdir(args.dir):
        #filename = args.dir+'%s_q00.json'%n
        #if os.path.isfile(filename):
        if filename.endswith('.json'):
            file = from_json(args.dir +'/'+filename)
            params = file['template_settings']['params']
            livetime = params['livetime']['value']
            for trial in file['trials']:
                mu = trial['mu_data']
                if mu > 0.3:
                    total +=1
                    q00s = np.append(q00s,trial['q'])
        #filename = args.dir+'%s_q01.json'%n
        #if os.path.isfile(filename):
        #    file = from_json(filename)
        #    q01s = np.append(q01s,float(file['trials'][0]['q']))


    bins = np.arange(0.0,x_range,0.1)
    fig = plt.figure()
    fig.patch.set_facecolor('none')
    ax = fig.add_subplot(111)
    histo, bins, _ = ax.hist(q00s, bins=bins, normed=1, linewidth=0, alpha=0.5)
    #histo2, bins, _ = ax.hist(q01s, bins=bins, normed=1, color='r')
    bin_centers = 0.5*(bins[1:]+bins[:-1])

    def chi(x, dof):
        return chi2.pdf(x, dof)

    popt, pcov = optimize.curve_fit(chi , bin_centers[1:], histo[1:])
    
    print 'chi2 dof = ', popt[0], '+/-', np.sqrt(np.diag(pcov))[0]
    x_fit = np.arange(0,x_range,0.1)
    y_fit = chi(x_fit, popt[0])
    y_1d = chi(x_fit, 1.0)
    ax.plot(x_fit, y_fit, color='r')
    #ax.plot(x_fit, y_1d, color='g')
    a_text = AnchoredText('x2 d.o.f. = %.4f +/- %.4f'%(popt[0],np.sqrt(np.diag(pcov))[0]), loc=3, frameon=False)
    ax.add_artist(a_text)
    a_text2 = AnchoredText(r'$\nu_\tau$ appearance'+'\n%s years, %s trials\nPreliminary'%(livetime,total), loc=2, frameon=False)
    ax.add_artist(a_text2)

    #ax.set_ylabel(key)
    ax.set_xlabel('q')
    ax.set_yscale("log", nonposy='clip')
    plt.grid(True)
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('-')
        line.set_alpha(0.2)
    plt.show()
    plt.savefig('q0.pdf',edgecolor='none',facecolor=fig.get_facecolor())
    plt.savefig('q0.png',edgecolor='none',facecolor=fig.get_facecolor())

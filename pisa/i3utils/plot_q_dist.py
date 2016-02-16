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

    x_range = 9

    all_results = {}
    q00s = np.array([])
    q01s = np.array([])

    #for n in xrange(1000,5000):
    for filename in os.listdir(args.dir):
        #filename = args.dir+'%s_q00.json'%n
        #if os.path.isfile(filename):
        if filename.endswith('q00.json'):
            file = from_json(filename)
            q00s = np.append(q00s,float(file['trials'][0]['q']))
        #filename = args.dir+'%s_q01.json'%n
        #if os.path.isfile(filename):
        #    file = from_json(filename)
        #    q01s = np.append(q01s,float(file['trials'][0]['q']))


    bins = np.arange(0.1,x_range,0.1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    histo, bins, _ = ax.hist(q00s, bins=bins, normed=1)
    #histo2, bins, _ = ax.hist(q01s, bins=bins, normed=1, color='r')
    bin_centers = 0.5*(bins[1:]+bins[:-1])

    def chi(x, dof):
        return chi2.pdf(x, dof)

    popt, pcov = optimize.curve_fit(chi , bin_centers, histo)
    
    print 'chi2 dof = ', popt[0], '+/-', np.sqrt(np.diag(pcov))[0]
    x_fit = np.arange(0,x_range,0.1)
    y_fit = chi(x_fit, popt[0])
    ax.plot(x_fit, y_fit, color='r')
    a_text = AnchoredText('x2 d.o.f. = %.4f +/- %.4f'%(popt[0],np.sqrt(np.diag(pcov))[0]), loc=2)
    ax.add_artist(a_text)

    #ax.set_ylabel(key)
    ax.set_xlabel('q')
    ax.set_yscale("log", nonposy='clip')
    plt.grid(True)
    plt.show()
    plt.savefig('q00.png')

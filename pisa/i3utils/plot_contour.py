#! /usr/bin/env python
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os, sys
from scipy.stats import chi2
from scipy import optimize
from scipy.interpolate import griddata
from matplotlib.offsetbox import AnchoredText
from matplotlib import colors, ticker, cm
from pisa.utils.jsons import from_json
from pisa.utils.params_MH import select_hierarchy


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-d','--dir',metavar='dir',help='directory containg output json files', default='.') 
    parser.add_argument('--dist',action='store_true') 
    parser.add_argument('-x',help='outer loop variable', default='') 
    parser.add_argument('-y',help='inner loop variable', default='') 
    args = parser.parse_args()

    x = []
    y = []
    q = []
    livetime = 0

    # get llh denominators for q for each seed
    for filename in os.listdir(args.dir):
        if filename.endswith('.json'):
            file = from_json(args.dir +'/'+filename)
            x.append(file['trials'][0][args.x][0])
            y = file['trials'][0][args.y]
            q.append(file['trials'][0]['q'])
            livetime = file['template_settings']['params']['livetime']['value']



    q = [z for (k,z) in sorted(zip(x,q))]
    x.sort()
    q = np.array(q)
    #print x, y ,q
    x = np.sin(x)
    x = np.square(x)

    levels = [2.3,4.61,5.99,9.21]
    flevels = 2*np.logspace(-2,2,100)
    fmt = {2.3:'68%',4.61:'90%',5.99:'95%',9.21:'99%'}

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    CS = ax1.contour(x,y,q.T, levels=levels, linewidth=2, colors=['r','orchid','blue','green'])
    #CF = plt.contourf(x,y,q.T, cmap=cm.Set3,levels=flevels)
    ax1.clabel(CS, inline=1, fontsize=10, fmt=fmt)
    ax1.set_xlabel(r'$\sin^2(\theta_{23})$')
    ax1.set_ylabel(r'$\Delta m_{31}^2\ \rm{(eV^2)}$')
    a_text = AnchoredText(r'$\nu_\tau$ appearance'+'\n%s years\nPreliminary'%livetime, loc=2, frameon=False)
    ax1.add_artist(a_text)
    ax1.grid()
    ax1.set_xlim(0.3,0.7)
    ax1.set_ylim(0.0021,0.0029)
    plt.show()
    plt.savefig('contour.pdf')

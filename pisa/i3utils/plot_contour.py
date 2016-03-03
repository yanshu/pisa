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

    # get llh denominators for q for each seed
    for filename in os.listdir(args.dir):
        if filename.endswith('.json'):
            file = from_json(args.dir +'/'+filename)
            x.append(file['trials'][0][args.x][0])
            y = file['trials'][0][args.y]
            q.append(file['trials'][0]['q'])



    q = [z for (k,z) in sorted(zip(x,q))]
    x.sort()
    q = np.array(q)
    print x, y ,q

    levels = [2.3,4.61,5.99,9.21]
    fmt = {2.3:'68%',4.61:'90%',5.99:'95%',9.21:'99%'}

    fig = plt.figure()
    ax = fig.add_subplot(111)
    CS = ax.contour(x,y,q.T, levels)
    ax.clabel(CS, inline=1, fontsize=10, fmt=fmt)
    ax.set_xlabel(args.x)
    ax.set_ylabel(args.y)
    plt.show()
    plt.savefig('test.png')

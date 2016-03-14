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


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-d','--dir',metavar='dir',help='directory containg output json files', default='.') 
    args = parser.parse_args()

    all= {}
    # get llh denominators for q for each seed
    for filename in os.listdir(args.dir):
        if filename.endswith('.json'):
            file = from_json(args.dir +'/'+filename)
            name,_ = filename.split('.') 
            if name == 'nufit_prior' or name == 'no_prior':
                name = 'nominal'
            else:
                name = name.replace('_nufit_prior','').replace('_no_prior','')
            all[name] = file['trials'][0]['q'][0]
    print '''{| class="wikitable"
!colspan="3"|Systematics Impact (N-1)
|-
|systematic fixed||significance||impact on significance
|-'''

    for name in all.keys():
        print '|-'
        print '| %s || %.2f || %.2f %%'%(name, np.sqrt(all[name]), (np.sqrt(all[name])-np.sqrt(all['nominal']))/np.sqrt(all[name])*100)
    print '|}'

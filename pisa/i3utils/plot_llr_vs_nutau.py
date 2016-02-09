#! /usr/bin/env python
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.utils.jsons import from_json


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-d','--dir',metavar='dir',help='directory containg output json files', default='.') 
    args = parser.parse_args()

    norms = np.arange(0,2.1,0.1)

    all_results = {}

    for norm in norms:
        filename = args.dir+'%.1f.json'%norm
        file = from_json(filename)
        if norm == 0.0:
            all_results = file['results']['data_tau']['hypo_notau']
            params = file['template_settings_up']['params']
            keys = all_results.keys()
            all_results['nutau_norm'] = norms
        else:
            result = file['results']['data_tau']['hypo_notau']
            for key in keys:
                all_results[key] = np.append(all_results[key],result[key][0])

    for key in keys:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(norms,all_results[key])
        if params.has_key(key):
            params_range = params[key]['range']
            params_value = params[key]['value']
            ax.axhline(params_value, color='r')
            if params[key]['prior'].has_key('sigma'):
                params_sigma = params[key]['prior']['sigma']
                ymin = params_value - params_sigma
                ymax = params_value + params_sigma
            else:
                ymin = params_range[0]
                ymax = params_range[1]
            ax.axhline(ymin, color='r', linestyle='--')
            ax.axhline(ymax, color='r', linestyle='--')
            ax.set_ylim(ymin-0.2*(params_value-ymin), ymax+0.2*(ymax-params_value))
        ax.set_ylabel(key)
        ax.set_xlabel('nutau norm')
        plt.grid(True)
        plt.show()
        plt.savefig(key+'.png')

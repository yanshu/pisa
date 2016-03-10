#! /usr/bin/env python
from pisa.i3utils.plot_template import plotter
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pisa.utils.log import set_verbosity,logging,profile
from pisa.analysis.TemplateMaker_nutau import TemplateMaker
from pisa.analysis.stats.Maps_nutau import get_true_template, get_burn_sample
from pisa.utils.jsons import from_json
from pisa.utils.params import get_values, select_hierarchy_and_nutau_norm
import copy
set_verbosity(0)
parser = ArgumentParser(description='''Quick check if all components are working reasonably well, by
                                        making the final level hierarchy asymmetry plots from the input settings file. ''')
parser.add_argument('-t','--template_settings',metavar='JSON',
                    help='Settings file to use for template generation')
parser.add_argument('-no_logE','--no_logE',action='store_true',default=False,
                    help='Energy in log scale.')
parser.add_argument('-o','--outdir',metavar='DIR',default='',
                    help='Directory to save the output figures.')
parser.add_argument('-f', '--fit-results', default=None, dest='fit_file',
                    help='use post fit parameters from fit result json file (nutau_norm = 1)')
parser.add_argument('-s','--sys', default='')
parser.add_argument('-d','--delta',default=None)
args = parser.parse_args()
sys = args.sys

# get settings file for nutau norm = 1
template_settings = from_json(args.template_settings)
template_settings['params'] = select_hierarchy_and_nutau_norm(template_settings['params'],normal_hierarchy=True,nutau_norm_value=1.0)
if sys != 'atmos_mu_scale':
    template_settings['params']['atmos_mu_scale']['value'] = 0


# get binning info
anlys_ebins = template_settings['binning']['anlys_ebins']
czbins = template_settings['binning']['czbins']
livetime = template_settings['params']['livetime']['value']

# get template
template_maker = TemplateMaker(get_values(template_settings['params']),
                                    **template_settings['binning'])

templates = {}

templates['nominal'] = template_maker.get_template(get_values(template_settings['params']))


value = template_settings['params'][sys]['value']
if not args.delta:
    sigma = template_settings['params'][sys]['prior']['sigma']
    up = value + sigma
    down = value - sigma
    names = {'nominal':'nominal','up':sys+r'$+1\sigma$','down':sys+r'$-1\sigma$'}

else:
    delta = args.delta
    assert(delta.startswith('['))
    assert(delta.endswith(']'))
    down,up = eval(delta)
    names = {'nominal':'nominal','up':sys+' = %s'%up,'down':sys+' = %s'%down}

#up template
template_settings['params'][sys]['value'] = up
templates['up'] = template_maker.get_template(get_values(template_settings['params']))
# down template
template_settings['params'][sys]['value'] = down
templates['down'] = template_maker.get_template(get_values(template_settings['params']))

myPlotter = plotter(livetime,args.outdir,logy=False) 
myPlotter2 = plotter(livetime,args.outdir,logy=False,fmt='png') 

colors = {'nominal':'k','up':'red','down':'b'}

runs = ['nominal','up','down']

for axis, bins in [('energy',anlys_ebins),('coszen',czbins)]:
    for channel in ['cscd','trck']:
        plot_maps = []
        plot_sumw2 = []
        plot_colors=[]
        plot_names=[]
        plot_linestyles=[]
        for run in runs:
            plot_maps.append(templates[run][channel]['map'])  
            plot_sumw2.append(templates[run][channel]['sumw2'])
            plot_colors.append(colors[run])
            plot_linestyles.append('-')
            plot_names.append(names[run])

        myPlotter.plot_1d_projection(plot_maps,plot_sumw2,plot_colors,plot_names ,axis, bins, channel,outname=sys,linestyles=plot_linestyles)
        myPlotter2.plot_1d_projection(plot_maps,plot_sumw2,plot_colors,plot_names ,axis, bins, channel,outname=sys,linestyles=plot_linestyles)

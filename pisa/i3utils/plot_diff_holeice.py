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
parser.add_argument('-bs','--burn_sample_file',metavar='FILE',type=str,
                    default='',
                    help='''HDF5 File containing burn sample.'
                    inverted corridor cut data''')
args = parser.parse_args()

# get settings file for nutau norm = 1
template_settings = from_json(args.template_settings)
template_settings['params'] = select_hierarchy_and_nutau_norm(template_settings['params'],normal_hierarchy=True,nutau_norm_value=1.0)
template_settings['params']['atmos_mu_scale']['value'] = 0


# get binning info
anlys_ebins = template_settings['binning']['anlys_ebins']
czbins = template_settings['binning']['czbins']
livetime = template_settings['params']['livetime']['value']

# get template
mod_templates = {}
template_maker = TemplateMaker(get_values(template_settings['params']),
                                    **template_settings['binning'])
mod_templates[60] = template_maker.get_template(get_values(template_settings['params']))


mod_template_settings = copy.deepcopy(template_settings)
mod_template_settings['params']['hole_ice']['value'] = 0.02
mod_template_settings['params']['dom_eff']['value'] = 0.91

mod_templates[50] = template_maker.get_template(get_values(mod_template_settings['params']))
mod_template_settings['params']['hole_ice']['value'] = 0.0
mod_templates[70] = template_maker.get_template(get_values(mod_template_settings['params']))
mod_template_settings['params']['hole_ice']['value'] = 0.0333333
mod_templates[71] = template_maker.get_template(get_values(mod_template_settings['params']))
mod_template_settings['params']['hole_ice']['value'] = 0.01
mod_templates[72] = template_maker.get_template(get_values(mod_template_settings['params']))
mod_template_settings['params']['hole_ice']['value'] = 0.02
mod_template_settings['params']['dom_eff']['value'] = 0.95
mod_templates[61] = template_maker.get_template(get_values(mod_template_settings['params']))
mod_template_settings['params']['dom_eff']['value'] = 1.1
mod_templates[64] = template_maker.get_template(get_values(mod_template_settings['params']))
mod_template_settings['params']['dom_eff']['value'] = 1.05
mod_templates[65] = template_maker.get_template(get_values(mod_template_settings['params']))

templates = {}

# Get templates from 8 MC sets
runs = [50,60,61,64,65,70,71,72]
for run_num in runs:
    templates[str(run_num)] = {'trck':{},
                             'cscd':{}
                             }
    print "run_num = ", run_num
    aeff_mc_file = '~/pisa/pisa/resources/aeff/1X%i_aeff_mc.hdf5' % run_num
    pid_param_file_up = '~/pisa/pisa/resources/pid/1X%i_pid.json' % run_num
    reco_mc_file = '~/pisa/pisa/resources/events/1X%i_weighted_aeff_joined_nu_nubar.hdf5' % run_num
    pid_param_file_down = '~/pisa/pisa/resources/pid/1X%i_pid_down.json' % run_num
    DH_template_settings = copy.deepcopy(template_settings)
    DH_template_settings['params']['aeff_weight_file']['value'] = aeff_mc_file
    DH_template_settings['params']['reco_mc_wt_file']['value'] = reco_mc_file
    DH_template_settings['params']['pid_paramfile_up']['value'] = pid_param_file_up
    DH_template_settings['params']['pid_paramfile_down']['value'] = pid_param_file_down
    DH_template_settings['params']['atmos_mu_scale']['value'] = 0

    DH_template_maker = TemplateMaker(get_values(DH_template_settings['params']), **DH_template_settings['binning'])

    template = DH_template_maker.get_template(get_values(DH_template_settings['params']),no_sys_maps=True)

    templates[str(run_num)]['trck'] = template['trck']
    templates[str(run_num)]['cscd'] = template['cscd']



myPlotter = plotter(livetime,args.outdir,logy=False) 

colors = {50:'r', 60:'g', 70:'b', 71:'m' ,72:'c',61:'b',64:'m',65:'c'}
names = {50:'(0.91,50)',60:'(1.0,50)',70:'(0.91,no)',71:'(0.91,30)',72:'(0.91,100)',61:'(0.95,50)', 64:'(1.1,50)', 65:'(1.05,50)'}

runs = [50,70,71,72]
#runs = [50,60,61,64,65]

for axis, bins in [('energy',anlys_ebins),('coszen',czbins)]:
    for channel in ['cscd','trck']:
        plot_maps = []
        plot_sumw2 = []
        plot_colors=[]
        plot_names=[]
        plot_linestyles=[]
        for run in runs:
            plot_maps.append(templates[str(run)][channel]['map'])  
            plot_sumw2.append(templates[str(run)][channel]['sumw2'])
            plot_colors.append(colors[run])
            plot_linestyles.append('-')
            plot_names.append(names[run])
            plot_maps.append(mod_templates[run][channel]['map'])  
            plot_sumw2.append(mod_templates[run][channel]['sumw2'])
            plot_colors.append(colors[run])
            plot_names.append(names[run]+' calc')
            plot_linestyles.append('--')

        myPlotter.plot_1d_projection(plot_maps,plot_sumw2,plot_colors,plot_names ,axis, bins, channel,linestyles=plot_linestyles)
        #myPlotter.plot_1d_slices(plot_maps,plot_sumw2,plot_colors,plot_names ,axis, bins, channel)

myPlotter.outdir +='domeff'

runs = [50,60,61,64,65]
for axis, bins in [('energy',anlys_ebins),('coszen',czbins)]:
    for channel in ['cscd','trck']:
        plot_maps = []
        plot_sumw2 = []
        plot_colors=[]
        plot_names=[]
        plot_linestyles=[]
        for run in runs:
            plot_maps.append(templates[str(run)][channel]['map'])  
            plot_sumw2.append(templates[str(run)][channel]['sumw2'])
            plot_colors.append(colors[run])
            plot_linestyles.append('-')
            plot_names.append(names[run])
            plot_maps.append(mod_templates[run][channel]['map'])  
            plot_sumw2.append(mod_templates[run][channel]['sumw2'])
            plot_colors.append(colors[run])
            plot_names.append(names[run]+' calc')
            plot_linestyles.append('--')

        myPlotter.plot_1d_projection(plot_maps,plot_sumw2,plot_colors,plot_names ,axis, bins, channel,linestyles=plot_linestyles)

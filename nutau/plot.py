from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np

from pisa import ureg, Q_
from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.log import set_verbosity
from pisa.utils.plotter import plotter
from pisa.utils.fileio import from_file

parser = ArgumentParser()
parser.add_argument('-t', '--template-settings',
		    metavar='configfile', required=True,
		    action='append',
		    help='''settings for the template generation''')
parser.add_argument('-d', '--data-settings', type=str,
		    metavar='configfile', default=None,
		    help='settings for the generation of "data"')
parser.add_argument('-sp', '--set-param', type=str, default='',
                    help='Set a param to a certain value.')
parser.add_argument('-f', '--fit', type=str,
		    metavar='fit file', default=None,
		    help='settings for the generation of "data"')
parser.add_argument('-f0', '--fit0', type=str,
		    metavar='fit file for null hypo', default=None,
		    help='settings for the generation of "data"')
parser.add_argument('-v', action='count', default=None,
                    help='set verbosity level')
parser.add_argument('--livetime', type=float, default=None)
args = parser.parse_args()
set_verbosity(args.v)

template_configs = []
for template_setting in args.template_settings:
    template_config = from_file(template_setting)
    template_configs.append(template_config)

if args.data_settings is not None:
    data_settings = args.data_settings

    print "data_settings = ", data_settings
    data_configs = []
    if type(data_settings)==str:
        data_settings = [data_settings]
    for data_setting in data_settings:
        data_config = from_file(data_setting)
        data_configs.append(data_config)

if args.set_param != [''] and args.set_param!='':
    print "what "
    if type(args.set_param)==str:
        args.set_param = [args.set_param]
    for item in args.set_param:
        print "item = ", item
        p_name,value = item.split("=")
        p_name = str(p_name)
        print "template_setting = ", template_setting
        for template_config in template_configs:
            for section in template_config.sections():
                if template_config.has_option(section, 'param.%s'%p_name):
                    print "         change ", section , " ", p_name, " to ", value
                    template_config.set(section, 'param.%s'%p_name, str(value))

        if args.data_settings is not None:
            print "data_setting = ", data_settings
            for data_config in data_configs:
                for section in data_config.sections():
                    if data_config.has_option(section, 'param.%s'%p_name):
                        print "         change ", section , " ", p_name, " to ", value
                        data_config.set(section, 'param.%s'%p_name, str(value))
if args.data_settings is not None:
    data_maker = DistributionMaker(data_configs)
    #data = data_maker.get_outputs()
    #print "data = ", data
    data = data_maker.get_total_outputs()
    print "data = ", data
    data.set_poisson_errors()
    stamp=r'$\nu_\tau$ appearance'+'\nBurnsample comparison'
    if args.fit is None:
        stamp+='\nPrefit Distributions'
        out_name = 'prefit'
    else:
        stamp+='\nPostfit Distributions'
        out_name = 'postfit'
else:
    data = None
    stamp=r'$\nu_\tau$ appearance'+'\nMC'
print "\n"

template_maker = DistributionMaker(template_configs)


my_plotter = plotter(stamp=stamp, outdir='plots', fmt='pdf', log=False, annotate=True, symmetric=False, ratio=True)

#template_maker = DistributionMaker(args.template_settings)
#template_maker_H0 = DistributionMaker(args.template_settings)
template_maker = DistributionMaker(template_configs)
template_maker_H0 = DistributionMaker(template_configs)

nutau_cc_norm = template_maker_H0.params['nutau_cc_norm']
nutau_cc_norm.value = 0.0 * ureg.dimensionless
template_maker_H0.update_params(nutau_cc_norm) 

def update(template_maker, fitfile):
    params = from_file(fitfile)
    for key,val in params[0].items():
        if key in template_maker.params.names:
            p = template_maker.params[key]
            if len(val[1]) == 0:
                p.value = Q_(val[0])
            else:
                p.value = Q_(val[0],'%s^%s'%(val[1][0][0],val[1][0][1]))
            template_maker.update_params(p)


if args.fit is not None:
    update(template_maker, args.fit)

if args.fit0 is not None:
    update(template_maker_H0, args.fit0)

if args.livetime is not None:
    for tm in [template_maker, template_maker_H0]:
        livetime = tm.params['livetime']
        livetime.value = args.livetime * ureg.common_year
        tm.update_params(livetime) 

template_nominal = template_maker.get_outputs()[0]
print "template_nominal = ", template_nominal
for map in template_nominal: map.tex = 'MC'
template_notau = template_maker_H0.get_outputs()[0]
for map in template_notau: map.tex = r'MC\ (\nu_\tau^{CC}=0)'

#my_plotter.plot_1d_array(template_nominal, 'reco_coszen', fname='p_coszen')
#my_plotter.plot_1d_array(template_nominal, 'reco_energy', fname='p_energy')
#my_plotter.plot_2d_array(template_nominal, fname='nominal_spicehd_set',cmap='PiYG')
my_plotter.plot_2d_array(template_nominal, fname='background',cmap='PiYG')

#lower_level_variables = ['linefit_speed']
#mc_lowlevel_params = template_maker.pipelines[0].stages[0].get_device_arrays()
#icc_lowlevel_params = template_maker.pipelines[1].stages[0].get_fields(fields=lower_level_variables)

#file_name='test'
#bdt_cut = 0.2
#for param in lower_level_variables:
#    if param not in mc_lowlevel_params['nue_cc'].keys():
#        continue
#    my_plotter.plot_low_level_quantities(mc_lowlevel_params, icc_lowlevel_params, None, param, fig_name='%s'%(file_name), outdir='ll_data/',title='bdt>%s'%bdt_cut,logy=False)
#
out_name = './1d_plots/'
if args.data_settings is not None:
    my_plotter.plot_1d_slices_array(mapsets=[template_nominal,template_notau, data], plot_axis='reco_coszen', fname=out_name+'_s_coszen')
    my_plotter.plot_1d_slices_array(mapsets=[template_nominal,template_notau, data], plot_axis='reco_energy', fname=out_name+'_s_energy')
    my_plotter.plot_1d_cmp(mapsets=[template_nominal,template_notau, data], plot_axis='reco_energy', fname=out_name+'_p_energy')
    my_plotter.plot_1d_cmp(mapsets=[template_nominal,template_notau, data], plot_axis='reco_coszen', fname=out_name+'_p_coszen')

"""
function to produce fit results for sets of disctrete systematics
(i.e. for example several simulations for different DOM efficiencies)

the parameters and settings going into the fit are given by an external
cfg file (fit settings)

n-dimensional MapSets are supported to be fitted with m-dimesnional polynomials,
that can either be forced to go through the nominal data point or not
"""


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import copy
import itertools
from uncertainties import unumpy as unp

import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter, generic_filter
from scipy import interpolate
from scipy.stats import chi2

from pisa import ureg, Q_
from pisa.core.map import Map, MapSet
from pisa.core.pipeline import Pipeline
from pisa.utils.fileio import from_file, to_file
from pisa.utils.log import set_verbosity
from pisa.utils.config_parser import parse_pipeline_config, parse_quantity, parse_string_literal
from pisa.utils.plotter import Plotter


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', '--template-settings', type=str,
                        metavar='configfile', required=True,
                        help='settings for the generation of templates')
    parser.add_argument('-f', '--fit-settings', type=str,
                        metavar='configfile', required=True,
                        help='settings for the generation of templates')
    parser.add_argument('-sp', '--set-param', type=str, default='',
                        help='Set a param to a certain value.')
    parser.add_argument('--tag', type=str, default='deepcore',
                        help='Tag for the filename')
    parser.add_argument('-o', '--out-dir', type=str, default='pisa/resources/discr_sys',
                        help='Set output directory')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='plot')
    parser.add_argument('-v', action='count', default=None,
                        help='set verbosity level')
    args = parser.parse_args()
    set_verbosity(args.v)

    if args.plot:
        import matplotlib as mpl
        mpl.use('pdf')
        import matplotlib.pyplot as plt
        from pisa.utils.plotter import Plotter

    cfg = from_file(args.fit_settings)
    sys_list = cfg.get('general','sys_list').replace(' ','').split(',')
    stop_idx = cfg.getint('general','stop_after_stage')

    for sys in sys_list:
        # parse info for given systematic
        nominal = cfg.getfloat(sys,'nominal')
        degree = cfg.getint(sys, 'degree')
        force_through_nominal = cfg.getboolean(sys, 'force_through_nominal')
        runs = eval(cfg.get(sys,'runs'))
        smooth = cfg.get(sys, 'smooth')

        x_values = np.array(sorted(runs))

        # build fit function
        if force_through_nominal:
            function = "fit_fun = lambda x, *p: np.polynomial.polynomial.polyval(x, [1.] + list(p))"
        else:
            function = "fit_fun = lambda x, *p: np.polynomial.polynomial.polyval(x, list(p))"
            # add free parameter for constant term
            degree += 1
        exec(function)

        # instantiate template maker
        template_maker = Pipeline(args.template_settings)

        if not args.set_param == '':
            p_name,value = args.set_param.split("=")
            print "p_name,value= ", p_name, " ", value
            value = parse_quantity(value)
            value = value.n * value.units
            print "value ", value
            test = template_maker.params[p_name]
            print "test.value = ", test.value
            test.value = value
            print "test.value = ", test.value
            template_maker.update_params(test)

        inputs = {}
        map_names = None
        # get sys templates
        for run in runs:
            for key, val in cfg.items('%s:%s'%(sys, run)):
                if key.startswith('param.'):
                    _,pname = key.split('.')
                    param = template_maker.params[pname]
                    try:
                        value = parse_quantity(val)
                        param.value = value.n * value.units
                    except ValueError:
                        value = parse_string_literal(val)
                        param.value = value
                    param.set_nominal_to_current_value()
                    template_maker.update_params(param)
            # retreive maps
            template = template_maker.get_outputs(idx=stop_idx)
            if map_names is None: map_names = [map.name for map in template]
            inputs[run] = {}
            for map in template:
                inputs[run][map.name] = map.hist

        # numpy acrobatics:
        arrays = {}
        for name in map_names:
            arrays[name] = []
            for x in x_values:
                arrays[name].append(inputs[x][name]/unp.nominal_values(inputs[nominal][name]))
            a = np.array(arrays[name])
            arrays[name] = np.rollaxis(a,0,len(a.shape))

        # shift to get deltas
        x_values -= nominal

        # binning object (assuming they're all the same)
        binning = template.maps[0].binning

        shape = [d.num_bins for d in binning] + [degree]
        shape_small = [d.num_bins for d in binning]

        outputs = {}
        errors = {}
        for name in map_names:
            # now actualy perform some fits
            outputs[name] = np.ones(shape)
            errors[name] = np.ones(shape)

            for idx in np.ndindex(*shape_small):
            #for i, j in np.ndindex(inputs[name][nominal].shape):
                y_values = unp.nominal_values(arrays[name][idx])
                y_sigma = unp.std_devs(arrays[name][idx])
                popt, pcov = curve_fit(fit_fun, x_values,
                        y_values, sigma=y_sigma, p0=np.ones(degree))
                perr = np.sqrt(np.diag(pcov))
                for k, p in enumerate(popt):
                    outputs[name][idx][k] = p
                    errors[name][idx][k] = perr[k]

                # maybe plot
                if args.plot:
                    fig_num = i + nx * j
                    if fig_num == 0:
                        fig = plt.figure(num=1, figsize=( 4*nx, 4*ny))
                    subplot_idx = nx*(ny-1-j)+ i + 1
                    plt.subplot(ny, nx, subplot_idx)
                    #plt.snameter(x_values, y_values, color=plt_colors[name])
                    plt.gca().errorbar(x_values, y_values, yerr=y_sigma, fmt='o', color=plt_colors[name], ecolor=plt_colors[name], mec=plt_colors[name])
                    # plot nominal point again in black
                    plt.snameter([0.0], [1.0], color='k')
                    f_values = fit_fun(x_values, *popt)
                    fun_plot, = plt.plot(x_values, f_values,
                            color=plt_colors[name])
                    plt.ylim(np.min(unp.nominal_values(arrays[name]))*0.9, np.max(unp.nominal_values(arrays[name]))*1.1)
                    if i > 0:
                        plt.setp(plt.gca().get_yticklabels(), visible=False)
                    if j > 0:
                        plt.setp(plt.gca().get_xticklabels(), visible=False)

        if smooth == 'gauss':
            for name in map_names:
                for d in range(degree):
                    outputs[name][...,d] = gaussian_filter(outputs[name][...,d],sigma=1)

        # save the raw ones anyway
        outputs['pname'] = sys
        outputs['nominal'] = nominal
        outputs['function'] = function
        outputs['map_names'] = map_names
        outputs['binning_hash'] = binning.hash
        to_file(outputs, '%s/%s_sysfits_%s_%s.json'%(args.out_dir,sys, args.tag, smooth))

        for d in range(degree):
            maps = []
            for name in map_names:
                maps.append(Map(name='%s_raw'%name, hist=outputs[name][...,d],
                            binning=binning))
            maps = MapSet(maps)
            my_plotter = Plotter(stamp='PISA cake test', outdir='.',
                fmt='pdf',log=False, label='')
            my_plotter.plot_2d_array(maps, fname='%s_%s_%s_%s'%(sys,args.tag,d,smooth), split_axis='pid',cmap='RdBu')
            #for name in map_names:
            #    data = (raw_outputs[name][...,d] - outputs[name][...,d]).ravel()
            #    fig = plt.figure()
            #    ax = fig.add_subplot(111)
            #    h,b,p = ax.hist(data,20, linewidth=2, histtype='step', color='k',normed=True)
            #    ax.ticklabel_format(useOffset=False)
            #    #p = chi2.fit(data,floc=0, scale=1)
            #    #print p
            #    #x = np.linspace(b[0], b[-1], 100)
            #    #f = chi2.pdf(x, *p)
            #    #ax.plot(x,f, color='r')
            #    plt.savefig('diff_%s_%s.png'%(sys,name))

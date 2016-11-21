from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import copy
import itertools
import sys
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
from pisa.utils.plotter import plotter


def main():
    parser = ArgumentParser()
    parser.add_argument('-t', '--template-settings', type=str,
                        metavar='configfile', required=True,
                        help='settings for the generation of templates')
    parser.add_argument('-f', '--fit-settings', type=str,
                        metavar='configfile', required=True,
                        help='settings for the generation of templates')
    parser.add_argument('-sp', '--set-param', type=str, default='',
                        help='Set a param to a certain value.')
    parser.add_argument('-o', '--out-dir', type=str, default='pisa/resources/sys',
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
        from pisa.utils.plotter import plotter

    cfg = from_file(args.fit_settings)
    sys_list = cfg.get('general','sys_list').replace(' ','').split(',')
    categories = cfg.get('general','categories').replace(' ','').split(',')
    idx = cfg.getint('general','stop_after_stage')

    # setup plotting colors
    colors = itertools.cycle(["r", "b", "g"])
    plt_colors = {}
    for cat in categories:
        plt_colors[cat] = next(colors)

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
        template_maker_settings = from_file(args.template_settings)
        template_maker_configurator = parse_config(template_maker_settings)
        template_maker = Pipeline(template_maker_configurator)

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
        for cat in categories:
            inputs[cat] = {}

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
            template = template_maker.get_outputs(idx=idx)
            for cat in categories:
                inputs[cat][run] = sum([map.hist for map in template if
                    map.name.endswith(cat)])

        # binning object
        binning = template.maps[0].binning

        # numpy acrobatics:
        arrays = {}
        for cat in categories:
            arrays[cat] = []
            for x in x_values:
                arrays[cat].append(inputs[cat][x]/unp.nominal_values(inputs[cat][nominal]))
            arrays[cat] = np.array(arrays[cat]).transpose(1,2,0)

        nx, ny = inputs[categories[0]][nominal].shape
        bins_x = np.arange(nx)
        bins_y = np.arange(ny)

        grid_x, grid_y = np.meshgrid(bins_x, bins_y)
        grid_x = np.ravel(grid_x)
        grid_y = np.ravel(grid_y)

        # shift to get deltas
        x_values -= nominal

        # array to store params
        outputs = {}
        errors = {}

        # now actualy perform some fits
        for cat in categories:
            outputs[cat] = np.ones((nx, ny, degree))
            errors[cat] = np.ones((nx, ny, degree))

            for i, j in np.ndindex((nx,ny)):
            #for i, j in np.ndindex(inputs[cat][nominal].shape):
                y_values = unp.nominal_values(arrays[cat][i,j,:])
                y_sigma = unp.std_devs(arrays[cat][i,j,:])
                popt, pcov = curve_fit(fit_fun, x_values,
                        y_values, sigma=y_sigma, p0=np.ones(degree))
                perr = np.sqrt(np.diag(pcov))
                for k, p in enumerate(popt):
                    outputs[cat][i,j,k] = p
                    errors[cat][i,j,k] = perr[k]

                # maybe plot
                if args.plot:
                    fig_num = i + nx * j
                    if fig_num == 0:
                        fig = plt.figure(num=1, figsize=( 4*nx, 4*ny))
                    subplot_idx = nx*(ny-1-j)+ i + 1
                    plt.subplot(ny, nx, subplot_idx)
                    #plt.scatter(x_values, y_values, color=plt_colors[cat])
                    plt.gca().errorbar(x_values, y_values, yerr=y_sigma, fmt='o', color=plt_colors[cat], ecolor=plt_colors[cat], mec=plt_colors[cat])
                    # plot nominal point again in black
                    plt.scatter([0.0], [1.0], color='k')
                    f_values = fit_fun(x_values, *popt)
                    fun_plot, = plt.plot(x_values, f_values,
                            color=plt_colors[cat])
                    plt.ylim(np.min(unp.nominal_values(arrays[cat]))*0.9, np.max(unp.nominal_values(arrays[cat]))*1.1)
                    if i > 0:
                        plt.setp(plt.gca().get_yticklabels(), visible=False)
                    if j > 0:
                        plt.setp(plt.gca().get_xticklabels(), visible=False)

        # save the raw ones anyway
        outputs['pname'] = sys
        outputs['nominal'] = nominal
        outputs['function'] = function
        outputs['categories'] = categories
        #outputs['binning'] = binning
        to_file(outputs, '%s/%s_sysfits_raw.json'%(args.out_dir,sys))

        # smoothing
        if not smooth == 'raw':
            raw_outputs = copy.deepcopy(outputs)
            for cat in categories:
                for d in range(degree):
                    if smooth == 'spline':
                        spline = interpolate.SmoothBivariateSpline(grid_x, grid_y,
                                np.ravel(outputs[cat][:,:,d]), kx=2, ky=2)
                        outputs[cat][:,:,d] = spline(bins_x, bins_y)
                    elif smooth == 'gauss':
                        outputs[cat][:,:,d] = gaussian_filter(outputs[cat][:,:,d],
                                sigma=1)
                    elif smooth == 'smart':
                        values = outputs[cat][:,:,d]
                        sigmas = errors[cat][:,:,d]
                        for (x,y), sig in np.ndenumerate(sigmas):
                            n = 0.
                            o = 0.
                            val = values[x,y]
                            nx, ny = sigmas.shape
                            width = 1.
                            f = 8.
                            for dx in [-2,-1,0,1,2]:
                                for dy in [-2,-1,0,1,2]:
                                    if not (x+dx < 0 or x+dx >= nx) and not (y+dy < 0 or y+dy >= ny):
                                        v = values[x+dx,y+dy]
                                        s = sigmas[x+dx,y+dy]
                                        #dist = np.sqrt(dx**2 + dy**2)
                                        dist = np.exp(-(dx**2+dy**2)/(2.*width**2))/(2.*np.pi*width**2)
                                        dist_v = np.exp(-(val-v)**2/(2*(sig*f)**2))/(np.sqrt(2*np.pi)*sig*f)
                                        w = dist*dist_v*(sig/s)
                                        o += w*v
                                        n += w
                            outputs[cat][x,y,d] = o/n

                if args.plot:
                    for i, j in np.ndindex((nx,ny)):
                        fig_num = i + nx * j
                        if fig_num == 0:
                            fig = plt.figure(num=1, figsize=( 4*nx, 4*ny))
                        subplot_idx = nx*(ny-1-j)+ i + 1
                        plt.subplot(ny, nx, subplot_idx)
                        p_smooth = outputs[cat][i,j,:]
                        f_values = fit_fun(x_values, *p_smooth)
                        fun_plot, = plt.plot(x_values, f_values,
                                color=plt_colors[cat], linestyle='--')

            outputs['pname'] = sys
            outputs['nominal'] = nominal
            outputs['function'] = function
            outputs['categories'] = categories
            #outputs['binning'] = binning
            to_file(outputs, '%s/%s_sysfits_%s.json'%(args.out_dir,sys,smooth))

        if args.plot:
            fig.subplots_adjust(hspace=0)
            fig.subplots_adjust(wspace=0)
            plt.show()
            plt.savefig('%s_sysfits_%s.pdf'%(sys,smooth))
            plt.clf()

            if not smooth == 'raw':
                for d in range(degree):
                    maps = []
                    for cat in categories:
                        maps.append(Map(name='%s_raw'%cat, hist=raw_outputs[cat][:,:,d],
                                    binning=binning))
                        maps.append(Map(name='%s_smooth'%cat, hist=outputs[cat][:,:,d],
                                    binning=binning))
                        maps.append(Map(name='%s_dfff'%cat,
                                    hist=(raw_outputs[cat][:,:,d] - outputs[cat][:,:,d]),
                                    binning=binning))
                        maps.append(Map(name='%s_chi2'%cat,
                                    hist=np.square(raw_outputs[cat][:,:,d] - outputs[cat][:,:,d])/np.square(errors[cat][:,:,d]),
                                    binning=binning))
                    maps = MapSet(maps)
                    my_plotter = plotter(stamp='PISA cake test', outdir='.',
                        fmt='pdf',log=False, label='')
                    my_plotter.plot_2d_array(maps, fname='%s_smooth_%s_p%s'%(sys,smooth,d),cmap='RdBu')
                    for cat in categories:
                        data = (raw_outputs[cat][:,:,d] - outputs[cat][:,:,d]).ravel()
                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        h,b,p = ax.hist(data,20, linewidth=2, histtype='step', color='k',normed=True)
                        ax.ticklabel_format(useOffset=False)
                        #p = chi2.fit(data,floc=0, scale=1)
                        #print p
                        #x = np.linspace(b[0], b[-1], 100)
                        #f = chi2.pdf(x, *p)
                        #ax.plot(x,f, color='r')
                        plt.savefig('diff_%s_%s.png'%(sys,cat))


if __name__ == '__main__':
    main()

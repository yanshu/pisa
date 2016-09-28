#! /usr/bin/env python
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib.offsetbox import AnchoredText
from pisa.utils.utils import get_bin_centers, get_bin_sizes
from pisa.utils.plot import show_map, plot_one_map
from pisa.utils.plot import show_map_swap, sum_map, ratio_map, delta_map
from pisa.analysis.stats.LLHStatistics_nutau import get_binwise_llh, get_binwise_smeared_llh, get_barlow_llh, get_chi2
import pisa.utils.utils as utils

class plotter(object):

    def __init__(self,livetime, outdir, logy=True, fmt='pdf'):
        self.logy= logy
        self.outdir = outdir
        self.livetime=livetime
        self.fmt = fmt
        self.channel = None

    def get_1D_projection(self,map_2d, axis):
        if axis == 'coszen':
            output_array = np.zeros(map_2d.shape[1])
            for i in range(0, map_2d.shape[0]):
                output_array += map_2d[i,:]
        if axis == 'energy':
            output_array = np.zeros(map_2d.shape[0])
            for i in range(0, map_2d.shape[1]):
                output_array += map_2d[:,i]
        return output_array

    def get_1D_average(self,map_2d, axis, bins):
        weights = bins[1:] - bins[:-1]
        total = bins[-1]-bins[0]
        if axis == 'coszen':
            output_array = np.zeros(map_2d.shape[1])
            for i in range(0, map_2d.shape[0]):
                output_array += weights[i]*map_2d[i,:]
        if axis == 'energy':
            output_array = np.zeros(map_2d.shape[0])
            for i in range(0, map_2d.shape[1]):
                output_array += weights[i]*map_2d[:,i]
        return output_array/total

    def plot_1d(self,maps, errors, colors, names, axis, x_bin_edges, outname,linestyles=None, ratio=True, yaxis_label='# entries'):
        if not linestyles:
            linestyles=['-']*len(maps)
        x_bin_centers = get_bin_centers(x_bin_edges)
        x_bin_width = get_bin_sizes(x_bin_edges)
        fig = plt.figure(figsize=(8,8))
        fig.patch.set_facecolor('none')
        if len(maps) == 1 or not ratio:
            ax1 = fig.add_subplot(111)
        else:
            ax1 = plt.subplot2grid((4,1), (0,0), rowspan=3)
        for map, error, color, name, linestyle in zip(maps,errors,colors,names,linestyles):
            if name == 'data':
                ax1.errorbar(x_bin_centers,map,yerr=error,fmt='o',color='black', markersize='4',label=name)
            else:
                #print map
                hist,_,_ = ax1.hist(x_bin_centers,weights= map,bins=x_bin_edges,histtype='step',lw=1.5,color=color,linestyle=linestyle, label=name)
                if error is not None:
                    ax1.bar(x_bin_edges[:-1],2*error, bottom=map-error, width=x_bin_width, color=color, alpha=0.25, linewidth=0)
        ax1.grid()
        gridlines = ax1.get_xgridlines() + ax1.get_ygridlines()
        for line in gridlines:
            line.set_linestyle('-')
            line.set_alpha(0.2)
        ax1.set_ylabel(yaxis_label)
        minimum = min([min(map) for map in maps])
        maximum = max([max(map) for map in maps])
        if self.logy:
            if minimum == 0:
                minimum = 1
            ax1.set_yscale('log')
            #ax1.set_ylim(np.power(10,0.8*(np.log10(minimum))),np.power(10,0.8**(np.log10(maximum))))
            maximum = np.power(10,0.5+np.log10(maximum))
            ax1.set_ylim(minimum, maximum)
        else:
            ax1.set_ylim(minimum, 1.5*maximum)
        ax1.legend(loc='upper right',ncol=2, frameon=False,numpoints=1,fontsize=10)
        text = r'$\nu_\tau$ appearance'
        if self.livetime:
            text += '\n%s years'%(self.livetime)
        if self.channel:
            text += ',%s'%(self.channel)
        text += '\nExpected'
        a_text = AnchoredText(text, loc=2, frameon=False)
        ax1.add_artist(a_text)
        if axis == 'energy':
            ax1.set_xlabel('Energy (GeV)')
            ax1.set_xscale('log')
            ax1.set_xlim(x_bin_edges[0],x_bin_edges[-1])
        if axis == 'coszen':
            ax1.set_xlabel('cos(zen)')

        if len(maps) >1 and ratio:
            ax2 = plt.subplot2grid((4,1), (3,0),sharex=ax1)
            minimum = 10
            maximum = 0
            for map, error, color,linestyle in zip(maps, errors, colors,linestyles):
                ratio = np.zeros_like(map)
                ratio_error = np.zeros_like(map)
                for i in range(len(map)):
                    if map[i]==0 and maps[0][i]==0:
                        ratio[i] = 1
                        ratio_error[i] = 1
                    elif map[i]!=0 and maps[0][i]==0:
                        print " non zero divided by 0 !!!"
                    else:
                        ratio[i] = map[i]/maps[0][i]
                        ratio_error[i] = error[i]/maps[0][i]
                        minimum = min(minimum,ratio[i])
                        maximum = max(maximum,ratio[i])
                hist,_,_ = ax2.hist(x_bin_centers,weights= ratio,bins=x_bin_edges,histtype='step',lw=1,color=color,linestyle=linestyle)
                ax2.bar(x_bin_edges[:-1],2*ratio_error, bottom=ratio-ratio_error, width=x_bin_width, color=color, alpha=0.25, linewidth=0)
            
            if axis == 'energy':
                ax2.set_xlabel('Energy (GeV)')
                ax2.set_xscale('log')
                ax2.set_xlim(x_bin_edges[0],x_bin_edges[-1])
            if axis == 'coszen':
                ax2.set_xlabel('cos(zen)')
            ax2.grid()
            gridlines = ax2.get_xgridlines() + ax2.get_ygridlines()
            for line in gridlines:
                line.set_linestyle('-')
                line.set_alpha(0.2)
            minimum = 1+ 2.*(minimum-1)
            maximum = 1+ 2.*(maximum-1)
            if maximum == 1:
                maximum += 0.5*(maximum-minimum)
            minimum = min(minimum,0.98)
            minimum = max(minimum,0)
            maximum = max(maximum,1.02)
            ax2.set_ylim(minimum, maximum)
            #ax2.set_ylim(0.8,1.2)
            ax2.set_ylabel('ratio over %s'%names[0])
            fig.subplots_adjust(hspace=0)
            plt.setp(ax1.get_xticklabels(), visible=False)

        plt.savefig(self.outdir+'/'+outname+'.'+self.fmt, dpi=150, edgecolor='none',facecolor=fig.get_facecolor())
        plt.clf()
        plt.close(fig)


    def plot_1d_projection(self,maps, errors, colors, names, axis, x_bin_edges, channel,outname='',linestyles=None):
        # iterate
        outname = '%s%s_%s'%(outname,channel, axis)
        self.channel = channel
        pmaps = []
        perrors = []
        for map, error in zip(maps,errors):
            pmaps.append(self.get_1D_projection(map,axis))
            perrors.append(np.sqrt(self.get_1D_projection(error,axis)))
        self.plot_1d(pmaps, perrors, colors, names, axis, x_bin_edges, outname,linestyles=linestyles)

    def plot_map(self,map,names,outname=''):
        fig = plt.figure(figsize=(8,8))
        fig.patch.set_facecolor('none')

        show_map(map,logE=True,annotate_no_evts=False)   

        plt.savefig(self.outdir+'/'+outname+'.'+self.fmt, dpi=150, edgecolor='none',facecolor=fig.get_facecolor())
        plt.clf()
        plt.close(fig)
        

    def plot_1d_slices(self,maps, errors, colors, names, axis, x_bin_edges, channel):
        self.channel = channel
        # iterate
        if axis == 'coszen':
                idx = range(0, maps[0].shape[0])
        if axis == 'energy':
                idx = range(0, maps[0].shape[1])
        for i in idx:
            pmaps = []
            perrors = []
            outname = '%s_%s_%s'%(channel, axis, i)
            for map, error, color, name in zip(maps,errors,colors,names):
                if axis == 'coszen':
                    pmaps.append(map[i,:])
                    perrors.append(np.sqrt(error[i,:]))
                else:
                    pmaps.append(map[:,i])
                    perrors.append(np.sqrt(error[:,i]))
            self.plot_1d(pmaps, perrors, colors, names, axis, x_bin_edges, outname)

def savetxt(tmpl, czbins, ebins, name):
    cz_bin_centers = get_bin_centers(czbins)
    e_bin_centers = get_bin_centers(anlys_ebins)
    print "e_bin_centers = ", e_bin_centers
    print "cz_bin_centers = ", cz_bin_centers
    cz_centers = []
    e_centers = []
    z_values = []
    for i in range(0,len(e_bin_centers)):
        for j in range(0,len(cz_bin_centers)):
            cz_centers.append(cz_bin_centers[j])
            e_centers.append(e_bin_centers[i])
            z_values.append(tmpl[i,j])
    output = np.column_stack((e_centers,cz_centers,z_values))
    np.savetxt('%s.csv'%name, output, delimiter=',')

if __name__ == '__main__':
    from pisa.utils.log import set_verbosity,logging,profile
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    #from pisa.analysis.TemplateMaker_nutau import TemplateMaker
    from pisa.analysis.TemplateMaker_MC import TemplateMaker 
    from pisa.analysis.stats.Maps_nutau import get_true_template, get_burn_sample_maps
    from pisa.utils.jsons import from_json
    from pisa.utils.params import get_values, select_hierarchy_and_nutau_norm
    set_verbosity(0)
    parser = ArgumentParser(description='''Quick check if all components are working reasonably well, by
                                            making the final level hierarchy asymmetry plots from the input settings file. ''')
    parser.add_argument('-t','--template_settings',metavar='JSON',
                        help='Settings file to use for template generation')
    parser.add_argument('-t2','--template_settings_2',metavar='JSON',
                        help='Settings file to use for template generation, the second template')
    parser.add_argument('--title',metavar='str',default='IC86',
                        help='Title of figure.')
    parser.add_argument('-no_logE','--no_logE',action='store_true',default=False,
                        help='Energy in log scale.')
    parser.add_argument('-o','--outdir',metavar='DIR',default='.',
                        help='Directory to save the output figures.')
    parser.add_argument('-f', '--fit-results', default=None, dest='fit_file',
                        help='use post fit parameters from fit result json file (nutau_norm = 1)')
    parser.add_argument('-bs','--burn_sample_file',metavar='FILE',type=str,
                        default='',
                        help='''HDF5 File containing burn sample.'
                        inverted corridor cut data''')
    args = parser.parse_args()

    utils.mkdir(args.outdir)
    utils.mkdir(args.outdir+'/pdf')

    # get settings file for nutau norm = 1
    template_settings = from_json(args.template_settings)
    print "type args.template_settings = ", type(args.template_settings)
    if args.template_settings_2:
        template_settings_2 = from_json(args.template_settings_2)

    if args.fit_file:
        # replace with parameters determ,ined in fit
        fit_file = from_json(args.fit_file)
        syslist = fit_file['trials'][0]['fit_results'][0].keys()
        for sys in syslist:
            if not sys == 'llh':
                val = fit_file['trials'][0]['fit_results'][0][sys][0]
                if sys == 'theta23' or sys =='deltam23' or sys =='deltam31':
                    sys += '_nh'
                print '%s at %.4f'%(sys,val)
                template_settings['params'][sys]['value'] = val

    # get binning info
    anlys_ebins = template_settings['binning']['anlys_ebins']
    print "anlys_ebins = ", anlys_ebins
    czbins = template_settings['binning']['czbins']
    livetime = template_settings['params']['livetime']['value']

    if args.template_settings_2:
        assert(np.all(anlys_ebins == template_settings_2['binning']['anlys_ebins']))
        assert(np.all(czbins == template_settings_2['binning']['czbins']))
        assert(livetime==template_settings_2['params']['livetime']['value'])

    # get template
    new_template_settings = get_values(select_hierarchy_and_nutau_norm(template_settings['params'],normal_hierarchy=True,nutau_norm_value=1.0))
    template_maker = TemplateMaker(new_template_settings, **template_settings['binning'])
    true_template = template_maker.get_template(new_template_settings, num_data_events=None)
    #print true_template

    if args.template_settings_2:
        new_template_settings_2 = get_values(select_hierarchy_and_nutau_norm(template_settings_2['params'],normal_hierarchy=True,nutau_norm_value=1.0))
        template_maker_2 = TemplateMaker(new_template_settings_2, **template_settings_2['binning'])
        true_template_2 = template_maker_2.get_template(new_template_settings, num_data_events=None)

    true_template['tot'] = {}
    true_template['tot']['map'] = true_template['cscd']['map'] + true_template['trck']['map']
    true_template['tot']['map_nu'] = true_template['cscd']['map_nu'] + true_template['trck']['map_nu']
    true_template['tot']['map_mu'] = true_template['cscd']['map_mu'] + true_template['trck']['map_mu']
    true_template['tot']['sumw2'] = true_template['cscd']['sumw2'] + true_template['trck']['sumw2']
    true_template['tot']['sumw2_nu'] = true_template['cscd']['sumw2_nu'] + true_template['trck']['sumw2_nu']
    true_template['tot']['sumw2_mu'] = true_template['cscd']['sumw2_mu'] + true_template['trck']['sumw2_mu']

    if args.burn_sample_file:
        burn_sample_maps = get_burn_sample_maps(burn_sample_file= args.burn_sample_file, anlys_ebins= anlys_ebins, czbins= czbins, output_form ='map', cut_level='L6', channel=template_settings['params']['channel']['value'], pid_remove=template_settings['params']['pid_remove']['value'], pid_bound=template_settings['params']['pid_bound']['value'], sim_version=template_settings['params']['sim_ver']['value'])
        burn_sample_maps['tot'] = {}
        burn_sample_maps['tot']['map'] = burn_sample_maps['cscd']['map'] + burn_sample_maps['trck']['map']

    myPlotter = plotter(livetime,args.outdir,logy=False) 

    for axis, bins in [('energy',anlys_ebins),('coszen',czbins)]:
        for channel in ['cscd','trck','tot']:
            if args.burn_sample_file:
                plot_maps = [burn_sample_maps[channel]['map']] 
                plot_sumw2 = [burn_sample_maps[channel]['map']]
                plot_colors = ['k']
                plot_names = ['data']
            else:
                plot_maps = []
                plot_sumw2 = []
                plot_colors = []
                plot_names = []
            plot_maps.extend([true_template[channel]['map'],true_template[channel]['map_nu'],true_template[channel]['map_mu']])
            plot_sumw2.extend([true_template[channel]['sumw2'],true_template[channel]['sumw2_nu'],true_template[channel]['sumw2_mu']])
            plot_colors.extend(['b','g','r'])
            plot_names.extend(['total','neutrinos','atmospheric muons'])
            #myPlotter.plot_1d_projection(plot_maps,plot_sumw2,plot_colors,plot_names ,axis, bins, channel)
            #myPlotter.plot_1d_slices(plot_maps,plot_sumw2,plot_colors,plot_names ,axis, bins, channel)


    # print
    for channel in ['cscd', 'trck']:
        # plot template
        cscd_max = int(500 * livetime)
        trck_max = int(420 * livetime)
        all_max = int(600 * livetime)
        true_template[channel]['ebins'] =  anlys_ebins
        true_template[channel]['czbins'] = czbins 
        data_map = true_template[channel]['map_nu']+ np.ones(np.shape(true_template[channel]))
        data_sumw2 = true_template[channel]['map_nu']
        print "data_map = ", data_map
        print "data_sumw2 = ", data_sumw2
        print "template = ", true_template[channel]['map_nu']
        print "template sumw2 = ", true_template[channel]['sumw2_nu']
        unscaled_best_fit_vals = []
        priors = []
        mod_chi2, chi2_p, dof = get_chi2(data_map, true_template[channel]['map_nu'], data_sumw2, true_template[channel]['sumw2_nu'], unscaled_best_fit_vals, priors)
        print "getting mod_chi2 ", mod_chi2
        plot_one_map(true_template[channel], args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_PISA_map1_final_NutauCCNorm_1_%s'% (channel), fig_title=r'${\rm %s \, yr \, PISA \, map \, %s \, (Nevts: \, %.1f) }$'%(livetime, channel, np.sum(true_template[channel]['map'])), save=True, max=cscd_max if channel=='cscd' else trck_max)

        #print "[true_template[channel]['map'] = ", true_template[channel]['map']
        #print "[true_template[channel]['sumw2'] = ", true_template[channel]['sumw2']
        #print "sqrt [true_template[channel]['sumw2'] = ", np.sqrt(true_template[channel]['sumw2'])
        #print "\n"
        #print "[true_template[channel]['map_nu'] = ", true_template[channel]['map_nu']
        #print "[true_template[channel]['sumw2_nu'] = ", true_template[channel]['sumw2_nu']
        #print "sqrt [true_template[channel]['sumw2_nu'] = ", np.sqrt(true_template[channel]['sumw2_nu'])
        #print "\n"

        # save to txt
        tmpl = true_template[channel]['map_nu']
        name = '%s/true_template_%s'% (args.outdir, channel)
        savetxt(tmpl, czbins, anlys_ebins, name)

        error_nu = {}
        error_nu['map'] = np.sqrt(true_template[channel]['sumw2_nu'])
        error_nu['ebins'] =  anlys_ebins
        error_nu['czbins'] = czbins 
        plot_one_map(error_nu, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_PISA_nu_error_final_NutauCCNorm_1_%s'% (channel), fig_title=r'${\rm %s \, yr \, PISA \, \, nu \, error \, map \, %s }$'%(livetime, channel), save=True)
        name = '%s/error_%s'% (args.outdir, channel)

        #savetxt(error_nu['map'], czbins, anlys_ebins, name)



            #############  Compare Final Stage Template ############

    if args.template_settings_2:
        for channel in ['cscd','trck']:
            print "livetime = ", livetime
            cscd_max = int(500 * livetime)
            trck_max = int(420 * livetime)
            all_max = int(600 * livetime)
            print "cscd_max = ", cscd_max
            print "trck_max = ", trck_max
            print "all_max = ", all_max
            print "true_template[channel].keys = ", true_template[channel].keys()
            true_template[channel]['ebins'] =  anlys_ebins
            true_template[channel]['czbins'] = czbins 
            true_template_2[channel]['ebins'] =  anlys_ebins
            true_template_2[channel]['czbins'] = czbins 
            plot_one_map(true_template[channel], args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_PISA_map1_final_NutauCCNorm_1_%s'% (channel), fig_title=r'${\rm %s \, yr \, PISA \, map \, %s \, (Nevts: \, %.1f) }$'%(livetime, channel, np.sum(true_template[channel]['map'])), save=False, max=cscd_max if channel=='cscd' else trck_max)

            plot_one_map(true_template_2[channel], args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_PISA_map2_final_NutauCCNorm_1_%s'% (channel), fig_title=r'${\rm %s \, yr \,  PISA \, map \, %s \, (Nevts: \, %.1f) }$'%(livetime, channel, np.sum(true_template_2[channel]['map'])), save=False, max=cscd_max if channel=='cscd' else trck_max)

            ratio_template_1_2 = ratio_map(true_template[channel], true_template_2[channel])
            plot_one_map(ratio_template_1_2, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_Ratio_final_map_%s'% (channel), fig_title=r'${\rm Ratio \, of \, (map \, 1 \, / \, map \, 2 ) \, %s }$'%(channel), save=False, annotate_prcs=3) 
            delta_template_1_2 = delta_map(true_template[channel], true_template_2[channel])
            plot_one_map(delta_template_1_2, args.outdir, logE=not(args.no_logE), fig_name=args.title+ '_Delta_final_map_%s'% (channel), fig_title=r'${\rm Delta \, of \, (map \, 1 \, - \, map \, 2 ) \, %s }$'%(channel), save=False, annotate_prcs=3) 

            print "In Final stage, from first template: \n",
            print "no.of cscd (f = 1) in first template = ", np.sum(true_template['cscd']['map'])
            print "no.of trck (f = 1) in first template = ", np.sum(true_template['trck']['map'])
            print "total no. of cscd and trck  = ", np.sum(true_template['cscd']['map'])+np.sum(true_template['trck']['map'])
            print "\n"
            print "In Final stage, from first template: \n",
            print "no.of cscd (f = 1) in first template = ", np.sum(true_template_2['cscd']['map'])
            print "no.of trck (f = 1) in first template = ", np.sum(true_template_2['trck']['map'])
            print "total no. of cscd and trck  = ", np.sum(true_template_2['cscd']['map'])+np.sum(true_template_2['trck']['map'])
            print "\n"

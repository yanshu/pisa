import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib.offsetbox import AnchoredText
from pisa.utils.jsons import from_json
from pisa.utils.plot import show_map, delta_map, sum_map, ratio_map, distinguishability_map, s_over_sqrt_b
from pisa.i3utils.plot_template import plotter
import numpy as np

parser = ArgumentParser(description='''Quick check if all components are working reasonably well, by
                                        making the final level hierarchy asymmetry plots from the input settings file. ''')
parser.add_argument('-f','--json_file',metavar='JSON',
                    help='file to use for template generation')
parser.add_argument('-s','--step',
                    help='step(s) to be plotted')
args = parser.parse_args()


file = from_json(args.json_file)
fig = plt.figure(figsize=(12,6))
fig.patch.set_facecolor('none')

def dump(name,text='Expected',livetime=None):
    name = 'nutau_appearance_analysis_psu_'+name
    if text:
        if livetime:
            a_text = AnchoredText(r'$\nu_\tau$ appearance'+'\n%s years\n%s'%(livetime,text), loc=2, frameon=False)
        else:
            a_text = AnchoredText(r'$\nu_\tau$ appearance'+'\n%s'%text, loc=2, frameon=False)
        plt.gca().add_artist(a_text)
    plt.show()
    plt.savefig('%s.pdf'%name,edgecolor='none',facecolor=fig.get_facecolor())
    plt.savefig('%s.png'%name,edgecolor='none',facecolor=fig.get_facecolor())
    plt.clf()

def cutmap(inmap):
    outmap = {}
    # cut away the lowest and highest 2 energy bins
    outmap['ebins'] = inmap['ebins'][2:-2]
    outmap['czbins'] = inmap['czbins']
    outmap['map'] = inmap['map'][2:-2]
    return outmap

try:
    livetime = file[3]['params']['livetime']
except:
    livetime = None

if 'osc' in args.step:
    nue = sum_map(file[1]['nue'],file[1]['nue_bar'])
    numu = sum_map(file[1]['numu'],file[1]['numu_bar'])
    nutau = sum_map(file[1]['nutau'],file[1]['nutau_bar'])
    bkgd = sum_map(nue,numu)
    ratio = s_over_sqrt_b(nutau,bkgd)
    disting = distinguishability_map(bkgd,nutau)
    show_map(disting,logE=True,annotate_no_evts=False,cmap='RdGy',interpolation='nearest',zlabel='distinguishability e+mu vs. tau flavours')
    dump('disting_nutau_osc','Oscillation')

if args.step == 'aeffplots':
    my_plotter = plotter(livetime, '.', logy=True, fmt='pdf')
    my_plotter2 = plotter(livetime, '.', logy=True, fmt='png')
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.5)
    for axis in ['energy','coszen']:
        maps = []
        names = []
        for i,t in enumerate(['','_bar']):
            for j,flav in enumerate(['nue','numu','nutau','all']):
                mapp = None
                if flav == 'all':
                    sum = sum_map(file['nue'+t]['nc'],file['numu'+t]['nc'])
                    mapp = sum_map(sum,file['nutau'+t]['nc'])
                    cmap = cutmap(mapp)
                    names.append(flav+t+' NC')
                else:
                    ccmap = file[flav+t]['cc']
                    names.append(flav+t+' CC')
                    cmap = cutmap(ccmap)
                if axis == 'energy':
                    plt.subplot(2,4,i*4+j+1)
                    show_map(cmap,logE=True,annotate_no_evts=False,log=True,title=names[-1],cmap='Spectral',interpolation='nearest',zlabel=r'$A_{\rm{eff}}\ \rm{(m^2)}$',fontsize=6,vmin=-8.5,vmax=-4)

                if axis == 'energy':
                    bins = cmap['ebins']
                    av_bins = cmap['czbins']
                else:
                    bins = cmap['czbins']
                    av_bins = cmap['ebins']
                maps.append(my_plotter.get_1D_average(cmap['map'],axis,av_bins))
        #my_plotter.plot_1d(maps, [False]*len(maps), ['crimson','forestgreen','blue','k','orange','lime','dodgerblue','slategrey'], names, axis, bins, 'aeff_%s'%axis,ratio=False, yaxis_label=r'$A_{\rm{eff}}\ \rm{(m^2)}$')
        my_plotter.plot_1d(maps, [False]*len(maps), ['crimson','forestgreen','blue','k']*2, names, axis, bins, 'aeff_%s'%axis,ratio=False,linestyles=['-']*4+['--']*4 ,yaxis_label=r'$A_{\rm{eff}}\ \rm{(m^2)}$')
        my_plotter2.plot_1d(maps, [False]*len(maps), ['crimson','forestgreen','blue','k']*2, names, axis, bins, 'aeff_%s'%axis,ratio=False,linestyles=['-']*4+['--']*4 ,yaxis_label=r'$A_{\rm{eff}}\ \rm{(m^2)}$')
    plt.tight_layout()
    dump('Aeff_plots',None)

elif 'aeff' in args.step:
    nue_cc = sum_map(file[2]['nue']['cc'],file[2]['nue_bar']['cc'])
    numu_cc = sum_map(file[2]['numu']['cc'],file[2]['numu_bar']['cc'])
    nutau_cc = sum_map(file[2]['nutau']['cc'],file[2]['nutau_bar']['cc'])
    nue_nc = sum_map(file[2]['nue']['nc'],file[2]['nue_bar']['nc'])
    numu_nc = sum_map(file[2]['numu']['nc'],file[2]['numu_bar']['nc'])
    nutau_nc = sum_map(file[2]['nutau']['nc'],file[2]['nutau_bar']['nc'])
    bkgd = sum_map(nue_cc,numu_cc)
    bkgd = sum_map(bkgd,nue_nc)
    bkgd = sum_map(bkgd,numu_nc)
    bkgd = sum_map(bkgd,nutau_nc)
    ratio = s_over_sqrt_b(nutau_cc,bkgd)
    ratio = cutmap(ratio)
    show_map(ratio,logE=True,annotate_no_evts=False,cmap='OrRd',interpolation='nearest',zlabel=r'$s/\sqrt{b}$')
    dump('sb_nutau_cc_aeff','MC truth',livetime)

if 'reco' in args.step: 
    nue_cc = file[3]['nue_cc']
    numu_cc = file[3]['numu_cc']
    nutau_cc = file[3]['nutau_cc']
    nuall_nc = file[3]['nuall_nc']
    bkgd = sum_map(nue_cc,numu_cc)
    bkgd = sum_map(bkgd,nuall_nc)
    ratio = s_over_sqrt_b(nutau_cc,bkgd)
    show_map(ratio,logE=True,annotate_no_evts=False,cmap='OrRd',interpolation='nearest',zlabel=r'$s/\sqrt{b}$')
    dump('sb_nutau_cc_reco','Reco level',livetime)

if 'tot' in args.step: 
    nue_cc = file[3]['nue_cc']
    numu_cc = file[3]['numu_cc']
    nutau_cc = file[3]['nutau_cc']
    nuall_nc = file[3]['nuall_nc']
    muon_cscd_bla = {}
    muon_cscd_bla['ebins'] = file[5]['cscd']['ebins']
    muon_cscd_bla['czbins'] = file[5]['cscd']['czbins']
    muon_cscd_bla['map'] = file[5]['cscd']['map_mu']
    muon_trck_bla = {}
    muon_trck_bla['ebins'] = file[5]['trck']['ebins']
    muon_trck_bla['czbins'] = file[5]['trck']['czbins']
    muon_trck_bla['map'] = file[5]['trck']['map_mu']
    bkgd = sum_map(nue_cc,numu_cc)
    bkgd = sum_map(bkgd,nuall_nc)
    bkgd = sum_map(bkgd,muon_cscd_bla)
    bkgd = sum_map(bkgd,muon_trck_bla)
    ratio = s_over_sqrt_b(nutau_cc,bkgd)
    show_map(ratio,logE=True,annotate_no_evts=False,cmap='OrRd',interpolation='nearest',zlabel=r'$s/\sqrt{b}$')
    dump('sb_nutau_cc_icc','Reco + ICC bkgd',livetime)


plt.close(fig)


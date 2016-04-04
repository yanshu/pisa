#! /usr/bin/env python
#
# author: Timothy C. Arlen
#
# This scripts gets the 1D parameterized effective area as a function
# of Energy directly from the simulations using the simulation weight
# (OneWeight), and creates effective area .dat files needed by PISA.
#
# To complete the effective area parameterization, one must
# parameterize the zenith dependence separately.
#

import os,sys
import numpy as np
from glob import glob
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from scipy.interpolate import splrep, splev
import tables

from pisa.i3utils.hdfchain import HDFChain
from pisa.i3utils.sim_utils import get_arb_cuts,get_aeff1D_zen
from pisa.utils.utils import get_bin_centers
from pisa.utils.log import logging, set_verbosity

def LoadData(directory,geometry,flavor):
    '''
    Expects path to files to be "<directory/<geometry>_<flavor>.hd5"
    returns a PyTables filehandle either to the specific flavor file
    or to the chain of files for flavor == 'NC'.

    NOTE: that flavor takes values of ['nue','nue_bar','numu',...]
    but filenames are only one of 'nue', 'numu', or 'nutau'
    '''

    ext = '.hd5'
    nue_file = os.path.join(directory,geometry+"_nue"+ext)
    numu_file = os.path.join(directory,geometry+"_numu"+ext)
    nutau_file = os.path.join(directory,geometry+"_nutau"+ext)
    if flavor == 'NC':
        # First open the filehandles:
        data_nue = tables.openFile(nue_file,'r')
        data_numu = tables.openFile(numu_file,'r')
        data_nutau = tables.openFile(nutau_file,'r')

        return HDFChain([nue_file,numu_file,nutau_file])

    else:
        if 'nue' in flavor: return tables.openFile(nue_file,'r')
        elif 'numu' in flavor: return tables.openFile(numu_file,'r')
        elif 'nutau' in flavor: return tables.openFile(nutau_file,'r')
        else: raise ValueError("flavor: %s is unknown!"%s)


def SaveAeff(aeff,aeff_err,egy_bin_edges,flavor,out_dir):

    # Correct for nutau/nutaubar:
    for i in range(len(egy_bin_edges)-1):
        if(aeff_err[i] < 1.0e-12): aeff_err[i] = 1.0e-12

    ecen = get_bin_centers(egy_bin_edges)
    splinefit = splrep(ecen,aeff,w=1./np.array(aeff_err), k=3, s=100)
    fit_aeff = splev(ecen,splinefit)

    outfile = os.path.join(out_dir,"a_eff_vs_cz_"+flavor+".dat")
    print "Saving spline fit to file: "+outfile
    fh = open(outfile,'w')
    for i,energy in enumerate(ecen):
        fh.write(str(energy)+' '+str(fit_aeff[i])+'\n')
    fh.close()

    outfile_data = os.path.join(out_dir,"a_eff_vs_cz_"+flavor+"_data.dat")
    print "Saving data to file: "+outfile_data
    fh = open(outfile_data,'w')
    for i,energy in enumerate(ecen):
        fh.write(str(energy)+' '+str(aeff[i])+' '+str(aeff_err[i])+'\n')
    fh.close()

    return

#########################################################


set_verbosity(0)
parser = ArgumentParser('''Creates the aeff_<flavor>.dat files for use in the PISA code. This
script expects to be handed a directory to <geom_str>_<flav>.hd5
files, where <geom_str> (i.e. 'V36' or 'V15') is a cmd line argument,
and <flav> is in ['nue',numu','nutau'].''',
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('data_dir',metavar='DIR',type=str,
                    help='Directory to the I3 hdf5 simulation files.')
parser.add_argument('geom_str',type=str,help='Geometry tag for hdf5 file labeling.')
parser.add_argument('--old_pid',action='store_true',default=False,
                    help='Use old style particle id numbers.')
parser.add_argument('-o','--outdir',type=str,default='',
                    help='Output file directory [default: cwd]')

parser.add_argument('--ne',metavar='INT',type=float,required=True,
                    help='Number of i3 sim files combined into the nue hdf5 file.')
parser.add_argument('--nmu',metavar='INT',type=float,required=True,
                    help='Number of i3 sim files combined into the numu hdf5 file.')
parser.add_argument('--ntau',metavar='INT',type=float,required=True,
                    help='Number of i3 sim files combined into the nutau hdf5 file.')
parser.add_argument('--all_cz',action='store_true', default=False,
                    help='Use all downgoing events (i.e. Do NOT cut cz < 0 events).')
parser.add_argument('--mcnu',metavar='STR',type=str,default='MCNeutrino',
                    help='Key in hdf5 file from which to extract MC True information')
# egy binning options:
parser.add_argument('--ebins',metavar='FUNC',type=str,
                    default='np.logspace(np.log10(1.0),np.log10(80.0),40)',
                    help='''Python text to convert to lambda function for energy
                    bin edges. i.e. np.linspace(1,40,40)''')
parser.add_argument('--czbins',metavar='FUNC',type=str,
                    default='np.linspace(-1.0,0.0,21)',
                    help='''Python text to convert to lambda function for coszen
                    bin edges. i.e. np.linspace(-1,0,21)''')
# Step1/Step2 cuts options (or NONE - for most DeepCore analyses):
hcut = parser.add_mutually_exclusive_group(required=False)
hcut.add_argument('--v3cuts',action='store_true',default=False,
                  help='Use V3 version of the cuts')
hcut.add_argument('--v4cuts',action='store_true',default=False,
                  help='Use V4 version of the cuts')
hcut.add_argument('--v5truth',action='store_true',default=False,
                  help='Use step2 V5 truth information')
hcut.add_argument('--nocuts',action='store_true',default=False,
                  help='Do not use any stage of the selection cuts on the files.')
hcut.add_argument('--custom',action='store_true',default=False,
                  help='Use custom cuts, given by args.custom_str')
parser.add_argument('--custom_str',metavar='STR',type=str,
                    default="[('IC86_Dunkman_L6','result',True)]",
                    help='''Python string which will be evaluated as a list of tuples of cuts
                    where tuple is (Attribute,ColName,bool) See function
                    sim_utils.get_arb_cuts for more details if needed.''')
parser.add_argument('-v', '--verbose', action='count', default=0,
                    help='set verbosity level')

args = parser.parse_args()
set_verbosity(args.verbose)

print "FILE NORMALIZATION: "
print "  >> nue: ",args.ne
print "  >> numu: ",args.nmu
print "  >> nutau: ",args.ntau

czbins = eval(args.czbins)
ebins = eval(args.ebins)

print "czbins = ", czbins
print "ebins = ", ebins
e_min = np.min(ebins)
e_max = np.max(ebins)

# Cut definitions:
cut_list = []
if args.v4cuts:
    logging.warn("Using cuts V4!")
    cut_list = [("Cuts_V4_Step1",'value',True),("Cuts_V4_Step2",'value',True)]
elif args.v3cuts:
    logging.warn("Using cuts V3!")
    cut_list = [('NewestBgRejCutsStep1','value',True), ('NewestBgRejCutsStep2','value',True)]
elif args.v5truth:
    logging.warn("USING V5 TRUTH information")
    cut_list = [('Cuts_V5_Step2_upgoing_Truth','value',True)]
elif args.nocuts:
    logging.warn("Using no selection cuts!")
    cut_list = []
elif args.custom:
    logging.warn("Using CUSTOM cuts: %s..."%args.custom_str)
    cut_list = eval(args.custom_str)
else:
    logging.warn("Using cuts V5!")
    cut_list= [("Cuts_V5_Step1",'value',True),("Cuts_V5_Step2",'value',True)]

nuDict = {}
if args.old_pid:
    nuDict = {'nue':66,'numu':68,'nutau':133,'nuebar':67,'numubar':69,'nutaubar':134}
else:
    nuDict = {'nue':12,'numu':14,'nutau':16,'nuebar':-12,'numubar':-14,'nutaubar':-16}


aeff_list = []
aeff_nu_nc_array = np.zeros(len(czbins)-1)
aeff_nubar_nc_array = np.zeros(len(czbins)-1)
aeff_err_list = []
aeff_nu_nc_err_sq_array = np.zeros(len(czbins)-1) 
aeff_nubar_nc_err_sq_array = np.zeros(len(czbins)-1) 
flavor_list = []

cut_e_range = True 
cut_sim_down = True
solid_angle = 2.0*np.pi
if args.all_cz:
    # Then use all sky, don't remove simulated downgoing events:
    cut_sim_down = False
    solid_angle = 4.0*np.pi

# Loop over all neutrino flavours, and get cc Aeff:
for flav,val in nuDict.items():

    logging.info("Loading data for %s..."%flav)
    data = LoadData(args.data_dir,args.geom_str,flav)

    cc_cuts = list(cut_list)
    cc_cuts.append(("I3MCWeightDict","InteractionType",1))
    cc_cuts.append((args.mcnu,"type",val))
    nc_cuts = list(cut_list)
    nc_cuts.append(("I3MCWeightDict","InteractionType",2))
    nc_cuts.append((args.mcnu,"type",val))

    arb_cc_cut_list = get_arb_cuts(data, cc_cuts,e_min,e_max,mcnu=args.mcnu, cut_sim_down=cut_sim_down, cut_e_range=cut_e_range)
    arb_nc_cut_list = get_arb_cuts(data, nc_cuts,e_min,e_max,mcnu=args.mcnu, cut_sim_down=cut_sim_down, cut_e_range=cut_e_range)

    logging.info("  NEvents CC: %d"%np.sum(arb_cc_cut_list))
    logging.info("  NEvents NC: %d"%np.sum(arb_nc_cut_list))

    if 'nue' in flav: nfiles = args.ne
    elif 'numu' in flav: nfiles = args.nmu
    elif 'nutau' in flav: nfiles = args.ntau
    else: raise ValueError("Unrecognized flav: %s"%flav)

    aeff_cc,aeff_cc_err,xedges = get_aeff1D_zen(data,arb_cc_cut_list, czbins,nfiles,
                                            mcnu=args.mcnu,solid_angle=solid_angle)
    aeff_nc,aeff_nc_err,xedges = get_aeff1D_zen(data,arb_nc_cut_list, czbins,nfiles,
                                            mcnu=args.mcnu,solid_angle=solid_angle)

    aeff_list.append(aeff_cc)
    aeff_err_list.append(aeff_cc_err)
    flavor_list.append(flav+'_cc')
    if 'bar' in flav:
        aeff_nubar_nc_array += aeff_nc
        aeff_nubar_nc_err_sq_array += np.square(aeff_nc_err)
    else:
        aeff_nu_nc_array += aeff_nc
        aeff_nu_nc_err_sq_array += np.square(aeff_nc_err)

aeff_list.append(aeff_nu_nc_array)
aeff_err_list.append(np.sqrt(aeff_nu_nc_err_sq_array))
flavor_list.append('nuall_nc')
aeff_list.append(aeff_nubar_nc_array)
aeff_err_list.append(np.sqrt(aeff_nubar_nc_err_sq_array))
flavor_list.append('nuallbar_nc')

for i,flavor in enumerate(flavor_list):
    logging.info("Saving: %s to %s"%(flavor,args.outdir))
    SaveAeff(aeff_list[i],aeff_err_list[i], czbins,flavor,args.outdir)

print "\nFINISHED...\n"

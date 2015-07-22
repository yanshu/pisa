import tables, h5py
import numpy as np
from pisa.utils.log import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def write_group(nu_group,intType,data_nu):
    '''
    Helper function to write all sim_wt arrays to file
    '''
    if intType not in ['cc','nc']:
        raise Exception('intType: %s unexpected. Expects cc or nc...'%intType)

    sub_group = nu_group.create_group(intType)
    sub_group.create_dataset('true_energy',data=data_nu[0],dtype=np.float32)
    sub_group.create_dataset('true_coszen',data=data_nu[1],dtype=np.float32)
    sub_group.create_dataset('reco_energy',data=data_nu[2],dtype=np.float32)
    sub_group.create_dataset('reco_coszen',data=data_nu[3],dtype=np.float32)

    return

def write_to_hdf5(outfilename,flavor,data_nu_cc,data_nu_nc):
    '''
    Writes the sim wt arrays to outfilename, for single flavour's cc/nc fields.
    '''
    fh = h5py.File(outfilename,'a')
    nu_group = fh.create_group(flavor)

    write_group(nu_group,'cc',data_nu_cc)
    write_group(nu_group,'nc',data_nu_nc)
    fh.close()
    return

parser = ArgumentParser(description='''Takes the simulated (and reconstructed) data files (in hdf5 format) as input and writes out the sim_wt arrays for use in the aeff and reco stage of the template maker.''',formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('fraction', metavar='float',type=float,
                    help='''The percentage of MC events to be selected''')
parser.add_argument('out_dir', metavar='str',type=str,
                    help='''The output directory''')
args = parser.parse_args()
frac = args.fraction
out_dir = args.out_dir

for run_num in [50,60,61,64,65,70,71,72]:
    # Ensure overwrite of existing filename...
    outfilename = '%s/1X%i_weighted_aeff_joined_nu_nubar_%i_percent_up.hdf5'% (out_dir,run_num,int(frac*100))
    fh = h5py.File(outfilename,'w')
    fh.close()
    logging.info("Writing to file: %s",outfilename)
    
    data_file = h5py.File('1X%i_weighted_aeff_joined_nu_nubar.hdf5'% run_num)
    
    data_nc_trueE = np.array(data_file['numu']['nc']['true_energy'])
    data_nc_trueCZ = np.array(data_file['numu']['nc']['true_coszen'])
    data_nc_recoE = np.array(data_file['numu']['nc']['reco_energy'])
    data_nc_recoCZ = np.array(data_file['numu']['nc']['reco_coszen'])
    assert(len(data_nc_trueE)==len(data_nc_recoE))
    assert(len(data_nc_trueCZ)==len(data_nc_recoCZ))
    assert(len(data_nc_trueCZ)==len(data_nc_trueE))
    #cut = data_nc_trueCZ < 0
    cut = data_nc_recoCZ < 0
    data_nc_trueE = data_nc_trueE[cut]
    data_nc_recoE = data_nc_recoE[cut]
    data_nc_trueCZ = data_nc_trueCZ[cut]
    data_nc_recoCZ = data_nc_recoCZ[cut]
    if frac !=1:
        len_data = len(data_nc_trueE)
        print "len(upgoing nc) = ", len_data
        len_new_data = int(frac*len(data_nc_trueE))
        print "new len(nc) = ", len_new_data
        selection = list(np.random.choice(len_data,len_new_data))
        new_data_nc_trueE = data_nc_trueE[selection]
        new_data_nc_recoE = data_nc_recoE[selection]
        new_data_nc_recoCZ = data_nc_recoCZ[selection]
        new_data_nc_trueCZ = data_nc_trueCZ[selection]
        arrays_nc = [new_data_nc_trueE,new_data_nc_trueCZ,new_data_nc_recoE,new_data_nc_recoCZ]
    else:
        arrays_nc = [data_nc_trueE,data_nc_trueCZ,data_nc_recoE,data_nc_recoCZ]
    
    for flavor in ['nutau','numu','nue']:
        data_cc_trueE = np.array(data_file[flavor]['cc']['true_energy'])
        data_cc_trueCZ = np.array(data_file[flavor]['cc']['true_coszen'])
        data_cc_recoE = np.array(data_file[flavor]['cc']['reco_energy'])
        data_cc_recoCZ = np.array(data_file[flavor]['cc']['reco_coszen'])
        assert(len(data_cc_trueE)==len(data_cc_recoE))
        assert(len(data_cc_trueCZ)==len(data_cc_recoCZ))
        assert(len(data_cc_trueCZ)==len(data_cc_trueE))
        #cut = data_cc_trueCZ < 0
        cut = data_cc_recoCZ < 0
        data_cc_trueE = data_cc_trueE[cut]
        data_cc_recoE = data_cc_recoE[cut]
        data_cc_trueCZ = data_cc_trueCZ[cut]
        data_cc_recoCZ = data_cc_recoCZ[cut]
        if frac !=1:
            len_data = len(data_cc_trueE)
            print "len( upgoing ", flavor , ") = ",  len_data
            len_new_data = int(frac*len(data_cc_trueE))
            print "new len(", flavor , ") = ",  len_new_data
            selection = list(np.random.choice(len_data,len_new_data))
            new_data_cc_trueE = data_cc_trueE[selection]
            new_data_cc_recoE = data_cc_recoE[selection]
            new_data_cc_recoCZ = data_cc_recoCZ[selection]
            new_data_cc_trueCZ = data_cc_trueCZ[selection]
            arrays_cc = [new_data_cc_trueE,new_data_cc_trueCZ,new_data_cc_recoE,new_data_cc_recoCZ]
        else:
            arrays_cc = [data_cc_trueE,data_cc_trueCZ,data_cc_recoE,data_cc_recoCZ]
    
        logging.info("Saving %s..."%flavor)
        write_to_hdf5(outfilename,flavor,arrays_cc,arrays_nc)
    
        # Duplicate and write to <flavor>_bar
        flavor+='_bar'
        logging.info("Saving %s..."%flavor)
        write_to_hdf5(outfilename,flavor,arrays_cc,arrays_nc)
    
    data_file.close()
    

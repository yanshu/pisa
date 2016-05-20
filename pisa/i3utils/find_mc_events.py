
#! /usr/bin/env python
import h5py
import numpy as np

file_name_list = []
run_list = [ '600', '601', '603', '604', '605', '606', '608', '610', '611', '612', '613', '620', '621', '622', '623', '624']
for run_num in run_list: 
    file_name_list.append("make_events_file_output_with_weight/events__deepcore__IC86__runs_12%s1-12%s3,14%s1-14%s3,16%s1-16%s3__proc_v5digit__unjoined_with_fluxes.hdf5" % (run_num,run_num,run_num,run_num,run_num,run_num))


Emax = 79.43282347
Emin = 5.01187234

#for file_name in file_name_list:
no_mc_events = np.zeros(len(run_list)) 
for i in range(0, len(file_name_list)):
    file_name = file_name_list[i]
    file = h5py.File(file_name, "r")
    #for prim in file.keys():
    for prim in ['nue', 'numu', 'nutau']:
        for int_type in file[prim]:
            nu_reco_e = np.array(file[prim][int_type]['reco_energy'])
            nu_reco_e = nu_reco_e[np.logical_and(nu_reco_e<=Emax, nu_reco_e>=Emin)]
            nubar_reco_e = np.array(file[prim+'_bar'][int_type]['reco_energy'])
            nubar_reco_e = nubar_reco_e[np.logical_and(nubar_reco_e<=Emax, nubar_reco_e>=Emin)]
            #no_mc_events[i]+= len(file[prim][int_type]['reco_energy']) + len(file[prim+'_bar'][int_type]['reco_energy'])
            #print "Run: ", run_list[i], " no_mc_events in ", prim, "+ ", prim+"_bar ", int_type, " : ", len(file[prim][int_type]['reco_energy'])+len(file[prim+'_bar'][int_type]['reco_energy'])
            no_mc_events[i] += len(nu_reco_e) + len(nubar_reco_e)
            print "Run: ", run_list[i], " no_mc_events in ", prim, "+ ", prim+"_bar ", int_type, " : ", len(nu_reco_e)+ len(nubar_reco_e)
    file.close()
    print "\n"

for i in range(0, len(file_name_list)):
    print "Run: ", run_list[i] , " total no_mc_events = ", no_mc_events[i]


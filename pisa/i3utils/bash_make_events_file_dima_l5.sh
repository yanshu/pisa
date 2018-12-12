#python make_events_file_all_without_genie_l5.py --det "DeepCore" --proc "5digit" --nue /storage/group/dfc13_collab/Matt_level5_mc/hdf5/DC_level5_12600.hdf5 --numu /storage/group/dfc13_collab/Matt_level5_mc/hdf5/DC_level5_14600.hdf5 --nutau /storage/group/dfc13_collab/Matt_level5_mc/hdf5/DC_level5_16600.hdf5 --outdir /storage/group/dfc13_collab/Matt_level5_mc/make_events_file_output_with_low_level_variables/

#python add_stuff_to_event_file.py -fp /storage/group/dfc13_collab/Matt_level5_mc/make_events_file_output_with_low_level_variables/events__deepcore__IC86__runs_126001-126003,146001-146003,166001-166003__proc_v5digit__unjoined.hdf5 -t ~/work/pisa/pisa/resources/settings/template_settings/DC12_event_15syst_2.5yr_no_prior_5digit_600_blind_fit_good.json -o /storage/group/dfc13_collab/Matt_level5_mc/make_events_file_output_with_low_level_variables_with_fluxes/ --add_fluxes --add_weights

#0930
#indir=/gpfs/group/dfc13/default/Matt_level5_mc/hdf5
#python make_events_file_all_without_genie_l5.py --det "DeepCore" --proc "5digit" --nue ${indir}/DC_level5_12600.hdf5 --numu ${indir}/DC_level5_14600.hdf5 --nutau ${indir}/DC_level5_16600.hdf5 --outdir /gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L2/ --run-settings 'events/mc_sim_run_settings_l5.json' --cut 'l2' --data-proc-params 'events/data_proc_params_l5.json'
#python make_events_file_all_without_genie_l5.py --det "DeepCore" --proc "5digit" --nue ${indir}/DC_level5_12600.hdf5 --numu ${indir}/DC_level5_14600.hdf5 --nutau ${indir}/DC_level5_16600.hdf5 --outdir /gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L3/ --run-settings 'events/mc_sim_run_settings_l5.json' --cut 'l3' --data-proc-params 'events/data_proc_params_l5.json'
#python make_events_file_all_without_genie_l5.py --det "DeepCore" --proc "5digit" --nue ${indir}/DC_level5_12600.hdf5 --numu ${indir}/DC_level5_14600.hdf5 --nutau ${indir}/DC_level5_16600.hdf5 --outdir /gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L34/ --run-settings 'events/mc_sim_run_settings_l5.json' --cut 'l34' --data-proc-params 'events/data_proc_params_l5.json'
#python make_events_file_all_without_genie_l5.py --det "DeepCore" --proc "5digit" --nue ${indir}/DC_level5_12600.hdf5 --numu ${indir}/DC_level5_14600.hdf5 --nutau ${indir}/DC_level5_16600.hdf5 --outdir /gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L345/ --run-settings 'events/mc_sim_run_settings_l5.json' --cut 'l345' --data-proc-params 'events/data_proc_params_l5.json'


#indir=/gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L2/
#outdir=/gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L2/
#mkdir $outdir

#indir=/gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L3/
#outdir=/gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L3/
#mkdir $outdir

#indir=/gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L34/
#outdir=/gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L34/

#indir=/gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L345/
#outdir=/gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L345/
#mkdir $outdir

#python add_stuff_to_event_file.py -fp ${indir}/events__deepcore__IC86__runs_126001-126003,146001-146003,166001-166003__proc_v5digit__unjoined.hdf5 -t ~/work/pisa/pisa/resources/settings/template_settings/DC12_event_15syst_2.5yr_no_prior_5digit_600_blind_fit_good.json -o ${outdir} --add_fluxes




# 0616
#python change_run_num_subruns.py -i /gpfs/group/dfc13/default/sim/icecube/dima/pisa/level5/

indir=/gpfs/group/dfc13/default/sim/icecube/dima/pisa/level5/

#outdir=/gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L345/
#mkdir -p $outdir
#python make_events_file_all_without_genie_l5.py --det "DeepCore" --proc "5digit" --nue ${indir}/nue/Level5_IC86.2013_genie_nue.12600.hdf5 --numu ${indir}/numu/Level5_IC86.2013_genie_numu.14600.hdf5 --nutau ${indir}/nutau/Level5_IC86.2013_genie_nutau.16600.hdf5 --outdir ${outdir} --run-settings 'events/mc_sim_run_settings_l5.json' --cut 'l345' --data-proc-params 'events/data_proc_params_l5.json'

#outdir=/gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L34/
#mkdir -p $outdir
#python make_events_file_all_without_genie_l5.py --det "DeepCore" --proc "5digit" --nue ${indir}/nue/Level5_IC86.2013_genie_nue.12600.hdf5 --numu ${indir}/numu/Level5_IC86.2013_genie_numu.14600.hdf5 --nutau ${indir}/nutau/Level5_IC86.2013_genie_nutau.16600.hdf5 --outdir ${outdir} --run-settings 'events/mc_sim_run_settings_l5.json' --cut 'l34' --data-proc-params 'events/data_proc_params_l5.json'
#
#outdir=/gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L3/
#mkdir -p $outdir
#python make_events_file_all_without_genie_l5.py --det "DeepCore" --proc "5digit" --nue ${indir}/nue/Level5_IC86.2013_genie_nue.12600.hdf5 --numu ${indir}/numu/Level5_IC86.2013_genie_numu.14600.hdf5 --nutau ${indir}/nutau/Level5_IC86.2013_genie_nutau.16600.hdf5 --outdir ${outdir} --run-settings 'events/mc_sim_run_settings_l5.json' --cut 'l3' --data-proc-params 'events/data_proc_params_l5.json'

outdir=/gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L3_no_direct_dom_cut/
mkdir -p $outdir
python make_events_file_all_without_genie_l5.py --det "DeepCore" --proc "5digit" --nue ${indir}/nue/Level5_IC86.2013_genie_nue.12600.hdf5 --numu ${indir}/numu/Level5_IC86.2013_genie_numu.14600.hdf5 --nutau ${indir}/nutau/Level5_IC86.2013_genie_nutau.16600.hdf5 --outdir ${outdir} --run-settings 'events/mc_sim_run_settings_l5.json' --cut 'l3_no_direct_dom_cut' --data-proc-params 'events/data_proc_params_l5.json'

##### L2 ######
#indir=/gpfs/group/dfc13/default/sim/icecube/dima/pisa/level2p/
#outdir=/gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L2/
#mkdir -p $outdir
#python make_events_file_all_without_genie_l2.py --det "DeepCore" --proc "5digit" --nue ${indir}/nue/Level2p_IC86.2013_genie_nue.12600.hdf5 --numu ${indir}/numu/Level2p_IC86.2013_genie_numu.14600.hdf5 --nutau ${indir}/nutau/Level2p_IC86.2013_genie_nutau.16600.hdf5 --outdir ${outdir} --run-settings 'events/mc_sim_run_settings_l5.json' --cut 'l2' --data-proc-params 'events/data_proc_params_l2.json'

python add_stuff_to_event_file.py -fp ${outdir}/events__deepcore__IC86__runs_126001-126003,146001-146003,166001-166003__proc_v5digit__unjoined.hdf5 -t ~/work/pisa/pisa/resources/settings/template_settings/DC12_event_15syst_2.5yr_no_prior_5digit_600_blind_fit_good.json -o ${outdir} --add_fluxes

#note: now the intermediate files and the directory /gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L2/ have been deleted. Only the final files are saved here(their names end with "L2", "L3", etc.):  /gpfs/group/dfc13/default/sim/icecube/dima/pisa/

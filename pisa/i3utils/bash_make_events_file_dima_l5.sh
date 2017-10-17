#python make_events_file_all_without_genie_l5.py --det "DeepCore" --proc "5digit" --nue /storage/group/dfc13_collab/Matt_level5_mc/hdf5/DC_level5_12600.hdf5 --numu /storage/group/dfc13_collab/Matt_level5_mc/hdf5/DC_level5_14600.hdf5 --nutau /storage/group/dfc13_collab/Matt_level5_mc/hdf5/DC_level5_16600.hdf5 --outdir /storage/group/dfc13_collab/Matt_level5_mc/make_events_file_output_with_low_level_variables/

#python add_stuff_to_event_file.py -fp /storage/group/dfc13_collab/Matt_level5_mc/make_events_file_output_with_low_level_variables/events__deepcore__IC86__runs_126001-126003,146001-146003,166001-166003__proc_v5digit__unjoined.hdf5 -t ~/work/pisa/pisa/resources/settings/template_settings/DC12_event_15syst_2.5yr_no_prior_5digit_600_blind_fit_good.json -o /storage/group/dfc13_collab/Matt_level5_mc/make_events_file_output_with_low_level_variables_with_fluxes/ --add_fluxes --add_weights

#0930
indir=/gpfs/group/dfc13/default/Matt_level5_mc/hdf5
python make_events_file_all_without_genie_l5.py --det "DeepCore" --proc "5digit" --nue ${indir}/DC_level5_12600.hdf5 --numu ${indir}/DC_level5_14600.hdf5 --nutau ${indir}/DC_level5_16600.hdf5 --outdir /gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L2/ --run-settings 'events/mc_sim_run_settings_l5.json' --cut 'l2' --data-proc-params 'events/data_proc_params_l5.json'
#python make_events_file_all_without_genie_l5.py --det "DeepCore" --proc "5digit" --nue ${indir}/DC_level5_12600.hdf5 --numu ${indir}/DC_level5_14600.hdf5 --nutau ${indir}/DC_level5_16600.hdf5 --outdir /gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L3/ --run-settings 'events/mc_sim_run_settings_l5.json' --cut 'l3' --data-proc-params 'events/data_proc_params_l5.json'
#python make_events_file_all_without_genie_l5.py --det "DeepCore" --proc "5digit" --nue ${indir}/DC_level5_12600.hdf5 --numu ${indir}/DC_level5_14600.hdf5 --nutau ${indir}/DC_level5_16600.hdf5 --outdir /gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L34/ --run-settings 'events/mc_sim_run_settings_l5.json' --cut 'l34' --data-proc-params 'events/data_proc_params_l5.json'
#python make_events_file_all_without_genie_l5.py --det "DeepCore" --proc "5digit" --nue ${indir}/DC_level5_12600.hdf5 --numu ${indir}/DC_level5_14600.hdf5 --nutau ${indir}/DC_level5_16600.hdf5 --outdir /gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L345/ --run-settings 'events/mc_sim_run_settings_l5.json' --cut 'l345' --data-proc-params 'events/data_proc_params_l5.json'


indir=/gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L2/
outdir=/gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_with_fluxes_L2/
mkdir $outdir

#indir=/gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L3/
#outdir=/gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_with_fluxes_L3/
#mkdir $outdir

#indir=/gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L34/
#outdir=/gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_with_fluxes_L34/

#indir=/gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_L345/
#outdir=/gpfs/group/dfc13/default/Matt_level5_mc/make_events_file_output_with_low_level_variables_with_fluxes_L345/
#mkdir $outdir

python add_stuff_to_event_file.py -fp ${indir}/events__deepcore__IC86__runs_126001-126003,146001-146003,166001-166003__proc_v5digit__unjoined.hdf5 -t ~/work/pisa/pisa/resources/settings/template_settings/DC12_event_15syst_2.5yr_no_prior_5digit_600_blind_fit_good.json -o ${outdir} --add_fluxes

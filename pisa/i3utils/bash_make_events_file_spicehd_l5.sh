#date 0214
#for run in 544 545 549
#do
#    echo $run
#    python make_events_file_all_without_genie.py --det "DeepCore" --proc "5digit" --nue /storage/group/dfc13_collab/Matt_level5b_mc/Spice_HD_l5/l5_hdf5/SpiceHD_level5_12${run}.hdf5 --numu /storage/group/dfc13_collab/Matt_level5b_mc/Spice_HD_l5/l5_hdf5/SpiceHD_level5_14${run}.hdf5 --nutau /storage/group/dfc13_collab/Matt_level5b_mc/Spice_HD_l5/l5_hdf5/SpiceHD_level5_16${run}.hdf5 --outdir /storage/group/dfc13_collab/Matt_level5b_mc/Spice_HD_l5/make_events_file_output_with_low_level_variables/
#done

#for run in 544 545 549
#do
#    echo $run
#    python add_stuff_to_event_file.py -fp /storage/group/dfc13_collab/Matt_level5b_mc/Spice_HD_l5/make_events_file_output_with_low_level_variables/events__deepcore__IC86__runs_12${run}1-12${run}3,14${run}1-14${run}3,16${run}1-16${run}3__proc_v5digit__unjoined.hdf5 -t ~/work/pisa/pisa/resources/settings/template_settings/DC12_event_15syst_2.5yr_no_prior_5digit_600_blind_fit_good.json -o /storage/group/dfc13_collab/Matt_level5b_mc/Spice_HD_l5/make_events_file_output_with_low_level_variables_with_fluxes/ --add_fluxes --add_weight
#done

#date 0412 (no files: nue 300; numu 750; nutau 60)
#for run in 534
#do
#    python make_events_file_all_without_genie_l5.py --det "DeepCore" --proc "5digit" --nue /storage/group/dfc13_collab/sim/icecube/spice_lea_hd/pisa/level5/nue/DC_spicehd_12${run}_l5.hdf5 --numu /storage/group/dfc13_collab/sim/icecube/spice_lea_hd/pisa/level5/numu/DC_spicehd_14${run}_l5.hdf5 --nutau /storage/group/dfc13_collab/sim/icecube/spice_lea_hd/pisa/level5/nutau/DC_spicehd_16${run}_l5.hdf5 --outdir /storage/group/dfc13_collab/sim/icecube/spice_lea_hd/pisa/level5/make_events_file_output_with_low_level_variables
#done
#for run in 534
#do
#    python add_stuff_to_event_file.py -fp /storage/group/dfc13_collab/sim/icecube/spice_lea_hd/pisa/level5/make_events_file_output_with_low_level_variables/events__deepcore__IC86__runs_12${run}1-12${run}3,14${run}1-14${run}3,16${run}1-16${run}3__proc_v5digit__unjoined.hdf5 -t ~/work/pisa/pisa/resources/settings/template_settings/DC12_event_15syst_2.5yr_no_prior_5digit_600_blind_fit_good.json -o /storage/group/dfc13_collab/sim/icecube/spice_lea_hd/pisa/level5/make_events_file_output_with_low_level_variables/ --add_fluxes --add_weight
#done

# 0412
# 25 numu, 10 nue, 2 nutau
#for run in 534
#do
#    python make_events_file_all_without_genie_l5.py --det "DeepCore" --proc "5digit" --nue /storage/group/dfc13_collab/sim/icecube/spice_lea_hd/pisa/level5_all_bdt/DC_spicehd_12${run}_l5_allbdt.hdf5 --numu /storage/group/dfc13_collab/sim/icecube/spice_lea_hd/pisa/level5_all_bdt/DC_spicehd_14${run}_l5_allbdt.hdf5 --nutau /storage/group/dfc13_collab/sim/icecube/spice_lea_hd/pisa/level5_all_bdt/DC_spicehd_16${run}_l5_allbdt.hdf5 --outdir /storage/group/dfc13_collab/sim/icecube/spice_lea_hd/pisa/level5_all_bdt/make_events_file_output_with_low_level_variables
#done
#for run in 534
#do
#    python add_stuff_to_event_file.py -fp /storage/group/dfc13_collab/sim/icecube/spice_lea_hd/pisa/level5_all_bdt/make_events_file_output_with_low_level_variables/events__deepcore__IC86__runs_12${run}1-12${run}3,14${run}1-14${run}3,16${run}1-16${run}3__proc_v5digit__unjoined.hdf5 -t ~/work/pisa/pisa/resources/settings/template_settings/DC12_event_15syst_2.5yr_no_prior_5digit_600_blind_fit_good.json -o /storage/group/dfc13_collab/sim/icecube/spice_lea_hd/pisa/level5_all_bdt/make_events_file_output_with_low_level_variables/ --add_fluxes
#done
#

##0422
#for run in 534
##0427
for run in 533
do
    hd5dir=/storage/group/dfc13_collab/sim/icecube/spice_lea_hd/pisa/level5p/make_events_file_output_with_low_level_variables/
    python make_events_file_all_without_genie.py --det "DeepCore" --proc "5digit" --nue /storage/group/dfc13_collab/sim/icecube/spice_lea_hd/pisa/level5p/nue/Level5p_IC86.2013_genie_nue.012${run}.hdf5 --numu /storage/group/dfc13_collab/sim/icecube/spice_lea_hd/pisa/level5p/numu/Level5p_IC86.2013_genie_numu.014${run}.hdf5 --nutau /storage/group/dfc13_collab/sim/icecube/spice_lea_hd/pisa/level5p/nutau/Level5p_IC86.2013_genie_nutau.016${run}.hdf5 --outdir ${hd5dir}
    python add_stuff_to_event_file.py -fp ${hd5dir}events__deepcore__IC86__runs_12${run}1-12${run}3,14${run}1-14${run}3,16${run}1-16${run}3__proc_v5digit__unjoined.hdf5 -t ~/work/pisa/pisa/resources/settings/template_settings/DC12_event_15syst_2.5yr_no_prior_5digit_600_blind_fit_good.json -o ${hd5dir} --add_fluxes
done

#570
#python make_events_file_all_without_genie.py --det "DeepCore" --proc "5digit" --nue /storage/group/dfc13_collab/Matt_level5b_mc/new_sets/DC12_12570.hdf5 --numu /storage/group/dfc13_collab/Matt_level5b_mc/new_sets/DC12_14570.hdf5 --nutau /storage/group/dfc13_collab/Matt_level5b_mc/new_sets/DC12_16570.hdf5 --outdir /storage/group/dfc13_collab/Matt_level5b_mc/make_events_file_output/

#python add_stuff_to_event_file.py -fp /storage/group/dfc13_collab/Matt_level5b_mc/make_events_file_output/events__deepcore__IC86__runs_125701-125703 145701-145703 165701-165703__proc_v5digit__unjoined.hdf5 -t ~/work/pisa/pisa/resources/settings/template_settings/DC12_event_15syst_2.5yr_no_prior_5digit_600_blind_fit_good.json -o /storage/group/dfc13_collab/Matt_level5b_mc/make_events_file_output_with_fluxes_GENIE_Barr/ --add_fluxes

# dima
#for run in 601 603 604 605 606 608 610 611 612 613 620 621 622 623 624
for run in 624
do
    python ~/pisa/pisa/scripts/make_events_file_all_without_genie_1.py --det "deepcore" --proc "5digit" --run 12${run} /storage/group/dfc13_collab/sim/icecube/dima/pisa/level5p/nue/DC12_12${run}.hdf5 --run 14${run} /storage/group/dfc13_collab/sim/icecube/dima/pisa/level5p/numu/DC12_14${run}_merged.hdf5 --run 16$run} /storage/group/dfc13_collab/sim/icecube/dima/pisa/level5p/nutau/DC12_16${run}.hdf5 --outdir /storage/group/dfc13_collab/sim/icecube/dima/pisa/level5p/Matt_mc_bdt_score_saved/all_lower_level_params/
done

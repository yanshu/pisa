run=600
indir=/gpfs/group/dfc13/default/sim/icecube/dima/pisa/level5p/
outdir=/gpfs/group/dfc13/default/sim/icecube/dima/pisa/level5p/Matt_mc_bdt_score_saved/all_lower_level_params/
python make_events_file_all_with_genie.py --det "DeepCore" --proc "5digit" --nue ${indir}/nue/DC12_12${run}_withGENIE.hdf5 --numu ${indir}/numu/DC12_14${run}_withGENIE.hdf5 --nutau ${indir}/nutau/DC12_16${run}_withGENIE.hdf5 --outdir ${outdir}

# add fluxes:
indir=/gpfs/group/dfc13/default/sim/icecube/dima/pisa/level5p/Matt_mc_bdt_score_saved/all_lower_level_params/
outdir=/gpfs/group/dfc13/default/sim/icecube/dima/pisa/level5p/Matt_mc_bdt_score_saved/all_lower_level_params/
python add_stuff_to_event_file.py -fp ${indir}/events__deepcore__IC86__runs_12$run\1-12$run\3,14$run\1-14$run\3,16$run\1-16$run\3__proc_v5digit__unjoined.hdf5 -t ~/work/pisa/pisa/resources/settings/template_settings/DC12_event_15syst_2.5yr_no_prior_5digit_600_blind_fit_good.json -o ${outdir} --add_fluxes --add_GENIE_Barr

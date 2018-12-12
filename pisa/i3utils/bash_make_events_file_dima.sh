#570
#python make_events_file_all_without_genie.py --det "DeepCore" --proc "5digit" --nue /storage/group/dfc13_collab/Matt_level5b_mc/new_sets/DC12_12570.hdf5 --numu /storage/group/dfc13_collab/Matt_level5b_mc/new_sets/DC12_14570.hdf5 --nutau /storage/group/dfc13_collab/Matt_level5b_mc/new_sets/DC12_16570.hdf5 --outdir /storage/group/dfc13_collab/Matt_level5b_mc/make_events_file_output/

#python add_stuff_to_event_file.py -fp /storage/group/dfc13_collab/Matt_level5b_mc/make_events_file_output/events__deepcore__IC86__runs_125701-125703 145701-145703 165701-165703__proc_v5digit__unjoined.hdf5 -t ~/work/pisa/pisa/resources/settings/template_settings/DC12_event_15syst_2.5yr_no_prior_5digit_600_blind_fit_good.json -o /storage/group/dfc13_collab/Matt_level5b_mc/make_events_file_output_with_fluxes_GENIE_Barr/ --add_fluxes

# dima
#for run in 601 603 604 605 606 608 610 611 612 613 620 621 622 623 624
for run in 624
do
    #python ~/pisa/pisa/scripts/make_events_file_all_without_genie_1.py --det "deepcore" --proc "5digit" --run 12${run} /gpfs/group/dfc13/default/sim/icecube/dima/pisa/level5p/nue/DC12_12${run}.hdf5 --run 14${run} /gpfs/group/dfc13/default/sim/icecube/dima/pisa/level5p/numu/DC12_14${run}_merged.hdf5 --run 16${run} /gpfs/group/dfc13/default/sim/icecube/dima/pisa/level5p/nutau/DC12_16${run}.hdf5 --outdir /gpfs/group/dfc13/default/sim/icecube/dima/pisa/level5p/Matt_mc_bdt_score_saved/make_events_file_output_with_fluxes_GENIE_Barr/
    #python /storage/home/fxh140/work/pisa/pisa/i3utils/make_events_file_all_without_genie.py --det "deepcore" --proc "5digit" --nue /gpfs/group/dfc13/default/sim/icecube/dima/pisa/level5p/nue/DC12_12${run}.hdf5 --numu /gpfs/group/dfc13/default/sim/icecube/dima/pisa/level5p/numu/DC12_14${run}_merged.hdf5 --nutau /gpfs/group/dfc13/default/sim/icecube/dima/pisa/level5p/nutau/DC12_16${run}.hdf5 --outdir /gpfs/group/dfc13/default/sim/icecube/dima/pisa/level5p/Matt_mc_bdt_score_saved/make_events_file_output_with_fluxes_GENIE_Barr/ --data-proc-params events/data_proc_params_without_genie_all_bdt.json
    python /storage/home/fxh140/work/pisa/pisa/i3utils/make_events_file_all_without_genie.py --det "deepcore" --proc "5digit" --nue /storage/home/fxh140/scratch/Level5p_IC86.2013_genie_nue.12${run}.hdf5 --numu /gpfs/group/dfc13/default/sim/icecube/dima/pisa/level5p/numu/DC12_14${run}_merged.hdf5 --nutau /gpfs/group/dfc13/default/sim/icecube/dima/pisa/level5p/nutau/DC12_16${run}.hdf5 --outdir /gpfs/group/dfc13/default/sim/icecube/dima/pisa/level5p/Matt_mc_bdt_score_saved/make_events_file_output_with_fluxes_GENIE_Barr/ --data-proc-params events/data_proc_params_without_genie_all_bdt.json
    hd5dir=/gpfs/group/dfc13/default/sim/icecube/dima/pisa/level5p/Matt_mc_bdt_score_saved/make_events_file_output_with_fluxes_GENIE_Barr/
    python add_stuff_to_event_file.py -fp ${hd5dir}events__deepcore__IC86__runs_12${run}1-12${run}3,14${run}1-14${run}3,16${run}1-16${run}3__proc_v5digit__unjoined.hdf5 -t ~/work/pisa/pisa/resources/settings/template_settings/DC12_event_15syst_2.5yr_no_prior_5digit_600_blind_fit_good.json -o ${hd5dir} --add_fluxes
done

#run=600
indir=/gpfs/group/dfc13/default/sim/icecube/dima/pisa/level5p/
outdir=/gpfs/group/dfc13/default/sim/icecube/dima/pisa/level5p/0618/
mkdir -p $outdir
python make_events_file_all_with_genie.py --det "DeepCore" --proc "5digit" --nue ${indir}/DC12_12600_withGENIE.hdf5 --numu ${indir}/DC12_14600_withGENIE.hdf5 --nutau ${indir}/DC_level5b_16600_withGENIE.hdf5 --outdir ${outdir}

# bulk ice sets
#run=681
#run=683
#run=640
run=682
#python ~/work/pisa/pisa/i3utils/change_run_num_subruns_file.py -i /gpfs/group/dfc13/default/sim/icecube/dima/pisa/level5p/nutau/Level5p_IC86.2013_genie_nutau.16${run}.hdf5
#python ~/work/pisa/pisa/i3utils/change_run_num_subruns_file.py -i /gpfs/group/dfc13/default/sim/icecube/dima/pisa/level5p/numu/Level5p_IC86.2013_genie_numu.14${run}.hdf5
#python ~/work/pisa/pisa/i3utils/change_run_num_subruns_file.py -i /gpfs/group/dfc13/default/sim/icecube/dima/pisa/level5p/nue/Level5p_IC86.2013_genie_nue.12${run}.hdf5
# change no of files in ../resources/events/mc_sim_run_settings.json
#for run in 681 683 640
for run in 683
do
    hd5dir=/gpfs/group/dfc13/default/sim/icecube/dima/pisa/level5p/Matt_mc_bdt_score_saved/make_events_file_output_with_fluxes_GENIE_Barr/
    indir=/gpfs/group/dfc13/default/sim/icecube/dima/pisa/level5p/
    echo $hd5dir
    #python make_events_file_all_without_genie.py --det "DeepCore" --proc "5digit" --nue ${indir}/nue/Level5p_IC86.2013_genie_nue.12${run}.hdf5 --numu ${indir}/numu/Level5p_IC86.2013_genie_numu.14${run}.hdf5 --nutau ${indir}/nutau/Level5p_IC86.2013_genie_nutau.16${run}.hdf5 --outdir ${hd5dir} --data-proc-params events/data_proc_params_without_genie_all_bdt.json
    #python add_stuff_to_event_file.py -fp ${hd5dir}events__deepcore__IC86__runs_12${run}1-12${run}3,14${run}1-14${run}3,16${run}1-16${run}3__proc_v5digit__unjoined.hdf5 -t ~/work/pisa/pisa/resources/settings/template_settings/DC12_event_15syst_2.5yr_no_prior_5digit_600_blind_fit_good.json -o ${hd5dir} --add_fluxes
done

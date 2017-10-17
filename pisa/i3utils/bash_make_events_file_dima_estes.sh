#0322
#python make_events_file_all_without_genie.py --det "DeepCore" --proc "5digit" --nue /storage/home/fxh140/work/muon_background/veto/hdf5_files/estes_12600_nue.hdf5 --numu /storage/home/fxh140/work/muon_background/veto/hdf5_files/estes_14600_numu.hdf5 --nutau /storage/home/fxh140/work/muon_background/veto/hdf5_files/estes_16600_nutau.hdf5 --outdir /storage/home/fxh140/work/muon_background/veto/hdf5_files/

#python add_stuff_to_event_file.py -fp /storage/home/fxh140/work/muon_background/veto/hdf5_files/events__deepcore__IC86__runs_126001-126003,146001-146003,166001-166003__proc_v5digit__unjoined.hdf5 -t ~/work/pisa/pisa/resources/settings/template_settings/DC12_event_15syst_2.5yr_no_prior_5digit_600_blind_fit_good.json -o /storage/home/fxh140/work/muon_background/veto/hdf5_files/ --add_fluxes

#0427
python change_run_num_subruns.py -i /storage/home/fxh140/work/muon_background/veto/hdf5_files/

python make_events_file_all_without_genie.py --det "DeepCore" --proc "5digit" --nue /storage/home/fxh140/work/muon_background/veto/hdf5_files/estes_nue_12600.hdf5 --numu /storage/home/fxh140/work/muon_background/veto/hdf5_files/estes_numu_14600.hdf5 --nutau /storage/home/fxh140/work/muon_background/veto/hdf5_files/estes_nutau_16600.hdf5 --outdir /storage/home/fxh140/work/muon_background/veto/hdf5_files/

python add_stuff_to_event_file.py -fp /storage/home/fxh140/work/muon_background/veto/hdf5_files/events__deepcore__IC86__runs_126001-126003,146001-146003,166001-166003__proc_v5digit__unjoined.hdf5 -t ~/work/pisa/pisa/resources/settings/template_settings/DC12_event_15syst_2.5yr_no_prior_5digit_600_blind_fit_good.json -o /storage/home/fxh140/work/muon_background/veto/hdf5_files/ --add_fluxes

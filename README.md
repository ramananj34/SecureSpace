### Current Status: 
- Acquired Datasets and trained LTSMs
- Reproduced Telemanon results
- Built DVB-S2 LDPCs
- PGD baseline attack

### Next steps:
- Null-space PGD

```
└── .gitignore
└── enviroment.yaml
└── lab_setup.sh
└── secrets.sh
└── README.txt
├── smap_msl_data/
│   └── channel_manifest.csv
│   └── channel_manifest.py
│   └── download_data.sh
│   └── inspect_data.ipynb
│   └── labeled_anomalies.csv
│   └── smap_msl_dataset_api.py
│   └── smap_msl_dataset_api_end2endtests.py
│   └── smap_msl_dataset_api_realtests.py
│   └── smap_msl_dataset_api_unit_tests.py
│   └── test/...
│   └── train/...
│   └── 2018-05-19_15.00.10/...
└── runs/...
├── telemanom_reproduction/
│   ├── VENDOR_telemanom/
│   │   └── LICENSE.txt
│   │   └── NOTICE.txt
│   │   └── __init__.py
│   │   └── aggregation.py
│   │   └── channel.py
│   │   └── errors.py
│   │   └── vendor_config.py
│   └── analyze_all_channels.py
│   └── eval_all_channels.py
│   └── ltsm_trainer.py
│   └── pipeline.py
│   └── pipeline_spot_test.py
│   └── smapmsl_data_pytorch_wrapper.py
│   └── synth_check.py
│   └── telemanom_lstm.py
│   └── train_all_channels.py
│   └── train_one_channel.py
│   └── vendor_smoke_test.py
├── ldpc/
│   └── __init__.py
│   └── codeword_enumeration.py
│   └── dvb_s2_ldpc.py
│   └── dvb_s2_short.py
│   └── tanner_graph.py
│   ├── tests/
│   │   └── __init__.py
│   │   └── exaust_tanner.py
│   │   └── test_codeword_enumeration.py
│   │   └── test_encoding.py
│   │   └── test_encoding_short.py
│   │   └── test_tanner.py
├── baseline_fgsm_pgd/
│   └── runs/...
│   └── ceil_attack.py
│   └── fgsm_pgd_attacks.py
│   └── test_attacks.py
│   └── e1_ceil.py
│   └── fgsm_pgd_attacks_viz.ipynb
├── nullspace_attack_utils/
│   └── __init__.py
│   └── frame_packing.py
│   └── gf2.py
│   └── ldpc_ops.py
│   └── projection.py
│   ├── tests/
│   │   └── dryrun_pipeline.py
│   │   └── gf2_tests.py
│   │   └── job.sbatch
│   │   └── ldpc_ops_tests.py
│   │   └── nullspace_longGE_log.txt
│   │   └── profile_nullspace.py
│   │   └── test_frame_packing.py
│   │   └── test_projection.py
│   │   └── verify_null_space_long_ge_gpu.py
│   │   └── verify_null_space_long_ge.py
│   │   └── verify_null_space.py
├── nullspace_attack/
│   └── __init__.py
│   └── e2_pilot.py
│   └── frame_ops.py
│   └── nullspace_attack_viz.ipynb
│   └── nullspace_attack.py
│   └── oracle_sweep.py
│   └── run_e2.py
│   └── weight_analysis.py
│   └── range_sensitivity.py
│   └── packed_frame_demo.py
│   ├── tests/
│   │   └── test_frame_ops.py
│   │   └── test_lift_snap.py
│   └── runs_pilot/...
│   └── runs_e2/...
│   └── runs_oracle/...
│   └── runs_e3/...
│   └── runs_range/...
```
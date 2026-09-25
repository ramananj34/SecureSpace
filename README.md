### Current Status: 
- Acquired Datasets and trained LTSMs
- Reproduced Telemanon results
- Built DVB-S2 LDPCs
- Implemented PGD baseline attack
- Implemented Null-space PGD
- Implemented Defenses
- Verified Theorems and Robustness
- Federated Learning Extension
- Demos

### Next steps:
- Write a Paper

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
│   └── runs_e1/...
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
├── nullspace_explorations/
│   └── dvb_s2_rates.py
│   └── exploration_viz.ipynb
│   └── frame_ops_rates.py
│   └── test_rates.py
│   └── run_e4.py
│   └── run_e5.py
│   └── run_e6.py
│   └── run_flat_train.py
│   └── runs_e4/
│   └── runs_e5/
│   └── runs_e6/
│   └── runs_flat/
├── theorem_1/
│   └── combined_defense.py
│   └── e8_helpers.py
│   └── run_e7.py
│   └── run_e8.py
│   └── secret_integrity.py
│   └── test_combined_defense.py
│   └── test_e8.py
│   └── test_secret_integrity.py
│   └── theorem_1_viz.ipynb
│   └── runs_e7/
│   └── runs_e8/
├── amrcc/
│   └── __init__.py
│   └── adv_train.py
│   └── amrcc_viz.ipynb
│   └── eta_w.py
│   └── counter.py
│   └── codec.py
│   └── fy_uniformity.py
│   └── keyed_permutation.py
│   └── run_e9.py
│   └── run_e11.py
│   └── run_e12.py
│   └── run_e14.py
│   └── sampling.py
│   └── fy_fast.py
│   └── operational.py
│   └── tests/
│   │   └── __init__.py
│   │   └── test_codec.py
│   │   └── test_codespace.py
│   │   └── test_counter.py
│   │   └── test_eta_w.py
│   │   └── test_keyed_permutation.py
│   │   └── test_sampling.py
│   │   └── test_fy_fast.py
│   └── runs_advtrain/
│   └── runs_e9/
│   └── runs_e9_adv/
│   └── runs_e14/
│   └── runs_e14_adv/
├── dither_expiriments/
│   └── __init__.py
│   └── dither.py
│   └── legit_cost.py
│   └── info_measures.py
│   └── sa_search.py
│   └── recovery_metrics.py
│   └── analytic_attacker.py
│   └── lstm_plausibility.py
│   └── discriminator.py
│   └── run_e10.py
│   └── qdither_consolidate.py
│   └── dither_viz.ipynb
│   └── tests/
│   │   └── __init__.py
│   │   └── test_dither.py
│   │   └── test_legit_cost.py
│   │   └── test_info_measures.py
│   │   └── test_sa_search.py
│   │   └── test_recovery_metrics.py
│   │   └── test_analytic_attacker.py
│   │   └── test_lstm_plausibility.py
│   │   └── test_discriminator.py
│   │   └── test_run_e10.py
│   │   └── test_qdither_consolidate.py
│   └── runs_e10/
├── theory_helpers/
│   └── security_game.py
│   └── __init__.py
│   └── tightness_scatter.py
│   └── tightness_scatter.png
│   └── qdither_bound_check.png
│   └── tests/
│   │   └── test_security_game.py
│   │   └── test_qdither_bound_check.py
├── prng_testing/
│   └── weak_prng.py
│   └── prng_distinguish.py
│   └── attribution.py
│   └── run_e13.py
│   └── e13_verify.py
│   └── e13_verify.json
│   └── prng_viz.ipynb
│   └── tests/
│   │   └── test_weak_prng.py
│   │   └── test_prng_distinguish.py
│   │   └── test_attribution.py
│   │   └── test_run_e13.py
│   │   └── test_e13_verify.py
│   └── runs_e13/
├── code_smoothing_cert/
│   └── certificate.py
│   └── cohen_smoothing.py
│   └── compare_norms.py
│   └── run_e14_full.py
│   └── run_cohen.py
│   └── e14_verify.py
│   └── e14_viz.ipynb
│   └── tests/
│   │   └── test_certificate.py
│   │   └── test_cohen_smoothing.py
│   │   └── test_compare_norms.py
│   │   └── test_run_cohen.py
│   │   └── test_e14_verify.py
│   └── runs_e14_full/
│   └── runs_cohen/
├── adv_subspace/
│   └── jacobian_dadv.py
│   └── theorem4_energy.py
│   └── energy_analysis.py
│   └── run_e15.py
│   └── e15_verify.py
│   └── e15_verify.json
│   └── 15_viz.ipynb
│   └── tests/
│   │   └── test_jacobian_dadv.py
│   │   └── test_theorem4_energy.py
│   │   └── test_energy_analysis.py
│   │   └── test_run_e15.py
│   │   └── test_e15_verify.py
│   └── runs_e15/
├── fl_core/
│   └── constellation.py
│   └── federated.py
│   └── grad_channel.py
│   └── dadv_param.py
│   └── theorem5_energy.py
│   └── run_fl_clean.py
│   └── run_e16.py
│   └── fl_attack.py
│   └── theorem5prime_energy.py
│   └── e16_verify.py
│   └── e16_verify.json
│   └── e16_viz.ipynb
│   └── tests/
│   │   └── test_constellation.py
│   │   └── test_federated.py
│   │   └── test_grad_channel.py
│   │   └── test_dadv_param_theorem5_energy.py
│   │   └── test_run_fl_clean.py
│   │   └── test_run_e16.py
│   │   └── test_fl_attack.py
│   │   └── test_theorem5prime_energy.py
│   │   └── test_e16_verify.py
│   └── runs_fl_clean/
│   └── runs_e16/
├── demos/
│   └── uplink.py
│   └── bp_decoder.py
│   └── link_budget.py
│   └── run_e17e18.py
│   └── e17e18_verify.py
│   └── e17e18_verify.json
│   └── demos_viz.ipynb
│   └── tests/
│   │   └── test_uplink.py
│   │   └── test_bp_decoder.py
│   │   └── test_link_budget.py
│   │   └── test_run_e17e18.py
│   │   └── test_e17e18_verify.py
│   └── runs_e17e18/
```
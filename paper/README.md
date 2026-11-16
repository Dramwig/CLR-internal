# Paper artifacts (M10c read-only cost update)

Experiments are permanently frozen after M10b. Rebuild: `.venv-paper/bin/python -I scripts/paper/make_all.py configs/M10c/paper_build.yaml`. Independent environment lock: `paper/requirements.lock.txt`.

Each product consumes hash-frozen CSVs; no model calls. Fixed trained seeds, 10000 problem/group bootstrap draws.

|Product|Script|YAML|CSV and SHA256|
|---|---|---|---|
|t1_gsm8k|`scripts/paper/make_m10b_t1_gsm8k.py`|`configs/M10b/paper/t1_gsm8k.yaml`|`data/M10b_v1/paper_inputs/main.csv` `dcb77fc2c0ed9854146461a5bfb19788f15eb7e423f29b81960a791bde93f85e`|
|t1_reader_val|`scripts/paper/make_m10b_t1_reader_val.py`|`configs/M10b/paper/t1_reader_val.yaml`|`data/M10b_v1/paper_inputs/main.csv` `dcb77fc2c0ed9854146461a5bfb19788f15eb7e423f29b81960a791bde93f85e`|
|t2_conformal|`scripts/paper/make_t2_conformal.py`|`configs/M10b/paper/t2_conformal.yaml`|`data/M10_v1/paper_inputs/conformal.csv` `3ed842626e241feb104d17c0c131564fd685246404ece6a49896230bdcd981ce`|
|t2_paired_sizes|`scripts/paper/make_t2_paired_sizes.py`|`configs/M10b/paper/t2_paired_sizes.yaml`|`data/M10_v1/paper_inputs/conformal_paired.csv` `b3c96ee070b7d91f3c7cd0a5ecba276b50ec1daba454b2d8567129ee1f7c2d08`|
|t3_shift|`scripts/paper/make_t3_shift.py`|`configs/M10b/paper/t3_shift.yaml`|`data/M9_v1/paper_inputs/shift.csv` `21b9287434721bbd630495efeca5fc0406ea6947bbb38e65868271078230a084`|
|t4_audit|`scripts/paper/make_m10_t4_audit.py`|`configs/M10b/paper/t4_audit.yaml`|`data/M10_v1/paper_inputs/communication_detection.csv` `9ec66f4eded81493f12c27480a9098429c0c2fe2c5831f06d5517c75aa178a5a`|
|t4_qa|`scripts/paper/make_m10_t4_qa.py`|`configs/M10b/paper/t4_qa.yaml`|`data/M10_v1/paper_inputs/communication_qa.csv` `5963890cac586f949fcdb61c623364d4c3a3e53db2154daa41340e19dca095ce`|
|t4_localization|`scripts/paper/make_t4_localization.py`|`configs/M10b/paper/t4_localization.yaml`|`data/M9_v1/paper_inputs/localization.csv` `139b429df229f75127bf62c83d242eaef824771e4f05468ed3f9b77e3253691c`|
|t5_cost|`scripts/paper/make_t5_cost.py`|`configs/M10b/paper/t5_cost.yaml`|`data/M10b_v1/paper_inputs/costs.csv` `681777e6d09eaeb6db5565b559382e7161dfe0035df2b1629daf7402ea51db93`|
|f1_transfer|`scripts/paper/make_f1_transfer.py`|`configs/M10b/paper/f1_transfer.yaml`|`data/M9_v1/paper_inputs/type_transfer.csv` `04e94ad186330fb145ca9bc6aaaea6cc875abb09f495eaf1f1ed5f1c2b7a1d2f`<br>`data/M9_v1/paper_inputs/source_transfer.csv` `7dddfa2ec27dabb2b45d6817d2f2664029229fd30827645fea05f32ea39ee3be`|
|f2_rq1|`scripts/paper/make_f2_rq1.py`|`configs/M10b/paper/f2_rq1.yaml`|`data/M9_v1/paper_inputs/early_exit.csv` `4ea6d9db8126d3a8c206b7f44e91733bb3aecc241c9f4fc561043bd779f37d47`<br>`data/M9_v1/paper_inputs/ranking.csv` `5ffc6776c2da3a3d5970c55c0cb5d7f9e7ddff7dc617d95eaa0aaf69ca99dadc`<br>`data/M9_v1/paper_inputs/search.csv` `477dfff06ee2bdcbf2cc303a196e481c45da11c0fc5a964e213f56c3db3f7ca7`|
|f3_robustness|`scripts/paper/make_f3_robustness.py`|`configs/M10b/paper/f3_robustness.yaml`|`data/M9_v1/paper_inputs/weak_D0.csv` `76fb6abdc7255785031cb6b2d659836e8defea88594bdc7cb222a051ce835023`<br>`data/M9_v1/paper_inputs/weak_designs.csv` `36de02b589bd3063a0814416f494da0657b65cbc8ec156bc92c8de7c0fc3f364`<br>`data/M9_v1/paper_inputs/dependent.csv` `948376557068c5bd084cd811ccd5de29ca9f4566e8ab02743a95f0418ff42dcb`|
|f3_r1_questions|`scripts/paper/make_f3_r1_questions.py`|`configs/M10b/paper/f3_r1_questions.yaml`|`data/M9_v1/paper_inputs/weak_designs.csv` `36de02b589bd3063a0814416f494da0657b65cbc8ec156bc92c8de7c0fc3f364`|
|appendix_noise|`scripts/paper/make_appendix_noise.py`|`configs/M10b/paper/appendix_noise.yaml`|`data/M9_v1/paper_inputs/noise_gpt2.csv` `3d5124db2f07d48ecd1b1901593c27c04213ae0398ce82b78074d34856e3af0d`<br>`data/M9_v1/paper_inputs/noise_llama.csv` `609b27728f9f49c487aa10b72f7eeecb185e8cf9bdaf69a05bcf782fc08bfdcb`|
|appendix_protocol|`scripts/paper/make_appendix_protocol.py`|`configs/M10b/paper/appendix_protocol.yaml`|`data/M9_v1/paper_inputs/protocol_alignment_v2.csv` `d9f6d401a5b08b33f9ac40f4c350aab7316bfed02a7cfb4861875208a66accf0`|
|appendix_dedup|`scripts/paper/make_appendix_dedup.py`|`configs/M10b/paper/appendix_dedup.yaml`|`data/M9_v1/paper_inputs/deduplication.csv` `7242af78d68b18a73a3a0f7c6b63c269db107ab6414dbf73133476b9946a1057`|
|appendix_labels|`scripts/paper/make_appendix_labels.py`|`configs/M10b/paper/appendix_labels.yaml`|`data/M9_v1/paper_inputs/label_statistics.csv` `266116dab5eb26ac9cbf79bf65f5bec5f1cfd47f6cb07fad1efbbef4d791a57f`|
|appendix_data_scales|`scripts/paper/make_appendix_data_scales.py`|`configs/M10b/paper/appendix_data_scales.yaml`|`data/M9_v1/paper_inputs/data_scales.csv` `91487c098c645781fe25f88337ec8cea4e4c1cb4e17b46713cfcc4702bca8c8c`|
|appendix_q1_clean_traj|`scripts/paper/make_appendix_q1_clean_traj.py`|`configs/M10b/paper/appendix_q1_clean_traj.yaml`|`data/M10_v1/paper_inputs/reader_val_q1_clean_traj.csv` `1e10a6fa42aa1ac676e97adaf6ee3989a14d991c68f6d9e5225a4244c3c86bb2`|
|t4_hiddenbench|`scripts/paper/make_t4_hiddenbench.py`|`configs/M10b/paper/t4_hiddenbench.yaml`|`data/M10_v1/paper_inputs/hiddenbench.csv` `72a13f31b108a686b1b31c58725f2d46f18c690df36d52a00cb75c959e8c5709`|
|appendix_per_type|`scripts/paper/make_appendix_per_type.py`|`configs/M10b/paper/appendix_per_type.yaml`|`data/M10_v1/paper_inputs/main.csv` `251d12ea3eb9b428f540557ca2d02774acc80e83bd910245ce854f37c4f46b2d`|
|f4_cross_use_transfer|`scripts/paper/make_f4_cross_use_transfer.py`|`configs/M10b/paper/f4_cross_use_transfer.yaml`|`data/M10_v1/paper_inputs/cross_use_transfer.csv` `57ecd82c794652b3bc4e01649b0cdc25afa3fee215334c2028df9d1b78941ebe`|

T1 includes GPT2 B5c Q3/Q4 (one complete training pass, r16 LoRA). Q1/Q2 and all LLaMA B5c cells are NA because excluded by scope; audit B5c remains NR. Audit claims concern the typed framework (question decomposition + public expectations + conformal thresholds), with CLR and B6 reported together; no CLR detection superiority claim. F4 transfer and HiddenBench are appendix-only; HiddenBench is too small to support a conclusion.

M9 original bytes: `data/M10_v1/m9_paper_snapshot/mapping.json`; M10 original bytes: `data/M10b_v1/m10_paper_snapshot/mapping.json`. Freeze manifests are unchanged. One pre-existing M10 pytest stdout hash differs (modified after M10 freeze, before M10b); original bytes unavailable, preserved and disclosed in preexisting_log_drift.json. All model/data/metric hashes match. Reproduction: `data/M10b_v1/reproducibility.json`; integrity: `data/M10b_v1/upstream_audit.json`; review: `reports/M10b_REVIEW.md`. TeX compilation remains unperformed (no local compiler); deterministic TeX, CSV and vector PDFs rebuild byte-for-byte.

M10c adds no training or scientific predictions. T6 raw measurement rebuild: `source scripts/m7_env.sh && .venv/bin/python scripts/m10c_measure.py configs/M10c/measure.yaml` (refuses overwrite; use a new version for remeasurement). Aggregation: `.venv/bin/python scripts/m10c_summarize.py configs/M10c/summarize.yaml`; no calibration. Cost scope and concurrency limitations are in T6 and reports/M10c_details.md.

|Product|Script|YAML|CSV and SHA256|
|---|---|---|---|
|t6_cost_precision|`scripts/paper/make_t6_cost_precision.py`|`configs/M10c/paper/t6_cost_precision.yaml`|`data/M10c_v1/paper_inputs/cost_precision.csv` `180486e76ff97283311876d5e238d4cd1265e3e167d003d86b158a9ba7da3efb`|
|t6_conformal|`scripts/paper/make_t6_conformal.py`|`configs/M10c/paper/t6_conformal.yaml`|`data/M10c_v1/paper_inputs/conformal_paired.csv` `871a2ed2fc48fd228f64826123ca530fe7aebb760e17bac993a23ffe8b0a9eb5`|

M10c original M10b paper bytes: `data/M10c_v1/m10b_paper_snapshot/mapping.json`. Current reproduction: `data/M10c_v1/reproducibility.json`; current integrity exceptions: `data/M10c_v1/upstream_audit.json` (M10 pytest stdout and two pre-existing M10b administrative files; scientific hashes intact).

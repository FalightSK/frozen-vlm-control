# Stage 2 paired video examples

These are diagnostic reruns of the already trained epoch 250 ALOHA and epoch 120 LIBERO heads. Each row compares GR00T and Qwen on the same recorded reset within its training seed. The rerun reset hash, initial image hash, policy sampling seed, and success/failure outcome matched the original final-test record for all 12 videos. ALOHA step counts also matched. The three LIBERO Qwen successes differed by 1–4 steps from the original records, so use these clips to inspect behavior, not to replace original result counts.

The ALOHA reset was selected because the winner reverses across seeds. The LIBERO start in each seed was selected as a Qwen-only success on the wine-rack task. These examples were selected to show contrasts; they do not estimate how often those behaviors occur.

| Benchmark and seed | Start | GR00T result and video | Qwen result and video |
|---|---:|---|---|
| ALOHA, seed 3 | 10000 | Failure · [video](C:/Users/SK/AppData/Local/Temp/action-omni-stage2-pilot/recordings/aloha-exp05-seed3/videos/qwen-only-10000/episode_01.mp4) | Success · [video](C:/Users/SK/AppData/Local/Temp/action-omni-stage2-pilot/recordings/aloha-exp06-seed3/videos/qwen-only-10000/episode_01.mp4) |
| ALOHA, seed 4 | 10000 | Failure · [video](C:/Users/SK/AppData/Local/Temp/action-omni-stage2-all-seeds-v3/aloha-seed4-exp05/same-reset-10000/videos/result/episode_01.mp4) | Failure · [video](C:/Users/SK/AppData/Local/Temp/action-omni-stage2-all-seeds-v3/aloha-seed4-exp06/same-reset-10000/videos/result/episode_01.mp4) |
| ALOHA, seed 5 | 10000 | Success · [video](C:/Users/SK/AppData/Local/Temp/action-omni-stage2-all-seeds-v3/aloha-seed5-exp05/same-reset-10000/videos/result/episode_01.mp4) | Failure · [video](C:/Users/SK/AppData/Local/Temp/action-omni-stage2-all-seeds-v3/aloha-seed5-exp06/same-reset-10000/videos/result/episode_01.mp4) |
| LIBERO-Goal wine rack, seed 3 | 28 | Failure · [video](C:/Users/SK/AppData/Local/Temp/action-omni-stage2-pilot/recordings/libero-exp05-seed3/wine-rack-qwen-only-28/videos/t09_canonical_ep0.mp4) | Success · [video](C:/Users/SK/AppData/Local/Temp/action-omni-stage2-pilot/recordings/libero-exp06-seed3/wine-rack-qwen-only-28/videos/t09_canonical_ep0.mp4) |
| LIBERO-Goal wine rack, seed 4 | 11 | Failure · [video](C:/Users/SK/AppData/Local/Temp/action-omni-stage2-all-seeds-v3/libero-seed4-exp05/qwen-only-wine-rack/videos/t09_canonical_ep0.mp4) | Success · [video](C:/Users/SK/AppData/Local/Temp/action-omni-stage2-all-seeds-v3/libero-seed4-exp06/qwen-only-wine-rack/videos/t09_canonical_ep0.mp4) |
| LIBERO-Goal wine rack, seed 5 | 21 | Failure · [video](C:/Users/SK/AppData/Local/Temp/action-omni-stage2-all-seeds-v4/libero-seed5-exp05/qwen-only-wine-rack/videos/t09_canonical_ep0.mp4) | Success · [video](C:/Users/SK/AppData/Local/Temp/action-omni-stage2-all-seeds-v5/libero-seed5-exp06-qwen-only-wine-rack/videos/t09_canonical_ep0.mp4) |

Contact sheets show matched frames sampled at 10%, 35%, 60%, and 85% of each clip. Columns are normalized by clip duration, not synchronized by simulator step.

- [ALOHA reset 10000 across all three seeds](C:/Users/SK/AppData/Local/Temp/action-omni-stage2-all-seeds-v3/aloha_selected_cases_all_seeds.png)
- [LIBERO wine-rack examples across all three seeds](C:/Users/SK/AppData/Local/Temp/action-omni-stage2-all-seeds-v3/libero_selected_cases_all_seeds.png)

## Saved transition diagnostic videos

The outcome-stratified transition probe is now preserved under `asset/durable/libero/diagnostics/stage2_transition_probe_v5/`. Initial observations and binary outcomes matched all 12 canonical records; 10/12 reruns also matched the original step count. The videos are selected examples, not a prevalence sample.

| Task and seed | Matched start | GR00T | Qwen |
|---|---:|---|---|
| Cheese in bowl, seed 3 | 13 | [video](../../../asset/durable/libero/diagnostics/stage2_transition_probe_v5/videos/task6_seed3_groot_start13.mp4) | [video](../../../asset/durable/libero/diagnostics/stage2_transition_probe_v5/videos/task6_seed3_qwen3vl_start13.mp4) |
| Cheese in bowl, seed 4 | 11 | [video](../../../asset/durable/libero/diagnostics/stage2_transition_probe_v5/videos/task6_seed4_groot_start11.mp4) | [video](../../../asset/durable/libero/diagnostics/stage2_transition_probe_v5/videos/task6_seed4_qwen3vl_start11.mp4) |
| Cheese in bowl, seed 5 | 15 | [video](../../../asset/durable/libero/diagnostics/stage2_transition_probe_v5/videos/task6_seed5_groot_start15.mp4) | [video](../../../asset/durable/libero/diagnostics/stage2_transition_probe_v5/videos/task6_seed5_qwen3vl_start15.mp4) |
| Wine on rack, seed 3 | 28 | [video](../../../asset/durable/libero/diagnostics/stage2_transition_probe_v5/videos/task9_seed3_groot_start28.mp4) | [video](../../../asset/durable/libero/diagnostics/stage2_transition_probe_v5/videos/task9_seed3_qwen3vl_start28.mp4) |
| Wine on rack, seed 4 | 11 | [video](../../../asset/durable/libero/diagnostics/stage2_transition_probe_v5/videos/task9_seed4_groot_start11.mp4) | [video](../../../asset/durable/libero/diagnostics/stage2_transition_probe_v5/videos/task9_seed4_qwen3vl_start11.mp4) |
| Wine on rack, seed 5 | 21 | [video](../../../asset/durable/libero/diagnostics/stage2_transition_probe_v5/videos/task9_seed5_groot_start21.mp4) | [video](../../../asset/durable/libero/diagnostics/stage2_transition_probe_v5/videos/task9_seed5_qwen3vl_start21.mp4) |

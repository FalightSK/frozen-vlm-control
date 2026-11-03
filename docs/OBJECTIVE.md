# frozen-vlm-control: research objective

Do frozen features from the robot-adapted GR00T N1.7 endpoint help a newly trained action policy compared with features from Qwen3-VL-2B, its stock lineage origin?

We evaluate feature usefulness through closed-loop control with matched action-head architectures within each benchmark. Both backbones are frozen and read at language layer 16; a token adapter and flow-matching decoder are independently trained for seeds 3–5. ALOHA and LIBERO use different head sizes and action spaces.

ALOHA Transfer Cube tests visually guided bimanual transfer with a fixed instruction. Two-camera LIBERO-Goal tests instruction-conditioned manipulation and responses to familiar goal changes. Language adaptability here means behavior under the tested familiar requests, not generalization to unseen language or tasks.

The main contributions are the three-seed endpoint comparison, behavioral localization of success differences, and an assessment of validation flow loss against development rollout success. Timing, masking, and action-prediction probes provide selected clues; they do not establish a causal representation mechanism. The intermediate Cosmos-Reason2 model prevents attribution to one pretraining stage. Cross-benchmark differences cannot isolate language, cameras, or embodiment.

See [results](RESULTS.md), [project overview](../README.md), and the [manuscript](../manuscript/main.tex). The previous component-ablation framing is retained only as a [historical record](archive/historical_objective.md).

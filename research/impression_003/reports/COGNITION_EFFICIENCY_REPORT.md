# [SYSTEM_NAME] Cognition Efficiency Report

This report records the verified efficiency findings from the IMPRESSION-003 baseline and its IMPRESSION-004 continuation. The current full report and receipts are in [IMPRESSION_004_REPORT.md](../../impression_004/reports/IMPRESSION_004_REPORT.md).

## Finding

The previous 213-train-token / 156-validation-token CPU cycle used one thread and took about 28 minutes. IMPRESSION-004 instrumented the complete optimizer step. Dense full-vocabulary backward computation cost 603.57 seconds; the AdamW update over 24,576 LoRA parameters cost effectively zero. The full-vocabulary head alone took 53.76 seconds for training and 39.30 seconds for validation.

The exact supervised-position output head preserves all supervised causal next-token targets and the full vocabulary. It matches dense logits, loss and adapter gradients bit for bit on the real [BASE_MODEL] backbone. In a matched full cycle it reduced wall time from 1558.85 seconds to 1050.94 seconds, a measured 32.58% improvement. Backward time fell 60.78%, and peak sampled RSS fell from 1.725 GB to 1.530 GB. Fresh-process reload passed, frozen backbone fingerprints remained unchanged, and the dense and sparse paths serialized the same adapter hash.

The two-thread sparse candidate took 1067.10 seconds and did not improve over one thread. A guarded CUDA output-head / CPU suffix cycle took 908.49 seconds, 13.55% faster than sparse CPU1, and passed fresh-process verification. The selector retained it using an explicit `0.0001` held-out-loss regression bound; the measured difference versus CPU1 was `0.00005174`. This profile optimizes execution placement within the existing graph. It does not move, resize, reconnect or dynamically repartition neural blocks.

## Generation limitation

Archived HF eager CPU generation calls returned one non-EOS token after 524.6–776.7 seconds despite `max_time=180`. The first full forward/prefill blocked before the time stopping criterion could be checked. Eight threads were only 4.3% faster in one sample. This is a runtime throughput problem; there is no evidence that EOS caused the one-token output.

## Learning limitation

One bounded update improved training loss but regressed family-heldout loss by 0.00133634. No raw symbolic generation improvement or reusable expression gain has been demonstrated. The sparse path optimizes the same update and produces the same adapter; it does not establish new cognitive capability.

## Artifacts

- Full operation profile: [training_profile.json](../../impression_004/training_profile.json)
- Hardware profile: [hardware_profiles.json](../../impression_004/hardware_profiles.json)
- Numerical verification: [numerical_verification.json](../../impression_004/numerical_verification.json)
- Scheduler decisions: [scheduler_decisions.jsonl](../../impression_004/scheduler_decisions.jsonl)
- Phase and resumable state: [run_state.json](../../impression_004/run_state.json)

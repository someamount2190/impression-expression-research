# IMPRESSION-004 — Adaptive Hardware Optimization & Parallel Partition Learning

**Status:** complete within the allotted runtime; measured sparse and CUDA-head execution profiles passed fresh-process verification.  
**Backbone:** [BASE_MODEL], pinned revision `392a6f57223e7ccfe6ef4ebdb2ff101a42d57364`.  
**Budget:** hard stop `2026-11-04 02:16 +08:00`; no checkpoint was overwritten.

## Executive findings

The measured bottleneck was the training backward pass through a dense full-vocabulary output projection at every sequence position, including positions excluded by the exact supervised label mask. On the matched 213-token training and 156-token held-out example, projecting only the 48 and 49 supervised positions preserves the same full vocabulary and causal cross-entropy objective. Real [BASE_MODEL] logits, loss, and LoRA A/B gradients matched dense computation bit for bit in the numerical probe. The matched complete cycle improved from **1558.85 s to 1050.94 s (32.58%)** on one CPU thread. Backward fell from **603.57 s to 236.78 s**, and train LM-head projection from **53.76 s to 12.06 s**. Peak sampled RSS fell from **1.725 GB to 1.530 GB**. Fresh-process sparse reload passed; the adapter SHA and held-out post-update loss exactly matched the dense run.

The improvement is real for the measured one-step setup. It is not evidence of a symbolic reasoning gain: train loss improved slightly, while family-held-out loss regressed by `0.00133634`. The controller used a strict zero held-out-regression bound relative to the dense run, where the two post-update held-out losses are identical; it selected the exact sparse CPU1 configuration for execution efficiency.

## Root cause of one-token generation

The archived HF eager CPU generation calls requested up to 512 new tokens with `max_time=180 s`; the actual call took `524.6–776.7 s`, returned one token ID `517` (decoded as `"{\n"`), ended without EOS, and did not hit the token limit. The eight-thread follow-up produced one token in `649.6 s`, only about 4.3% faster than the one-thread `679.0 s` sample. The evidence supports a very slow first forward/prefill on the eager CPU path: Transformers checks the time stopping criterion during generation iterations, after the blocking model forward has produced a token. The time limit therefore did not interrupt the first expensive forward. There is no evidence that EOS caused the one-token output or that the model emitted EOS. The archived request and results are in `experiments/expression_growth_001/operational_evaluation/generation_abc_run01/` and `generation_abc_run02_threads8/`.

The Ollama `[BASE_MODEL]-3b:latest` endpoint is Q4_K_M and is a different precision/runtime from raw BF16. Its archived 16-token sample does not establish trained-adapter parity. IMPRESSION-004 did not complete an identical trained-adapter HF-versus-Ollama multistep generation comparison; no generation competence claim is made.

## Training-cycle profile

| Configuration | Complete cycle | Backward | Train LM head | Validation LM head | Peak sampled RSS | Held-out loss after |
|---|---:|---:|---:|---:|---:|---:|
| Dense CPU, 1 thread | 1558.85 s | 603.57 s | 53.76 s | 39.30 s | 1.725 GB | 3.796094656 |
| Exact supervised-position CPU, 1 thread | 1050.94 s | 236.78 s | 12.06 s | 12.30 s | 1.530 GB | 3.796094656 |
| Exact supervised-position CPU, 2 threads | 1067.10 s | 241.42 s | 12.24 s | 12.32 s | 1.531 GB | 3.796094656 |
| Exact sparse CUDA head / CPU suffix, 1 thread | 908.49 s | 132.96 s | 0.217 s | 0.307 s | 1.799 GB | 3.796146393 |
| Controller-selected CUDA head / CPU suffix repeat | 902.05 s | 127.90 s | 0.215 s | 0.314 s | 1.798 GB | 3.796146393 |

Both completed cycles used the same pinned raw BF16 frozen backbone, 213/156 token inputs, 48/49 supervised positions, one AdamW step, one-thread dense / sparse execution as listed, and the same initial verified run02 adapter (SHA256 `589ce0f66f9f72530521662a839daab616014ffede3273b51b2c1373fdeda8b9`). Dense and sparse completed adapter file SHA256: `1261d4226a6ff146998388bff43896c631dd2b074c3b49cb1a578b3130125a3a`. Frozen backbone hashes remained unchanged on fresh-process verification.

The train backward reduction is 60.78%, the training head reduction is 77.58%, and complete-cycle wall reduction is 32.58%. Prefix work still takes roughly 711 seconds across train and validation, so it is now the largest remaining cycle floor. Activation reuse is in-process only; persistent cross-run prefix caching has not been demonstrated.

The CPU2 sparse candidate completed at 1067.10 s, 1.54% slower than CPU1, with the same adapter hash and held-out post-update loss. It was rejected by the selector. Two controller-executed CUDA-head / CPU-suffix cycles completed at `908.49 s` and `902.05 s` (mean `905.27 s`, 0.71% relative difference), an average 13.86% faster than sparse CPU1 and 41.93% faster than dense CPU. Both produced the same adapter SHA and held-out loss, and both passed fresh-process verification. The BF16 CPU/CUDA numerical probe passed declared tolerances. Their held-out loss was `0.00005174` higher than CPU1 sparse; the controller accepted this under the configured `0.0001` bound. The embedding allocation was 788,529,152 bytes; the highest sampled nvidia-smi device-memory use was 1,077 MiB (spot sample, not a continuous high-water counter). Free VRAM after embedding load was 2,662,753,895 bytes. The preflight 20% VRAM reserve and extra 256 MiB guard remained satisfied.

## Hardware and resource use

- CPU: AMD Ryzen 5 7535HS, 6 physical / 12 logical cores. Dense and sparse CPU1 cycles used one PyTorch intra-op thread; process CPU time was 1528.38 s and 1027.58 s respectively. CPU2 sparse took 1067.10 s and was rejected as slower.
- RAM: 24.92 GB total. Sparse run minimum available RAM was 13.18 GB and peak process RSS was 1.53 GB, comfortably beyond the 20% reserve requirement.
- GPU: RTX 2050, 4 GB VRAM. CPU cycles left the GPU idle (0% utilization, 0 VRAM observed). A historical single q_proj probe measured CPU `0.112 s` versus CUDA `0.164 s` including transfer, so the controller did not favor CUDA. No full-cycle CUDA or hybrid run was measured; no GPU gain is claimed.
- Storage: C: had approximately 131 GiB free at initial profiling. Training-cycle I/O counters are recorded in the run receipts, but the machine did not receive a sustained sequential-storage benchmark. Model weights were largely served from the OS cache in the short initial sample.
- Thermal: GPU temperature was observed at 47 °C in initial profiling. CPU thermal sensors were unavailable because the ACPI query returned access denied. No CPU temperature is inferred.
- Protection: preflight reserves at least 20% system RAM plus explicit headroom; processes were run sequentially, and no unrelated process was stopped.

## Adaptive execution scheduling and structural boundary

The existing `experiments/impression_002/controller/efficiency_controller.py` was reused. `tools/select_measured_configuration.py` turns completed, fresh-process-verified cycle receipts into versioned `BlockCostProfile` inputs and calls the existing selector. It selected the CUDA LM-head / CPU-suffix execution profile with a `0.0001` maximum held-out-loss regression bound; its measured loss difference from CPU1 was `0.00005174`. Evidence and receipts are in `scheduler_decisions.jsonl`, with the retained execution profile in `selected_configuration.json`.

This selection is based on complete cycle measurements and quality, not a fixed device preference. `tools/execute_selected_configuration.py` launches one bounded run from the selected profile, refuses to overwrite a run directory, and records the choice. It does not recursively launch work.

This is **execution scheduling for the inherited partition graph**, not [SYSTEM_NAME]'s structural self-reorganization. The current trainer still uses fixed `PREFIX_PARTITIONS` through layers 0–26, a layer-27 suffix, and a fixed layer-27 q_proj LoRA target. No code in this experiment autonomously moves or resizes neural blocks, changes partition boundaries, rewires dependencies, or promotes new trainable regions. The optimizer chooses CPU1 sparse versus CUDA-head/CPU-suffix placement for a measured cycle; it does not choose new neural block structure. This distinction is a remaining architecture task, not a demonstrated feature.

The future structural controller needs versioned candidate partition graphs and parameter snapshots, explicit inter-block activation/dependency contracts, relocation and resize cost estimates, frozen-backbone compatibility checks, and transactional reintegration with held-out validation. Each candidate graph should be profiled and compared end to end before promotion; this experiment does not implement those operations.

## Numerical integrity and learning outcome

- Dense-vs-sparse numerical probe on the real checkpoint: exact equality for selected full-vocabulary logits, causal loss, and LoRA A and B gradients. See `numerical_verification.json` and `numerical_checks/sparse_supervised_reference.json`.
- Deterministic train-forward reuse probe: exact logits, loss, and gradients. See `numerical_checks/gradient_reuse_reference.json`.
- Dense and sparse complete cycles produce the same adapter file hash and exact same held-out post-update loss. The held-out delta is slightly worse than the pre-update value; the one-step update is not an accepted symbolic learning improvement.
- CUDA-vs-CPU numerical probe passed within explicitly recorded BF16 tolerances: selected-logit maximum absolute difference `0.0625` (relative L2 `0.000139`), loss difference `2.29e-5`, and LoRA gradient relative L2 differences `0.00138` (A) / `0.00278` (B). Both complete CUDA cycles also passed fresh-process reload and frozen-backbone integrity checks. A strict first attempt correctly rejected these nonzero backend differences; the retained tolerance values and failed attempt remain in the numerical receipts.
- The existing 41 primitive library was preserved. No new reusable expression was validated. Sealed evaluation was not used for training or configuration selection.
- Fresh-process adapter reload verifies serialized adapter identity, logits, frozen partition fingerprints, and cleanup; both completed paths passed.

## Parallel partition learning and supervision

Routine resource tracking used the existing deterministic JSONL telemetry and stage timers. Two lower-cost read-only reviewers previously audited controller and partition integration. They did not launch training or delegate further. Agent token costs are not exposed by this runtime. Configurable limits are in `config.yaml` (`max_active_analysis_agents: 2`, one GPU training job, no recursive delegation).

Only layer-27 q_proj has been independently updated in the IMPRESSION-002E training pathway. No second independently trainable partition update or integration compatibility test was completed. The multi-partition acceptance level is therefore **not met**; monitoring concurrency is not treated as neural training parallelism. Runs in this experiment were sequential.

## Acceptance status

| Level | Result | Evidence / limitation |
|---|---|---|
| 1 — Bottleneck attribution | Passed for measured stages | Timed prefix, attention, MLP, RMSNorm, full/selected head, loss, backward, optimizer, serialization and full cycle. Tokenization and cache lookup remain untimed. |
| 2 — Numerical preservation | Passed for sparse head | Bitwise logits/loss/gradient probe; same adapter hash and held-out output on matched full update; fresh-process verification passed. |
| 3 — Hardware optimization | Passed for CPU sparse path | Reproducible matched one-cycle comparison: 32.58% less wall time and lower RSS on CPU1. |
| 4 — Adaptive scheduling | Partially passed | Existing controller selects and executes a repeatable measured CUDA-head/CPU-suffix profile within a declared quality bound. Structural block reorganization remains unimplemented. |
| 5 — Parallel partition learning | Not met | Only one partition target trained; no independent second partition update/integration was validated. |
| 6 — EXPRESSION held-out improvement | Not met | Family-heldout loss rose `0.00133634`; no generation or expression reuse improvement was demonstrated. |

## Reproduction

From the repository root, with dependencies installed and the pinned local snapshot available:

```powershell
$env:PYTHONPATH='.'
python experiments/impression_004/tools/verify_reused_forward_gradients.py
python -m experiments.impression_002e.streaming_training.train_and_verify train `
  --run-dir experiments/impression_004/partition_training_receipts/cpu1_sparse01 `
  --metadata experiments/impression_003/receipts/smoke_training_metadata.json `
  --initial-adapter experiments/expression_growth_001/impression_training/run02/adapter/adapter.safetensors `
  --sparse-supervised-logits --cpu-threads 1
python -m experiments.impression_002e.streaming_training.train_and_verify verify `
  --run-dir experiments/impression_004/partition_training_receipts/cpu1_sparse01
python experiments/impression_004/tools/select_measured_configuration.py
python experiments/impression_004/tools/select_measured_configuration.py --max-quality-regression 0.0001
python experiments/impression_004/tools/execute_selected_configuration.py --run-name cuda_head_reproduction
python -m experiments.impression_002e.streaming_training.train_and_verify verify `
  --run-dir experiments/impression_004/partition_training_receipts/cuda_head_reproduction
```

For the selected CUDA head path, set `$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'` before starting Python. The trainer sets this default before CUDA initialization as well, because deterministic CuBLAS requires a workspace configuration.

The trainer resolves the pinned local snapshot using IMPRESSION-001 provenance. It refuses to overwrite an existing run directory. See each `training_receipt.json`, `training_cycle_profile.json`, resource telemetry, and `fresh_process_verification.json` for hashes and complete values.

## Remaining work

Run the IMPRESSION-002E regression suite; finalize run state and report. Do not resume extensive EXPRESSION updates without an agreed held-out regression bound and an evaluation that measures raw model-generated symbolic plans. Structural block reorganization, persistent prefix caching, multi-partition update integration, and trained-adapter multistep generation remain follow-up work unless independently verified within the budget.

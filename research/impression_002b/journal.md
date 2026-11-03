# IMPRESSION-002B Journal

## Scope and constraints

- Continue from IMPRESSION-001B and IMPRESSION-002. Pinned [BASE_MODEL] BF16 revision `392a6f57223e7ccfe6ef4ebdb2ff101a42d57364`; structural GGUF registry has 255 entries.
- Maximum runtime 8 hours; one lower-cost executor (`gpt-6-luna`) at a time; no recursive/parallel delegation. Token accounting is unavailable and is not inferred.
- Preserve 20% physical RAM reserve. RTX 2050 has 4 GiB VRAM, less than the 6.43 GB BF16 parameter lower bound; full-model GPU execution is not feasible.
- A pre-existing unrelated full pytest process was observed and left untouched. No 002B training was active at experiment start.

## Phase 0 — Baseline

- Baseline details in `reports/STREAMING_BASELINE.md`.
- Previous distributed transformer code is demo sized and does not use [BASE_MODEL] checkpoint tensors.
- The full-model AutoModel load had failed preflight in IMPRESSION-002. Current experiment avoids creating an AutoModel and uses the index-backed residency manager.

## Phase 1 — Residency manager

- Added `residency/manager.py`: maps registry tensor names to exact HF Safetensors names, verifies pinned revision, shape and BF16 dtype, checks active system RAM/VRAM reserve, permits bounded resident partition count, hashes loaded tensor bytes, tracks explicit residency states, and clears its owned storage on release.
- A retained execution view initially held tensor references after release. Corrected to a read-only proxy over the owned mapping; release invalidates it. Added regression coverage.
- Probe executed three real layer-27 partitions (RMSNorm, Q projection, K projection) sequentially on CPU. Each ended `UNLOADED`; manager resident set ended empty. Largest observed RSS was 391,397,376 bytes for Q loaded, decreasing to 372,756,480 after release. Process RSS does not return exactly to baseline because native allocator/cache behavior is measured separately from manager residency.
- Probe evidence: `benchmarks/residency_probe.json`.

## Phase 2 — Transformer block and full forward

- Layer 27 streamed all nine registered partitions in order and matched the official Transformers 4.56.2 `LlamaDecoderLayer` on identical BF16 input, mask, positions, RoPE settings and pinned weights: max/mean absolute diff 0.0. Stream and reference ran in separate processes.
- The full-forward runner completed all 28 [BASE_MODEL] layers on an 8-token prompt, using 255 operations across 254 distinct registered partitions; the tied embedding was reused for output logits. At most one registry partition was resident at any time. It produced finite `[1, 8, 128258]` logits without creating an AutoModel.
- Full-run preflight passed before any parameter tensor was materialized: available RAM 12,790,210,560 bytes; reserve 4,983,970,201 bytes; largest partition embedding 788,017,152 bytes; required available RAM including activation estimate and 512 MiB headroom 6,310,431,129 bytes.
- Full pass: 87.479 seconds; aggregated measured loading 10.122 s, component operations 16.669 s, release/GC 30.440 s; peak sampled process RSS 1,849,311,232 bytes; minimum sampled system available RAM 12,788,379,648 bytes. CPU only, no GPU computation.
- The actual-prefix layer-27 output matched an official Transformers block on the exact captured layer input with max/mean diff 0.0. An independent full-model reference was not run; full-model logit parity remains unresolved.
- Full-forward evidence, reproducible prompt/token IDs and logits are in `benchmarks/full_forward_cpu_streamed.json`, `benchmarks/full_forward_preflight.json`, and `references/`.

## Phase 3 — Streamed selected-region training

- One rank-4 LoRA update was executed for registered layer-27 Q projection using a 16-token causal-LM example from `trace_edge_0035`; the held-out `trace_schema_0034` record was also captured. Adapter and pre/post logits are preserved in `checkpoints/streamed_lora_layer27_qproj_run02/`.
- From saved logits, the training record CE moved 7.881416 to 7.881065; held-out CE moved 6.622949 to 6.623010. The latter is a small regression, so task-quality improvement is not established.
- Adapter file fingerprint independently matches metadata (`195ca27851660551ed8e639e91c9ab727ed509e5e5c85697c399080124332850`). This verifies saved bytes only, not model reload or recomputation.
- This is a measured training-feasibility improvement for one bounded update: the IMPRESSION-002 full-model preflight had required about 16.38 GB against 12.97 GB available and did not start; 002B streamed training preflight required 6.78 GB against 12.83 GB available and the adapter optimizer step ran. Fresh-process restore remains unverified.
- Fresh-process reload remains unverified. The final standalone attempt exited `3221225477` (`0xC0000005`) with a Windows access violation in Torch storage indexing during `ResidencyManager.load` at the first norm tensor. Preflight showed adequate memory; this is an unresolved native read/runtime failure. Original attempt and repair diagnostics remain preserved.
- `training_report.json` was not written because the original runner exited before report serialization. Gradient norms and optimizer timing/state telemetry are absent. Failed `run01` preflight/startup evidence is retained; no training retry was performed.

## Phase 4 — Measured CPU/CUDA placement component

- Ran the same genuine layer-27 Q projection, pinned weight, and saved real activation three times on CPU and CUDA BF16 after RAM/VRAM preflight. RTX 2050 CUDA execution was feasible; this benchmark did use the GPU, while the full 28-layer forward above did not.
- Median total operation wall time: CPU 0.176832 s; CUDA 0.063504 s (64.1% lower for this operation only). CUDA transfer cost was 0.003312 s and 19,070,976 bytes per measured repeat; peak measured VRAM allocation was 27,590,656 bytes. Scaled max output error was 0.002525, below the configured one-BF16-epsilon bound 0.0078125.
- The existing Adaptive Efficiency Controller selected CUDA for the Q projection microbenchmark. This does not demonstrate full-model acceleration or a complete heterogeneous CPU/GPU schedule. `benchmarks/layer27_q_placement_comparison.json` and `references/layer27_q_placement_outputs.npz` preserve results.

## Final status and tests

- Added artifact-only regression tests for the full streamed forward, exact layer-27 reference parity, CPU/CUDA scheduler choice, saved training artifacts, and the explicit unresolved fresh-process reload state. The tests do not execute training, GPU kernels, or the verifier.
- Full streamed inference succeeded with max one resident registered partition and 20% RAM reserve maintained. Independent full-model logits parity is unresolved. The highest fully verified rung is Level 1 (bounded residency); Level 2 has an executed full forward but not independent end-to-end parity. Level 3 update artifacts exist but reload and telemetry acceptance are incomplete. Level 4 heterogeneous full-model execution and Level 5 broad adaptive efficiency are not demonstrated.
- Component efficiency was measured only for one fixed layer-27 Q projection: CUDA median end-to-end time was 64.1% lower over three repeats. No full-model efficiency gain was measured.
- Final commands are recorded in the handoff report; no broad repository test suite was run.
- Final checks: `python -m pytest experiments/impression_002b/tests -q` → 9 passed; `py_compile` over all 14 IMPRESSION-002B Python files → passed. These checks read artifacts and compile syntax; they did not rerun model workloads.

# IMPRESSION-004 Journal

## 2026-11-03 — Recovery workspace initialized

- Rechecked active work before creating this workspace. IMPRESSION-003 `smoke_1thread` completed one optimizer update; its fresh-process verification is still running. Latest observed status: `fresh_process_verification_in_progress`, recomputing the pinned raw BF16 prefix. Do not interrupt or overwrite that run.
- Checkpoint: `[BASE_MODEL_REPOSITORY]`, revision `392a6f57223e7ccfe6ef4ebdb2ff101a42d57364`.
- Preserved existing EXPRESSION-GROWTH-001 run02 and IMPRESSION-002E run03 artifacts. The new workspace owns only IMPRESSION-004 outputs.
- Earlier measurements identify a one-thread CPU path and five dense suffix passes, but detailed suffix-stage timing was absent. IMPRESSION-004 will instrument that path before optimization.
- Completed Ollama Q4_K_M prompt check: 16 generated tokens at 43.6 tok/s after 14.4 s cold load; this is a different precision/runtime from pinned BF16 and is not a numerical match.
- EXPRESSION 213-token train / 156-token disjoint-validation smoke update took about 28 minutes end-to-end for training. Loss moved up slightly on both train (+0.00203085) and held-out (+0.00008845); this is not evidence of learning improvement. Adapter file SHA256: `b0ff3d2aafb015055953b1fc204133770cb6494bc97f3fbbebc0a28669de7253`. It was initialized from the existing deterministic zero-output adapter rather than the earlier EXPRESSION run02 adapter and is treated as a benchmark branch, not a resume checkpoint.
- Smoke preflight estimated 1.48 GB peak working set and required 7.00 GB available; system had about 13.3 GB available. CPU-only, one-thread, 11 suffix partitions resident. RTX 2050 stayed idle; no temperature sensors for CPU were available (ACPI query denied).
- Agent work: prior read-only generation/hardware/training bottleneck analyses completed. Current IMPRESSION-004 controller and partition audits are bounded, read-only tasks; no agent is running inference/training or delegating recursively.
- Controller and partition audits completed: `experiments/impression_002/controller/efficiency_controller.py` is an existing safe selector that can compare measured, comparable block profiles and reject infeasible/regressive candidates, but it does not launch complete training cycles. The current CUDA integration profiles only layer-27 q_proj; it records one run per placement and uses CPU as the measured winner (0.112s CPU vs 0.164s CUDA including transfers). Preserve and extend this controller rather than replacing it.
- No full-cycle GPU/hybrid gain or independently trained multi-partition integration has been verified. The IMPRESSION-002C full-pass attempt failed before model execution; do not treat it as evidence for GPU scheduling.
- The small smoke run's end-to-end time per train+validation token is approximately 4.55 seconds, near the historic long run's 4.65 seconds. This is not a matched benchmark, but confirms the dense CPU baseline is not materially more efficient.
- At 18:30 local the fresh verifier has completed training-prefix recomputation and is running the suffix logits check. No competing compute job is active.

## Next actions

1. Finish fresh-process verification without starting another training workload.
2. Extend the existing Adaptive Efficiency Controller with run-level complete-cycle profiles and retained decisions; do not duplicate its feasibility policy.
3. Add stage timers for complete training cycle, including LM head, backward, optimizer and serialization.
4. Benchmark an exact supervised-position/chunked vocabulary loss against dense forward loss and gradients.
5. Compare a bounded CPU thread set and GPU feasibility using end-to-end costs and memory preflight.
6. Resume from a verified EXPRESSION adapter only after a faster path and generation/reload behavior are validated.

## 2026-11-03 — Complete-cycle attribution and exact sparse head

- The latest dense run `partition_training_receipts/cpu1_resume01` finished and passed fresh-process adapter reload, with unchanged frozen-backbone hashes. It resumed from the earlier verified EXPRESSION run02 adapter. One update reduced train loss by 0.00254679 but increased family-heldout loss by 0.00133634, so it remains a trial result, not a promoted learning checkpoint.
- Complete-cycle stage timing attributed the dominant cost to the 213-token train suffix backward pass (603.57 s). The dense LM head cost 53.76 s for train and 39.30 s for validation; AdamW cost was effectively zero. Prefix train+validation forward remained roughly 647 s, so projection-only optimization cannot remove that floor.
- Implemented an opt-in exact sparse supervised-position output projection in `experiments/impression_002e/streaming_training/train_and_verify.py`. It performs the same full-vocabulary projection only at shifted causal positions with supervision; the causal objective, labels, averaging, frozen BF16 weights and adapter update remain intact. Dense remains the default and prior receipts remain readable.
- Numerical comparison on the actual [BASE_MODEL] 3 checkpoint and historical adapter showed bitwise-identical selected dense logits, causal loss, and both LoRA gradients on a 16-token probe (15 supervised positions). Receipt: `numerical_checks/sparse_supervised_reference.json`.
- A matched sparse CPU1 cycle completed from the same historical adapter and dataset. It measured 1050.945 s vs 1558.851 s dense (32.58% less wall time); backward 236.78 s vs 603.57 s; train head 12.06 s vs 53.76 s. Peak sampled RSS fell from 1.725 GB to 1.530 GB. Both runs produced the same adapter file SHA256. The heldout delta remains +0.00133634, requiring explicit regression-bound policy before selecting it for future learning.
- Sparse fresh-process reload verification is running by recomputing the raw-BF16 prefix, with no concurrent training job. Until it passes, the sparse configuration is measured but not retained as the selected safe configuration.
- Multi-agent usage remains bounded: two low-cost read-only reviewers completed earlier; no subordinate is running expensive workloads. Existing telemetry collectors provide routine resource observations. Agent-token accounting is not exposed by the runtime.
- The two-thread exact-sparse cycle completed and passed its own fresh-process verification. Its wall time was 1067.10 s (1.54% slower than CPU1), backward 241.42 s, and it produced the same adapter SHA and heldout loss. The controller rejected it and retained CPU1.
- Began a separately preflighted CUDA output-head probe. It runs only the BF16 embedding/LM-head matrix on RTX 2050 while prefix and suffix computation stay on CPU; the probe compares full-vocabulary logits, loss and LoRA gradients against the CPU dense reference. GPU head availability is capped by a 20% VRAM reserve plus 256 MiB explicit headroom. No full GPU training will run unless this numerical probe passes.
- Architectural clarification: the user correctly noted that selecting a device placement is not [SYSTEM_NAME]/IMPRESSION structural self-reorganization. This experiment's measured selector changes only cycle execution placement under the existing registered graph. The fixed layer0–26 prefix, layer27 suffix, and q_proj adapter target were inherited and remain unchanged. No dynamic block movement, resizing, rewiring, partition-boundary search, or multi-partition integration is claimed; these are future architecture work and are called out in the final report.

# IMPRESSION-002B Final Report

## Outcome

The bounded partition manager and real CPU streamed inference worked within the memory reserve. The complete 28-layer forward produced finite logits, and the exact layer-27 block output matched the official Transformers block on the activation captured from the full run. Independent full-model logit parity was not produced. Streaming also passed the bounded training preflight and executed one real LoRA Q-projection update after IMPRESSION-002's full-model preflight had rejected the available RAM budget; the adapter and logits were saved. Fresh-process recomputation remains unverified after a Windows `0xC0000005` access violation, and gradient/optimizer timing telemetry was not persisted. The GPU was used only for a three-repeat Q-projection microbenchmark, not for the full forward.

## Implementation evidence summary

1. **Can the manager load individual registered tensors without loading the checkpoint?** Yes. It extracts named tensors from the pinned Safetensors index and validates registry identity, revision, name, shape, dtype, and content. Real layer-27 norm, Q, and K partitions were sequentially loaded and released.
2. **Can it bound residency and release tensors?** Yes in the tested path: one registered partition resident at most, final resident set empty. The release-view regression test checks that a yielded view is invalidated too.
3. **Can a genuine [BASE_MODEL] block be streamed?** Yes. Layer 27 ran from its nine registered partitions using pinned [BASE_MODEL] BF16 weights.
4. **Does the streamed block match an independent implementation?** Yes for the tested block. Official Transformers `LlamaDecoderLayer` parity was exact (max and mean absolute difference 0.0) on identical input and settings. The full-run layer-27 stage was also exactly equal to the official block on the captured actual-prefix activation.
5. **Did the full 28-layer forward complete?** Yes, on CPU, for the deterministic 8-token sequence. It executed 255 registered-weight operations over 254 unique partitions, at maximum concurrency one, and produced finite logits shaped `[1, 8, 128258]`.
6. **Is complete-model numerical parity established?** No. No independent full-model reference forward was run. Full-model logit parity remains unresolved; the one-block parity result cannot establish full-model parity.
7. **What were the full-forward memory measurements?** Peak sampled process RSS was 1,849,311,232 bytes; minimum available system RAM was 12,788,379,648 bytes on a 24,919,851,008-byte system. The 20% reserve check passed. Full forward used CPU and no model VRAM.
8. **What were the full-forward timings?** Wall time 87.479 s; aggregated partition load 10.122 s; component compute 16.669 s; release/GC 30.440 s. These are measured end-to-end run figures, not a speedup comparison.
9. **Was the full-model forward run on GPU?** No. The full 28-layer pass was CPU-only. RTX 2050 was used only in the separate Q-projection component benchmark.
10. **Was an actual task-data optimizer update performed?** Yes, one rank-4 LoRA update on layer-27 `q_proj`, using a 16-token causal-LM sample from `trace_edge_0035`; `trace_schema_0034` was used as a disjoint held-out record. Adapter and before/after logits were saved.
11. **Did training loss improve?** Saved-logit causal CE for the training record moved from 7.8814157 to 7.8810652. Held-out CE moved from 6.6229493 to 6.6230100. This is a tiny train improvement and tiny held-out regression; task-quality improvement is not established.
12. **Was the saved adapter independently fingerprinted?** Yes. Safetensors content fingerprint `195ca27851660551ed8e639e91c9ab727ed509e5e5c85697c399080124332850` matches the adapter metadata. This validates file identity only.
13. **Did a fresh process reload and recompute logits successfully?** No. The last standalone diagnostic exited with code `3221225477` (`0xC0000005`) and no stdout. Faulthandler recorded a Windows access violation in Torch storage indexing while `ResidencyManager.load` read the first norm tensor. RAM preflight was feasible. The verifier failure artifact and earlier failed attempts are preserved; reload remains explicitly unverified.
14. **Were gradient and optimizer measurements retained?** No. The training runner exited before writing `training_report.json`; gradient norms and optimizer timing/state telemetry are absent. No training rerun was made.
15. **Did streaming eliminate the earlier RAM preflight blocker?** It eliminated the blocker for this bounded run. IMPRESSION-002's conservative full-model CPU preflight required about 16.38 GB against about 12.97 GB available and did not start training. The streamed one-step training preflight required 6.78 GB including reserve/headroom against 12.83 GB available; the optimizer update ran and saved an adapter. This is a measured **training-feasibility improvement** for one late-layer adapter update, not proof of successful fresh-process reload or full-model training acceptance. The 28-layer streamed CPU inference also completed within reserve.
16. **Was CPU versus CUDA placement measured on the same component?** Yes. A genuine layer-27 Q projection with the same checkpoint weight and saved real activation was repeated three times per placement. Preflight passed for both RAM and VRAM.
17. **What efficiency difference was measured?** Median total operation wall time was 0.176832 s on CPU and 0.063504 s on CUDA, about 64.1% lower for this isolated Q projection. CUDA transfer cost was 0.003312 s and 19,070,976 bytes per repeat. Peak CUDA allocation was 27,590,656 bytes. Output scaled max error was 0.002525, within the configured 0.0078125 one-BF16-epsilon bound. This is not a full-model speedup.
18. **What did the Adaptive Efficiency Controller select?** It selected CUDA for this Q-projection configuration based on the three measured repeats, lower total cost, and quality error within the configured bound. No full-model placement schedule or reusable broad configuration was validated.
19. **What remains before broader claims or deployment?** Resolve the native Safetensors/Torch read access violation in a fresh process, then verify adapter reload and recomputed logits; preserve gradient, optimizer, and timing telemetry in a report; run an independently credible complete-model reference if resources permit; and only then evaluate full-model CPU/GPU placement or adaptive scheduling. There is no full-model efficiency-gain measurement in this experiment.

## Direct answers to the morning report

1. Genuine [BASE_MODEL] tensor partitions were independently loaded from the pinned Safetensors index without constructing the full model; layer-27 norm, Q, and K were loaded/executed/released sequentially.
2. The full pass made 255 partition operations across 254 distinct partitions (the tied embedding was used twice); the separate full layer-27 check streamed 9 partitions.
3. Yes. The CPU pass executed all 28 layers and produced finite `[1, 8, 128258]` logits.
4. Layer-27 reference agreement was exact: max/mean absolute error 0.0 on both the fixed block input and the full-pass captured input. Full-model logit parity is unresolved.
5. Full-pass maximum sampled process RSS was 1,849,311,232 B; training was observed near 2.11 GB RSS, but its peak was not persisted. Minimum sampled system available RAM in full inference was 12,788,379,648 B.
6. The full forward and training used 0 VRAM. The Q-projection CUDA microbenchmark peaked at 27,590,656 B allocated VRAM.
7. Yes: the RTX 2050 ran the real layer-27 Q projection. It did not run the complete model.
8. Yes: one actual-data rank-4 LoRA optimizer update ran on layer-27 `q_proj` with shifted causal-LM loss.
9. The adapter was saved and its Safetensors fingerprint matches metadata. Fresh-process model reload/logit parity is unverified because of `0xC0000005` native crashes.
10. Only LoRA A/B matrices were passed to the optimizer; frozen suffix weights were used as non-trainable tensors. A persisted before/after frozen-weight fingerprint report was not produced, so independent backbone integrity proof remains incomplete.
11. For this one-step experiment, yes: full-model preflight previously needed about 16.38 GB against 12.97 GB available and rejected execution; streamed training preflight needed 6.78 GB against 12.83 GB available and one update ran. That is a training-feasibility improvement, with reload acceptance still open.
12. Full inference took 87.479 s (10.122 s partition loading, 16.669 s component computation, 30.440 s release/GC); those sub-times omit about 30.248 s of Python orchestration and other unclassified overhead. CUDA Q projection median was 0.063504 s including 0.017970 s load, 0.003312 s transfer (19,070,976 B), 0.000315 s kernel, and 0.041618 s release. Training step timing was not persisted.
13. For the matched Q projection, CUDA was faster: 0.063504 s versus CPU 0.176832 s median total (64.1% lower). No full-model speed comparison exists.
14. CUDA was the measured lower-cost placement for this fixed Q projection. Whole-model stages were only benchmarked/selected for CPU streaming; a full heterogeneous placement comparison was not performed.
15. Yes. IMPRESSION's Adaptive Efficiency Controller selected the CUDA Q-projection profile after three repeats under its measured cost and BF16-scaled output-error bounds.
16. Fresh-process native Safetensors/Torch access violation, lack of an independent full-model reference, missing persisted training gradient/optimizer telemetry, and 4 GiB VRAM prevented stronger acceptance.
17. Unimplemented: full-model reference parity, verified fresh-process adapter reintegration, full-model CPU/GPU co-scheduling, prefetch/pipelining, phone-node execution, and adaptive structural relocation/resizing/reconnection.
18. One lower-cost `gpt-6-luna` executor handled seven bounded tasks sequentially. Token usage was not available from tooling.
19. Next: resolve the Windows fresh-process tensor-read fault without retraining; validate adapter reload and logits; preserve training telemetry before any verifier step; then pursue a credible independent full-model reference and repeated end-to-end efficiency comparison.

## Demonstrated levels 1–5

| Level | Status | Evidence and limit |
|---|---|---|
| 1 — Bounded residency | **Passed** | Named genuine tensors load and release; release invalidates views; one registered partition maximum in probes and full forward. |
| 2 — Streamed inference | **Partial** | Genuine 28-layer CPU forward completed and the actual layer-27 stage matched official reference exactly. Full-model independent logits parity is unresolved, so Level 2 is not fully verified. |
| 3 — Streamed training | **Partial** | One real-data adapter update, saved adapter and logits, fingerprint matched; fresh-process logits reload is unverified and gradient/optimizer telemetry is absent. |
| 4 — Heterogeneous execution | **Not demonstrated end to end** | CUDA ran the isolated Q-projection microbenchmark only. No full streamed model combined CPU and GPU placements. |
| 5 — Adaptive efficiency | **Not demonstrated for a model** | Controller chose CUDA for the measured Q-projection microbenchmark. No broad block reorganization, full-model improvement, or reusable end-to-end policy result was measured. |

**Highest fully verified level: Level 1.** Higher-level work produced useful partial evidence described above, but its unresolved checks are not counted as full acceptance.

## Reproduction and final checks

From the repository root, the focused artifact and residency tests were run with:

```powershell
python -m pytest experiments/impression_002b/tests -q
```

The new IMPRESSION-002B Python files were syntax-compiled with `py_compile`. No broad repository suite, training rerun, GPU benchmark rerun, or fresh verifier retry was performed for finalization. Artifact tests assert the saved status—including that reload is still unverified—without launching model work.

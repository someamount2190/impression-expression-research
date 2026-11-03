# IMPRESSION-002C Final Report

## Outcome

IMPRESSION-002C continued the pinned [BASE_MODEL] implementation at revision `392a6f57223e7ccfe6ef4ebdb2ff101a42d57364` and preserved the 255-entry registry, the 002B adapter, receipts, and earlier experiment artifacts.

The experiment established a raw BF16 tensor-loading route and reproduced the saved layer-27 adapter computation from verified activation caches. It also added an explicit CPU/CUDA streamed placement runner. The runner's first CPU warmup crashed in the native loading path before any full pass completed, so the GPU run and matched end-to-end benchmark were deliberately stopped.

## Acceptance levels

| Level | Result | Evidence |
|---|---|---|
| 1 — Memory-safe persistence | **Passed** | Atomic legacy receipt and references survived a deliberately terminated verifier test. Receipt remains independent of the later verification result. |
| 2 — Streamed adapter reintegration | **Passed at the cached layer-27 boundary** | Fresh CPU process loaded BF16 weights through the raw reader, loaded the saved adapter through NumPy, and reproduced saved pre/post logits exactly for both cached samples. |
| 3 — Full training-to-reload parity | **Not passed** | The prefix was not recomputed and the historical training runner did not persist backward/optimizer telemetry. This result is not full-model training-to-reload parity. |
| 4 — GPU-enabled streaming | **Not passed in 002C** | CUDA runner and preflight exist, but no GPU child ran because the CPU warmup crashed first. The earlier isolated q-projection microbenchmark is not streamed-runtime evidence. |
| 5 — Adaptive heterogeneous optimization | **Not passed** | No matched complete CPU/CUDA run, controller selection, or end-to-end gain was measured. |

## Findings

### 1. Windows Safetensors access violation

The preserved 002B adapter file is intact: its 98,488-byte file SHA-256 is `1aa308661c400a08fe664dd147c6f56707f74e64c367d801d1739ff7f4f46558`; its tensor fingerprint matches metadata. The NumPy Safetensors backend read both adapter tensors successfully. NumPy could not interpret BF16 directly, so 002C added a read-only parser that validates the pinned revision, index, shard confinement, header, offsets, tensor shape/dtype/size, endianness, and RAM reserve, then copies only selected raw BF16 bytes through mmap/NumPy uint16 into Torch BF16.

The single pinned 6 KiB BF16 norm read succeeded. A later full streamed CPU warmup reproduced `0xC0000005`: the captured stack includes an access violation in Pandas/PyArrow initialization and another in Torch storage indexing inside `PartitionResidencyManager.load` at layer 0. The evidence does not resolve causal order or prove which native fault terminated the process. Therefore the original Torch-backed manager read issue remains unresolved; it was not retried after the warmup crash.

### 2. Saved adapter reintegration

One fresh isolated process passed preflight and evaluated the existing cached layer-27 activations using the saved adapter. It loaded 989,362,176 bytes of suffix, final norm, and tied-embedding tensors; available RAM was 13,779,054,592 bytes against a 7,047,074,201-byte requirement, including the 20% reserve and headroom. Peak sampled process RSS was 1,443,303,424 bytes; minimum sampled free RAM was 12,804,222,976 bytes.

For `trace_edge_0035` and `trace_schema_0034`, both pre-adapter and post-adapter output arrays matched their preserved references exactly (max absolute difference 0.0). The adapter fingerprint also matched its metadata. This confirms cached-boundary suffix reintegration and does not claim that the model prefix was rerun.

### 3. Training telemetry and measured costs

The 002B update was a genuine one-step layer-27 q-projection LoRA update. Its durable receipt records the adapter's initial and post-update fingerprints and the saved pre/post loss references. Its backward duration, optimizer duration, gradient norm, and optimizer state are explicitly null because the original process exited before writing those values. No values were reconstructed or inferred, and no new training was run.

Preserved 002B measurements remain component scoped: the q-projection microbenchmark median was 0.176832 seconds CPU and 0.063504 seconds CUDA, with 0.003312 seconds median CUDA transfer and 27,590,656 bytes peak VRAM. The earlier complete CPU streamed forward took 87.479 seconds with peak process RSS 1,849,311,232 bytes. These are separate measurements; they do not demonstrate that CUDA improves the complete streamed forward.

### 4. CUDA placement and controller

The additive runner executes all non-target registered operations through the existing CPU streamer and defines explicit transfers for a genuine registered layer-27 q-projection CUDA operation. Metadata preflight found 13,916,360,704 bytes RAM available with the 4,983,970,201-byte reserve; CUDA BF16 was supported with 3,461,768,807 bytes VRAM free and a 287,358,976-byte estimated Q-projection working set including headroom. Despite feasible preflight, execution did not reach the model forward: the CPU warmup process crashed at the existing manager's layer-0 tensor materialization. No CUDA child, measured CPU baseline, repeats, or controller selection followed. Static tests verify the scheduler policy/fallback and benchmark artifact shape only.

## Regression checks

- 002C focused suite: **12 passed**.
- IMPRESSION-001B parity harness, IMPRESSION-002 controller, all IMPRESSION-002B tests, and all IMPRESSION-002C tests together: **29 passed**.
- EXPRESSION integration and execution-hardening compatibility: **27 passed, 1 skipped**.
- All 12 IMPRESSION-002C Python files compiled with `py_compile`.
- No checkpoint or 002B artifact was modified by 002C.

These are the named focused suites, not a whole-repository regression pass. An unrelated repository-wide pytest process was already running and was left untouched.

## Answers to the experiment questions

1. **Crash cause:** unresolved at the native root-cause level. The 002C full-stream attempt reproduced Torch storage failure during manager loading and captured a co-occurring PyArrow/Pandas import access violation; causality is undetermined.
2. **Can the saved adapter be loaded safely?** Yes through the NumPy Safetensors adapter path used by the isolated verifier.
3. **Were adapter parameters identical after serialization?** Yes. The loaded tensor fingerprint matches metadata.
4. **Component parity?** The layer-27 cached-boundary computation through logits passed exactly for pre/post adapter outputs on both references. No independent new CUDA Q-only parity run occurred in 002C.
5. **Full streamed training-to-reload parity?** No; cached prefix activations were used rather than recomputing the full prefix.
6. **Were all original backbone parameters preserved?** No backbone mutation was performed in this verification. A full parameter-by-parameter post-run audit was not performed in 002C.
7. **Did telemetry survive verification?** Yes. Durable receipt/references survived the forced verifier exit test. Missing legacy gradient/backward/optimizer timings remain null.
8. **Forward/backward/optimizer times?** Full CPU streamed forward: 87.479 s. The legacy training receipt lacks per-update forward/backward/optimizer timings; backward and optimizer are null. The isolated q-projection microbenchmark medians are 0.176832 s CPU and 0.063504 s CUDA, not training timing.
9. **Did the RTX 2050 execute operations through the streamed runtime?** No in 002C. Previous isolated Q-projection CUDA benchmark remains component-only evidence.
10. **Which partitions ran on each device?** No partition completed in the 002C full placement attempt. The previous full forward used CPU. No CUDA partition ran through the 002C stream.
11. **Peak RAM/VRAM?** Cached suffix evaluation RSS peak 1,443,303,424 B, min system free RAM 12,804,222,976 B. Previous CPU forward peak RSS 1,849,311,232 B. Previous isolated Q-projection CUDA peak VRAM 27,590,656 B. The 002C placement preflight estimated 287,358,976 B needed but no placement runtime VRAM peak exists.
12. **Transfer overhead?** Previous isolated Q-projection CUDA median transfer was 0.003312 s. No full streamed transfer overhead was measured.
13. **Did GPU-assisted streaming improve end-to-end runtime?** Not measured.
14. **Did the controller select the best feasible configuration?** No placement comparison reached controller selection.
15. **Were configuration changes reproducible after restart?** Existing IMPRESSION-002 persistence tests passed; no new 002C runtime placement was persisted or restored.
16. **Did held-out task performance improve?** Not established. Legacy held-out CE shifted from 6.6229493 to 6.6230100, a slight worsening; this is not evidence of generalization improvement.
17. **Previous experiment regressions?** Named focused IMPRESSION suites passed; EXPRESSION compatibility passed with one skip. No checkpoint files were modified.
18. **Acceptance levels:** Level 1 passed; Level 2 passed at the cached layer-27 boundary; Levels 3, 4, and 5 not passed.
19. **Unresolved architecture constraints:** Torch-backed native loading in the existing partition manager; full-prefix parity; runtime GPU placement after native reader stabilization; optimizer telemetry; complete-path measured controller evaluation.
20. **Next experiment:** isolate and repair the Windows native failure with a raw-reader-backed residency manager or a compatible Torch/Safetensors runtime, starting with one bounded shard tensor. Then verify a CPU streamed pass before attempting one serialized CUDA-assisted pass and matched repeats.

No efficiency improvement is claimed from partitioning. The only observed acceleration is the earlier isolated Q-projection measurement, and it must not be generalized to the whole model.


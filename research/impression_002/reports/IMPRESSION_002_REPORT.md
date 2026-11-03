# IMPRESSION-002 — Adaptive Computational Efficiency

## Result

**Status: partial foundation verified; full-model acceptance criteria not met.** The full pinned [BASE_MODEL] model could not safely be loaded for training under the measured RAM/VRAM envelope, so no end-to-end model training, task-level quality comparison, or whole-model efficiency result is claimed. The implementation provides a tested measurement controller, versioned configuration store, append-only experience ledger, and one real-tensor component probe selection/restore cycle.

The measured local microbenchmark comparison selected the layer-zero attention query projection. Its one-run component-probe wall time was **6.5845 s**, compared with **14.1085 s** for the layer-zero FFN gate projection (about 53.3% lower in this isolated probe). Query peak process RSS was **488,005,632 bytes**, gate was **543,678,464 bytes**; both recorded 0 VRAM use and 0 bytes of transfer. These are measured component-probe results on different projection shapes, one run each, under a synthetic objective. They do not establish a full-model speedup or a robust repeated-run gain.

The local validation residual got worse for both probes: query **0.0003508 → 0.0080111**, gate **0.0003465 → 0.0099472**. The query residual after the probe was lower than the gate residual after the probe, which allowed the controller's configured non-regression comparison against the gate baseline. Neither probe demonstrated learning improvement over its own initial value. No task-level learning gain was measured.

## Acceptance questions

1. **Candidate blocks profiled:** Two genuine pinned BF16 [BASE_MODEL] tensors were profiled: `model.layers.0.mlp.gate_proj` (25,165,824 base parameters; 45,056 adapter parameters) and `model.layers.0.self_attn.q_proj` (9,437,184 base parameters; 24,576 adapter parameters). Both ran as isolated CPU component probes.
2. **Promising regions:** Query was cheaper in this local measurement and had lower post-probe residual than gate. This is a probe-level candidate only; it is not validated as a task-training region.
3. **Baseline performance:** No full-model training baseline was run. Component total wall times were 14.1085 s (gate) and 6.5845 s (query), including selected tensor load, warmup, local training, validation, adapter save and reload.
4. **GPU / offload:** Full BF16 parameters alone require 6,425,512,064 bytes, exceeding the RTX 2050's 4 GiB VRAM. GPU full-model execution was rejected before execution. CPU full-model conservative preflight also failed: available RAM was about 12.97 GB, while estimated model load plus the 20% reserve required about 16.38 GB. No CPU/GPU offload training was attempted.
5. **Configuration performance:** Query was faster than gate in this one-repeat local component comparison. No full-model configuration winner is established.
6. **Learning gains:** None measured. Both synthetic held-out residuals increased. The final untouched synthetic residual also increased for both candidates. No IMPRESSION task-record holdout was run.
7. **Controller selection:** The deterministic measurement-based controller selected query using lower measured total cost while meeting the configured post-probe quality comparison against gate. It is not a learned policy.
8. **Physical relocation:** None performed.
9. **Connections / capacity:** No full architecture dependencies, network connections, or distributed capacity were changed. The stored configuration describes one component probe only.
10. **Save / load / restore:** The probe query config was retained by the versioned store, then restored by a fresh store instance. Registry validation passed and restored component output SHA-256 matched the applied output.
11. **IMPRESSION-001B parity:** The CPU adapter reload behavior remains intact; the parity implementation was not changed in this work. Focused parity/regression suite results are listed below.
12. **Memory:** Local measured peak RSS was 543,678,464 bytes (gate) and 488,005,632 bytes (query); local VRAM peak was 0. These process peaks exclude loading the entire model. Full-model preflight was blocked as described above.
13. **Time saved:** No whole-model training time savings were measured. The observed 7.5240 s difference is only between two isolated, differently sized projection probes and includes setup and serialization overhead.
14. **Best configuration:** Query is retained as the current probe configuration under this narrow measurement. There is no identified best full-model training configuration.
15. **Adaptation policy:** Deterministic measured-cost policy, not learned.
16. **Execution budget:** One lower-cost `gpt-6-luna` executor was used sequentially. Token usage was unavailable from tooling. The existing full pytest process observed during the task was left running; this report does not treat it as a completed test result.

## Verification

- `experiments/impression_002/tests/test_impression_002_controller.py`: **6 passed**.
- Focused IMPRESSION-001B regression set: **43 passed, 1 skipped in 36.86 s**.
- `py_compile` on the new runner/controller/configuration modules and `training/impression_selective_training.py`: passed.
- Both successful component adapters passed serialization/reload parity with maximum output difference 0.0; base tensor fingerprints were unchanged.
- Two full-model candidate profiles are explicitly `rejected_resource_preflight` and did not start training.

## Remaining work / IMPRESSION-002 schedule

1. Revisit the full-model experiment on a host with sufficient RAM or a verified memory-mapped/offload path, first checking the complete peak-memory envelope.
2. Run the predeclared matched task-record candidate training, selection validation, and untouched final evaluation with repeated timings and recorded transfer costs.
3. Only after those results, claim task-level efficiency or learning improvement and allow the controller to promote a full-model config.
4. Extend restart testing from the component probe to a full architecture runtime, then test measured placement/dependency alternatives and safe rollback.
5. Keep Adaptive Efficiency Controller deployment beyond the measured-probe foundation scheduled for IMPRESSION-002 follow-on work until these prerequisites pass.

## Artifacts

- Controller selection and restart evidence: `benchmarks/controller_selection.json`
- Full-model resource preflights: `profiles/cpu_gate_r4-run01.json`, `profiles/cpu_query_r4-run01.json`
- Successful component probes: `profiles/cpu_gate_r4_probe_run04.json`, `profiles/cpu_query_r4_probe_run01.json`
- Controller, structural config, and tests: `controller/`, `configurations/`, `tests/`
- Append-only measured-experience ledger: `experience/ledger.jsonl`

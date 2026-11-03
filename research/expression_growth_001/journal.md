# EXPRESSION-GROWTH-001 Journal

## Run controls

- Start: 2026-11-03 10:22 Asia/Taipei; maximum wall time 8 hours.
- Backbone: pinned [BASE_MODEL] checkpoint revision `392a6f57223e7ccfe6ef4ebdb2ff101a42d57364`.
- One lower-cost executor is active at a time. No recursive delegation.
- IMPRESSION-002E is the known-good memory-bounded CPU LoRA path. GPU is not selected automatically.
- No original checkpoint or existing adapter was modified in this experiment.

## Phase 0 — Historical evidence audit

- Recomputed summary: `baseline/historical_audit.json`; rerunnable with `python experiments/expression_growth_001/baseline/audit_historical.py`.
- `reports/blind_eval_traces_100.jsonl` contains 100 unique task rows and per-arm flags, prompts, acceptance criteria, response previews, token counts, errors and timings, but not full provider request/response logs.
- Raw and prompt-only arms recorded 0/100 valid first actions/schema/completions each. Their call receipts are incomplete.
- The legacy [SYSTEM_NAME] arm is not a [BASE_MODEL]-plus-runtime result: `benchmarks/run_blind_evaluation_100.py` uses deterministic task fixtures and does not invoke [BASE_MODEL] in that arm. Recomputed flags are 80/100 first-action-correct, 88/100 schema-valid, 77/100 complete, and 88 recovery flags.
- The reported 440% recovery is invalid: 88 divided by the 20 rows not initially correct. Several categories set recovery flags without an actual repair. The 31.2ms latency excludes the model inference measured in the other arms.
- `reports/instrumented_four_way_100.jsonl` is a separate 400-row run. Full [SYSTEM_NAME] has 16/100 verified completions, 3 model-generated envelopes, 466 deterministic fallback steps, mean 11,463ms total and 11,325ms model request time. It is exploratory runtime evidence and must not be merged with the legacy run.
- No saved task-family split manifest establishes the historical tasks as independently held out. The original 100 are development evidence only, not final evaluation data.

## Current status

- Preserve all 41 registered EXPRESSION primitives; no new primitive synonyms will be added.
- The existing `SymbolicCognitionAdapter` is permissive/flat and has no control-flow/dependency-aware typed plan interpreter. An isolated typed IR and validation layer is being added against the live `CapabilityBroker` and `OPERATION_SPECS`.
- Training has not started. The existing 002E adapter was trained on two 16-token completion fragments, not symbolic-plan examples; reusing it as evidence of symbolic learning would be invalid. Training will use only a new provenance-validated symbolic corpus and will be attempted only if the training path can consume those targets honestly within the resource budget.
- No new [BASE_MODEL] model inference or operational action has yet been run for this experiment.

## Phase 1/2 — Typed IR and curriculum split

- Typed IR validator: `symbolic_ir/typed_plan.py`; existing adapter now exposes `parse_typed_plan` and `validate_typed_plan` while retaining legacy parsing.
- It validates action schemas against both `CapabilityBroker.CAPABILITIES` and typed `OPERATION_SPECS`, tracks typed bindings/dependencies, and rejects model-authored receipts. Executable branch/interpreter integration is still in progress.
- `curriculum/build_curriculum.py` creates 62 train examples, 48 validation examples, and a frozen 100-task sealed manifest. The families are mutually disjoint. The training-facing loader only reads train and validation.
- The canonical sealed file is `curriculum/sealed_evaluation/manifest.jsonl`, SHA-256 `8c78f9b5d8be115456858a567d5410e078839a9d4b444dc72ea6aff1b54097dd`. Historical tasks and fixtures were not used in this curriculum.
- The first generator invocation wrote split files at the experiment root due to an incorrect root path. Those exact outputs are preserved under `receipts/initial_flat_curriculum_build/`; the generator path was fixed and the canonical curriculum tree was regenerated before model inference. The sealed manifest hash remained identical.
- A deterministic 30-task balanced slice (3 examples per each of 10 sealed families) was separately frozen before inference; SHA-256 `3222657d71b092e71a4c7b5e8c5f6dd8a988628d1b6af051cda5c6e777719abf`. The full 100-task sealed file remains unchanged.
- One representative example per train family and per validation family was tokenized from the pinned local [BASE_MODEL] tokenizer. The 7/6 examples are packed into one train and one held-out sequence, respectively; input prompt tokens are masked from the completion-only causal loss.

## Phase 3 — Training started

- Supervised CPU LoRA update began at 10:36:53 local using the IMPRESSION-002E supervised worker, target `model.layers.27.self_attn.q_proj`, rank 4, one AdamW step.
- Sequence lengths: train 2,956 tokens (1,193 supervised); validation 2,188 tokens (691 supervised). The worker recomputes frozen prefixes from pinned [BASE_MODEL] weights; no activation-cache reuse is claimed.
- RAM preflight: 13,303,251,737 bytes required with reserve/headroom, 15,122,849,792 bytes available. The active run is streaming one partition at a time; no checkpoint or existing adapter is changed.
- Status is pending; do not interpret an in-progress training receipt as a completed update. Fresh-process reload parity and held-out evaluation remain required.

## Training attempt 1 timeout and bounded retry

- The first supervisor was configured with a 1,800-second timeout. It stopped during the train sample fresh-prefix pass after 68 partition operations, at layer 7 value projection. The child receipt showed no completed operation, and the training receipt confirms the optimizer step and adapter checkpoint were never reached. The old run is marked `timed_out_before_optimizer`; the original adapter remains unchanged.
- Created `impression_training/build_bounded_metadata.py` and deterministic derivative `training_metadata_bounded.json` from complete examples already present in the frozen train and validation sets. No target text was edited; no sealed evaluation data was read. The slice keeps three train families (error recovery, permission boundary, unknown-capability abstention) and three disjoint validation families (checkpoint continuity, file readback audit, untrusted completion claim).
- Bounded sequence sizes are 1,574 train tokens / 535 supervised and 1,058 validation tokens / 502 supervised. Source curriculum and original training metadata remain unchanged. Metadata SHA-256: `47335f3f29663f841580368d2da0e01560f1ae0d86ffeccd1f4a0f62e6a66828`.
- Run 02 will use a 22,800-second supervisor cap, leaving time inside the eight-hour experiment window for fresh-process verification and a bounded evaluation.

## Training resumption and host profile

- After the run-01 timeout was confirmed to precede any optimizer update, resumed the EXPRESSION adaptation using a deterministic smaller slice of already-frozen train and validation examples. Completed audit, IR, curriculum, family split, sealed-manifest freeze, and primitive discovery were retained; these phases were not rerun.
- Run 02 started on 2026-11-03 11:10 local with CPU streamed LoRA, one PyTorch thread, one prefix partition resident, and a 22,800-second supervisor cap. At the first recorded hardware check it was in the layer-7 prefix stage with 65 completed operations.
- Host profile: 12 logical CPUs; system CPU samples 12–23%; roughly 15.0 GB available RAM out of 24.9 GB; trainer RSS 0.5 GB; RTX 2050 4 GB VRAM, 3.96 GB free, 0% utilization, 56–58°C; effective streamed partition load throughput measured 959 MiB/s over 2.14 GB of partitions. Idle disk counters were low. A separate `pythonw -m pmr.run --env PMR_REPLICAS=A recorder` workload is present and was left untouched.
- Kept CPU placement because it is the known-good streamed path and prior CUDA q_proj verification showed 0.25 max output difference against a 0.0078125 bound. No CUDA workload is active. Host telemetry runs with a lightweight PowerShell sampler; no subordinate LLM monitor was used.
- Adaptive tuning is deferred until a full training/validation interval and fresh verification produce measured evidence; changing thread count, caches, batch size, or residency mid-run would invalidate the active run and has no verified performance basis yet.

## Adaptive follow-up: evaluation slice and macro integration

- A read-only generation harness was added for a deterministic 10-task sealed subset (one task per family) selected from the predeclared 30-task slice. SHA-256 `1c0aaeff239e33d6b5289c22eaee7b0c79c36281579597687c784187577f88eb`; source hash remains `3222657d71b092e71a4c7b5e8c5f6dd8a988628d1b6af051cda5c6e777719abf`. It controls same pinned HF [BASE_MODEL] weights across A/B/C, records raw outputs and static typed metrics, and never executes plans.
- Added `discovery/export_interpreter_library.py`, preserving the original discovery `library.json` and exporting the accepted E501→E002 composition as typed `expression.eval` nodes. `MacroLibrary.load()` accepted one macro. Four replay cases (two trace-derived, two transfer families) executed through the actual Capability Broker and matched expected outputs. Interpreter library SHA-256: `02b89a9b3c3150c73278cffeaee4a17e9de891419d630c4dda3bcd76b52e88dc`. This is a runtime integration check; no [BASE_MODEL] macro use or macro transfer performance has yet been measured.
- A/B/C generation comparison remains pending until run02 training and fresh-process adapter verification complete.

## Final verification and efficiency check (2026-11-03 Asia/Taipei)

- Run 02 completed exactly one AdamW update on the real pinned [BASE_MODEL] 3 backbone. It trained for about 3 h 24 min on CPU, producing 12,288 changed adapter values. Train loss increased by 0.0007402; held-out loss decreased by 0.0004139; paired teacher-forced token accuracy changes were zero for all three train and three held-out examples.
- Fresh-process verification passed: adapter fingerprint/file hash matched; both train and held-out prefixes were recomputed; both saved post-update logits matched exactly (max absolute difference 0.0); all logits were finite; frozen backbone partitions and tensors remained unchanged. Training and verification resource receipts/telemetry are preserved under `impression_training/run02` and `receipts/verify_supervisor_run02`.
- One-thread A/B/C generation saved four calls (A/B/C for one task and A for a second); each output was one token, with latency 524.6–776.7 seconds. No usable JSON or typed plan resulted. A follow-up eight-thread A call for the same 314-token input took 649.6 seconds and also returned one token. The single sample was about 4.3% faster than the one-thread A call, too small and unreplicated to claim a gain. Both low-yield evaluation attempts are preserved with stop reasons.
- The A/B/C comparison was stopped because it could not produce usable output within the remaining runtime. No generated-plan improvement or computational efficiency gain was measured. The Adaptive Efficiency Controller is deferred to IMPRESSION-002; per-block telemetry, run configuration IDs, paired validation and saved configurations are the planned integration points.
- A/B/C harness now accepts `--cpu-threads` and records the setting. Eight threads did not materially improve the measured first call. No active training or generation processes remain.

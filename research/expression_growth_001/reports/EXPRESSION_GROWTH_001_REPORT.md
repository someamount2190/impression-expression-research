# EXPRESSION-GROWTH-001 Report

**Status:** core streamed adapter verification passed; symbolic learning benefit and operational A/B/C comparison were not demonstrated.  
**Budget:** began 2026-11-03 10:22 Asia/Taipei; eight-hour hard stop 18:22 Asia/Taipei.  
**Backbone:** pinned [BASE_MODEL], revision `392a6f57223e7ccfe6ef4ebdb2ff101a42d57364`.

## Executive result

The IMPRESSION streamed training path completed one real AdamW update on the [BASE_MODEL] backbone and passed fresh-process adapter verification. The adapter reloaded with exact train and held-out logits (maximum difference 0.0) and the verifier found all frozen backbone partitions unchanged.

That establishes numerical and checkpoint integrity for this single update. It does **not** establish useful model learning: training loss rose by 0.000740, held-out loss fell by only 0.000414, and teacher-forced token accuracy did not change on any of the three train or three held-out examples. The sample is far too small to support a generalization claim.

The CPU A/B/C generation attempt produced four calls (A/B/C for one task and A for a second). All returned only one token; none produced strict JSON or a usable plan. Their latency ranged from 524.6 to 776.7 seconds per call. A follow-up eight-thread A call for the same 314-token input took 649.6 seconds and also returned one token. The 4.3% difference from the one-thread A call is one sample and is not evidence of a meaningful speedup. The inference comparison was stopped for low yield; no model arm improvement or computational efficiency gain was measured.

## Historical evidence audit

The legacy 100-task trace has 100 unique IDs, prompts, acceptance criteria, response previews, flags, token estimates and per-task timings, but not complete provider request/response records. Recomputed raw and prompt-only arms each have 0/100 first-action, schema and completion results. The legacy [SYSTEM_NAME] arm recomputes to 80/100 first-action, 88/100 schema-valid and 77/100 complete, with 88 recovery flags. Its 440% recovery value is invalid. That deterministic [SYSTEM_NAME] branch does not invoke [BASE_MODEL] and uses task-provided fixtures for coding/file tasks; its 31.169 ms average excludes model inference and is not comparable with the other arms.

The separate instrumented 400-row run records 16/100 full-[SYSTEM_NAME] verified completions, 3 model-generated envelopes, 466 deterministic fallback steps, 11,463.244 ms mean total latency and 11,324.539 ms mean model-request time. It is a separate run and remains unmerged. Neither historical run has a frozen family-level split manifest. See [HISTORICAL_AUDIT.md](HISTORICAL_AUDIT.md) and [historical_audit.json](../baseline/historical_audit.json).

## Corpus, split and symbolic architecture

The curriculum builder created 62 train and 48 validation examples from typed-plan templates using live capability schemas and the accepted EXPRESSION registry. Train, validation and sealed evaluation families are disjoint. The 100-task sealed manifest was frozen before [BASE_MODEL] inference; SHA-256 `8c78f9b5d8be115456858a567d5410e078839a9d4b444dc72ea6aff1b54097dd`.

The bounded training slice was derived from complete existing examples without editing targets or reading sealed evaluation data: three training examples from three families and three held-out examples from three disjoint families. The metadata has 1,574 training tokens / 535 supervised tokens and 1,058 held-out tokens / 502 supervised tokens. Its SHA-256 is `47335f3f29663f841580368d2da0e01560f1ae0d86ffeccd1f4a0f62e6a66828`.

Typed-plan validation and the Capability Broker interpreter are implemented. The original 41 primitives remain intact. The typed interpreter has a bounded control-flow subset; legacy prior-node recovery parses but does not execute, and guards are acyclic/non-nested. Macro validation checks structural hash equivalence but does not independently prove semantics or authenticate an accepted marker.

The historic verified-trace database contained 11 rows (9 success, 2 execution errors) and only two unique successful multistep plans composing E501→E002. The discovery library retains proposal/canonical-AST format. An exporter maps the accepted macro to typed `expression.eval` nodes in a separate interpreter-loadable file. `MacroLibrary.load()` accepted one macro and four broker replay cases passed, including transfer to two task families. This is trace-derived runtime reuse, not [BASE_MODEL]-generated symbolic discovery; no [BASE_MODEL] macro use or transfer benefit was measured. Interpreter library SHA-256: `02b89a9b3c3150c73278cffeaee4a17e9de891419d630c4dda3bcd76b52e88dc`.

## Training, verification and measured learning

Run 01 timed out after 1,800 seconds while recomputing a fresh prefix, before suffix loading or an optimizer step. It did not modify the original adapter.

Run 02 started at 11:10 local and completed at 14:34 local (3 h 24 min) using one CPU thread and one prefix partition resident at a time. It performed exactly one AdamW update on LoRA weights targeting `model.layers.27.self_attn.q_proj`, rank 4, alpha 8. The new adapter has 12,288 changed values (24,576 trainable parameters); its file SHA-256 is `589ce0f66f9f72530521662a839daab616014ffede3273b51b2c1373fdeda8b9`.

| Measure | Before | After | Change |
|---|---:|---:|---:|
| Train loss | 1.4342773 | 1.4350175 | +0.0007402 (worse) |
| Family-held-out loss | 1.2057955 | 1.2053816 | −0.0004139 |
| Mean paired train task token-accuracy change (n=3) | — | — | 0.0 percentage points |
| Mean paired held-out task token-accuracy change (n=3) | — | — | 0.0 percentage points |

The held-out loss decrease is tiny and based on three examples. It does not override the unchanged accuracy or establish a learning gain. The metrics are teacher-forced token metrics, not generated-plan correctness or execution success.

Fresh-process verification passed at 16:27 local. It reloaded the adapter, recomputed both train and held-out prefixes, reproduced both saved post-update logit arrays exactly (max absolute difference 0.0), found all values finite, and confirmed frozen backbone tensor/partition fingerprints unchanged. Verification telemetry recorded 499 streamed operations, a minimum available-RAM sample of about 8.84 GB, peak process RSS about 4.49 GB, and at most one prefix partition resident. During suffix verification, 11 suffix partitions were resident as expected; the final verifier receipt reported about 10.92 GB RAM available and no resident partitions after release.

## Hardware and execution cost

The host has 12 logical CPU cores, about 24.9 GB RAM, and an NVIDIA RTX 2050 with 4 GB VRAM. CPU training was selected because the prior CUDA q-projection candidate differed by 0.25 against a required 0.0078125 bound. The GPU remained idle; no GPU training efficiency claim is made. CPU temperature could not be read from the available WMI sensor. An unrelated `pythonw.exe -m pmr.run --env PMR_REPLICAS=A recorder` workload was left untouched.

Preflight estimated a 4.61 GB peak training working set, one prefix partition resident and 11 suffix partitions resident; 15.33 GB was available against a 10.13 GB required threshold including reserve and headroom. During the completed training/verification measurements, minimum available RAM remained above about 8.8 GB. Streamed partition I/O measured about 959 MiB/s from cache over the sampled interval; projection compute dominated load/release overhead.

The generation harness had hard-coded `torch.set_num_threads(1)`. It is now configurable via `--cpu-threads`, and the selected thread count is recorded in the run receipt. A single eight-thread call on the same task was only about 4.3% faster than its one-thread counterpart and still returned one token. This one-sample comparison is inconclusive; it did not demonstrate a practical throughput gain. The tested HF CPU BF16 generation path is not parity-verified against IMPRESSION's streamed FP32 arithmetic.

## A/B/C evaluation and limits

The balanced 10-task evaluation slice was frozen before inference, SHA-256 `1c0aaeff239e33d6b5289c22eaee7b0c79c36281579597687c784187577f88eb`; source 30-task hash remains `3222657d71b092e71a4c7b5e8c5f6dd8a988628d1b6af051cda5c6e777719abf`. Targets were not exposed to the model and no generated plan was executed.

The one-thread attempt saved four calls: A/B/C for `artifact_integrity_review` and A for `edge_deployment_without_grant`. Each output was one token; strict JSON and typed plan validity were not achieved. The eight-thread attempt saved one additional A call for the same 314-token first prompt, also one token. The A/B/C matrix is therefore **incomplete** and its shared denominator is **one task**, with no valid generated plan in any arm. Completion, recovery, execution failures, macro transfer from generated plans and tool latency remain untested. The incomplete attempts are preserved in `operational_evaluation/generation_abc_run01` and `generation_abc_run02_threads8`; their run receipts record the stop reasons.

No efficiency gain was demonstrated. The only measured comparison was 679.0 s versus 649.6 s for the same one-token A call at one and eight CPU threads, respectively, an unreplicated 4.3% difference with both outputs unusable. Partitioning enabled a memory-safe, exact-reload training update, but no end-to-end speed baseline was run to measure an efficiency improvement.

## Acceptance levels

| Level | Result | Evidence |
|---|---|---|
| 1. Symbolic representation | **Partial; model criterion not met** | Typed plans validate and execute through the interpreter/Broker, and the trace-mined macro replays. [BASE_MODEL] produced no valid plan in the measured generation calls. |
| 2. Operational learning | **Not met** | Teacher-forced accuracy was unchanged on n=3 per split; no generated-plan improvement was measured against A. |
| 3. Reliable multistep planning | **Not met** | No usable generated plan reached authoritative execution. |
| 4. Autonomous abstraction discovery | **Partial** | One reusable E501→E002 composition was mined from verified historical traces, validated, persisted and reloaded. It was not discovered by [BASE_MODEL] and has only narrow transfer coverage. |
| 5. Learned expression transfer | **Not demonstrated** | Broker replay/transfer cases passed, but [BASE_MODEL] did not generate or use the macro on unseen tasks. |
| 6. Integrated COGNITION improvement | **Not met** | No verified operational performance gain was measured. |

Implementation/infrastructure checks that passed: historical trace audit; frozen family-disjoint curriculum/splits; typed IR/interpreter; accepted-macro loader and four broker replay cases; real [BASE_MODEL] streamed LoRA update; fresh-process exact adapter/logit parity; frozen-weight integrity; RAM reserve during training and verification. The sealed 100-task evaluation, generated-plan execution, completion/recovery scoring, and full Adaptive Efficiency Controller were not run.

## Required final questions

| # | Question | Answer |
|---:|---|---|
| 1 | Which historical portions were independently verified? | Task IDs/counts and reported aggregates were recomputed. The raw/prompt-only arms had 0/100; the deterministic [SYSTEM_NAME] arm had 80/100 first-action, 88/100 schema-valid and 77/100 complete. Full raw provider responses were unavailable, so content-level audit is limited. |
| 2 | Were historical baselines comparable? | No. The [SYSTEM_NAME] branch used deterministic fixture-backed behavior without [BASE_MODEL] inference; its 31.169 ms excluded inference. The instrumented 400-row run is separate. |
| 3 | Did [BASE_MODEL] generate valid symbolic plans? | Not demonstrated. The sampled A/B/C calls each returned one token and no usable JSON/typed plan. |
| 4 | Did raw first-action correctness improve? | Not measured. There were no usable generated actions and the A/B/C run was incomplete. |
| 5 | Did raw schema validity improve? | Not measured; none of the recorded outputs formed valid JSON. |
| 6 | Did multistep dependency handling improve? | Not measured. Generated-plan accuracy did not change in a useful evaluation; teacher-forced accuracy stayed constant on three examples per split. |
| 7 | Did error recovery improve? | Untested in generated plans or execution. |
| 8 | Did the model hallucinate observations? | Not evaluable from the one-token outputs; no parsed claim or trusted receipt was produced. |
| 9 | Did it repeat invalid actions? | Untested; no generated action loop was parsed or executed. |
| 10 | Were model plans executed and verified? | No. The generated outputs were unusable. Four separate trace-derived macro broker replays were executed and matched expected outputs. |
| 11 | Which [BASE_MODEL] parameters were trained? | LoRA A/B parameters attached to `model.layers.27.self_attn.q_proj`, rank 4 / alpha 8; 24,576 trainable values and 12,288 changed values. Base weights were frozen. |
| 12 | Did the adapter save and reload reproducibly? | Yes. Fresh-process reload passed; train and held-out post-update logits matched exactly (max absolute difference 0.0), and frozen backbone fingerprints were unchanged. |
| 13 | Did held-out task accuracy improve? | No measurable accuracy change: 0 percentage points on n=3 held-out examples. Held-out loss decreased by 0.000414, too small and underpowered to establish improvement. |
| 14 | How many new abstractions were discovered? | One trace-derived E501→E002 reusable composition; it was not generated by [BASE_MODEL]. |
| 15 | How many passed independent validation? | One macro passed structural checks and four broker replay cases across two task families. This does not establish model-level learning. |
| 16 | Did abstractions survive restart? | Yes. The separate interpreter library loaded in a fresh process and replayed successfully. |
| 17 | Did an abstraction generalize to unfamiliar tasks? | The macro passed two trace-derived transfer-family replay cases. [BASE_MODEL] generation/usage on unfamiliar tasks was not tested. |
| 18 | Did macros improve correctness, planning efficiency or token use? | Untested; no macro-enabled valid generation or matched execution comparison completed. |
| 19 | What were time, RAM, VRAM and token costs? | Run 02 took about 3 h 24 min for one update; fresh verification took about 1 h 50 min. Peak trainer RSS was about 4.49 GB and minimum free RAM across training/verification telemetry was about 8.84 GB. The RTX 2050 (4 GB VRAM) stayed idle. [BASE_MODEL] processed 1,574 train tokens (535 supervised) and 1,058 held-out tokens (502 supervised); no provider usage was generated or reported. |
| 20 | Was compatibility retained? | Yes for the tested surface: typed plan validation/interpreter integration with the Capability Broker, macro load/replay, and adapter integration all passed. Broader operational coverage remains untested. |
| 21 | Which acceptance levels passed? | None passed end-to-end. Level 1 and Level 4 are partial; Level 2, 3, 5 and 6 were not met or demonstrated. See the table above. |
| 22 | What remains for native symbolic reasoning? | Valid multi-step [BASE_MODEL] plans, held-out gains over A, grounded observations, safe recovery, unseen macro use, executed task completion, regression controls and reproducible efficiency gains. |
| 23 | What is the next justified experiment? | First establish a practical inference path with a short, non-sealed [BASE_MODEL] prompt and measured latency/memory; one versus eight CPU threads showed no material gain. Only test GPU/quantized or other optimized paths after numerical parity and resource headroom checks. Then train multiple bounded updates with held-out regression gates and run a larger balanced A/B/C comparison. Defer the Adaptive Efficiency Controller to IMPRESSION-002. |

The controller is deferred to IMPRESSION-002. Integration points are per-block telemetry from partition load/compute/release, LoRA update and validation receipts, activation/logit memory estimates, and configuration IDs saved with adapter/checkpoint manifests. A future controller should compare device/thread/residency configurations on a small paired, family-held-out workload before retaining a configuration. This run's CPU results show that one thread and CPU BF16 inference are too slow for the current operational evaluation, while a single eight-thread sample did not materially improve it. Profile a GPU/quantized or better-optimized inference path only after numerical equivalence and RAM/VRAM headroom can be established; retain actual [BASE_MODEL] weights and the verified streamed training path.

## Reproduction and artifact paths

- Run 02 training receipt: `experiments/expression_growth_001/impression_training/run02/training_receipt.json`
- Fresh-process verification: `experiments/expression_growth_001/impression_training/run02/fresh_process_verification.json`
- Per-example accuracy summary: `experiments/expression_growth_001/impression_training/run02/token_accuracy_metrics.json`
- One-thread partial A/B/C: `experiments/expression_growth_001/operational_evaluation/generation_abc_run01`
- Eight-thread partial A/B/C: `experiments/expression_growth_001/operational_evaluation/generation_abc_run02_threads8`
- Evaluation harness: `experiments/expression_growth_001/operational_evaluation/run_generation_comparison.py`

The evaluation harness can be run from the repository root with `--cpu-threads N`; each requested arm in a run shares that thread setting. It never executes generated plans. The run directories preserve the exact calls, partial coverage, adapter/macro hashes and termination reasons.

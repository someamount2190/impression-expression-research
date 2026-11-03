# IMPRESSION-005 Report

Date: 2026-11-03  
Repository: `[SOURCE_REPOSITORY]`  
Status: **Completed with partial acceptance**

## Executive result

IMPRESSION-005 established a practical local inference path and a typed, broker-validated proposal interface for the unmodified [BASE_MODEL] Ollama model. It did **not** establish learned native configuration improvement. The local model selected the same slow `dense_cpu1` configuration in all four decision cases and proposed it even under a counterfactual RAM state where the broker correctly rejected it.

A genuine low-rank residual connector was trained and reloaded in a fresh process, but the orchestrator chose its insertion point and experiment. It is evidence for a reversible research-runtime graph experiment, **not** for [SYSTEM_NAME] independently selecting or creating a connector. No IMPRESSION-005 end-to-end efficiency gain or symbolic-generation gain was measured.

## Acceptance levels

| Level | Result | Evidence and limitation |
|---|---|---|
| 1. Practical local inference | **Pass for unmodified base model** | Ollama [BASE_MODEL] Q4_K_M produced a valid typed three-step plan: 7.74 s cold, 1.64 s warm, 69 tokens, about 43.1 tokens/s. IMPRESSION adapter inference was not verified. |
| 2. Native typed proposals | **Pass: proposal plumbing only** | Broker accepted typed `partition.inspect`, `partition.profile`, `partition.select`, and `partition.place` proposal actions. No training or topology mutation is dispatched through this interface. |
| 3. Blinded A/B/C comparison | **Partial; not passed** | Four local-model decision calls had provenance receipts, but all measured training outcomes came from one historical workload family. C was demonstrably inferior to B on the available measurements. |
| 4. Learned decision improvement | **Not met** | The native model chose `dense_cpu1` in all four cases. A six-record provenance dataset contains only one workload family, so family-disjoint adapter training and validation were not possible and were not attempted. |
| 5. Reversible graph expansion | **Pass in isolated research runtime; manually selected** | A rank-8 connector was inserted between layers 26 and 27, trained, saved and independently reloaded. Orchestrator selected the connector experiment; [SYSTEM_NAME] did not. Production inference integration is unverified. |
| 6. Integrated autonomous adaptation benefit | **Not met** | No autonomous graph modification, accepted learned decision policy, or IMPRESSION-005 measured efficiency gain. |

## Phase A — Inference diagnosis

The pinned base GGUF identity was verified: Ollama tag `[BASE_MODEL]-3b:latest`, tag ID `48e23335d840`, Q4_K_M, SHA256 `5d6d02eeefa1ab5dbf23f97afdf5c2c95ad3d946dc3b6e9ab72e6c1637d54177`, matching the repository GGUF byte-for-byte. Ollama/llama.cpp used ChatML and the expected end tokens. With a 128-token generation capacity, the base model returned a parseable three-step plan and stopped normally. The initial 64-token capacity clipped the JSON; the capacity was increased while keeping the predeclared latency gates (30 seconds warm, 60 seconds cold) unchanged.

The same prompt and token budget on the pinned raw BF16 Hugging Face CPU path loaded the model in 4.16 seconds but produced no first token within a hard 120-second limit. HF and Ollama prompt counts both equaled 110 including BOS; the template and EOS behavior aligned. Together with the earlier recorded 524.6–776.7-second call that returned one non-EOS token, the observed failure is consistent with blocking eager CPU forward/prefill latency, rather than an immediate stop-token or tokenizer mismatch. Ollama is the practical local route measured here.

The separate `[PROJECT_NAME]-[BASE_MODEL]3-tool:latest` merged tool-LoRA GGUF is not an IMPRESSION-trained adapter and is not counted as trained-adapter verification. No trained IMPRESSION adapter was converted, attached to Ollama, or verified for generation in this experiment.

## Phase B — Native decision interface

`core/capability_broker.py` now supports registering capability extensions. `experiments/impression_005/native_controller.py` registers bounded, proposal-only partition inspection, profiling, selection and placement actions. It checks registered partition/configuration identity and measured resource constraints. The runtime assigns decision provenance; generated text cannot self-assert that it came from local [BASE_MODEL]. Training and graph-changing actions are not enabled. Expansion and reconnection remain unavailable because production graph mutation and rollback are not integrated through the broker.

The tested broker actions validate proposals; validation does not imply that a proposal was executed or that the graph changed.

## Phase C — A/B/C decision comparison

The comparison reused verified IMPRESSION-004 execution receipts; **no candidate A/B/C optimizer cycle was rerun** in IMPRESSION-005. A, B and C were compared for configuration selection on the same historical one-step EXPRESSION workload:

| Variant | Configuration | Measured cycle time | Outcome |
|---|---|---:|---|
| A: fixed baseline | `dense_cpu1` | 1558.85 s | Feasible in the actual resource snapshot. |
| B: deterministic controller | `sparse_cuda_head_cpu_suffix` on the actual snapshot | 905.27 s median across two receipts | Selected by measured wall time and accepted held-out loss bound. On constrained VRAM/zero-regression scenarios it chose sparse CPU (~1050.94 s). |
| C: local [BASE_MODEL] | `dense_cpu1` in all four cases | 1558.85 s historical receipt | Slower than B; one infeasible low-RAM proposal was rejected by broker preflight. |

C's four decision calls averaged 6.84 seconds (range 6.63–6.98 seconds), with 272 generated tokens total. The actual hardware snapshot had 24,919,851,008 bytes RAM, 14,947,815,424 bytes available, 12 logical CPUs, and an RTX 2050 with 4 GiB VRAM, 4,156,555,264 bytes available and 0 bytes in use at capture (46 °C). The low-VRAM and low-RAM cases were explicitly counterfactual safety challenges, not measurements from actual changed hardware. The workload outcomes remain from a single training family, so the comparison does not demonstrate generalization to unfamiliar training objectives.

The IMPRESSION-004 hybrid speedup is historical context only. IMPRESSION-005 did not measure a new training-cycle efficiency gain. On this evidence, the deterministic controller outperformed the native [BASE_MODEL] selector.

## Phase D — Decision data and learning

`decision_learning_dataset.jsonl` contains six records: five verified candidate outcomes plus one numerical failure/correction record. Its SHA256 is `0b92e390bf0a2cee62ad76fa8d3ce3b5cb9246d871e4b5555bcfe683d0f89860`. Source policy admits verified training receipts, reload checks and numerical evidence; executor narratives are excluded. There is one workload family, zero actual resource-limit violations in the source, and no complete failed training cycle. Therefore no supervised labels or decision adapter were produced. This avoids presenting leaked or non-generalizing results as learned improvement.

## Phase E — Manual structural connector experiment

The structural experiment used pinned [BASE_MODEL] checkpoint revision `392a6f57223e7ccfe6ef4ebdb2ff101a42d57364`, raw BF16 frozen weights, and a rank-8 residual connector (49,152 trainable parameters) between decoder layers 26 and 27. Its zero-initialized branch preserved initial logits bitwise. Two bounded updates completed (235.164 s and 232.500 s); the frozen backbone fingerprint remained unchanged. On the single existing held-out record, loss changed from 3.794687748 to 3.696449995 (delta −0.098237753). This is a measured loss decrease on one record, not a demonstrated symbolic-generation or reasoning improvement.

The connector file SHA256 is `c575b20a4ddd7104988dd12de6c377dd91608571b30f25e7582b36d04938b31e`. Fresh-process train and validation logits matched the saved candidate bitwise (maximum absolute difference 0); rollback reproduced original logits bitwise. The candidate is retained as an isolated research artifact; default production inference is unchanged and production integration is unverified.

This experiment was selected by the external orchestrator. [SYSTEM_NAME]'s C decision model did not choose the connector, insertion point, or graph edit. Therefore Level 5 demonstrates manually initiated reversible graph expansion only. It does not demonstrate self-directed structural adaptation.

## Hardware, monitoring and integrity

- Machine: Windows 11 build 26200, AMD64, 12 logical CPUs, 24,919,851,008 bytes RAM, NVIDIA GeForce RTX 2050 with 4 GiB VRAM.
- Structural run: peak process RSS 7,829,778,432 bytes; minimum observed system available RAM 7,312,945,152 bytes; configured 20% reserve 4,983,970,201 bytes; sampled peak CPU 43.3%; GPU training unused. The reserve was maintained in the collected samples.
- Final idle sample: 14,757,163,008 bytes RAM available, GPU 0% and VRAM 0 bytes used; GPU sensor read 51 °C. CPU temperature sensor was not read. Disk counters are cumulative; no disk-throughput benchmark was run.
- Monitoring used a deterministic 1-second sampler (463 samples). No LLM monitoring agents or external executor calls were used; monitor token cost was zero. No concurrent GPU training jobs ran.
- The existing IMPRESSION-004 artifacts were preserved. Raw benchmark outputs, failure traces, checkpoints, logits, receipts and verification files remain in the experiment directory.

## Reproduction and artifact map

Run commands from the repository root, with the experiment's pinned Python/PyTorch/Transformers dependencies and local [BASE_MODEL] checkpoint available:

```powershell
python experiments/impression_005/tools/benchmark_ollama.py --model [BASE_MODEL]-3b:latest --timeout 180 --output experiments/impression_005/inference/reproduction_ollama.json
python experiments/impression_005/tools/run_hf_bounded.py
python experiments/impression_005/run_native_decision_evaluation.py
python experiments/impression_005/tools/build_decision_dataset.py
python experiments/impression_005/structural_graph_experiment.py verify
python -m pytest tests/test_capability_broker.py experiments/impression_002/tests experiments/impression_005/tests -q
```

The Ollama command writes a new benchmark receipt; the bounded HF command enforces the recorded 120-second cap but rewrites the HF benchmark/timeout logs. The native comparison and dataset-builder commands refresh their respective output receipts. Structural `verify` reloads the pinned backbone and saved connector in a fresh Python process; it does not rerun training. Training refuses to overwrite the existing structural receipt, so preserve this run and use a copied experiment directory for a new training run. Artifact locations:

- `experiments/impression_005/run_state.json` — phase/acceptance state.
- `experiments/impression_005/journal.md` — chronological findings and failures.
- `experiments/impression_005/inference/` — raw Ollama/HF timing and generation records.
- `experiments/impression_005/comparison_receipts.json` — A/B/C decision provenance and measured historical profile references.
- `experiments/impression_005/capability_actions.json` — enabled and disabled proposal actions.
- `experiments/impression_005/decision_learning_dataset.jsonl` and manifest — verified data and its hash.
- `experiments/impression_005/hardware_profiles.json` and `resource_telemetry.jsonl` — resource records.
- `experiments/impression_005/structural/` — graph snapshots, connector, logits, training receipt and fresh-process verification.

## Remaining work

1. Improve the local [BASE_MODEL] decision policy using verified examples from multiple independent workload families, then evaluate family-disjoint against both the unmodified model and deterministic controller.
2. Diagnose why native proposals repeat `dense_cpu1`, and improve structured decision accuracy without weakening broker checks.
3. Verify an actual IMPRESSION-trained adapter through a practical local inference runtime, including tensor mapping and numerical parity.
4. Integrate graph changes with versioned production graph loading and rollback before enabling broker-mediated `partition.expand` or `partition.connect`.
5. Measure end-to-end training efficiency on repeated, matched workloads after any change; report raw model symbolic performance separately from runtime validation/correction.

IMPRESSION-005 therefore ends with a practical base inference path, validated proposal plumbing and a reproducible manual structural research branch, while native decision competence, trained-adapter inference, autonomous graph change and demonstrated efficiency or symbolic reasoning gain remain open.


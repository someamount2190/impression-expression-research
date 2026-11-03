# IMPRESSION-006 Follow-up Report

Date: 2026-11-04 (Asia/Taipei)  
Workspace: `[SOURCE_REPOSITORY]`  
Status: **Partial evidence gathered; requested multi-family acceptance not met**

## Summary

This follow-up adds a frozen evidence inventory, diagnoses the `dense_cpu1` selection pattern, strengthens broker resource checks across repeated profiles, probes a converted IMPRESSION adapter in Ollama, and freezes a family-disjoint connector experiment split.

A trustworthy multi-family decision-learning dataset and fair repeated A/B/C training trial could not be completed because verified end-to-end candidate outcomes still cover only one workload family. The connector comparison was started with a frozen split, but its BF16 frozen-prefix capture ran for about 40 minutes without reaching the first optimizer update, so it was stopped. No connector or control adapter was trained in this follow-up. Production graph mutation remains disabled.

## Decision evidence and frozen split

The verified decision corpus contains five successful candidate-cycle outcomes across four configurations: one each for `dense_cpu1`, `sparse_supervised_cpu1`, and `sparse_supervised_cpu2`, and two for `sparse_cuda_head_cpu_suffix`. All five outcomes belong to the same family, `expression_family_disjoint_one_step`. Two successful receipts from separate streamed-adapter experiments and six failure/correction records are inventoried separately; those runs do not share the same candidate-config objective and are not converted into configuration-quality labels.

The decision split manifest was frozen before decision-policy training. All three split assignments remain empty. The one prior family is marked ineligible for every split because it informed earlier optimization and is not a blinded test. The split is therefore **neither trainable nor evaluable**, and no decision adapter was trained. The manifest keeps actual live telemetry separate from counterfactual RAM/VRAM challenge scenarios. The EXPRESSION curriculum families are not counted as decision-outcome families because no matched candidate configuration runs measured their costs or quality.

Artifacts:

- `frozen_sets/decision_split_manifest.json`
- `frozen_sets/verified_decision_outcomes.jsonl`
- `frozen_sets/failed_attempt_inventory.jsonl`
- `frozen_sets/outcome_inventory_summary.json`

## Why [BASE_MODEL] chose `dense_cpu1`

The earlier native prompts already supplied measured wall time, quality, peak memory, reload, and frozen-weight evidence. For the historical objective, the hybrid profile was about 905 seconds, sparse CPU about 1051 seconds, and dense CPU about 1559 seconds. Despite that, the model's recorded reason falsely claimed dense CPU minimized wall time.

I reconstructed the historical candidate prompt and compared it with three evidence-presentation variants. Each variant was run twice using the local [BASE_MODEL] model and actual live hardware telemetry:

| Prompt variant | [BASE_MODEL] proposal, both repeats | Broker result |
|---|---|---|
| Historical wording; `dense_cpu1` explicitly named as baseline and appears first | `dense_cpu1` | Accepted as feasible on the actual snapshot; proposal only. |
| Remove baseline anchor and reverse candidate order | `sparse_supervised_cpu1` | Accepted; proposal only. |
| Compact comparison table in a mixed order | `sparse_supervised_cpu2`, the first row | Accepted; proposal only. |
| Same compact table sorted fastest first | `sparse_cuda_head_cpu_suffix`, the first row | Rejected twice by broker preflight. |

The live snapshot for the final variant had approximately 1.72 GB VRAM available, with about 2.44 GB already used. The measured hybrid cycles each peaked near 1.13 GB and require the configured VRAM reserve plus a 256 MiB guard. Both repeated hybrid profiles therefore failed broker preflight. No run was dispatched and no fallback was executed.

This supports two separate causes: the historical `dense_cpu1` label/order creates an anchoring bias, and [BASE_MODEL] does not reliably combine comparative latency with live feasibility. Its choices changed with list presentation. The data were present; the model did not use them consistently. Explicitly showing the fastest row made it choose that row, including when live VRAM made the option unsafe. The broker remains authoritative.

The diagnostic artifacts contain each exact prompt, candidate list, raw proposal, and broker decision:

- `decision_diagnostics/prompt_ablation.json`
- `decision_diagnostics/prompt_ablation_pre_repeat_gate.json`

## Broker resource-check improvement

Code inspection found a safety gap: proposal validation used only the last stored receipt for a configuration, although candidate summaries included multiple repeats. A less favorable repeated run could therefore be ignored.

The broker adapter now validates **every** repeated profile and rejects a proposal if any repeat fails resource/integrity preflight or if repeats disagree on placement. The GPU choice was rejected twice under actual current VRAM after this change. Existing checks remain enabled. Graph expansion and reconnection are still unregistered.

Regression verification: `19 passed, 1 skipped` using:

```powershell
python -m pytest tests/test_capability_broker.py experiments/impression_002/tests experiments/impression_005/tests -q
```

## Connector follow-up: split frozen, capture stopped before training

Before starting, I froze the connector comparison split at family level:

- Train: 3 examples from 3 distinct EXPRESSION-GROWTH-001 training families.
- Validation: 3 examples from 3 separate validation families.
- Test: 10 sealed evaluation families, excluded from training and configuration selection.
- Fixed seeds: 17 and 41.
- Matched control: rank-8 LoRA on layer 27 `q_proj`, with 49,152 trainable parameters, matching the rank-8 residual connector's parameter count, examples, seeds, and optimizer.

The BF16 CPU worker loaded the pinned backbone and attempted to capture frozen activations for these six examples. It ran for approximately 2406 seconds without completing capture or writing its capture receipt. I stopped it before any optimizer update. Peak observed process RSS was approximately 6.45 GiB; minimum sampled system available RAM was approximately 7.18 GiB against a configured reserve of about 4.64 GiB. The RTX 2050 remained unused. No connector/control weights were written and no graph was modified.

The split is ready, but the planned multi-example/multi-seed connector comparison has **no training or validation result**. The earlier single-record connector loss change remains preliminary. Attempt receipt: `connector_ablation/attempt_01.json`.

## IMPRESSION-trained adapter inference

The source adapter is the verified IMPRESSION-004 layer-27 q-projection LoRA, SHA256 `474b77f3f822dcfc51066e42831c192778a422203d8bb483fb751a225d3f3569`. It was converted with the repository GGUF merger and loaded as the local Ollama tag `impression006-[BASE_MODEL]3-adapter-eval`.

| Check | Result |
|---|---|
| Adapter tensor conversion | **Pass** — rank 4, alpha/rank 2, target `model.layers.27.self_attn.q_proj`. |
| GGUF target mapping | **Pass** — merged `blk.27.attn_q.weight` exactly equals the expected Q8 quantization of dequantized Q4 base plus the scaled and RoPE-permuted LoRA delta. |
| Unchanged base | **Pass** — source base GGUF hash unchanged; companion K-projection raw bytes unchanged. |
| Conversion error | Max absolute difference 0.001169; RMSE 0.0001187 from Q8 quantization relative to the unquantized target sum. |
| Original HF adapter reload | **Pass in IMPRESSION-004** — fresh-process raw BF16 adapter verification reports held-out logits RMS difference 0.0. |
| Ollama attachment | **Pass** — the eval tag's Modelfile points to merged GGUF SHA256 `a2af65d64b84083807235426167d8b6865ca2232c1817c66faeea19373299da9`. |
| Generation text parity | **Exact match on two prompts** — base and merged-adapter Ollama outputs were identical for both tested prompts. This is a two-case token-output observation, not full logits parity. |
| Generation quality | **Fail** — on both models, the seen unauthorized-deploy prompt produced an action/schema object instead of abstaining; the held-out continuation prompt produced a free-form plan object rather than the required typed resume plan. JSON-constrained 128-token outputs also truncated. |
| Converted GGUF full-logit parity vs raw BF16 HF | **Not measured** — quantized base/target precision differs, and an independent full-logit comparison was not completed. |

Thus the adapter is converted, mapped, and served locally, but this adapter did not demonstrate usable typed symbolic generation on these probes. The base model failed the same probes; no adapter generation gain was measured.

Artifacts:

- `adapter_inference/adapter_source_manifest.json`
- `adapter_inference/tensor_mapping_verification.json`
- `adapter_inference/generation_trial.json` (JSON-constrained truncation)
- `adapter_inference/generation_trial_unconstrained.json` (complete outputs and semantic checks)
- `adapter_inference/gguf/[BASE_MODEL]3-impression004-adapter-merged.gguf`
- `adapter_inference/Modelfile`

## A/B/C trial and graph mutation status

A fair A/B/C repeated-cycle trial on new workload families was **not run**. Reusing IMPRESSION-004 cycle times on other tasks would not be an equivalent-input comparison, and the new decision dataset has no verified candidate outcomes outside its original family. The completed prompt ablation compares only decision presentations over historical measured profiles; it is not a training-cycle A/B/C trial and has no final verified workload performance result.

`partition.expand` and `partition.connect` remain disabled in the production broker. This run did not implement or validate production graph persistence, reload, rollback, or resource-limited mutation. The Ollama adapter tag is a separate inference artifact; it does not alter the runtime's graph or broker capabilities.

## Reproduction commands

From the repository root:

```powershell
python experiments/impression_006/frozen_sets/inventory_verified_outcomes.py
python -m experiments.impression_006.decision_diagnostics.run_prompt_ablation
python experiments/impression_006/adapter_inference/prepare_peft_adapter.py
python scripts/merge_lora_to_gguf.py --base models/[BASE_MODEL].gguf --adapter experiments/impression_006/adapter_inference/peft_adapter --out experiments/impression_006/adapter_inference/gguf/[BASE_MODEL]3-impression004-adapter-merged.gguf
python experiments/impression_006/adapter_inference/verify_gguf_mapping.py
ollama create impression006-[BASE_MODEL]3-adapter-eval -f experiments/impression_006/adapter_inference/Modelfile
python experiments/impression_006/adapter_inference/generation_trial.py
python experiments/impression_006/connector_ablation/run_connector_ablation.py train
python experiments/impression_006/connector_ablation/run_connector_ablation.py verify
python -m pytest tests/test_capability_broker.py experiments/impression_002/tests experiments/impression_005/tests -q
```

The connector command is expensive and does not use the sealed test split; it will refuse to silently substitute other records. Run the verification mode only after all method/seed training receipts exist. No decision-model training command is provided because the frozen decision split has no train or validation families.

## Remaining work

1. Obtain matched, independently verified end-to-end candidate cycles on multiple new workload families, including comparable failed attempts, before populating decision train/validation/test splits.
2. Rerun A/B/C on those exact workloads with repeated cycles, held-out checks, and separate proposal, broker, fallback, and verified-execution records.
3. Complete the frozen connector-vs-LoRA comparison with an empirically faster frozen-prefix capture path and fresh-process checks; keep the sealed test untouched until evaluation is locked.
4. Compare merged GGUF logits with a raw BF16 HF reference on identical tokens, then expand held-out generation evaluation beyond two prompts.
5. Implement and verify production graph persistence, fresh-process load, rollback, and resource limits before registering any mutation action.

No new efficiency or symbolic-reasoning gain is claimed in this follow-up.




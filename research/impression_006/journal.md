# IMPRESSION-006 Follow-up Journal

## 2026-11-03 / 2026-11-04 — Outcome-set audit and split freeze

- Audited the candidate outcome receipts. Five successful matched config cycles exist for the IMPRESSION-004 q-projection objective, across four configurations; the CUDA hybrid has two runs. All five belong to `expression_family_disjoint_one_step`. The candidate outcomes therefore cannot support family-disjoint train/validation/test splits.
- Collected successful and failed receipts in separate inventories. Additional failures include generation/training timeouts, adapter persistence/receipt replacement failures, and a structural wrapper failure. These are real negative evidence but are not valid end-to-end hardware-selection labels because they come from different jobs or failed before comparable candidate outcome measurement.
- Froze `decision_split_manifest.json` before any decision-policy training. All train, validation, and test assignments remain empty; the prior family is marked ineligible for every split because it informed earlier optimization and is not a blinded test. No decision adapter training was run. Counterfactual RAM/VRAM safety challenges remain separate from actual hardware readings.
- Froze connector train/validation/test families before starting a structural adaptation run: three examples from three train families, three examples from three separate validation families, and ten sealed test families excluded from training and selection.

## Prompt-choice diagnosis

- Preserved the historical reconstructed prompt, available options and measured profile summaries for four variants, with two repeated calls per variant on actual live hardware telemetry.
- Historical prompt: both calls chose `dense_cpu1`; generated reason falsely called it the minimum-wall-time option even though the supplied profiles list a faster hybrid and sparse CPU choices. The prompt explicitly named `dense_cpu1` as baseline and listed candidates alphabetically with it first.
- Removing the explicit baseline anchor and reversing choices changed both outputs to `sparse_supervised_cpu1`. A compact mixed-order table changed both choices to `sparse_supervised_cpu2`, the first row. A compact fastest-first table changed both choices to the first row, `sparse_cuda_head_cpu_suffix`.
- On the later diagnostic snapshot, the RTX 2050 had only 1.72 GB available VRAM. The broker rejected both hybrid proposals because both repeated measured profiles needed more VRAM plus the reserve and guard. The action remained proposal-only; no fallback or training execution occurred.
- This shows the model had comparative measurements but used them unreliably; the explicit baseline anchor and option order bias the selection. When presented an explicit speed criterion, it selected the fastest option but failed to account for live VRAM feasibility. Broker preflight remains necessary.

## Broker repeat-profile safety correction

- Code inspection found proposal validation used only the last stored profile for each configuration, even when the candidate summary represented multiple repeats. This could ignore a less favorable repeat's peak resource use.
- Changed proposal validation to check every repeated profile, and to reject if any repeat fails live resource/integrity preflight or if repeats disagree on placement. Added a regression test.
- Regression suite: 19 passed, 1 skipped.

## Connector adaptation attempt

- Started the predeclared matched comparison: rank-8 layer-26 residual connector versus rank-8 layer-27 q-projection LoRA, 49,152 trainable parameters each, two fixed seeds, identical three train examples, separate three validation examples. Frozen-prefix activation capture did not finish after approximately 40 minutes on the one-thread CPU path.
- Stopped before any optimizer update, before writing either candidate adapter, and before any production graph change. Peak process RSS observed was about 6.45 GiB; minimum sampled system available RAM was about 7.18 GiB, above the 20% reserve. RTX 2050 was unused.
- No follow-up connector result is claimed; the earlier single-record result remains preliminary only.

## IMPRESSION adapter runtime probe

- Adapted the verified IMPRESSION-004 layer-27 q-projection LoRA tensors to the repository's GGUF merge format and created an evaluation-only Ollama tag `impression006-[BASE_MODEL]3-adapter-eval` from the exact [BASE_MODEL] Q4 base plus the converted delta.
- Verified merged `blk.27.attn_q.weight` exactly matches the expected Q8 quantization of dequantized Q4 base plus scaled/permuted LoRA delta. An untouched K-projection tensor remained byte-identical; the original base GGUF hash did not change. The Q8 conversion error was max abs 0.001169 and RMSE 0.0001187 relative to the unquantized target sum.
- In the existing IMPRESSION-004 HF fresh-process verification, the original safetensors adapter reloaded on the pinned raw BF16 backbone with RMS difference 0.0 on the held-out logits. This is the reference adapter reload parity result.
- Ran identical Ollama generation prompts on base and merged-adapter tags. For both the seen training prompt and the family-held-out prompt, text outputs matched exactly between tags. Both generations were semantically invalid: the seen task should abstain from an unauthorized deploy but instead produced an action/schema object; the held-out continuation task should return a typed resume plan but emitted a free-form `typedSymbolicPlan` object. A 128-token JSON-constrained run truncated; a 256-token unconstrained run was complete JSON but still failed the typed safety/intent checks. Generation quality therefore fails this small probe. Full-logit parity between converted Q4/Q8 GGUF and raw BF16 HF remains unmeasured.

## Uncompleted requirements

- No new family-matched repeated A/B/C optimizer-cycle trial was run. The only verified candidate cycle family is the existing one-step expression case. Reusing those historical costs for different symbolic tasks would be false attribution, and the connector capture pilot shows the CPU reference cost is substantial.
- No decision adapter training was run; the frozen family-disjoint decision split is intentionally empty for train and validation.
- Graph expansion/reconnection remain disabled through the production broker. No production loader, persistence transaction, resource-limited mutation, or production rollback path was tested.


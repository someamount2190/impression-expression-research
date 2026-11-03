# IMPRESSION-002D — Native Runtime Stabilization Report

**Status:** CPU streaming and saved-adapter reintegration are repeatable through the raw BF16 reader. The legacy Torch SafeTensors materialization path still has an intermittent native access violation. The single CUDA probe executed safely but failed its numerical acceptance threshold.

## Acceptance levels

| Level | Result | Evidence |
| --- | --- | --- |
| 1. Native failure isolated or root cause demonstrated | **Partial: isolated to a specific operation/path; root cause unresolved.** A worker reproduced `0xC0000005` while `safetensors.safe_open(framework="pt").get_tensor` materialized `IMP_L04_ATTENTION_INPUT_NORM`, in `torch/storage.py:456 __getitem__` via the 002B manager. It occurred with about 12.62 GB available RAM and about 860 MB child RSS. A separate Transformer lazy-import/block case emitted a faulthandler access-violation diagnostic despite exit 0. | Phase B matrix and Phase E Torch-backend failure receipts |
| 2. Worker crash-isolated and recoverable | **Pass.** The supervisor recorded a deliberately terminated child (exit 86), remained alive, and ran a subsequent benign child. It also survived the actual Torch-backed native fault and preserved partial receipts. The raw-reader path then completed real workloads. | `reports/PHASE_C_WORKER_SUPERVISION.md`; Phase E receipts |
| 3. Two complete CPU streamed forwards | **Pass.** Two fresh child processes each completed 28 layers and 255 operations over 254 unique registered partitions, with one-partition maximum residency, no remaining residents, no native fault, and no Transformers/Pandas/PyArrow imports. Outputs were byte-identical: max, mean, and RMS absolute difference all 0.0 for layer-27 hidden states and logits. | `reports/PHASE_D_PURE_TORCH_LAYER0.md`; `diagnostics/worker/phase_d_comparison/pass1_vs_pass2.json` |
| 4. Full-prefix saved-adapter reintegration | **Pass for the two preserved records.** The raw-reader worker recomputed each prefix from the pinned checkpoint. Fresh prefix activations matched cached BF16 activations exactly; base logits matched saved pre-update logits and adapter logits matched saved post-update logits exactly (all max/mean/RMS differences 0.0). | `reports/PHASE_E_SAVED_ADAPTER_REINTEGRATION.md`; raw-run child receipt |
| 5. Fresh streamed training-to-reload parity | **Not demonstrated in 002D.** No new training step was run, as required by the initial no-retraining plan. The preserved 002B adapter was freshly loaded and its inference effect was verified after recomputing prefixes, but this run did not capture a new optimizer update followed by reload. | Phase E scope and run receipts |
| 6. Genuine CUDA streaming operation | **Not passed.** A real registered [BASE_MODEL] q-projection ran on CUDA and cleaned up, but max absolute output error was 0.25 against a limit of 0.0078125. | `reports/PHASE_F_QPROJ_CUDA_PROBE.md` |

## Findings and required questions

1. **Was the Windows native crash reproduced?** Yes. The Torch-backed materialization worker terminated with `0xC0000005`; the supervisor preserved its status and logs.
2. **Which exact operation triggered it?** Materializing `IMP_L04_ATTENTION_INPUT_NORM` for `trace_schema_0034` through the existing Torch SafeTensors manager, at `torch/storage.py:456 __getitem__` from `experiments/impression_002b/residency/manager.py:234`. The same genuine tensor was subsequently read successfully by the raw BF16 reader. This localizes the failing route; it does not establish why the native extension fault occurs.
3. **Did import order affect the outcome?** The tested Torch+SafeTensors and Torch+Pandas+PyArrow import orders passed. A and I harness startup errors were corrected and rerun once. J returned a valid block output but emitted a native access-violation faulthandler stack during Transformers' lazy import. There is not enough evidence to conclude that import order causes the Torch storage crash.
4. **Were Pandas or PyArrow required?** No. The repository's pure-Torch and raw-reader streaming workers do not import either package. The Phase B J warning traversed a Transformers lazy import through Pandas/PyArrow, but isolated Pandas and PyArrow imports passed and this was a separate observed warning.
5. **Did the failure involve Torch tensor storage?** Yes. The preserved traceback entered Torch storage `__getitem__` during SafeTensors `get_tensor` materialization.
6. **Could the fault be isolated to a subprocess?** Yes. The native fault occurred in a child; the parent survived and produced a structured receipt.
7. **Did the supervisor survive worker failure?** Yes, including the deliberate exit-86 recovery test and the actual access-violation worker termination.
8. **Were genuine [BASE_MODEL] tensors repeatedly materialized and released?** Yes. Phase B loaded/released the nine layer-0 partitions. The raw CPU path streamed 254 registered partitions in each full forward and performed 532 operations across two adapter records. Each job ended with zero resident partitions.
9. **Did layer-0 loading become stable?** Yes on the isolated raw-reader/pure-Torch worker path. The legacy Torch-backed reader remains unsafe under the observed intermittent failure.
10. **Did two complete CPU streamed forwards succeed?** Yes, in independent fresh processes.
11. **What was their numerical agreement?** Outputs were byte-identical; max/mean/RMS absolute differences were 0.0. Each also exactly matched the preserved 002B streamed layer-27 hidden and logits artifacts.
12. **Was independent full-model parity established?** No. The 002B references and the two new passes use the same streamed implementation family. The verified independent official-model parity remains component/layer-level only.
13. **Could the saved layer-27 adapter be reintegrated after a fresh prefix?** Yes for both fixed records. Prefix activation, base logits, adapted logits, and adapter fingerprint matched their saved references exactly.
14. **Was a new streamed training update performed?** No. Checkpoint and adapter were preserved; no optimizer step occurred in 002D.
15. **Did training-to-reload parity pass?** Not as a new 002D training test. Inference after fresh-prefix recomputation and loading the saved 002B adapter passed exactly. The original 002B update is preserved but this run did not repeat its update-and-reload lifecycle. Recomputed causal CE differs from legacy receipt values by approximately `5e-5` to `6e-5`, so exact objective/label-mask equivalence remains unresolved.
16. **Did CUDA execute a genuine streamed operation?** Yes. The genuine registered layer-27 q-projection executed on the RTX 2050 through the raw residency path. Numerical acceptance failed, so Level 6 is not passed.
17. **What were the memory and execution costs?** Full CPU forward pass 1: 40.82 s in child / 44.19 s supervised, peak sampled child RSS 1,170,489,344 B; preflight available RAM 13,642,293,248 B versus 6,312,090,041 B required, including 20% reserve. The two-record adapter reintegration took 129.98 s supervised with one partition resident. CUDA probe: 3.27 ms transfer-in, 65.93 ms kernel, 0.136 ms transfer-out, 100.87 ms CPU compute, 198.96 ms total measured operation wall, and 27.59/41.94 MB peak allocated/reserved VRAM. CPU/CUDA max error 0.25 failed the 0.0078125 limit.
18. **Were previous IMPRESSION acceptance tests preserved?** Yes. Combined focused regression: **49 passed** across 001B parity, 002 controller, 002B, 002C, and 002D. Relevant EXPRESSION compatibility previously passed 27 with 1 skipped; it was not rerun in the final combined command. No repository-wide completion is claimed.
19. **Which problems remain?** Root cause of the Torch SafeTensors native fault; independent full-model reference parity; new streamed training-to-reload parity; CUDA max-error discrepancy; explanation of the small historical loss/CE telemetry mismatch. The Transformer lazy-import faulthandler warning also remains unexplained.
20. **What next experiment is justified?** Keep the raw reader as the stable CPU route while isolating the Torch/SafeTensors native build/runtime interaction in a pinned, disposable environment. Separately investigate the BF16 CUDA q-projection discrepancy using persisted elementwise mismatch indices/values and operation-level precision controls before any GPU training. Then add an independent feasible full-model reference and a fresh bounded training-to-reload test. Do not claim scheduler speedup or promote CUDA placement until correctness passes.

## Performance conclusion

No end-to-end efficiency gain was established. The two CPU runs measure reproducibility of one configuration, not a controlled comparison. The CUDA kernel was faster than the one CPU compute sample, but transfer and lifecycle costs made the complete measured operation 198.96 ms, and the output failed the required max-error criterion. No measured speedup is claimed.

## Reproduction and validation

From the repository root:

```powershell
python -m pytest experiments/impression_002d/tests -q --disable-warnings --maxfail=3
python -m pytest experiments/impression_001b/test_verify_parity.py experiments/impression_002/tests/test_impression_002_controller.py experiments/impression_002b/tests experiments/impression_002c/tests experiments/impression_002d/tests -q --disable-warnings --maxfail=3
```

The final combined focused run completed with `49 passed in 4.37s`. Phase-specific execution commands, inputs, and immutable receipts are preserved in the corresponding phase reports and `diagnostics/worker/`. One lower-cost executor was used sequentially; no concurrent executor was active. No usage or token-consumption estimate is claimed.

## Evidence index

- Baseline: `reports/NATIVE_RUNTIME_BASELINE.md`
- Isolation matrix: `reports/PHASE_B_NATIVE_RUNTIME_ISOLATION.md`
- Worker supervision: `reports/PHASE_C_WORKER_SUPERVISION.md`
- CPU streaming: `reports/PHASE_D_PURE_TORCH_LAYER0.md`
- Saved-adapter reintegration: `reports/PHASE_E_SAVED_ADAPTER_REINTEGRATION.md`
- CUDA probe: `reports/PHASE_F_QPROJ_CUDA_PROBE.md`
- Aggregate state: `run_state.json`

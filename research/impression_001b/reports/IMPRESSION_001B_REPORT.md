# IMPRESSION-001B — Deterministic Reintegration Report

**Outcome: full success for the training-to-reload parity acceptance condition.** The new controlled CPU training reference matches two independent fresh CPU reloads exactly. The historical 001A mismatch was not recreated from its deleted reference, but its cause was identified and corrected in the shared fresh-load code path.

## Root cause

IMPRESSION-001 trained and captured logits on CPU. The fresh-load path loaded the base with `device_map={"": "cpu"}`, then called PEFT 0.21.2 `PeftModel.from_pretrained` without a device map. PEFT detects a CPU/disk `hf_device_map` and defaults to automatic re-dispatch. On this machine that moved some model parameters to `cuda:0`, despite CPU-only training. This mismatch changed the inference execution path and logits. An instrumented run directly detected a CUDA parameter after adapter load; prior reloads also emitted CuBLAS warnings. The previous verification did not assert every post-PEFT parameter device.

Both fresh reload calls in [impression_selective_training.py](/[REDACTED_LOCAL_PATH]) now pass `device_map={"": "cpu"}`. The 001B harness checks that all parameters remain on CPU, hashes the backbone after adapter attachment, and refuses to overwrite references. Strict deterministic settings are recorded.

## Evidence

| Check | Result |
|---|---|
| Pinned checkpoint | `[BASE_MODEL_REPOSITORY]`, revision `392a6f57223e7ccfe6ef4ebdb2ff101a42d57364` |
| Update | One real optimizer step; layer 0 FFN gate LoRA; BF16 CPU |
| Trainable adapter parameters | 45,056 |
| Frozen base fingerprint | `d31f019f638f0f90d2fbf6d9ed05e2dc8653410b016a392c9a20ac3e32a70feb` before and after |
| Live vs saved adapter tensors | Exact SHA-256 match for both A/B tensors |
| Training process vs CPU fresh reload | Max/mean absolute diff 0.0; RMSE 0.0; 0 of 2,052,128 mismatches |
| Independent second CPU reload | Max/mean absolute diff 0.0; RMSE 0.0; 0 mismatches |
| Same-process unload/reload | Max/mean absolute diff 0.0; RMSE 0.0; frozen-base fingerprint unchanged |
| CPU baseline vs adapter disabled | Exact match, max diff 0.0 |

The successful training reference is [controlled-02-training-process](../references/controlled-02-training-process/). The isolated adapter is [controlled-02](../../../checkpoints/impression_001b/controlled-02/). The parity harness and immutable captures are in [verify_parity.py](../verify_parity.py) and [references](../references/).

The unfixed controlled pair showed max diff `0.5625`, mean diff `0.0421749`, and 88.9% of values outside tolerance, despite matching base and serialized-adapter hashes. After explicit CPU pinning, the same saved training reference matches exactly. The original IMPRESSION-001 training logits are still unrecoverable; the new controlled run supplies the valid replacement evidence rather than fabricating historical data.

## Regression results

Focused command:

```powershell
python -m pytest tests/test_impression_registry.py tests/test_impression_selective_training.py tests/test_distributed_transformer_invariants.py tests/test_distributed_training_convergence.py tests/test_expression_integration.py tests/test_expression_exec_hardening.py experiments/impression_001b/test_verify_parity.py -q
```

**43 passed, 1 skipped** in 38.67 seconds. Harness compilation and CLI help also pass. A separate full repository pytest process was observed running without output; because its outcome could not be read, no full-suite pass is claimed.

## Limits and next experiment

- The controlled update trained one adapter step on a single record and established reintegration correctness, not task improvement or generalization.
- No inference/training efficiency gain was measured. No placement optimization or Adaptive Efficiency Controller was implemented.
- Next justified work is a multi-example quality evaluation of the verified adapter and a matched device/precision performance baseline. Structural changes should wait for that evaluation.

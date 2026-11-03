# IMPRESSION-002C Phase B — Legacy Training Receipt

## Materialized receipt

`training/receipts/run02_legacy_training_receipt.json` is a truthful receipt for the saved IMPRESSION-002B run02 update. It reads immutable JSON metadata and existing NumPy logits only. It does not materialize Safetensors adapter tensors or load any backbone weights.

The receipt captures the pinned model revision and target, adapter file SHA-256 and size, initial/post tensor fingerprints from the saved metadata, sample token IDs and cache identities, and references to each existing pre/post logits array. Each logits reference includes its SHA-256, size, shape, dtype, and shifted causal-LM cross-entropy recomputed from the saved array. No large logits were copied into the 002C receipt.

Reload status remains `unverified`; the preserved result is `0xC0000005` from the old diagnostic. Any later verification output has a separate reserved path under `training/receipts/verification_results/`. Gradient norm, backward seconds, optimizer seconds, and optimizer-state persistence are explicit `null`/false because the original 002B runner exited before serializing them.

## Resource and dependency provenance

`resource_measurements.jsonl` has three provenance-linked records: run02 training preflight, the complete CPU streamed forward, and the CPU/CUDA single-Q-projection placement microbenchmark. The Q-projection GPU figures are clearly scoped to that component; they are not full-model measurements. Appending is fsynced and the materializer avoids duplicate measurement IDs.

The Phase A runtime inventory now includes pandas 2.3.3 and pyarrow 24.0.0, dependencies implicated by the earlier crash stack. Preserved IMPRESSION-001B manifests show Torch 2.5.1+cu121 and Transformers 4.56.2 but do not record pandas or PyArrow, so historical comparison for those two packages is unavailable.

## Atomic-write durability check

The focused test atomically writes a receipt that references an existing logits file, starts a child process which immediately exits with code 73, then confirms the receipt and referenced file remain readable and hash-valid. Test command and result:

```powershell
python -m pytest experiments/impression_002c/tests -q
```

Result: **3 passed**. The new Python files also passed `py_compile`. No training, model load, GPU operation, or verifier retry was part of Phase B.

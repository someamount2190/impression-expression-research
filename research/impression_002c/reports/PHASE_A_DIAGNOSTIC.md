# IMPRESSION-002C Phase A — Safetensors Read Diagnosis

## Scope

This phase inspected the preserved IMPRESSION-002B run02 adapter and prior crash evidence without changing the adapter or any 002B artifact. It did not load the backbone, train, or run a GPU/scheduler experiment.

The parent inspector uses only the Python standard library for Safetensors header parsing, raw tensor/file hashing, and distribution-version discovery. The materialization checks ran in isolated child processes with `-X faulthandler`; both subprocess command lines, standard output/error, return codes, and elapsed times are preserved in [phase_a_safetensors_numpy_diagnostic.json](../diagnostics/phase_a_safetensors_numpy_diagnostic.json).

## Evidence

- Saved adapter header contains `lora_A` F32 `[4, 3072]` and `lora_B` F32 `[3072, 4]`, 49,152 bytes each. Whole adapter file SHA-256: `1aa308661c400a08fe664dd147c6f56707f74e64c367d801d1739ff7f4f46558`.
- Stdlib raw-data hashing and isolated NumPy materialization agree on tensor hashes. The combined adapter fingerprint equals the immutable metadata value `195ca27851660551ed8e639e91c9ab727ed509e5e5c85697c399080124332850`.
- Adapter subprocess: `safe_open(framework="numpy", device="cpu")`, exit 0, both tensors read with expected shapes/dtypes, fingerprint matched, no stderr.
- Because that succeeded, one distinct child attempted exactly one pinned layer-27 norm tensor selected by `model.safetensors.index.json`: `model.layers.27.input_layernorm.weight`, 3,072 BF16 values / 6,144 logical bytes. It exited 1 with Python `TypeError: data type 'bfloat16' not understood` in the NumPy backend. It did not crash and had no NTSTATUS.
- Prior 002B Torch-backed failure remains `0xC0000005`, with faulthandler pointing to Torch storage indexing inside `ResidencyManager.load`. It was inspected, not repeated.
- Runtime: Python 3.10.6, safetensors 0.6.2, NumPy 2.1.2, Torch 2.5.1+cu121, Transformers 4.56.2, huggingface-hub 0.35.1, psutil 7.1.0. These versions were obtained through `importlib.metadata` without importing Torch or Transformers in the inspector process.

## Diagnosis and limit

The adapter result shows that this adapter can be materialized by the NumPy Safetensors backend and that the saved bytes match their metadata fingerprint. This makes the prior adapter-read crash consistent with a Torch-storage-path-specific issue. It does **not** prove causation: the sole BF16 checkpoint-tensor comparison could not materialize because the installed NumPy backend does not recognize `bfloat16`. Therefore, the diagnosis of general checkpoint Safetensors materialization remains unresolved. No fallback, repeat, Torch retry, training, GPU, scheduler, or full model was attempted.

## Validation

The focused artifact test checks these saved facts without loading weights. Run it from the repository root:

```powershell
python -m pytest experiments/impression_002c/tests/test_phase_a_diagnostic_artifact.py -q
```

# IMPRESSION-002D Phase A — Native Runtime Baseline

## Collection scope

This is a metadata-only baseline. The capture script used Python standard-library APIs and `importlib.metadata`, Windows `GlobalMemoryStatusEx`, file metadata/hashes, JSON parsing for the checkpoint configuration/index, and `nvidia-smi`. It did **not** import Torch, Transformers, NumPy, Safetensors, pandas or PyArrow; did not read checkpoint tensor payloads; and did not modify the checkpoint or adapter.

The machine-readable record is [runtime_baseline.json](../runtime_baseline.json). Recreate it from the repository root with:

```powershell
python experiments/impression_002d/capture_runtime_baseline.py
```

## Host and runtime

- Microsoft Windows 11 Home Single Language, build `26200` (caption from `Win32_OperatingSystem` CIM); CPython 3.10.6, 64-bit AMD64. Python reports platform version `10.0.26200`, which is the Windows compatibility version, not the product caption.
- Physical RAM: 24,919,851,008 bytes; available at capture: 13,630,603,264 bytes.
- NVIDIA GeForce RTX 2050; driver 581.83; 4,096 MiB VRAM total, 3,806 MiB free at capture. `nvidia-smi` reports CUDA compatibility version 13.0.
- Installed PyTorch distribution is `2.5.1+cu121`, identifying its CUDA 12.1 wheel build. This is separate from the driver’s CUDA 13.0 compatibility header; Phase A did not import PyTorch or initialize a CUDA runtime.
- Previously captured Phase D/E preflight (not rerun here) reported device capability 8.6 and BF16 support, 3,461,768,807 bytes VRAM free, and 287,358,976 bytes required for the Q weight, activation and 256 MiB headroom.

## Runtime dependency path

The source for `streamed_placement.py` imports local 002B streaming/residency modules and calls those modules along the pass path. The direct critical-path dependencies, according to those source imports, are:

| Dependency | Installed version | Runtime source path |
|---|---:|---|
| PyTorch | 2.5.1+cu121 | `PartitionResidencyManager.load`, CPU/CUDA tensor operations |
| Safetensors | 0.6.2 | `residency/manager.py` imports `safetensors.safe_open` for tensor loads |
| NumPy | 2.1.2 | `run_full_forward.py` and `run_block_parity.py` import NumPy directly |
| psutil | 7.1.0 | Residency/resource measurements and preflight |
| Transformers | 4.56.2 | Runner imports `LlamaConfig`; streamed block imports the Llama norm, RoPE and attention helpers |

The captured warmup stack also shows lazy imports through `transformers.utils.import_utils` into `pandas.compat.pyarrow`, then the native `pyarrow` extension. Pandas 2.3.3 and PyArrow 24.0.0 were installed. Neither is imported directly by the repository runner or residency manager. The preserved Phase D/E stack contains both the PyArrow initialization access violation and a Torch storage indexing access violation in manager tensor loading; the artifacts do not establish the causal order or whether one triggered the other. Transformers distribution metadata and its declared requirements are included in the JSON baseline, while the crash stack provides the observed runtime import evidence.

## Pinned artifacts

- [BASE_MODEL] checkpoint: `[BASE_MODEL_REPOSITORY]`, revision `392a6f57223e7ccfe6ef4ebdb2ff101a42d57364`.
- Model config SHA-256: `e21ff53ea39726f972362beba869807216775d5e308bc2f531784846c06a0249`.
- Safetensors index SHA-256: `e6322ad005d27134ba439f7ddd572e632e5b2116ebfbdc88186596dada904c75`.
- Index reports 254 tensor names over 2 shards; shard sizes were obtained by file stat only. Neither shard payload was read.
- Saved 002B layer-27 Q-projection adapter is 98,488 bytes; file SHA-256 `1aa308661c400a08fe664dd147c6f56707f74e64c367d801d1739ff7f4f46558`. Metadata lists initial/post tensor fingerprints `349b671e…09956e` and `195ca278…332850`; no adapter tensors were loaded in Phase A.

## Existing evidence boundary

The 002B bounded residency probe, CPU 28-layer forward and exact layer-27 block parity are preserved. Independent full-model parity remains unresolved. 002C Phase C reloaded the saved adapter through the raw reader and exactly reproduced cached pre/post logits for both preserved examples. The later streamed placement experiment passed metadata RAM/VRAM preflight but stopped on its first CPU warmup’s native access violation; it ran no CUDA child, timed comparison, or controller selection. Consequently, no new end-to-end placement gain was measured.

Focused baseline schema test:

```powershell
python -m pytest experiments/impression_002d/tests/test_runtime_baseline.py -q
```

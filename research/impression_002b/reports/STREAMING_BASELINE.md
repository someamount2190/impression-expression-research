# IMPRESSION-002B Streaming Baseline

## Provenance and model

- GGUF structural registry: `experiments/impression_001/partitions/[BASE_MODEL]3_q4_registry.json`, 255 registered tensors/partitions, source SHA-256 `5d6d02eeefa1ab5dbf23f97afdf5c2c95ad3d946dc3b6e9ab72e6c1637d54177`.
- Full precision checkpoint: `[BASE_MODEL_REPOSITORY]`, immutable revision `392a6f57223e7ccfe6ef4ebdb2ff101a42d57364`, pinned BF16 Safetensors shards and exact hashes recorded in `experiments/impression_001/reports/selective_training.json`.
- Verified architecture metadata against registry in IMPRESSION-001: 28 decoder layers, hidden size 3072, intermediate size 8192, 24 attention heads, 8 key/value heads.
- IMPRESSION-001B reports exact CPU train/reload logit parity (max and mean absolute difference 0 across 2,052,128 values); focused regression evidence from the immediately preceding IMPRESSION-002 run is 43 passed, 1 skipped.
- IMPRESSION-002 controller, immutable structural configuration store, append-only ledger, and local component-probe artifacts are present under `experiments/impression_002/`.

## Runtime and resource state at 2026-11-03

- Host: Windows PC; system RAM 24,919,851,008 bytes (about 23.2 GiB); sampled free RAM during 002B setup varied around 11.5–12.4 GB.
- GPU: NVIDIA GeForce RTX 2050, 4,096 MiB VRAM; sampled free VRAM 3,806 MiB. Full BF16 parameter lower bound is 6,425,512,064 bytes, so full model GPU residency cannot fit.
- Required reserve retained from 002: 20% system RAM (about 4.98 GB).
- An unrelated full pytest process PID 9176 was found alive, at approximately 2.77 GB RSS and low CPU. It was left untouched. This is included in the observed system memory state and must not be represented as an experiment-owned workload.
- The existing `llm_inference/distributed_transformer.py` contains demo-sized 512-wide worker fragments and hard-coded generated responses. It is not a [BASE_MODEL] checkpoint executor and is not used as a reference.
- No active IMPRESSION training process was found at 002B start. No IMPRESSION-001B implementation or checkpoint was modified.

## Baseline decision

The full AutoModel CPU load path in `training/impression_selective_training.py` has a measured conservative 11.39 GB load peak before optimizer state/activations and a 4.98 GB reserve; at current free memory it is infeasible. 002B must avoid full-model construction and use the immutable Safetensors index to load only selected registered tensors. CPU is the safe initial placement. Any GPU trial must preflight the actual selected working set and preserve 256 MiB VRAM headroom.

The existing 001B CPU adapter reload parity remains the compatibility baseline. Full streamed-model parity is a new, separate requirement and must be measured against an independent component/reference implementation with identical tensors and execution settings.

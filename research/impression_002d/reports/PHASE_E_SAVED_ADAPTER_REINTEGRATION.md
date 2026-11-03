# IMPRESSION-002D Phase E — Saved Adapter Reintegration

## Scope and preserved inputs

This phase attempted read-only reintegration of the preserved IMPRESSION-002B rank-4 LoRA adapter for `model.layers.27.self_attn.q_proj`. It used both actual 16-token task records (`trace_edge_0035` and `trace_schema_0034`), their cache identities, and saved pre/post logits. The checkpoint and adapter were not modified; no optimizer step or GPU execution occurred.

## First CPU worker result: Torch SafeTensors backend

The single authorized full-prefix attempt is preserved at `diagnostics/worker/phase_e_adapter_reintegration_run01/`. Preflight passed with 13.70 GB available against 6.34 GB required, including a 20% RAM reserve, a 788,017,152-byte largest partition, a 536,870,912-byte runtime margin, saved activation caches, adapter tensors, and logits comparison buffers.

`trace_edge_0035` completed durably. The fresh streamed prefix matched its cached layer-27 input exactly after BF16 conversion. Base logits matched preserved pre-update logits exactly; adapter logits matched preserved post-update logits exactly. For both comparisons, max, mean, and RMS absolute differences were 0.0. The adapter fingerprint matched the preserved metadata. Its outputs, hashes, CE measurements, and per-partition telemetry are stored in the child receipt and output files.

The worker then terminated with Windows access violation `0xC0000005` while loading `IMP_L04_ATTENTION_INPUT_NORM` for `trace_schema_0034`. The faulthandler stack identifies `torch/storage.py:456 __getitem__`, reached through `experiments/impression_002b/residency/manager.py:234`. It occurred during tensor materialization, before a hash or successful load measurement for that attempted tensor. The last complete partition operation was layer 3 down projection. The second record is incomplete; no result is claimed for it. Immediately before the crash, free RAM was about 12.62 GB of 24.92 GB and child RSS about 860 MB, so measured memory pressure was not close to the reserve threshold. The parent supervisor remained alive and wrote exit/NTSTATUS, stdout/stderr, resource samples, and partial child receipt.

## Raw reader follow-up

Added `worker/raw_partition_manager.py`, which reuses the strict 002C raw BF16 parser, caches parsed shard headers, validates pinned revision/index/path, registered shape/dtype/size, content hashes, and RAM reserve, then releases its manager-owned mapping. It supports one CPU resident partition and mirrors the existing manager's lifecycle and telemetry interface.

The exact layer-4 norm partition that crashed above was loaded once through the raw reader in a separate supervised child. It passed shape/dtype and reserve checks, produced the same partition digest as the earlier Torch-backed successful loads (`1089a66f...e7720d8`), and ended with no resident partitions.

After exact snapshot-directory-name validation was added, one separately authorized raw-backend two-record worker run passed in `diagnostics/worker/phase_e_adapter_reintegration_raw_run01/`. It completed 532 serial operations, used a maximum of one resident partition, and ended with no resident partitions. Supervisor exit was 0 with no NTSTATUS and empty stderr; Transformers, pandas, and PyArrow were not imported. Preflight measured 13,524,000,768 B available versus 6,340,857,561 B required, including a 4,983,970,201 B reserve and 512 MiB explicit headroom.

Both real 16-token task records completed independently. Fresh prefix activations exactly matched the preserved BF16 caches. For each record, base logits exactly matched its saved pre-update logits and adapter logits exactly matched saved post-update logits; max, mean, and RMS absolute differences were 0.0. Adapter fingerprint matched metadata. The causal CE recomputed from those logits is recorded beside the legacy receipt's loss evidence. The values differ by roughly `5e-5` to `6e-5`; exact logits parity does not establish that the original training loss used the same label mask/objective details, and the earlier receipt lacks enough telemetry to resolve that difference. Recomputed adapter CE improved `trace_edge_0035` from 7.8813562 to 7.8810067 and rose on `trace_schema_0034` from 6.6229019 to 6.6229620.

## Validation

- `python -m py_compile` passed for the changed/new Phase E worker, supervisor, and tests.
- `python -m pytest experiments/impression_002d/tests -q`: **19 passed**.
- Tests cover cache identity mismatch rejection, the durable exact first-record result and explicit native-crash status, raw reader load/release/hash parity against a genuine layer-0 receipt, malformed shape and revision-path rejection, and the successful layer-4 raw-reader artifact.

## Conclusion

Both saved records now have exact end-to-end streamed adapter reintegration evidence through the raw Safetensors reader. The first Torch-backed attempt remains preserved as a reproducible native crash and was not retried on that backend. This is verification of the saved adapter's effect on these two fixed records, not a training result, a model-wide independent parity claim, or an efficiency measurement.

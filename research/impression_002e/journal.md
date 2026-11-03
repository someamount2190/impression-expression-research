# IMPRESSION-002E Journal

## Scope and controls

- Continued from IMPRESSION-002D at pinned revision `392a6f57223e7ccfe6ef4ebdb2ff101a42d57364`.
- The raw BF16 partition reader remains the execution path; the legacy Torch-backed SafeTensors materializer remains quarantined.
- No checkpoint or existing adapter was modified. One training update was saved as a new adapter and verified by a fresh process.
- The official Transformers layer-27 boundary reference is exact for the preserved activation. Full-model official reference remains open.

## Streamed training acceptance

- Attempts 01 and 02 failed before a durable completed receipt (read-only descriptor `fsync`, then transient Windows `os.replace` permission failure); both attempts and diagnostics were preserved.
- Attempt 03 completed one AdamW LoRA update and passed fresh-process reload parity for both selected records. Prefix hashes matched, post-update logits matched exactly, frozen suffix fingerprints were unchanged, and all resident partitions were released.
- Training loss decreased by `0.0003495216`; held-out loss increased by `0.0000600815`. This is one update and does not establish generalization improvement.

## Numerical and placement evidence

- The native CUDA BF16 q_proj path remains rejected: it violates the cancellation-sensitive accumulation bound.
- The corrected registered q_proj probe (BF16-rounded input/weight, FP32 accumulation, BF16 output) completed under the supervised worker. CPU and CUDA each had zero errors beyond the combined local BF16 rounding plus `gamma_(2n)` FP32 accumulation bound. CUDA had five strict nearest-even output differences near rounding midpoints; these are within the combined bound.
- Measured operation costs include partition load, operand conversion, compute, input/output transfers where applicable, and release. CPU: `0.1117164 s`; CUDA: `0.1639099 s` (about 46.7% slower). CUDA peak allocated VRAM: `65,732,608 B`; final resident partitions: none.
- The controller therefore retained CPU. No efficiency gain was measured; partitioning alone is not evidence of a speedup.
- `phase_g_candidate_run01` failed at parent import resolution before child work. `run02` exposed an overly strict nearest-even acceptance test; its genuine measurement was preserved. The acceptance rule was corrected to use the combined numerical error bound, then `run03` passed and produced the controller selection artifact.

## Verification

- `python -m pytest experiments\\impression_002d\\tests experiments\\impression_002e\\tests -q`: **39 passed**.
- Python compilation passed for the corrected CUDA worker, supervisor, and controller profile adapter.
- Evidence: `diagnostics/phase_g_candidate_run03/phase_g_qproj_cuda_fp32/child_receipt.json` and `controller_selection.json`.

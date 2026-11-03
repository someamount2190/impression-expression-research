# IMPRESSION-002D Phase C — Supervised Partition Worker

## Implementation

`worker/supervised_worker.py` provides a standard-library supervisor and a fresh child process for each batch. The parent writes an atomic request, enforces a timeout, samples child RSS and system RAM, captures stdout/stderr and exit status, reads the child's atomic receipt, and records timeout, abnormal exit, native crash, or success. On abnormal exit the parent remains available to start another child.

The child validates the checkpoint revision, explicit registered partition IDs, and exactly 20% physical-RAM reserve before importing the residency manager. It loads the batch sequentially on CPU with a one-partition cap, records a pre-operation marker before every load/release, and releases each partition before continuing. Receipts include resource measurements, partition hashes, tensor names/shapes/dtypes/byte counts, and last attempted/completed operations. The parent receives only JSON metadata; tensor objects and tensor values never cross process boundaries. The worker path imports no Transformers modules.

## Recovery exercise

The isolated recovery test deliberately terminated its first child with exit code 86 before Torch or Transformers imports. The supervisor recorded `abnormal_exit_before_completion` and retained the child's pre-import receipt. The same parent then launched a separate benign child job successfully; parent PID remained unchanged. This test does not access a checkpoint tensor.

## Genuine pinned workload

One supervised worker job loaded and released all nine registered IMPRESSION layer-0 partitions from the pinned revision `392a6f57223e7ccfe6ef4ebdb2ff101a42d57364`.

- Parent supervisor status: passed; child exit 0; no timeout or NTSTATUS.
- Runtime: 4.05 seconds; peak sampled child RSS: approximately 384.6 MB; final child RSS: 371,359,744 B.
- Before materialization: available physical RAM was 13,639,163,904 B; 20% reserve was 4,983,970,201 B; preflight passed.
- All nine requested partitions ended `UNLOADED`; final resident partition list was empty.
- Per-tensor metadata and SHA-256 fingerprints are in `diagnostics/worker/genuine_layer0/child_receipt.json`.
- The supervisor's only child response was compact JSON; tensor contents were not returned.

This confirms bounded sequential loading/release for this layer-0 batch in a worker process. It does not establish full-model execution safety or performance gain. No Transformers model construction, GPU work, training, or checkpoint modification occurred in Phase C. The Phase B J warning remains recorded separately: its block computation exited successfully while faulthandler stderr contained a native access-violation stack from Transformers' lazy Pandas/PyArrow import.

## Validation

`python -m pytest experiments/impression_002d/tests -q` passed all focused tests. `py_compile` passed for the worker and its test. JSON parsing passed for supervisor/child receipts and run state.

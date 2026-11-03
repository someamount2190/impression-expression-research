# IMPRESSION-002C Phase D/E — Streamed Placement Attempt

## Implementation

Added `streamed_placement.py` without changing IMPRESSION-002B. The runner reuses the 002B block streamer for ordinary CPU blocks and implements an additive layer-27 path that can place only the genuine registered q_proj operation on CUDA. The CPU manager remains responsible for one registered partition at a time; the CUDA path explicitly transfers the activation and Q weight to the RTX 2050, synchronizes around measured transfers/kernel work, returns the result to CPU, then releases the partition. The CPU route remains available.

The runner records end-to-end forward wall time including partition load, release, transfer and compute, plus per-partition memory and concurrency. If both placements complete, matched profiles are passed to the existing IMPRESSION-002 controller; its choice and measured effect determine the claim field. Static tests cover controller choice/CPU fallback and the benchmark schema.

## Preflight and stop

Before model work, metadata-only checks found both routes feasible:

- RAM available 13,916,360,704 bytes; 20% reserve 4,983,970,201 bytes; required available RAM 6,310,431,129 bytes.
- RTX 2050, compute capability 8.6; CUDA BF16 supported; 3,461,768,807 bytes free VRAM.
- Q projection weight + activation + 256 MiB allocator headroom: 287,358,976 bytes.
- Adapter not used; adapter allocation 0 bytes.

The first CPU warmup subprocess exited `3221225477` (`0xC0000005`) before a forward completed. Its faulthandler output contains an access violation during PyArrow initialization through pandas/Transformers and another in Torch storage indexing at `PartitionResidencyManager.load` on the first streamed layer-0 tensor. The captured evidence does not distinguish their causal sequence. The full attempt, including stdout, stderr, command, exit code and NTSTATUS, is preserved in `benchmarks/streamed_placement/comparison.json`; preflight is in `preflight.json`.

The native crash stopped the experiment. No CUDA child, full-pass baseline, timed repeat, or controller selection was started. There is no full-pass parity result and no measured end-to-end efficiency gain. This phase makes no Level 5 claim. A future run requires a separately authorized fix to the native runtime/reader path before placement comparisons resume.

## Validation

Before the warmup, the 002C suite had **11 passing tests** and all 11 002C Python files passed `py_compile`. The added crash-artifact test verifies the preserved CPU failure and confirms no CUDA run was attempted. After adding that test, run:

```powershell
python -m pytest experiments/impression_002c/tests -q
python -m py_compile experiments/impression_002c/streamed_placement.py experiments/impression_002c/tests/test_streamed_placement_scheduler.py experiments/impression_002c/tests/test_streamed_placement_crash_artifact.py
```

Result after adding the crash-artifact assertion: **12 passed**. `py_compile` passed for all 12 IMPRESSION-002C Python files; `run_state.json` parses successfully.

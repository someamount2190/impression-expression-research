# IMPRESSION-002D Phase B — Native Runtime Isolation (A–G)

## Scope

The Phase B parent process uses only Python standard-library modules. Each matrix entry runs in a fresh child with `faulthandler`, unbuffered output, flushed pre-operation markers, a per-case timeout, sampled child working-set RSS and system available RAM, and captured stdout, stderr, exit code, and NTSTATUS. Child result and crash receipts are written atomically. Each test configuration is recorded once; the initial A command-construction failure is preserved as a harness startup failure, then A was re-run once after correcting that harness defect.

Pinned checkpoint: `[BASE_MODEL_REPOSITORY]`, revision `392a6f57223e7ccfe6ef4ebdb2ff101a42d57364`.

## Results

| Case | Operation | Result | Wall seconds | Peak sampled RSS |
|---|---|---:|---:|---:|
| A | Import PyTorch only | Passed | 3.28 | 368,316,416 B |
| B | Import Safetensors only | Passed | 0.22 | 2,265,088 B |
| C | Import Pandas only | Passed | 1.24 | 73,138,176 B |
| D | Import PyArrow only | Passed | 0.42 | 30,138,368 B |
| E1 | PyTorch then Safetensors | Passed | 3.26 | 367,755,264 B |
| E2 | Safetensors then PyTorch | Passed | 3.25 | 368,386,048 B |
| F1 | PyTorch, Pandas, PyArrow | Passed | 3.90 | 409,812,992 B |
| F2 | PyArrow, Pandas, PyTorch | Passed | 3.86 | 409,018,368 B |
| G | Pinned layer-0 tensor metadata via SafeTensors `get_slice` | Passed | 3.27 | 368,349,184 B |
| H | Materialize one genuine layer-0 input norm tensor via SafeTensors Torch backend | Passed | 3.25 | 368,975,872 B |
| I (initial) | Sequentially load/release nine registered layer-0 partitions | Harness import failure before manager entry | 0.21 | 2,379,776 B |
| I (corrected) | Sequentially load/release nine registered layer-0 partitions | Passed | 4.08 | 384,577,536 B |
| J | Execute one genuine streamed layer-0 block on pinned eight-token input | Child passed; stderr contains native AV faulthandler diagnostic | 25.10 | 1,605,414,912 B |

All completed children exited 0. No timeout or native NTSTATUS occurred. Every child wrote a pre-operation marker. G read the SafeTensors header/slice metadata for `model.layers.0.input_layernorm.weight`; H then materialized exactly that tensor: BF16 `[3072]`, 6,144 logical bytes. H's RAM preflight observed 13,543,165,952 B available against 5,051,085,209 B required, including a 20% physical-memory reserve of 4,983,970,201 B.

## Preserved harness startup issue

The first A attempt exited with code 2 in argparse before importing any library because the parent command omitted the `--test` option. Its full stdout/stderr and return code remain in `diagnostics/phase_b_matrix/A/result.json`. After correcting command construction, A ran in `diagnostics/phase_b_matrix/A_attempt02/` and passed. This is reported as a harness repair, not a runtime failure.

## Boundary and next review

I's initial child exited 1 before manager entry because the repository root was not in child `sys.path`; its artifact remains preserved in `I/`. The harness path was corrected and a second, authorized I execution passed in `I_attempt02/`: all nine genuine layer-0 partitions were loaded and released serially, each was UNLOADED after release, the final resident list was empty, and the configured manager cap was one partition. Available system RAM remained approximately 13.3–13.5GB against a 20% reserve of 4,983,970,201 B. J then ran in one authorized child. Its preflight observed 12,424,765,440 B available against 6,308,858,265 B required (including 20% reserve) before embedding allocation. It executed the pinned eight-token input through a genuine layer-0 block, returned finite `[1,8,3072]` output from `[1,8,3072]` input, processed nine registered partitions with maximum concurrency one, and ended with no residents. J exited 0 in 25.10 seconds with peak sampled RSS 1,605,414,912 B. However, its stderr contains a `Windows fatal exception: access violation` faulthandler stack during Transformers lazy import via pandas/PyArrow. The child continued and completed the block, but the combination is ambiguous and is recorded as a native diagnostic despite the normal exit. No later workload was run. Review J's full stderr in `diagnostics/phase_b_matrix/J/result.json` before authorizing further work. The H/I/J observations do not establish a full model pass or performance gain.

No performance gain was measured. No training or GPU workload ran, and no checkpoint or adapter content was modified.

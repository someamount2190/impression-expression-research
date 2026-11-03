# IMPRESSION-002D Phase F — Single q_proj CUDA Probe

## Scope

One supervised CPU/CUDA placement probe used the genuine registered `IMP_L27_ATTENTION_QUERY_PROJECTION` BF16 partition and the validated cached layer-27 activation for `trace_edge_0035`. It did not load an adapter, train, run the full model, or import Transformers, pandas, or PyArrow. The partition was loaded and released through `RawPartitionResidencyManager`.

## Preflight and execution

Preflight passed before checkpoint tensor materialization:

- CPU RAM: 13,373,267,968 B available; 5,540,108,697 B required, including 20% reserve, 512 MiB headroom, q_proj weights, and activation allowance.
- RTX 2050: 3,461,768,807 B free of 4,294,443,008 B; 287,703,040 B required, including the q_proj weights, activation/output/temporary allowance, and 256 MiB headroom.

The worker computed the same BF16 linear projection on CPU and CUDA, synchronized around CUDA transfers and kernel timing, transferred the CUDA result back to CPU, and released all partition references. CPU output shape/dtype and CUDA output shape/dtype were both `[1,16,3072]` BF16. The worker exited with code 1 after recording the failed quality assertion; there was no native fault or NTSTATUS.

## Measurements and acceptance

| Measurement | Result |
| --- | ---: |
| CPU q_proj compute | 100.87 ms |
| CPU→CUDA transfer, weight + activation | 3.27 ms |
| CUDA kernel | 65.93 ms |
| CUDA→CPU output transfer | 0.136 ms |
| Measured operation wall time, including raw load and release | 198.96 ms |
| CUDA peak allocated / reserved VRAM | 27.59 MB / 41.94 MB |
| Max / mean / RMS absolute output difference | 0.25 / 0.00000720 / 0.001148 |
| Required maximum absolute difference | ≤ 0.0078125 |

The maximum error exceeded the declared threshold, so the CUDA placement did **not** pass numerical acceptance. The mean and RMS errors were much smaller, but they do not replace the maximum-error criterion. No scheduler choice or speedup claim is made from this probe. The CPU compute was about 100.9 ms and the CUDA kernel about 65.9 ms, while the full measured load/transfer/compute/release operation took about 199.0 ms; this single run is not a repeatable end-to-end speedup result.

All evidence is preserved in `diagnostics/worker/phase_f_qproj_cuda_run01/`, including preflight, transfer/kernel timings, VRAM peak, output comparison, supervisor exit, and final empty residency list. No retry was run.

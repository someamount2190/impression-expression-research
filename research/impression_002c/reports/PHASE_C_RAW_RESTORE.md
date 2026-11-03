# IMPRESSION-002C Phase C — Raw Restore and Cached-Input Verification

## Raw BF16 norm probe

Added a strict read-only reader at `training/raw_safetensors.py`. Before reading tensor bytes it checks pinned snapshot directory/revision, the exact index entry, resolved shard confinement, Safetensors header length and JSON, offsets and overlapping ranges, expected tensor shape/dtype/byte size, little-endian host, and current RAM reserve. It then maps the shard, copies only the selected raw BF16 bytes into a NumPy `uint16` buffer, and views that buffer as a Torch BF16 tensor. It does not use SafeTensors' Torch backend.

The one 6 KiB `model.layers.27.input_layernorm.weight` probe did materialize. Its raw SHA-256 was `18ea0d7ad090805b9e0acd48d595b39f90c9dc22ad92b4bfdb8fea41dd168908`. The subprocess initially exited 1 because the check compared this raw hash directly with the manager's *partition aggregate* hash. The preserved manager code hashes the ASCII tensor-digest string for a one-tensor partition; computing `SHA256(ASCII_HEX(raw_sha256))` yields exactly `ad3f8ff76e6837769d472d56aea6ab7e8097f1ae10aedd4bce72dc142e10dde9`. This correction uses only the hash captured by the first probe; no tensor reread or retry was performed. Original failure and correction are separately recorded in `training/verification_results/phase_c_raw_norm_probe.json` and `phase_c_raw_norm_probe_resolution.json`.

## One isolated saved-adapter evaluation

After correcting the digest interpretation, the one allowed evaluator ran in a fresh CPU process. Its metadata-only preflight passed before tensor allocation:

- Available RAM: 13,779,054,592 bytes.
- Reserve: 4,983,970,201 bytes (20% of physical RAM).
- Exact loaded tensor set: 989,362,176 bytes (layer 27, final norm, tied embedding).
- Additional headroom: 512 MiB, activation estimate: 512 MiB.
- Required available RAM: 7,047,074,201 bytes.
- Peak sampled process RSS: 1,443,303,424 bytes; minimum sampled free RAM: 12,804,222,976 bytes.

The saved adapter was read with the NumPy backend validated in Phase A; its tensor fingerprint matched metadata. Pure Torch CPU operations (without Transformers) evaluated existing layer-27 input caches. For both `trace_edge_0035` and `trace_schema_0034`, recomputed **pre-adapter and post-adapter logits were exactly equal** to the saved arrays (max and mean absolute difference 0.0). Causal losses also matched. Prefix layers were not recomputed, no training or GPU work ran, and this does not claim full-model parity for a newly computed prefix.

The isolated result and subprocess stdout/stderr/exit code are stored in `training/verification_results/phase_c_legacy_adapter_verification.json`. The Phase B training receipt remains immutable and retains its truthful as-of-materialization `unverified` status; this later verification is a separate artifact. No 002B file was changed.

## Tests

Parser tests use tiny synthetic headers only and exercise valid fields, dtype/shape mismatch, invalid/overlapping offsets, wrong revision, and shard path escape. Artifact tests validate the preserved norm comparison correction, exact two-record logits result, and separation from the legacy receipt. Run from the repository root:

```powershell
python -m pytest experiments/impression_002c/tests -q
python -m py_compile experiments/impression_002c/phase_a_inspect.py experiments/impression_002c/training/receipt_utils.py experiments/impression_002c/training/materialize_legacy_receipt.py experiments/impression_002c/training/raw_safetensors.py experiments/impression_002c/training/phase_c_raw_verify.py experiments/impression_002c/tests/test_phase_a_diagnostic_artifact.py experiments/impression_002c/tests/test_training_receipt.py experiments/impression_002c/tests/test_raw_safetensors.py experiments/impression_002c/tests/test_phase_c_verification_artifacts.py
```

Result: **8 passed** across the focused 002C test directory; `py_compile` passed for all nine 002C Python files listed in the final check.


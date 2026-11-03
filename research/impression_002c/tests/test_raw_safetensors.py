"""Synthetic metadata fixtures exercise parser invariants without model reads."""
import json
from pathlib import Path
import struct

import pytest

from experiments.impression_002c.training.raw_safetensors import (
    RawTensorError, read_header, tensor_location, validate_tensor_spec, validate_snapshot,
)


def _write_tensor(path: Path, *, name="w", dtype="BF16", shape=(2,), offsets=(0, 4), raw=b"\0\0\0\0"):
    header = {name: {"dtype": dtype, "shape": list(shape), "data_offsets": list(offsets)}}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw)


def test_header_and_tensor_spec_validation(tmp_path):
    path = tmp_path / "tiny.safetensors"
    _write_tensor(path)
    header, start, size = read_header(path)
    assert start == 8 + len(json.dumps({"w": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}, separators=(",", ":")).encode())
    assert size == start + 4
    assert validate_tensor_spec(header["w"], (2,), "BF16") == 4
    with pytest.raises(RawTensorError, match="shape mismatch"):
        validate_tensor_spec(header["w"], (3,), "BF16")
    with pytest.raises(RawTensorError, match="dtype mismatch"):
        validate_tensor_spec(header["w"], (2,), "F16")


def test_header_rejects_offsets_outside_data_and_overlapping_ranges(tmp_path):
    out = tmp_path / "outside.safetensors"
    _write_tensor(out, offsets=(0, 5), raw=b"\0\0\0\0")
    with pytest.raises(RawTensorError, match="out-of-range"):
        read_header(out)
    overlap = tmp_path / "overlap.safetensors"
    header = {"a": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
              "b": {"dtype": "BF16", "shape": [2], "data_offsets": [2, 6]}}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    overlap.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(6))
    with pytest.raises(RawTensorError, match="overlapping"):
        read_header(overlap)


def test_snapshot_revision_index_and_shard_confinement(tmp_path):
    snap = tmp_path / "rev1"
    snap.mkdir()
    shard = snap / "part.safetensors"
    _write_tensor(shard)
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "part.safetensors"}}))
    assert validate_snapshot(snap, "rev1").is_file()
    located, spec, _ = tensor_location(snap, "rev1", "w")
    assert located == shard.resolve() and spec["shape"] == [2]
    with pytest.raises(RawTensorError, match="does not equal pinned revision"):
        validate_snapshot(snap, "wrong")
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "../outside.safetensors"}}))
    with pytest.raises(RawTensorError, match="escapes pinned snapshot"):
        tensor_location(snap, "rev1", "w")

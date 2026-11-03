"""Versioned registry for real GGUF tensors in a configured model backbone.

GGUF stores tensor dimensions in GGML order. ``gguf_dimensions`` preserves that
declared order; ``array_shape`` is the logical NumPy/PyTorch shape after
dequantization (the dimensions are reversed). For quantized tensors,
``GGUFReader.data.shape`` is a packed-byte shape and is not the logical shape.
No dimensions are inferred.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import gguf
import numpy as np


SCHEMA_VERSION = 1
_LAYER_TENSOR = re.compile(r"^blk\.(\d+)\.([^.]+)\.weight$")
_COMPONENTS = {
    "attn_norm": "attention.input_norm",
    "attn_q": "attention.query_projection",
    "attn_k": "attention.key_projection",
    "attn_v": "attention.value_projection",
    "attn_output": "attention.output_projection",
    "ffn_norm": "feed_forward.input_norm",
    "ffn_gate": "feed_forward.gate_projection",
    "ffn_up": "feed_forward.up_projection",
    "ffn_down": "feed_forward.down_projection",
}


@dataclass(frozen=True)
class TensorRecord:
    name: str
    gguf_dimensions: Tuple[int, ...]
    array_shape: Tuple[int, ...]
    parameter_count: int
    ggml_type: str
    packed_bytes: int
    data_offset: int
    weight_reference: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TensorRecord":
        return cls(
            name=str(value["name"]),
            gguf_dimensions=tuple(int(x) for x in value["gguf_dimensions"]),
            array_shape=tuple(int(x) for x in value["array_shape"]),
            parameter_count=int(value["parameter_count"]),
            ggml_type=str(value["ggml_type"]),
            packed_bytes=int(value["packed_bytes"]),
            data_offset=int(value["data_offset"]),
            weight_reference=str(value["weight_reference"]),
        )


@dataclass(frozen=True)
class Partition:
    block_id: str
    model: str
    architecture: str
    source_layer: Optional[int]
    component: str
    tensor_names: Tuple[str, ...]
    parameter_count: int
    trainable: bool
    version: int
    connections: Tuple[str, ...]
    device: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Partition":
        layer = value.get("source_layer")
        return cls(
            block_id=str(value["block_id"]),
            model=str(value["model"]),
            architecture=str(value["architecture"]),
            source_layer=None if layer is None else int(layer),
            component=str(value["component"]),
            tensor_names=tuple(str(x) for x in value["tensor_names"]),
            parameter_count=int(value["parameter_count"]),
            trainable=bool(value["trainable"]),
            version=int(value["version"]),
            connections=tuple(str(x) for x in value["connections"]),
            device=str(value["device"]),
        )


def _metadata_value(reader: gguf.GGUFReader, key: str, default: Any = None) -> Any:
    field = reader.fields.get(key)
    return default if field is None else field.contents()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component_for_tensor(name: str) -> Tuple[Optional[int], str]:
    match = _LAYER_TENSOR.match(name)
    if match:
        layer = int(match.group(1))
        module = match.group(2)
        component = _COMPONENTS.get(module, f"transformer.{module}")
        return layer, component

    globals_by_name = {
        "token_embd.weight": "embedding.token",
        "output.weight": "language_model.output_projection",
        "output_norm.weight": "language_model.output_norm",
    }
    return None, globals_by_name.get(name, "model.other")


def _dependencies(layer: Optional[int], component: str, architecture: str) -> Tuple[str, ...]:
    """Describe only known Llama block dataflow dependencies by stable IDs."""
    if architecture != "llama" or layer is None:
        return ()
    base = f"IMP_L{layer:02d}_"
    prior = f"IMP_L{layer - 1:02d}_FEED_FORWARD_DOWN_PROJECTION" if layer else None

    def refs(*names: str) -> Tuple[str, ...]:
        return tuple(base + name for name in names)

    if component == "attention.input_norm":
        return () if prior is None else (prior,)
    if component in {"attention.query_projection", "attention.key_projection", "attention.value_projection"}:
        return refs("ATTENTION_INPUT_NORM")
    if component == "attention.output_projection":
        return refs("ATTENTION_QUERY_PROJECTION", "ATTENTION_KEY_PROJECTION", "ATTENTION_VALUE_PROJECTION")
    if component == "feed_forward.input_norm":
        return refs("ATTENTION_OUTPUT_PROJECTION")
    if component in {"feed_forward.gate_projection", "feed_forward.up_projection"}:
        return refs("FEED_FORWARD_INPUT_NORM")
    if component == "feed_forward.down_projection":
        return refs("FEED_FORWARD_GATE_PROJECTION", "FEED_FORWARD_UP_PROJECTION")
    return ()


class PartitionRegistry:
    """Serializable index of exact tensor content in an immutable GGUF file."""

    def __init__(
        self,
        *,
        source_path: str,
        source_size: int,
        source_sha256: str,
        model: str,
        architecture: str,
        metadata: Mapping[str, Any],
        tensors: Iterable[TensorRecord],
        partitions: Iterable[Partition],
    ) -> None:
        self.schema_version = SCHEMA_VERSION
        self.source_path = str(Path(source_path).resolve())
        self.source_size = int(source_size)
        self.source_sha256 = source_sha256
        self.model = model
        self.architecture = architecture
        self.metadata = dict(metadata)
        self.tensors: Dict[str, TensorRecord] = {t.name: t for t in tensors}
        self.partitions: Dict[str, Partition] = {p.block_id: p for p in partitions}
        self._validate()

    def _validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported registry schema: {self.schema_version}")
        if len(self.source_sha256) != 64:
            raise ValueError("source_sha256 must be a SHA-256 hex digest")
        if not self.tensors:
            raise ValueError("Registry must contain at least one tensor")
        for key, tensor in self.tensors.items():
            if key != tensor.name or tensor.parameter_count <= 0 or not tensor.array_shape:
                raise ValueError(f"Invalid tensor record: {key}")
            if int(np.prod(tensor.array_shape, dtype=np.int64)) != tensor.parameter_count:
                raise ValueError(f"Parameter count does not match shape for {key}")
            if not tensor.weight_reference.startswith("sha256:"):
                raise ValueError(f"Invalid content reference for {key}")
        for key, partition in self.partitions.items():
            if key != partition.block_id or not partition.tensor_names:
                raise ValueError(f"Invalid partition record: {key}")
            if any(name not in self.tensors for name in partition.tensor_names):
                raise ValueError(f"Partition {key} references an unknown tensor")
            expected = sum(self.tensors[n].parameter_count for n in partition.tensor_names)
            if expected != partition.parameter_count:
                raise ValueError(f"Parameter count does not match tensors for {key}")
            unknown = set(partition.connections) - set(self.partitions)
            if unknown:
                raise ValueError(f"Partition {key} has unknown dependencies: {sorted(unknown)}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": {
                "path": self.source_path,
                "size_bytes": self.source_size,
                "sha256": self.source_sha256,
            },
            "model": self.model,
            "architecture": self.architecture,
            "metadata": self.metadata,
            "tensors": {name: asdict(t) for name, t in sorted(self.tensors.items())},
            "partitions": {key: asdict(p) for key, p in sorted(self.partitions.items())},
        }

    def save(self, path: os.PathLike[str] | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_name: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=target.parent,
                prefix=f".{target.name}.", suffix=".tmp", delete=False,
            ) as stream:
                temp_name = stream.name
                json.dump(self.to_dict(), stream, indent=2, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, target)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)

    @classmethod
    def load(cls, path: os.PathLike[str] | str) -> "PartitionRegistry":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported registry schema: {data.get('schema_version')}")
        source = data["source"]
        return cls(
            source_path=source["path"],
            source_size=source["size_bytes"],
            source_sha256=source["sha256"],
            model=data["model"],
            architecture=data["architecture"],
            metadata=data.get("metadata", {}),
            tensors=(TensorRecord.from_dict(v) for v in data["tensors"].values()),
            partitions=(Partition.from_dict(v) for v in data["partitions"].values()),
        )

    def resolve_partition(
        self,
        block_id: str,
        *,
        source_path: Optional[os.PathLike[str] | str] = None,
        dequantize: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Resolve and verify exact tensors. Dequantization is opt-in."""
        partition = self.partitions.get(block_id)
        if partition is None:
            raise KeyError(f"Unknown partition: {block_id}")
        path = Path(source_path or self.source_path)
        if path.stat().st_size != self.source_size:
            raise ValueError("GGUF source size differs from the registered checkpoint")
        reader = gguf.GGUFReader(str(path))
        by_name = {t.name: t for t in reader.tensors}
        resolved: Dict[str, np.ndarray] = {}
        for name in partition.tensor_names:
            record = self.tensors[name]
            tensor = by_name.get(name)
            if tensor is None:
                raise ValueError(f"GGUF is missing registered tensor {name}")
            if tuple(int(x) for x in tensor.shape) != record.gguf_dimensions:
                raise ValueError(f"GGUF dimensions changed for {name}")
            raw = memoryview(reader.data[tensor.data_offset:tensor.data_offset + tensor.n_bytes])
            actual_hash = hashlib.sha256(raw).hexdigest()
            if f"sha256:{actual_hash}" != record.weight_reference:
                raise ValueError(f"Content hash changed for {name}")
            array = np.array(tensor.data, copy=True)
            if dequantize and tensor.tensor_type not in {
                gguf.GGMLQuantizationType.F32,
                gguf.GGMLQuantizationType.F16,
                gguf.GGMLQuantizationType.F64,
                gguf.GGMLQuantizationType.I8,
                gguf.GGMLQuantizationType.I16,
                gguf.GGMLQuantizationType.I32,
                gguf.GGMLQuantizationType.I64,
            }:
                array = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
            resolved[name] = array
        return resolved


def discover_registry(gguf_path: os.PathLike[str] | str) -> PartitionRegistry:
    """Index every real tensor and create stable component IDs from GGUF names."""
    path = Path(gguf_path).resolve(strict=True)
    reader = gguf.GGUFReader(str(path))
    architecture = str(_metadata_value(reader, "general.architecture", "unknown"))
    model = str(_metadata_value(reader, "general.name", path.stem))
    metadata_keys = (
        "llama.block_count",
        "llama.embedding_length",
        "llama.feed_forward_length",
        "llama.attention.head_count",
        "llama.attention.head_count_kv",
        "llama.rope.freq_base",
    )
    metadata = {key: _metadata_value(reader, key) for key in metadata_keys if key in reader.fields}

    records: List[TensorRecord] = []
    groups: Dict[Tuple[Optional[int], str], List[str]] = {}
    for tensor in reader.tensors:
        name = tensor.name
        gguf_dims = tuple(int(x) for x in tensor.shape)
        # GGUF dimensions are in GGML order; the logical tensor shape is reversed.
        # Quantized ReaderTensor.data instead has a packed-byte shape.
        array_shape = tuple(reversed(gguf_dims))
        count = int(tensor.n_elements)
        if count <= 0 or int(np.prod(array_shape, dtype=np.int64)) != count:
            raise ValueError(f"Invalid tensor dimensions in GGUF: {name}")
        start = int(tensor.data_offset)
        end = start + int(tensor.n_bytes)
        raw_digest = hashlib.sha256(memoryview(reader.data[start:end])).hexdigest()
        record = TensorRecord(
            name=name,
            gguf_dimensions=gguf_dims,
            array_shape=array_shape,
            parameter_count=count,
            ggml_type=tensor.tensor_type.name,
            packed_bytes=int(tensor.n_bytes),
            data_offset=start,
            weight_reference=f"sha256:{raw_digest}",
        )
        records.append(record)
        groups.setdefault(_component_for_tensor(name), []).append(name)

    model_fingerprint = _sha256_file(path)
    partitions: List[Partition] = []
    for (layer, component), names in sorted(groups.items(), key=lambda item: (item[0][0] is None, item[0][0] or 0, item[0][1])):
        if layer is None:
            block_id = "IMP_GLOBAL_" + re.sub(r"[^A-Z0-9]+", "_", component.upper()).strip("_")
        else:
            block_id = f"IMP_L{layer:02d}_" + re.sub(r"[^A-Z0-9]+", "_", component.upper()).strip("_")
        partitions.append(Partition(
            block_id=block_id,
            model=model,
            architecture=architecture,
            source_layer=layer,
            component=component,
            tensor_names=tuple(sorted(names)),
            parameter_count=sum(next(t.parameter_count for t in records if t.name == n) for n in names),
            trainable=False,
            version=1,
            connections=(),
            device="host",
        ))

    # Fill architecture-derived dependencies only when all referenced blocks exist.
    ids = {p.block_id for p in partitions}
    with_dependencies: List[Partition] = []
    for partition in partitions:
        deps = _dependencies(partition.source_layer, partition.component, architecture)
        with_dependencies.append(Partition(
            **{**asdict(partition), "connections": tuple(d for d in deps if d in ids)}
        ))

    registry = PartitionRegistry(
        source_path=str(path),
        source_size=path.stat().st_size,
        source_sha256=model_fingerprint,
        model=model,
        architecture=architecture,
        metadata=metadata,
        tensors=records,
        partitions=with_dependencies,
    )
    return registry

"""Portable raw-Float32 .slither-brain export; Rust independently recompiles admission.

This is a neural artifact, never a Python model/pickle or trainer-job checkpoint.
"""

from dataclasses import dataclass
import hashlib
import json
import struct
from typing import get_args

from slither_contracts import CONSTANTS, models as wire

import numpy as np

from .canonical_graph import CanonicalGraphManifest
from .distillation_transport import DistillationContractError, bounded_json, exact_keys

MAX_ARTIFACT_BYTES = CONSTANTS["maxBrainBytes"]
MAX_PROVENANCE_BYTES = 64 * 1024
HEADER = struct.Struct(CONSTANTS["brainHeader"])
MAGIC = CONSTANTS["brainMagic"].encode()


@dataclass(frozen=True)
class BrainArtifact:
    """Immutable canonical export bytes and their retry/content identity."""

    data: bytes
    sha256: str
    weights_sha256: str


def _provenance(value, depth=0, nodes=None):
    """Keep user/trainer provenance bounded without interpreting it as runtime behavior."""
    if nodes is None:
        nodes = [0]
    nodes[0] += 1
    if depth > 16 or nodes[0] > 4096:
        raise DistillationContractError("brain provenance shape admission")
    if isinstance(value, str):
        if len(value.encode("utf-8")) > 4096 or "\0" in value:
            raise DistillationContractError("brain provenance text admission")
    elif isinstance(value, list):
        for item in value:
            _provenance(item, depth + 1, nodes)
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or len(key.encode("utf-8")) > 128 or "\0" in key:
                raise DistillationContractError("brain provenance key admission")
            _provenance(item, depth + 1, nodes)
    elif value is not None and isinstance(value, (bool, int, float)):
        pass  # allow_nan=False in the bounded serializer rejects nonfinite values.
    elif value is not None:
        raise DistillationContractError("unsupported brain provenance value")


def encode_brain_artifact(manifest, weights, provenance):
    """Pack exact canonical bytes; layout validity remains independently checked by Rust."""
    if not isinstance(manifest, CanonicalGraphManifest) or not isinstance(weights, bytes):
        raise DistillationContractError("invalid canonical brain export inputs")
    if (
        len(weights) != manifest.total_parameters * 4
        or not np.isfinite(np.frombuffer(weights, dtype="<f4")).all()
    ):
        raise DistillationContractError("brain weight count/finite admission")
    exact_keys(provenance, wire.BrainProvenanceV1)
    if provenance["schemaVersion"] != 1 or isinstance(provenance["schemaVersion"], bool):
        raise DistillationContractError("unsupported brain provenance revision")
    if provenance["sourceKind"] not in get_args(wire.BrainSourceKind) or not isinstance(
        provenance["details"], dict
    ):
        raise DistillationContractError("invalid brain provenance kind/details")
    try:
        _provenance(provenance["details"])
        details = json.dumps(
            provenance, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        if len(details) > MAX_PROVENANCE_BYTES:
            raise DistillationContractError("brain provenance byte admission")
        encoded = json.dumps(
            {
                "schemaVersion": 1,
                "graphManifest": bounded_json(manifest.canonical_json),
                "provenance": provenance,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (UnicodeError, ValueError) as error:
        raise DistillationContractError("invalid finite brain metadata") from error
    if len(encoded) > 1024 * 1024:
        raise DistillationContractError("brain manifest byte admission")
    digest = hashlib.sha256(weights).digest()
    data = HEADER.pack(
        MAGIC, len(encoded), 0, manifest.total_parameters, len(weights), digest, digest
    )
    data += encoded + weights
    if len(data) > MAX_ARTIFACT_BYTES:
        raise DistillationContractError("brain artifact byte admission")
    return BrainArtifact(data, hashlib.sha256(data).hexdigest(), digest.hex())


def read_raw_brain_artifact(data):
    # Check every fixed header condition before allocating metadata/weight views.
    # pylint: disable=too-many-boolean-expressions
    """Validate a local raw export for checkpoint retention; compressed imports require Rust."""
    if not isinstance(data, bytes) or not HEADER.size <= len(data) <= MAX_ARTIFACT_BYTES:
        raise DistillationContractError("brain artifact byte admission")
    magic, json_bytes, encoding, count, stored, raw_digest, stored_digest = HEADER.unpack_from(data)
    if (
        magic != MAGIC
        or encoding != 0
        or data[13:16] != b"\0\0\0"
        or json_bytes > 1024 * 1024
        or count > 1000000
        or stored != count * 4
        or len(data) != HEADER.size + json_bytes + stored
    ):
        raise DistillationContractError("unsupported brain envelope/encoding/dimensions")
    weights = data[HEADER.size + json_bytes :]
    if hashlib.sha256(weights).digest() != raw_digest or raw_digest != stored_digest:
        raise DistillationContractError("brain immutable numeric digest mismatch")
    metadata = bounded_json(data[HEADER.size : HEADER.size + json_bytes])
    exact_keys(metadata, wire.BrainManifestV1)
    if metadata["schemaVersion"] != 1 or isinstance(metadata["schemaVersion"], bool):
        raise DistillationContractError("unsupported brain metadata revision")
    manifest = CanonicalGraphManifest.from_bytes(
        json.dumps(metadata["graphManifest"], separators=(",", ":")).encode("utf-8")
    )
    if manifest.total_parameters != count:
        raise DistillationContractError("brain manifest parameter count mismatch")
    checked = encode_brain_artifact(manifest, weights, metadata["provenance"])
    # Return the original immutable bytes; key order has no numeric significance.
    return (
        BrainArtifact(data, hashlib.sha256(data).hexdigest(), checked.weights_sha256),
        manifest,
        metadata["provenance"],
    )

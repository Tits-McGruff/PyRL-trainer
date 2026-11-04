"""Export saved canonical weights without restarting or advancing a distillation job.

A portable brain does not require source-equivalent trainer execution. Immutable
checkpoint role/replay/candidate digests are verified first. The original source
identity is retained as provenance; no source identity or checkpoint is rewritten.
"""

import argparse
import os
from pathlib import Path

from .brain_artifact import encode_brain_artifact
from .canonical_graph import CanonicalGraphManifest
from .distillation_checkpoint import _directory_sync, _manifest
from .distillation_transport import hex_identity, integer


def export(checkpoint, output):
    """Extract only finite admitted canonical weights into a new portable artifact."""
    metadata, payloads = _manifest(checkpoint)
    manifest = CanonicalGraphManifest.from_bytes(payloads["graph.json"])
    updates = integer(metadata["updates"], 0, (1 << 64) - 1)
    source = hex_identity(metadata["trainerSourceSha256"], 64)
    job = hex_identity(metadata["jobId"], 32, True)
    artifact = encode_brain_artifact(
        manifest,
        payloads["weights.f32"],
        {
            "schemaVersion": 1,
            "sourceKind": "distilled",
            "details": {
                "method": "checkpoint-weight-export-v1",
                "jobId": job,
                "updates": f"{updates:016x}",
                "trainerSourceSha256": source,
                "checkpointWeightSha256": metadata["digests"]["weights.f32"],
                "claim": "saved weights only; no trainer restart or further optimization",
            },
        },
    )
    with Path(output).open("xb") as stream:
        stream.write(artifact.data)
        stream.flush()
        os.fsync(stream.fileno())
    _directory_sync(Path(output).parent)
    return artifact


def main():
    """Explicit extraction CLI; refuses to overwrite any existing artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    artifact = export(args.checkpoint, args.output)
    print(f"artifactSha256={artifact.sha256}\nweightsSha256={artifact.weights_sha256}")


if __name__ == "__main__":
    main()

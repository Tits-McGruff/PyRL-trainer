"""Explicit large-brain admission-v1 migration; ordinary restart remains exact-source.

Only source trees whose individual files match the built-in reviewed revision
(or enumerated manual size-admission edits) can be migrated. Payloads are copied
byte for byte, then the ordinary loader validates the entire successor before
exclusive publication. No training step or network operation is performed.
"""

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

from slither_contracts import IDENTITY

from .distillation_checkpoint import _directory_sync, _json, _manifest, load_job_checkpoint
from .distillation_job import SOURCE_FILES, trainer_source_sha256
from .distillation_transport import DistillationContractError

APPROVAL = Path(__file__).with_name("large_brain_migration_v1.json")
DOMAIN = b"slither.distillation-trainer-source.v1\0"


def _sha(data):
    """Digest immutable bytes."""
    return hashlib.sha256(data).hexdigest()


def _file_sha(path):
    """Hash bounded checkpoint files without allocating the complete replay archive."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024**2):
            digest.update(block)
    return digest.hexdigest()


def _normalized(data):
    """Approve only platform line-ending differences; aggregate source hashes remain exact."""
    return data.replace(b"\r\n", b"\n")


def _source_identity(files, contract):
    """Use the unchanged runtime framing, including the actual old file bytes."""
    digest = hashlib.sha256(DOMAIN)
    digest.update(contract.encode("ascii") + b"\0")
    for name in SOURCE_FILES:
        digest.update(name.encode() + b"\0")
        digest.update(files[name])
    return digest.hexdigest()


def migrate(checkpoint, output, source_root):
    # The complete cold migration is one validation/publication transaction.
    # pylint: disable=too-many-branches,too-many-statements
    """Create a validated successor only for the fixed admission-only revision pair."""
    approval = json.loads(APPROVAL.read_bytes())
    target = Path(__file__).parent
    if list(SOURCE_FILES) != list(approval["targetFiles"]):
        raise DistillationContractError("migration source inventory changed; new review required")
    for name in SOURCE_FILES:
        if _sha(_normalized((target / name).read_bytes())) != approval["targetFiles"][name]:
            raise DistillationContractError("migration target revision is not approved")
    if IDENTITY not in approval["targetContractIdentities"]:
        raise DistillationContractError("migration target contracts are not approved")
    current_files = {name: (target / name).read_bytes() for name in SOURCE_FILES}
    if _source_identity(current_files, IDENTITY) != trainer_source_sha256():
        raise DistillationContractError("migration target source changed after process startup")
    source = Path(source_root)
    # Source root is the old pyrl_trainer package, retained before updating it.
    files = {}
    for name in SOURCE_FILES:
        path = source / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**2:
            raise DistillationContractError("migration source file admission")
        data = path.read_bytes()
        if _sha(_normalized(data)) not in approval["sourceFiles"][name]:
            raise DistillationContractError(f"unreviewed source change in {name}")
        files[name] = data
    metadata, _payloads = _manifest(checkpoint)
    old_source = metadata["trainerSourceSha256"]
    if old_source not in {
        _source_identity(files, contract) for contract in approval["sourceContractIdentities"]
    }:
        raise DistillationContractError(
            "old source bytes/contracts do not explain checkpoint identity"
        )
    checkpoint, output = Path(checkpoint), Path(output)
    if (
        checkpoint.resolve() == output.resolve()
        or output.exists()
        or output.with_suffix(output.suffix + ".migration.json").exists()
    ):
        raise DistillationContractError("migration requires a new output path")
    original_digest = _file_sha(checkpoint)
    metadata["trainerSourceSha256"] = trainer_source_sha256()
    # Temporary file is in the destination filesystem; no original is overwritten.
    descriptor, temporary = tempfile.mkstemp(
        prefix=".admission-migration-", suffix=".distill-job", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        with zipfile.ZipFile(checkpoint) as old, zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_STORED
        ) as new:
            for item in old.infolist():
                if item.filename == "job.json":
                    new.writestr(item.filename, _json(metadata))
                else:
                    # Bounded streaming retains even maximum-size replay checkpoints.
                    with old.open(item) as reader, new.open(
                        item.filename, "w", force_zip64=True
                    ) as writer:
                        while block := reader.read(1024**2):
                            writer.write(block)
        successor, _ = _manifest(temporary)
        if successor != metadata:
            raise DistillationContractError("migration metadata changed")
        with zipfile.ZipFile(checkpoint) as old, zipfile.ZipFile(temporary) as new:
            for name in old.namelist():
                if name == "job.json":
                    continue
                with old.open(name) as left, new.open(name) as right:
                    while block := left.read(1024**2):
                        if right.read(len(block)) != block:
                            raise DistillationContractError("migration altered immutable payload")
                    if right.read(1):
                        raise DistillationContractError("migration added immutable payload bytes")
        # Includes optimizer method/moments, numerical environment, replay/sampler
        # state, RNG, recurrent state and every pending candidate's graph/job identity.
        restored = load_job_checkpoint(temporary)
        if restored.trainer.completed_updates != metadata["updates"]:
            raise DistillationContractError("migration update count mismatch")
        if _file_sha(checkpoint) != original_digest:
            raise DistillationContractError("original changed during migration")
        with temporary.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        # link is atomic and refuses an existing target on both supported platforms.
        os.link(temporary, output)
        _directory_sync(output.parent)
        receipt = {
            "migration": "large-brain-admission-v1",
            "originalSha256": original_digest,
            "successorSha256": _file_sha(output),
            "fromSourceSha256": old_source,
            "toSourceSha256": metadata["trainerSourceSha256"],
            "jobId": metadata["jobId"],
            "updates": metadata["updates"],
            "payloadDigests": metadata["digests"],
            "claim": "validated admission-only continuation; same numerical environment required",
        }
        with output.with_suffix(output.suffix + ".migration.json").open(
            "x", encoding="utf-8"
        ) as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return receipt
    finally:
        temporary.unlink(missing_ok=True)


def main():
    """Explicit CLI, with no arbitrary hash override or mismatch switch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="retained pre-update pyrl_trainer package directory",
    )
    args = parser.parse_args()
    print(json.dumps(migrate(args.checkpoint, args.output, args.source_root), indent=2))


if __name__ == "__main__":
    main()

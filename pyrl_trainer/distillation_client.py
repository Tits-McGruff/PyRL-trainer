"""Bounded HTTP-only teacher consumer with local resumable cursor proposals.

Local cursor commits are checkpoint state. They send no simulation ACK, actor
join, action, controller lease or resume token. Whole missing chunks become gaps.
"""

from dataclasses import dataclass
import json
import math
from urllib import error, parse, request

from .distillation_transport import (
    DatasetManifest,
    DistillationContractError,
    TeacherChunk,
    TeacherContract,
    TeacherDescriptor,
    bounded_json,
    exact_keys,
    hex_identity,
    integer,
)


class DistillationHTTPError(RuntimeError):
    """Bounded transport failure, independent of all game/controller state."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class _NoRedirect(request.HTTPRedirectHandler):
    """Never follow a download to an unrelated host or path."""

    # Signature required by urllib; redirects must fail before following the URL.
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise DistillationHTTPError(code, "distillation redirects are not admitted")


@dataclass(frozen=True)
class DistillationFetch:
    """One bounded proposal committed with replay/sampler/optimizer checkpoint state."""

    manifest: DatasetManifest
    start_cursor: int
    next_cursor: int
    chunks: tuple
    gaps: tuple


class DistillationClient:
    """Independent metadata/lease/download client; never constructs ActorClient."""

    # Independently bounded HTTP admission settings are explicit keyword arguments.
    # pylint: disable=too-many-arguments,too-many-boolean-expressions
    def __init__(
        self,
        base_url,
        capture_epoch,
        *,
        page_size=16,
        owned_bytes=16 * 1024 * 1024,
        timeout=10.0,
        lease_ms=60000,
        checkpoint=None,
    ):
        if not isinstance(base_url, str):
            raise DistillationContractError("distillation base URL must be text")
        origin = parse.urlsplit(base_url)
        if (
            origin.scheme not in ("http", "https")
            or not origin.netloc
            or origin.username
            or origin.password
            or origin.query
            or origin.fragment
            or origin.path not in ("", "/")
        ):
            raise DistillationContractError("distillation base URL must be a server origin")
        self.base_url = base_url.rstrip("/") + "/api/distillation/v1/"
        self.capture_epoch = hex_identity(capture_epoch, 32, True)
        self.page_size = integer(page_size, 1, 256)
        self.owned_bytes = integer(owned_bytes, 2 * 1024 * 1024, 16 * 1024 * 1024)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or not 0 < timeout <= 60
        ):
            raise DistillationContractError("invalid bounded HTTP timeout")
        self.timeout = float(timeout)
        self.lease_ms = integer(lease_ms, 1, 300000)
        self.opener = request.build_opener(_NoRedirect())
        self.cursor = 0
        self.manifest = None
        self._restore_identity = None
        if checkpoint is not None:
            exact_keys(
                checkpoint,
                "schemaVersion origin captureEpoch cursor contract runSeed sensorContractJson",
            )
            integer(checkpoint["schemaVersion"], 1, 1)
            if (
                checkpoint["origin"] != self.base_url
                or checkpoint["captureEpoch"] != self.capture_epoch
            ):
                raise DistillationContractError("distillation checkpoint source mismatch")
            self.cursor = int(hex_identity(checkpoint["cursor"], 16), 16)
            self._restore_identity = (
                TeacherContract.from_object(checkpoint["contract"]),
                integer(checkpoint["runSeed"], 0, (1 << 32) - 1),
                checkpoint["sensorContractJson"],
            )

    def _http(self, path, *, method="GET", payload=None, binary=None, limit=1024 * 1024):
        """Bound response allocation independently of any Content-Length claim."""
        body = (
            None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
        )
        if binary is not None:
            if (
                payload is not None
                or not isinstance(binary, bytes)
                or not 88 <= len(binary) <= 64 * 1024**2
            ):
                raise DistillationContractError("candidate request byte admission")
            body = binary
        elif body is not None and len(body) > 16384:
            raise DistillationContractError("distillation request byte limit")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = (
                "application/octet-stream" if binary is not None else "application/json"
            )
            headers["Content-Length"] = str(len(body))
        req = request.Request(self.base_url + path, data=body, method=method, headers=headers)
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                data = response.read(limit + 1)
                if len(data) > limit:
                    raise DistillationContractError("distillation response byte limit")
                return data, response.headers
        except error.HTTPError as exc:
            raise DistillationHTTPError(exc.code, f"distillation HTTP {exc.code}") from exc
        except (error.URLError, TimeoutError, OSError) as exc:
            raise DistillationHTTPError(0, "distillation transport unavailable") from exc

    def _json(self, path, **kwargs):
        """Validate UTF-8 and reject duplicate fields/nonfinite values."""
        data, _headers = self._http(path, **kwargs)
        return bounded_json(data)

    def _accept_manifest(self, raw):
        """Bind immutable identity before accepting progress or allocating row arrays."""
        manifest = DatasetManifest.from_object(raw)
        if (
            manifest.contract.capture_epoch != self.capture_epoch
            or manifest.production_watermark < self.cursor
        ):
            raise DistillationContractError("dataset epoch/cursor regression")
        if self.manifest is not None:
            if not self.manifest.compatible(manifest):
                raise DistillationContractError("immutable dataset identity changed")
            if (
                manifest.production_watermark < self.manifest.production_watermark
                or manifest.resolved_through < self.manifest.resolved_through
                or manifest.storage_watermark < self.manifest.storage_watermark
            ):
                raise DistillationContractError("dataset watermark regression")
        if self._restore_identity is not None:
            contract, seed, sensors = self._restore_identity
            if (
                contract != manifest.contract
                or seed != manifest.run_seed
                or sensors != manifest.sensor_json
            ):
                raise DistillationContractError("checkpoint dataset identity mismatch")
            self._restore_identity = None
        self.manifest = manifest
        return manifest

    def fetch(self):
        # pylint: disable=too-many-branches,too-many-statements
        """Propose only settled objects/gaps; failures leave the local cursor unchanged."""
        cursor = self.cursor
        page = self._json(
            f"datasets/{self.capture_epoch}/chunks?after={cursor:016x}&limit={self.page_size}"
        )
        exact_keys(page, "dataset chunks nextCursor hasMore")
        manifest = self._accept_manifest(page["dataset"])
        if (
            not isinstance(page["hasMore"], bool)
            or not isinstance(page["chunks"], list)
            or len(page["chunks"]) > self.page_size
        ):
            raise DistillationContractError("invalid bounded dataset page")
        descriptors = tuple(
            TeacherDescriptor.from_object(raw, manifest.contract) for raw in page["chunks"]
        )
        previous = cursor
        for descriptor in descriptors:
            if (
                descriptor.sequence <= previous
                or descriptor.sequence > manifest.production_watermark
            ):
                raise DistillationContractError("dataset descriptor order/watermark mismatch")
            previous = descriptor.sequence
        if int(hex_identity(page["nextCursor"], 16), 16) != previous or (
            page["hasMore"] and not descriptors
        ):
            raise DistillationContractError("dataset pagination cursor mismatch")
        admitted, cost = [], 0
        for descriptor in descriptors:
            if descriptor.sequence > manifest.resolved_through:
                break
            next_cost = cost + 2 * descriptor.byte_count + 4096
            if next_cost > self.owned_bytes:
                break
            admitted.append(descriptor)
            cost = next_cost
        chunks = []
        if admitted:
            lease = self._json(
                f"datasets/{self.capture_epoch}/leases",
                method="POST",
                payload={
                    "sequences": [f"{item.sequence:016x}" for item in admitted],
                    "durationMs": self.lease_ms,
                },
            )
            exact_keys(lease, "id captureEpoch expiresAt chunks")
            lease_id = hex_identity(lease["id"], 32, True)
            try:
                if lease["captureEpoch"] != self.capture_epoch:
                    raise DistillationContractError("storage lease dataset mismatch")
                integer(lease["expiresAt"], 1, (1 << 53) - 1)
                if not isinstance(lease["chunks"], list) or len(lease["chunks"]) != len(admitted):
                    raise DistillationContractError("storage lease count mismatch")
                leased = tuple(
                    TeacherDescriptor.from_object(raw, manifest.contract) for raw in lease["chunks"]
                )
                if leased != tuple(admitted):
                    raise DistillationContractError("storage lease descriptor mismatch")
                for descriptor in admitted:
                    data, headers = self._http(
                        (
                            f"datasets/{self.capture_epoch}/chunks/"
                            f"{descriptor.sequence:016x}?lease={lease_id}"
                        ),
                        limit=descriptor.byte_count,
                    )
                    if (
                        headers.get("Content-Length") != str(descriptor.byte_count)
                        or headers.get("X-Slither-Chunk-Sha256") != descriptor.sha256
                    ):
                        raise DistillationContractError("immutable download header mismatch")
                    chunks.append(TeacherChunk.decode(data, descriptor))
                    if sum(chunk.byte_cost for chunk in chunks) > self.owned_bytes:
                        raise DistillationContractError("decoded transfer exceeds RAM admission")
            finally:
                self._json(f"leases/{lease_id}", method="DELETE")
        next_cursor, gaps = cursor, []
        for descriptor in admitted:
            if descriptor.sequence > next_cursor + 1:
                gaps.append((next_cursor + 1, descriptor.sequence - 1))
            next_cursor = descriptor.sequence
        eligible = sum(item.sequence <= manifest.resolved_through for item in descriptors)
        if (
            not page["hasMore"]
            and len(admitted) == eligible
            and manifest.resolved_through > next_cursor
        ):
            gaps.append((next_cursor + 1, manifest.resolved_through))
            next_cursor = manifest.resolved_through
        return DistillationFetch(manifest, cursor, next_cursor, tuple(chunks), tuple(gaps))

    def commit(self, fetched):
        """Commit local replay/checkpoint progress; send no server/simulation ACK."""
        if (
            not isinstance(fetched, DistillationFetch)
            or fetched.manifest.contract.capture_epoch != self.capture_epoch
        ):
            raise DistillationContractError("foreign distillation cursor proposal")
        if fetched.next_cursor == self.cursor and fetched.start_cursor <= self.cursor:
            return False
        if fetched.start_cursor != self.cursor or fetched.next_cursor < self.cursor:
            raise DistillationContractError("stale distillation cursor proposal")
        self.cursor = fetched.next_cursor
        return True

    def checkpoint_fields(self):
        """Job checkpoints save this with sampler/optimizer/RNG, not as a gameplay ACK."""
        if self.manifest is None:
            raise DistillationContractError("no validated dataset for checkpoint")
        return {
            "schemaVersion": 1,
            "origin": self.base_url,
            "captureEpoch": self.capture_epoch,
            "cursor": f"{self.cursor:016x}",
            "contract": self.manifest.raw["contract"],
            "runSeed": self.manifest.run_seed,
            "sensorContractJson": self.manifest.sensor_json,
        }

    def upload_candidate(self, artifact):
        """Send one immutable canonical artifact to the same explicitly configured server origin."""
        # Keep ordinary metadata consumption independent of Torch imports.
        # pylint: disable=import-outside-toplevel
        from .brain_artifact import (
            BrainArtifact,
            read_raw_brain_artifact,
        )  # pylint: disable=import-outside-toplevel

        if not isinstance(artifact, BrainArtifact):
            raise DistillationContractError("candidate upload requires immutable portable bytes")
        checked, expected, provenance = read_raw_brain_artifact(artifact.data)
        if checked != artifact or provenance["sourceKind"] != "distilled":
            raise DistillationContractError("candidate upload immutable provenance mismatch")
        record = self._json(f"candidates/{artifact.sha256}", method="POST", binary=artifact.data)
        exact_keys(record, "descriptor byteCount")
        if integer(record["byteCount"], 88, 64 * 1024**2) != len(artifact.data):
            raise DistillationContractError("candidate receipt byte count mismatch")
        descriptor = record["descriptor"]
        exact_keys(
            descriptor,
            "schemaVersion artifactSha256 weightsSha256 architectureKey layoutSha256 sensorSha256 "
            "inputSize parameterCount stateSize controlIntervalSeconds lifeStateRule sourceKind",
        )
        integer(descriptor["schemaVersion"], 1, 1)
        if (
            descriptor["artifactSha256"] != artifact.sha256
            or descriptor["weightsSha256"] != artifact.weights_sha256
            or descriptor["sourceKind"] != "distilled"
        ):
            raise DistillationContractError("native candidate identity receipt mismatch")
        for name in ("layoutSha256", "sensorSha256"):
            hex_identity(descriptor[name], 64)
        integer(descriptor["inputSize"], 1, 504)
        integer(descriptor["parameterCount"], 0, 1000000)
        integer(descriptor["stateSize"], 0, 16384)
        if (
            not isinstance(descriptor["architectureKey"], str)
            or len(descriptor["architectureKey"].encode("utf-8")) > 512 * 1024
            or descriptor["lifeStateRule"] not in ("reset", "preserve")
        ):
            raise DistillationContractError("invalid native candidate compatibility receipt")
        for name, value in (
            ("layoutSha256", expected.layout_sha256),
            ("sensorSha256", expected.sensor_sha256),
            ("inputSize", expected.input_size),
            ("parameterCount", expected.total_parameters),
            ("stateSize", expected.total_state_size),
            ("controlIntervalSeconds", expected.control_interval_seconds),
            ("lifeStateRule", expected.life_state_rule),
        ):
            if descriptor[name] != value:
                raise DistillationContractError(
                    "native candidate complete contract receipt mismatch"
                )
        return descriptor

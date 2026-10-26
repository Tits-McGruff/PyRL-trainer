"""Real bounded HTTP teacher consumption without any actor/game-control protocol."""

from copy import deepcopy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit

import pytest

from pyrl_trainer.distillation_replay import TeacherReplay
from pyrl_trainer.distillation_client import DistillationClient, DistillationHTTPError
from pyrl_trainer.distillation_transport import DistillationContractError

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "teacher_chunk_v1"
REFERENCE = json.loads((ROOT / "reference.json").read_text(encoding="utf-8"))
DATA = (ROOT / "chunk.bin").read_bytes()
EPOCH = REFERENCE["contract"]["captureEpoch"]
POLICY = {"schemaVersion": 1, "scope": "producer-global", "maxBytes": "0000000100000000",
          "maxChunks": 65536, "maxDatasets": 256, "maxLeases": 64, "maxPinnedChunks": 1024,
          "maxPinnedBytes": "0000000020000000", "maxLeaseMs": 300000,
          "diskReserveBytes": "0000000040000000"}



@pytest.fixture(name="endpoint")
def endpoint_fixture():
    """Serve only dedicated metadata/pin/download routes and record every request."""
    state = {
        "requests": [],
        "storage": deepcopy(POLICY),
        "storage_status": 200,
        "has_more": False,
        "resolved": 9,
        "production": 9,
        "descriptors": [deepcopy(REFERENCE["descriptor"])],
        "manifest_updates": {},
        "corrupt": False,
        "expired": False,
        "redirect": False,
    }

    class Handler(BaseHTTPRequestHandler):
        """An independent synthetic endpoint carrying actual immutable Rust codec bytes."""

        def log_message(self, format, *args):  # pylint: disable=redefined-builtin
            pass

        def do_GET(self):  # pylint: disable=invalid-name
            """Reject all non-dataset paths, including actor/player control."""
            state["requests"].append(("GET", self.path))
            path = urlsplit(self.path).path
            if path == "/api/distillation/v1/storage":
                self.respond(state["storage_status"], state["storage"])
            elif path.endswith("/chunks"):
                manifest = {
                    "schemaVersion": 1,
                    "runSeed": 42,
                    "contract": REFERENCE["contract"],
                    "sensorContractJson": REFERENCE["sensorContractJson"],
                    "state": "active",
                    "productionWatermark": f"{state['production']:016x}",
                    "storageWatermark": "0000000000000007",
                    "retentionFloor": "0000000000000007",
                    "resolvedThrough": f"{state['resolved']:016x}",
                    "nativeDroppedChunks": "0000000000000008",
                    "nativeDroppedSamples": "0000000000000000",
                    "nativeDroppedBytes": "0000000000000000",
                    "spoolDroppedChunks": "0000000000000000",
                    "retainedChunks": 1,
                    "retainedBytes": f"{len(DATA):016x}",
                    "error": None,
                }
                manifest.update(state["manifest_updates"])
                descriptors = state["descriptors"]
                self.respond(
                    200,
                    {
                        "dataset": manifest,
                        "chunks": descriptors,
                        "nextCursor": (
                            descriptors[-1]["sequence"] if descriptors else "0000000000000000"
                        ),
                        "hasMore": state["has_more"],
                    },
                )
            elif "/chunks/" in path:
                if state["expired"]:
                    self.respond(410, {"error": "expired"})
                elif state["redirect"]:
                    self.send_response(302)
                    self.send_header("Location", "/api/player/join")
                    self.end_headers()
                else:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(DATA)))
                    self.send_header("X-Slither-Chunk-Sha256", hashlib.sha256(DATA).hexdigest())
                    self.end_headers()
                    self.wfile.write(
                        DATA[:-1] + bytes([DATA[-1] ^ 1]) if state["corrupt"] else DATA
                    )
            else:
                self.respond(404, {"error": "not a distillation dataset route"})

        def do_POST(self):  # pylint: disable=invalid-name
            """Pins are diagnostic storage ownership, not controller leases."""
            state["requests"].append(("POST", self.path))
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert set(payload) == {"sequences", "durationMs"}
            self.respond(
                201,
                {
                    "id": "12" * 16,
                    "captureEpoch": EPOCH,
                    "expiresAt": 2000000000000,
                    "chunks": state["descriptors"],
                },
            )

        def do_DELETE(self):  # pylint: disable=invalid-name
            """Record pin cleanup even when a download fails."""
            state["requests"].append(("DELETE", self.path))
            self.respond(200, {"released": True})

        def respond(self, code, value):
            """Write one bounded complete JSON response."""
            data = json.dumps(value).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_cursor_proposals_gaps_pin_cleanup_and_exact_local_commit(endpoint):
    """Fetching sends no simulation ACK and preserves a resumable exact cursor."""
    origin, state = endpoint
    client = DistillationClient(origin, EPOCH)
    fetched = client.fetch()
    assert client.cursor == 0
    assert fetched.next_cursor == 9 and fetched.gaps == ((1, 6), (8, 9))
    assert fetched.chunks[0].raw_bytes == DATA
    replay = TeacherReplay()
    assert replay.admit(fetched)
    assert replay.cursor == fetched.next_cursor
    assert client.commit(fetched)
    assert not client.commit(fetched)
    restored = DistillationClient(origin, EPOCH, checkpoint=client.checkpoint_fields())
    assert restored.cursor == 9
    assert all(path.startswith("/api/distillation/v1/") for _method, path in state["requests"])
    assert [method for method, _path in state["requests"]] == [
        "GET", "GET", "POST", "GET", "DELETE"
    ]
    assert not any(
        "join" in path or "action" in path or "resume" in path
        for _method, path in state["requests"]
    )


def test_pagination_never_labels_an_unlisted_settled_tail_missing(endpoint):
    """hasMore keeps the trailing resolved range pending until available pages finish."""
    origin, state = endpoint
    state["has_more"] = True
    fetched = DistillationClient(origin, EPOCH).fetch()
    assert fetched.next_cursor == 7 and fetched.gaps == ((1, 6),)


def test_unsettled_object_does_not_skip_a_pending_predecessor(endpoint):
    """Stored high sequence alone does not establish missing earlier observations."""
    origin, state = endpoint
    state["resolved"] = 1
    client = DistillationClient(origin, EPOCH)
    fetched = client.fetch()
    assert not fetched.chunks and fetched.next_cursor == 1 and fetched.gaps == ((1, 1),)
    assert [method for method, _path in state["requests"]] == ["GET", "GET"]


@pytest.mark.parametrize("failure", ["corrupt", "expired", "redirect"])
def test_failed_download_does_not_advance_cursor_and_releases_pin(endpoint, failure):
    """Malformed bytes, eviction/expiry and redirection are explicit retry/continuity failures."""
    origin, state = endpoint
    state[failure] = True
    client = DistillationClient(origin, EPOCH)
    with pytest.raises((DistillationContractError, DistillationHTTPError)):
        client.fetch()
    assert client.cursor == 0
    assert state["requests"][-1][0] == "DELETE"
    assert not any("player" in path for _method, path in state["requests"])


def test_foreign_checkpoint_and_invalid_limits_reject_before_http(endpoint):
    """Source identity and allocation caps cannot be coerced by bool/float/string values."""
    origin, state = endpoint
    for options in (
        {"page_size": 257},
        {"page_size": True},
        {"owned_bytes": 1},
        {"timeout": float("nan")},
        {"lease_ms": 300001},
    ):
        with pytest.raises(DistillationContractError):
            DistillationClient(origin, EPOCH, **options)
    with pytest.raises(DistillationContractError):
        DistillationClient(origin + "/api/player", EPOCH)
    assert not state["requests"]


def test_checkpoint_binds_source_seed_and_full_native_contract(endpoint):
    """Restoring local progress cannot change the capture or its exact partition seed."""
    origin, state = endpoint
    client = DistillationClient(origin, EPOCH)
    fetched = client.fetch()
    client.commit(fetched)
    checkpoint = client.checkpoint_fields()
    calls = len(state["requests"])
    for update in (
        {"origin": "http://other/api/distillation/v1/"},
        {"captureEpoch": "22" * 16},
        {"runSeed": True},
    ):
        with pytest.raises(DistillationContractError):
            DistillationClient(origin, EPOCH, checkpoint={**checkpoint, **update})
    assert len(state["requests"]) == calls
    foreign = DistillationClient(origin, EPOCH, checkpoint={**checkpoint, "runSeed": 43})
    with pytest.raises(DistillationContractError, match="identity mismatch"):
        foreign.fetch()
    assert foreign.cursor == 9


def test_storage_policy_is_immutable_checkpoint_provenance_and_changes_precede_data(endpoint):
    """Restart and later fetches preserve recorded ceilings before reading new chunks."""
    origin, state = endpoint
    client = DistillationClient(origin, EPOCH, expected_spool_bytes=4 * 1024**3)
    fetched = client.fetch()
    client.commit(fetched)
    saved = client.checkpoint_fields()
    assert saved["schemaVersion"] == 2 and saved["storagePolicy"] == POLICY
    restored = DistillationClient(origin, EPOCH, checkpoint=saved)
    assert restored.producer_storage == client.producer_storage
    assert restored.cursor == client.cursor == 9
    state["requests"].clear()
    state["storage"]["maxChunks"] = 32768
    with pytest.raises(DistillationContractError, match="policy changed"):
        restored.fetch()
    assert restored.cursor == 9
    assert state["requests"] == [("GET", "/api/distillation/v1/storage")]
    state["requests"].clear()
    with pytest.raises(DistillationContractError, match="requested spool"):
        DistillationClient(origin, EPOCH, checkpoint=saved, expected_spool_bytes=8 * 1024**3)
    assert not state["requests"]


@pytest.mark.parametrize("change", [
    {"schemaVersion": True}, {"scope": "job-local"}, {"maxBytes": "0000000000000000"},
    {"maxBytes": "0000001000000001"}, {"maxBytes": 4 * 1024**3}, {"maxChunks": 65537},
    {"maxLeases": True}, {"maxPinnedBytes": "0000000020000001"}, {"maxLeaseMs": 300001},
    {"diskReserveBytes": "0020000000000000"}, {"unknown": 1},
])
def test_malformed_storage_policy_rejects_before_pages_leases_or_rows(endpoint, change):
    """Untrusted producer metadata cannot alter counters or allocation admission."""
    origin, state = endpoint
    state["storage"].update(change)
    client = DistillationClient(origin, EPOCH)
    with pytest.raises(DistillationContractError):
        client.fetch()
    assert client.cursor == 0 and client.manifest is None
    assert state["requests"] == [("GET", "/api/distillation/v1/storage")]


def test_requested_spool_budget_and_storage_unavailability_do_not_guess_legacy_defaults(endpoint):
    """Only legacy 404 means unreported; errors and explicit mismatches remain failures."""
    origin, state = endpoint
    with pytest.raises(DistillationContractError, match="requested spool"):
        DistillationClient(origin, EPOCH, expected_spool_bytes=8 * 1024**3).fetch()
    assert state["requests"] == [("GET", "/api/distillation/v1/storage")]
    state["requests"].clear()
    state["storage_status"] = 503
    with pytest.raises(DistillationHTTPError) as unavailable:
        DistillationClient(origin, EPOCH).fetch()
    assert unavailable.value.status == 503
    state["requests"].clear()
    state["storage_status"] = 404
    client = DistillationClient(origin, EPOCH)
    client.commit(client.fetch())
    assert client.producer_storage is None and client.checkpoint_fields()["storagePolicy"] is None
    legacy = client.checkpoint_fields()
    legacy.pop("storagePolicy")
    legacy["schemaVersion"] = 1
    restored = DistillationClient(origin, EPOCH, checkpoint=legacy)
    assert restored.refresh_storage_policy() is None
    with pytest.raises(DistillationContractError, match="requested spool"):
        DistillationClient(origin, EPOCH, expected_spool_bytes=4 * 1024**3).fetch()


def test_larger_advertised_inventory_obeys_the_actual_storage_policy(endpoint):
    """Synthetic large metadata can accompany a bounded real-codec page; no bulk allocation."""
    origin, state = endpoint
    state["storage"]["maxBytes"] = f"{64 * 1024**3:016x}"
    state["manifest_updates"] = {"retainedBytes": f"{64 * 1024**3:016x}",
                                 "retainedChunks": 65536}
    client = DistillationClient(origin, EPOCH, expected_spool_bytes=64 * 1024**3)
    fetched = client.fetch()
    assert len(fetched.chunks) == 1 and fetched.chunks[0].raw_bytes == DATA
    assert client.manifest.raw["retainedBytes"] == f"{64 * 1024**3:016x}"
    assert client.cursor == 0  # Fetch is still a proposal, never a simulation/storage ACK.
    saved = client.checkpoint_fields()
    restored = DistillationClient(origin, EPOCH, checkpoint=saved)
    # Checkpoint loading uses the same intake method without HTTP or decoder allocation.
    restored._accept_manifest(client.manifest.raw)  # pylint: disable=protected-access
    assert restored.manifest.raw["retainedBytes"] == f"{64 * 1024**3:016x}"
    smaller = deepcopy(saved)
    smaller["storagePolicy"]["maxBytes"] = f"{4 * 1024**3:016x}"
    rejected = DistillationClient(origin, EPOCH, checkpoint=smaller)
    with pytest.raises(DistillationContractError, match="inventory exceeds"):
        rejected._accept_manifest(client.manifest.raw)  # pylint: disable=protected-access
    assert rejected.manifest is None and rejected.cursor == client.cursor


@pytest.mark.parametrize("inventory,policy", [
    ({"retainedBytes": f"{4 * 1024**3 + 1:016x}"}, {}),
    ({"retainedChunks": 2}, {"maxChunks": 1}),
])
def test_inventory_over_reported_policy_rejects_before_binding_pins_or_rows(
    endpoint, inventory, policy
):
    """Global protocol limits cannot override a smaller selected producer quota."""
    origin, state = endpoint
    state["storage"].update(policy)
    state["manifest_updates"] = inventory
    client = DistillationClient(origin, EPOCH)
    with pytest.raises(DistillationContractError, match="inventory exceeds"):
        client.fetch()
    assert client.manifest is None and client.cursor == 0
    assert all(method == "GET" for method, _path in state["requests"])
    assert not any("/leases" in path or "/chunks/" in path for _method, path in state["requests"])

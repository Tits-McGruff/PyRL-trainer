"""Real bounded HTTP teacher consumption without any actor/game-control protocol."""

from copy import deepcopy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit

import pytest

from pyrl_trainer.distillation_client import DistillationClient, DistillationHTTPError
from pyrl_trainer.distillation_transport import DistillationContractError

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "teacher_chunk_v1"
REFERENCE = json.loads((ROOT / "reference.json").read_text(encoding="utf-8"))
DATA = (ROOT / "chunk.bin").read_bytes()
EPOCH = REFERENCE["contract"]["captureEpoch"]


@pytest.fixture(name="endpoint")
def endpoint_fixture():
    """Serve only dedicated metadata/pin/download routes and record every request."""
    state = {
        "requests": [],
        "has_more": False,
        "resolved": 9,
        "production": 9,
        "descriptors": [deepcopy(REFERENCE["descriptor"])],
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
            if path.endswith("/chunks"):
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
    assert client.commit(fetched)
    assert not client.commit(fetched)
    restored = DistillationClient(origin, EPOCH, checkpoint=client.checkpoint_fields())
    assert restored.cursor == 9
    assert all(path.startswith("/api/distillation/v1/") for _method, path in state["requests"])
    assert [method for method, _path in state["requests"]] == ["GET", "POST", "GET", "DELETE"]
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
    assert [method for method, _path in state["requests"]] == ["GET"]


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

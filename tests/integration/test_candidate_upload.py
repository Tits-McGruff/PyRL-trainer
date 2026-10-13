"""Bounded same-origin canonical candidate POST and exact receipt checks."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

import pytest

from pyrl_trainer.brain_artifact import encode_brain_artifact
from pyrl_trainer.canonical_graph import CanonicalGraphManifest
from pyrl_trainer.distillation_client import DistillationClient
from pyrl_trainer.distillation_transport import DistillationContractError

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1] / "fixtures/distillation_graph_v1"


def test_candidate_post_uses_only_dedicated_namespace_and_complete_native_receipt():
    """Only matching content identity/deployment can release an immutable pending export."""
    manifest = CanonicalGraphManifest.read(ROOT / "manifest.json")
    source = encode_brain_artifact(
        manifest,
        (ROOT / "weights.f32").read_bytes(),
        {
            "schemaVersion": 1,
            "sourceKind": "distilled",
            "details": {"scope": "synthetic reference"},
        },
    )
    state = {"calls": [], "forge": False}

    class Handler(BaseHTTPRequestHandler):
        """Controlled transport endpoint; native admission has actual-tool tests."""

        def log_message(self, format, *args):  # pylint: disable=redefined-builtin
            """Keep expected socket traffic out of test output."""

        def do_POST(self):  # pylint: disable=invalid-name
            """Verify exact opaque bytes and return bounded compiler metadata."""
            data = self.rfile.read(int(self.headers["Content-Length"]))
            assert data == source.data
            assert self.headers["Content-Type"] == "application/octet-stream"
            state["calls"].append(self.path)
            record = {
                "descriptor": {
                    "schemaVersion": 1,
                    "artifactSha256": source.sha256,
                    "weightsSha256": source.weights_sha256,
                    "architectureKey": "native-reference",
                    "layoutSha256": manifest.layout_sha256,
                    "sensorSha256": manifest.sensor_sha256,
                    "inputSize": manifest.input_size,
                    "parameterCount": manifest.total_parameters,
                    "stateSize": manifest.total_state_size,
                    "controlIntervalSeconds": manifest.control_interval_seconds,
                    "lifeStateRule": manifest.life_state_rule,
                    "sourceKind": "distilled",
                },
                "byteCount": len(data),
            }
            if state["forge"]:
                record["descriptor"]["parameterCount"] += 1
            encoded = json.dumps(record).encode("utf-8")
            self.send_response(201)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = DistillationClient(f"http://127.0.0.1:{server.server_port}", "11" * 16)
        for _retry in range(2):
            assert (
                client.upload_candidate(source)["artifactSha256"]
                == hashlib.sha256(source.data).hexdigest()
            )
        state["forge"] = True
        with pytest.raises(DistillationContractError, match="complete contract"):
            client.upload_candidate(source)
        assert state["calls"] == [f"/api/distillation/v1/candidates/{source.sha256}"] * 3
        assert client.cursor == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)

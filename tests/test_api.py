from fastapi.testclient import TestClient

from kairos.api.server import create_app


def test_rest_and_websocket_session(cfg, tmp_path):
    cfg = cfg.with_overrides({"telemetry.trace_dir": str(tmp_path / "traces"),
                              "telemetry.sqlite_path": str(tmp_path / "t.sqlite"),
                              "bench.results_dir": str(tmp_path / "bench")})
    client = TestClient(create_app(cfg))
    assert client.get("/api/health").json()["status"] == "ok"
    corpus = client.get("/api/corpus").json()
    assert corpus["status"] == "READY" and corpus["chunks"] > 0
    assert client.get("/api/benchmark/latest").json()["status"] == "NOT_MEASURED"
    assert any(s["id"] == "pune-demo" for s in client.get("/api/scripts").json())

    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        sid = hello["session_id"]
        ws.send_json({"type": "chunk", "utterance_id": "u1", "text": "Is alcohol served at company venues?",
                      "t0": 0.0, "t1": 1.0, "final": True})
        seen = []
        while True:
            msg = ws.receive_json()
            if msg["type"] == "event":
                seen.append(msg["event"]["event_type"])
                if msg["event"]["event_type"] == "TelemetryEmitted":
                    break
        assert "AnswerVersionCreated" in seen and "GroundingVerified" in seen
        trace = client.get(f"/api/sessions/{sid}/trace").json()
        assert trace["validation"]["coverage"] == 1.0
        ws.send_json({"type": "end_session"})
    assert client.delete(f"/api/sessions/{sid}").json()["ended"] is False  # already purged

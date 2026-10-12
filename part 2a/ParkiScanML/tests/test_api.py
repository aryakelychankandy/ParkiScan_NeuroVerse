import io
import json
import os

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sessions")
    os.environ["PARKISCAN_SESSION_DIR"] = str(tmp)
    os.environ["PARKISCAN_ALLOWED_RAW_ROOTS"] = str(tmp)
    os.environ["PARKISCAN_ENABLE_TTS"] = "false"
    import importlib

    import app.main as m
    importlib.reload(m)
    from fastapi.testclient import TestClient

    c = TestClient(m.app)
    c.tmp = tmp
    return c


def test_health_and_capabilities(client):
    assert client.get("/health").json()["status"] == "ok"
    cap = client.get("/capabilities").json()
    assert "spiral" in cap["features"]
    assert cap["privacy"]["participant_audio_to_third_parties"] is False
    assert "api_key" not in json.dumps(cap).lower()


def test_session_lifecycle_and_summary(client):
    r = client.post("/session/start", json={"session_id": "s-001", "participant_id": "P1", "language": "ta"})
    assert r.status_code == 200
    # a Unity task record plus its raw spiral CSV
    raw = client.tmp / "s-001_spiral.csv"
    th = np.linspace(0, 6 * np.pi, 400)
    rr = 0.05 + 0.02 * th
    with raw.open("w") as f:
        f.write("t,attempt,xNorm,yNorm,down,rNorm,thetaRad,rIdealNorm,radialDevNorm,nearestDistNorm,progress\n")
        for i, (a, b) in enumerate(zip(rr * np.cos(th), rr * np.sin(th))):
            f.write(f"{i / 60:.4f},1,{a:.5f},{b:.5f},1,0,0,0,0,0,0\n")
    rec = {"schema": "parkiscan.task.v1", "sessionId": "s-001", "levelId": "L2", "taskId": "L2_T1_TRACE_NEURAL_CIRCUIT",
           "completion": "complete", "durationSec": 6.6, "metrics": [{"name": "accuracy", "value": None}]}
    r = client.post("/tasks", json={"record": rec, "raw_csv_path": str(raw), "language": "ta"})
    assert r.status_code == 200
    an = r.json()["server_analysis"]
    assert an["analysed"] and an["extractor"] == "spiral"
    assert abs(an["features"]["loops"] - 3) < 0.2
    r = client.post("/features/cognition", json={"session_id": "s-001", "task": "nodes",
                                                 "trials": [{"correct": True, "rt_ms": 900}, {"correct": False, "rt_ms": 1500}]})
    assert r.json()["result"]["features"]["accuracy"] == 0.5
    s = client.get("/session/s-001/summary").json()
    assert s["counts"]["tasks"] == 1 and s["counts"]["feature_requests"] == 1
    assert "not a diagnosis" in s["notice"]
    assert client.post("/session/stop", json={"session_id": "s-001"}).json()["session"]["status"] == "stopped"


def test_raw_path_outside_allowed_roots_is_ignored(client):
    rec = {"sessionId": "s-002", "levelId": "L2", "taskId": "X"}
    r = client.post("/tasks", json={"record": rec, "raw_csv_path": "/etc/passwd"})
    assert r.status_code == 200 and r.json()["server_analysis"]["analysed"] is False


def test_bad_session_id_rejected(client):
    assert client.post("/session/start", json={"session_id": "../evil"}).status_code == 422
    assert client.get("/session/..%2Fevil/summary").status_code in (404, 422)


def test_voice_upload(client):
    sr = 16000
    t = np.arange(sr * 2) / sr
    y = (0.3 * np.sin(2 * np.pi * 140 * t) + 0.1 * np.sin(2 * np.pi * 280 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    r = client.post("/features/voice", files={"file": ("a.wav", buf.getvalue(), "audio/wav")}, data={"session_id": "s-003"})
    assert r.status_code == 200
    assert abs(r.json()["result"]["features"]["f0_mean_hz"] - 140) < 5
    assert r.json()["inputs"]["audio_stored"] is False
    assert client.post("/features/voice", files={"file": ("x.wav", b"not audio", "audio/wav")}).status_code == 415


def test_imu_ingest_simulated(client):
    from parkiscan.sensors.simulator import generate

    pk = [p.model_dump() for p in generate(seconds=6, fs=100, tremor_hz=5.0)]
    pk.append({"dev": "SIM-x", "seq": 1, "t_ms": 1, "ax": 99, "ay": 0, "az": 0})  # out of range
    r = client.post("/sensors/imu", json={"session_id": "s-004", "packets": pk})
    assert r.json()["accepted"] == 600 and r.json()["rejected"] == 1 and r.json()["simulated"] is True
    s = client.get("/session/s-004/summary").json()
    dev = s["sensors"]["SIM-esp32-01"]
    assert dev["features"]["simulated"] is True
    assert 4.5 <= dev["features"]["acc_peak_hz"] <= 5.5
    assert any("SIMULATED" in w for w in dev["warnings"])


def test_tts_disabled_by_default(client):
    assert client.post("/tts", json={"text": "hello"}).status_code == 403

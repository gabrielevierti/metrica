"""HTTP API: case lifecycle, evidence intake, analyst annotations, access restrictions.

These tests do not start analysis jobs (see test_pipeline_synthetic.py for the full pipeline)."""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from conftest import write_test_video  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from metrica.api.app import create_app  # noqa: E402
from metrica.evidence.audit import verify_chain  # noqa: E402
from metrica.evidence.case import Case  # noqa: E402

H = {"X-Metrica-Analyst": "pytest-analyst"}


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path / "ws")
    with TestClient(app) as c:
        yield c
    app.state.jobs._stop = True


def test_system_and_models(client):
    r = client.get("/api/system")
    assert r.status_code == 200
    d = r.json()
    assert d["version"] and set(d["profiles"]) == {"fast", "standard", "quality"}
    assert "bind_note" in d and d["hardware"]["onnx_providers"]
    m = client.get("/api/models").json()
    keys = {x["key"] for x in m["models"]}
    assert {"depth_anything_v2_s", "yolox_s", "yolo11n"} <= keys
    # copyleft model download requires an explicit opt-in
    r = client.post("/api/models/download", json={"keys": ["yolo11n"]})
    assert r.status_code == 400


def test_case_lifecycle_and_intake(client, tmp_path):
    r = client.post("/api/cases", json={"title": "API test", "reference": "T-1"}, headers=H)
    assert r.status_code == 200
    cid = r.json()["id"]
    assert any(c["id"] == cid for c in client.get("/api/cases").json())
    v1 = write_test_video(tmp_path / "in" / "a.mp4", seed=1)
    r = client.post(f"/api/cases/{cid}/videos/path", json={"path": str(v1), "copy_into_case": True}, headers=H)
    assert r.status_code == 200 and r.json()["video"]["id"] == "v1"
    # duplicate content is rejected
    assert client.post(f"/api/cases/{cid}/videos/path", json={"path": str(v1)}, headers=H).status_code == 409
    # upload path
    v2 = write_test_video(tmp_path / "in" / "b.mp4", seed=2)
    with open(v2, "rb") as f:
        r = client.post(f"/api/cases/{cid}/videos", files={"file": ("b.mp4", f, "video/mp4")}, headers=H)
    assert r.status_code == 200 and r.json()["video"]["id"] == "v2"
    assert "uploaded through the application" in r.json()["video"]["original_path"]
    # a non-video upload is refused
    r = client.post(f"/api/cases/{cid}/videos", files={"file": ("x.mp4", b"not a video", "video/mp4")}, headers=H)
    assert r.status_code == 400
    d = client.get(f"/api/cases/{cid}").json()
    assert [v["id"] for v in d["videos"]] == ["v1", "v2"] and not d["has_scene"]
    assert client.get(f"/api/cases/{cid}/videos/v1/probe").json()["video"]["width"] == 160
    # exact source frame decoded from the evidence file
    r = client.get(f"/api/cases/{cid}/frame/v1/3")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content[:4] == b"\x89PNG"
    # evidence verification
    ver = client.get(f"/api/cases/{cid}/verify").json()
    assert len(ver["evidence"]) == 2 and all(x["ok"] for x in ver["evidence"]) and ver["chain"]["valid"]
    case = Case(__import__("pathlib").Path(d["path"]))
    assert case.audit.entries()[-1]["action"] == "evidence.verified_on_request"
    # snapshots must be PNG
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    assert client.post(f"/api/cases/{cid}/snapshots", files={"file": ("s.png", png, "image/png")},
                       data={"caption": "view"}, headers=H).status_code == 200
    assert client.post(f"/api/cases/{cid}/snapshots", files={"file": ("s.png", b"GIF89a", "image/png")},
                       headers=H).status_code == 400


def test_annotations_audit_and_restrictions(client):
    cid = client.post("/api/cases", json={"title": "Annotations", "reference": ""}, headers=H).json()["id"]
    assert client.post(f"/api/cases/{cid}/notes", json={"text": "Lamp post visible at 0:12"}, headers=H).status_code == 200
    r = client.post(f"/api/cases/{cid}/verifications", json={"target": "u_v1-pole-1", "verdict": "verified",
                                                              "note": "checked on site photos"}, headers=H)
    assert r.status_code == 200
    assert client.post(f"/api/cases/{cid}/verifications", json={"target": "x", "verdict": "maybe"},
                       headers=H).status_code == 422
    notes = client.get(f"/api/cases/{cid}/annotations/notes").json()
    assert notes and notes[0]["text"].startswith("Lamp post")
    audit = client.get(f"/api/cases/{cid}/audit").json()
    entries = audit["entries"] if isinstance(audit, dict) else audit
    assert any(e["actor"] == "pytest-analyst" for e in entries)
    assert verify_chain(entries)["valid"]
    # no scene yet: measurements are refused rather than invented
    assert client.post(f"/api/cases/{cid}/measurements", json={"kind": "distance", "points": [[0, 0, 0], [1, 0, 0]]},
                       headers=H).status_code == 404
    # derived-file access cannot escape the case's derived folder
    assert client.get(f"/api/cases/{cid}/files/../case.json").status_code in (403, 404)
    assert client.get(f"/api/cases/{cid}/files/..%2F..%2Fcase.json").status_code in (403, 404)
    # invalid ids and configs
    assert client.get("/api/cases/NOT-AN-ID").status_code == 400
    assert client.post(f"/api/cases/{cid}/analyze", json={"profile": "standard", "overrides": {}}).status_code == 400

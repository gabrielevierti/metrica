"""End-to-end pipeline on procedurally generated footage (needs the default models; slow).

Run with:  pytest -m "slow" tests/integration
The synthetic scene contains no COCO-detectable people, so the static clip exercises the
fallback paths (no people -> no people-based metric scale) honestly: the test checks that the
status degrades to RELATIVE/ARBITRARY instead of claiming metres.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import models_available

pytestmark = [pytest.mark.slow, pytest.mark.models]


@pytest.mark.skipif(not models_available("depth_anything_v2_s", "yolox_s"), reason="default models not downloaded")
def test_static_synthetic_end_to_end(tmp_path):
    from metrica.config import AnalysisConfig
    from metrica.evidence.audit import verify_chain
    from metrica.evidence.case import Case
    from metrica.pipeline.runner import run_pipeline
    from metrica.reporting.exports import generate_report
    from metrica.synthetic.scene import generate

    info = generate(tmp_path / "in", kind="static", seconds=3.0, fps=8.0, width=480, height=270)
    case = Case.create(tmp_path / "case", "Synthetic static", "PYTEST")
    case.add_video(Path(info["video"]))
    cfg = AnalysisConfig.from_dict({"profile": "fast"})
    st = run_pipeline(case, cfg, actor="pytest")
    assert st["state"].startswith("completed"), st
    stages = {s["name"]: s["status"] for s in st["stages"]}
    assert stages["ingest"] == "ok" and stages["scene"] in ("ok", "degraded")
    scene = case.read_json("derived/scene/scene.json")
    assert scene["schema"] == "metrica-scene/1"
    cs = scene["coordinate_system"]
    # no people in the synthetic scene: a metric claim would be unsupported
    assert cs["status"] in ("RELATIVE", "ARBITRARY") and cs["unit"] == "su"
    assert scene["layers"] and all(set(l["files"]) >= {"positions"} for l in scene["layers"] if l["kind"] != "coverage")
    # every entity declares an evidence class
    assert all(e.get("evidence", {}).get("class") for e in scene["entities"])
    # binary layers load and evidence ids are valid
    lay = next(l for l in scene["layers"] if l["kind"] == "mesh")
    evi = np.fromfile(case.root / "derived" / "scene" / lay["files"]["evidence"], np.uint8)
    assert evi.size == lay["stats"]["vertices"] and evi.max() <= 5
    # manifest and integrity
    man = case.read_json("manifest.json")
    assert man["inputs"][0]["sha256_intake"] == case.videos()[0]["sha256_intake"]
    assert man["inputs"][0]["verified_at_manifest_time"] and man["audit"]["chain_valid"]
    assert all(r["ok"] for r in case.verify_originals())
    assert verify_chain(case.audit.entries())["valid"]
    # resume: a second run reuses every stage
    st2 = run_pipeline(case, cfg, actor="pytest")
    assert all(s.get("reused") for s in st2["stages"] if s["status"] in ("ok", "degraded"))
    # report
    rec = generate_report(case, {"analyst": "pytest", "purpose": "test"}, actor="pytest", pdf=True)
    files = rec["files"]
    for k in ("html", "pdf", "zip"):
        assert (case.root / files[k]).exists(), k
    html = (case.root / files["html"]).read_text()
    assert "not proof of authenticity" in html
    assert (case.root / files["pdf"]).read_bytes()[:4] == b"%PDF"
    assert any(e.endswith("objects.csv") for e in files["exports"])

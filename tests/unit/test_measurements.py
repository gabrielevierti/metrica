"""Measurement values, units, status labels and uncertainty propagation."""
from __future__ import annotations

import pytest

from metrica.analysis.measurements import POINT_REL_SIGMA, compute, record


def test_distance_value_and_status_label(scene_stub):
    r = compute(scene_stub(), "distance", [[0, 0, -10], [3, 0, -14]], ["OBSERVED_MULTI_VIEW"] * 2)
    assert r["value"] == pytest.approx(5.0) and r["unit"] == "m" and r["status"] == "ESTIMATED_METRIC"
    assert "ESTIMATED_METRIC" in r["display"] and r["sigma"] > 0
    rel = compute(scene_stub("RELATIVE", "su", None), "distance", [[0, 0, -10], [3, 0, -14]])
    assert "not a real-world measurement" in rel["display"] and rel["unit"] == "su"


def test_uncertainty_grows_with_weaker_evidence_and_scale_sigma(scene_stub):
    pts = [[0, 0, -10], [3, 0, -14]]
    obs = compute(scene_stub(sigma=0.0), "distance", pts, ["OBSERVED_MULTI_VIEW"] * 2)
    inf = compute(scene_stub(sigma=0.0), "distance", pts, ["INFERRED"] * 2)
    assert inf["sigma"] > obs["sigma"] * (POINT_REL_SIGMA["INFERRED"] / POINT_REL_SIGMA["OBSERVED_MULTI_VIEW"]) * 0.9
    with_scale = compute(scene_stub(sigma=0.2), "distance", pts, ["OBSERVED_MULTI_VIEW"] * 2)
    assert with_scale["sigma"] > 0.2 * 5.0 * 0.99


def test_components_angle_height(scene_stub):
    s = scene_stub()
    assert compute(s, "horizontal_distance", [[0, 0, 0], [3, 2, 4]])["value"] == pytest.approx(5.0)
    assert compute(s, "height_difference", [[0, 0, 0], [3, 2, 4]])["value"] == pytest.approx(2.0)
    assert compute(s, "height_above_ground", [[1, 2.5, -3]])["value"] == pytest.approx(2.5)
    a = compute(s, "angle", [[1, 0, 0], [0, 0, 0], [0, 0, 1]])
    assert a["value"] == pytest.approx(90.0) and a["unit"] == "deg"
    with pytest.raises(ValueError):
        compute(s, "distance", [[0, 0, 0]])
    with pytest.raises(ValueError):
        compute(s, "volume", [[0, 0, 0]])


def test_track_distance_refuses_to_interpolate_across_gaps(scene_stub):
    s = scene_stub()
    sm = [{"t": t / 10, "p": [t / 10, 0, -5]} for t in range(0, 11)] + [{"t": 4 + t / 10, "p": [4 + t / 10, 0, -5]} for t in range(11)]
    other = [{"t": t / 10, "p": [0, 0, -8]} for t in range(0, 51)]
    s["agents"] = [{"id": "A", "smoothed": sm, "summary": {"position_sigma_median": 0.1}},
                   {"id": "B", "smoothed": other, "summary": {"position_sigma_median": 0.1}}]
    ok = compute(s, "track_distance_at_time", targets=["A", "B"], t=0.5)
    assert ok["value"] == pytest.approx((0.5 ** 2 + 3 ** 2) ** 0.5, rel=1e-3)
    with pytest.raises(ValueError, match="gap"):
        compute(s, "track_distance_at_time", targets=["A", "B"], t=2.5)


def test_record_keeps_provenance(scene_stub):
    r = compute(scene_stub(), "distance", [[0, 0, 0], [1, 0, 0]])
    rec = record(r, [[0, 0, 0], [1, 0, 0]], None, "alice", "kerb", [[0, 0, 0], [1, 0, 0]])
    assert rec["id"].startswith("M") and rec["actor"] == "alice" and rec["active"] and rec["label"] == "kerb"

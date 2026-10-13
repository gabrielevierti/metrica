"""Tracker association, gap recording, interaction flags, fragment re-linking and kinematics."""
from __future__ import annotations

import numpy as np

from metrica.tracking.linking import link_fragments, renumber
from metrica.tracking.tracker import Tracker, iou_matrix


def _det(x, y, w=30, h=80, conf=0.9, cat="person"):
    return {"bbox": [x, y, x + w, y + h], "conf": conf, "label": 0, "category": cat}


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    assert np.allclose(iou_matrix(a, b), [[1.0, 1 / 3, 0.0]])


def test_single_subject_with_gap_is_one_track_and_gap_is_recorded():
    tr = Tracker(max_gap_steps=10, min_hits=3)
    for k in range(40):
        if 15 <= k < 20:  # occluded for 5 steps: no detections
            tr.update(k, k / 10, [])
            continue
        tr.update(k, k / 10, [_det(100 + 4 * k, 200)])
    out = tr.finish(gap_frames=1)
    assert len(out) == 1
    t = out[0]
    assert len(t["observations"]) == 35  # missing frames are not filled in
    assert len(t["gaps"]) == 1 and t["gaps"][0]["after_frame"] == 14 and t["gaps"][0]["before_frame"] == 20


def test_two_people_crossing_keep_ids_and_flag_interaction():
    tr = Tracker(max_gap_steps=5)
    for k in range(30):
        tr.update(k, k / 10, [_det(100 + 8 * k, 200), _det(340 - 8 * k, 205)])
    out = tr.finish()
    assert len(out) == 2
    starts = sorted(t["observations"][0]["bbox"][0] for t in out)
    assert starts == [100, 340]
    # each track keeps moving in its own direction
    for t in out:
        xs = [o["bbox"][0] for o in t["observations"]]
        assert np.all(np.diff(xs) > 0) or np.all(np.diff(xs) < 0)
    assert any(f["type"] == "close_interaction" for t in out for f in t["flags"])


def test_class_groups_are_not_mixed():
    tr = Tracker(max_gap_steps=5)
    for k in range(10):
        cat = "person" if k < 5 else "car"
        tr.update(k, k / 10, [_det(100 + k, 200, cat=cat)])
    out = tr.finish()
    assert {t["group"] for t in out} == {"person", "vehicle"}


def _frag(raw_id, t0, t1, x0, vx, y=200, h=80, app=None):
    ts = np.arange(t0, t1 + 1e-9, 0.1)
    obs = [{"bbox": [x0 + vx * (t - t0), y, x0 + vx * (t - t0) + 30, y + h], "conf": 0.9, "category": "person",
            "frame": int(round(t * 10)), "t": float(t)} for t in ts]
    return {"raw_id": raw_id, "category": "person", "group": "person", "observations": obs,
            "first_frame": obs[0]["frame"], "last_frame": obs[-1]["frame"], "t0": float(ts[0]), "t1": float(ts[-1]),
            "gaps": [], "flags": [], "links": [], "appearance": app}


def test_fragments_relinked_as_recorded_hypothesis():
    app = list(np.ones(48) / 48)
    a = _frag(1, 0.0, 2.0, 100, 40, app=app)            # ends at x=180 at t=2
    b = _frag(2, 3.0, 5.0, 100 + 40 * 3.0, 40, app=app)  # starts where motion predicts
    c = _frag(3, 3.0, 5.0, 700, -40, app=app)            # elsewhere: must not be linked
    out = link_fragments([a, b, c], max_gap_s=4.0)
    assert len(out) == 2
    merged = next(t for t in out if t.get("fragments") and len(t["fragments"]) == 2)
    assert merged["fragments"] == [1, 2]
    assert merged["links"][0]["method"].startswith("motion+appearance")
    assert any("hypothesis" in g["reason"] for g in merged["gaps"])
    ids = [t["id"] for t in renumber(out, "v1")]
    assert ids == ["v1-P1", "v1-P2"]


def _person_track(cam, n=40, speed=1.4, hide_feet=(), fps=10.0):
    obs = []
    for k in range(n):
        t = k / fps
        foot = np.array([-3.0 + speed * t, 0.0, -18.0])
        (uf, vf), = cam.project(foot[None])[0]
        (uh, vh), = cam.project((foot + [0, 1.72, 0])[None])[0]
        hgt = vf - vh
        y2 = vf - (0.45 * hgt if k in hide_feet else 0.0)  # lower body hidden behind something
        obs.append({"bbox": [uf - 0.18 * hgt, vh, uf + 0.18 * hgt, y2], "conf": 0.9, "category": "person",
                    "frame": k, "t": t})
    return {"id": "v1-P1", "category": "person", "group": "person", "observations": obs, "t0": 0.0,
            "t1": (n - 1) / fps, "gaps": [], "flags": [], "links": []}


def test_trajectory_speed_and_unreliable_boxes():
    from metrica.geometry.camera import Camera, Intrinsics, Pose, look_rotation
    from metrica.tracking.trajectories import build_track

    cam = Camera(Intrinsics.from_hfov(1280, 720, 60.0), Pose(look_rotation(np.radians(15), 0.0), np.array([0, 6.0, 0])))
    tr = build_track(_person_track(cam), lambda f: cam, 1280, 720, 0.05, {})
    assert abs(tr["summary"]["speed_median"] - 1.4) < 0.1
    assert abs(tr["stature"]["median"] - 1.72) < 0.05
    assert not any(f.get("type") == "implausible_speed" for f in tr["flags"])
    # samples whose feet are hidden are flagged and excluded instead of producing a fast, wrong jump
    tr2 = build_track(_person_track(cam, hide_feet={10, 11, 12}), lambda f: cam, 1280, 720, 0.05, {})
    bad = [s for s in tr2["samples"] if any("inconsistent" in x for x in s["flags"])]
    assert {s["f"] for s in bad} == {10, 11, 12} and all(s["s"] is None for s in bad)
    assert abs(tr2["summary"]["speed_median"] - 1.4) < 0.15
    # a physically impossible speed for a person is flagged
    tr3 = build_track(_person_track(cam, speed=9.0), lambda f: cam, 1280, 720, 0.05, {})
    assert any(f.get("type") == "implausible_speed" for f in tr3["flags"])

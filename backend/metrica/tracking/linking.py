"""Re-linking track fragments across occlusions (a machine *hypothesis*, always recorded).

Two fragments A -> B (B starts after A ends) are linked when:
same class group, gap below a limit, B's start is where A's motion predicts (image space,
or ground plane when 3D positions are available), and their appearance is similar.
The link, its score and the evidence used are stored on the merged track so an analyst
can review and undo it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .appearance import distance as app_distance


def _end_state(tr: dict[str, Any], n: int = 6) -> tuple[np.ndarray, np.ndarray, float]:
    obs = tr["observations"][-n:]
    c = np.array([[(o["bbox"][0] + o["bbox"][2]) / 2, o["bbox"][3]] for o in obs])
    ts = np.array([o["t"] for o in obs])
    v = (c[-1] - c[0]) / max(ts[-1] - ts[0], 1e-3) if len(obs) > 1 else np.zeros(2)
    h = float(np.median([o["bbox"][3] - o["bbox"][1] for o in obs]))
    return c[-1], v, h


def _start_state(tr: dict[str, Any]) -> tuple[np.ndarray, float]:
    o = tr["observations"][0]
    return np.array([(o["bbox"][0] + o["bbox"][2]) / 2, o["bbox"][3]]), float(o["bbox"][3] - o["bbox"][1])


def link_fragments(tracks: list[dict[str, Any]], max_gap_s: float, embed: dict[int, np.ndarray] | None = None,
                   max_app_dist: float = 0.35) -> list[dict[str, Any]]:
    """Merge fragments; returns the new track list (merged tracks keep all observations)."""
    if len(tracks) < 2:
        return tracks
    n = len(tracks)
    cost = np.full((n, n), np.inf)
    evidence: dict[tuple[int, int], dict[str, Any]] = {}
    for i, a in enumerate(tracks):
        pa, va, ha = _end_state(a)
        for j, b in enumerate(tracks):
            if i == j or a["group"] != b["group"]:
                continue
            gap = b["t0"] - a["t1"]
            if gap <= 0 or gap > max_gap_s:
                continue
            pb, hb = _start_state(b)
            # size continuity (same subject at similar distance)
            if not 0.6 < hb / max(ha, 1) < 1.6:
                continue
            pred = pa + va * gap
            err = np.linalg.norm(pb - pred) / max(ha, 1)
            radius = 1.0 + 0.6 * gap  # tolerance grows with the gap (subject may change direction)
            if err > radius:
                continue
            ad = app_distance(np.asarray(a["appearance"]) if a.get("appearance") else None,
                              np.asarray(b["appearance"]) if b.get("appearance") else None)
            if embed is not None and a["raw_id"] in embed and b["raw_id"] in embed:
                ad = 0.5 * ad + 0.5 * float(1 - embed[a["raw_id"]] @ embed[b["raw_id"]]) / 2
            if ad > max_app_dist:
                continue
            c = 0.5 * err / radius + 0.5 * ad / max_app_dist + 0.1 * gap / max_gap_s
            cost[i, j] = c
            evidence[(i, j)] = {"gap_s": round(gap, 2), "position_error_heights": round(float(err), 2),
                                "appearance_distance": round(float(ad), 3), "score": round(float(1 - c / 1.1), 3)}
    if not np.isfinite(cost).any():
        return tracks
    big = 1e6
    r, c = linear_sum_assignment(np.where(np.isfinite(cost), cost, big))
    nxt: dict[int, int] = {}
    for i, j in zip(r, c):
        if cost[i, j] < big:
            nxt[i] = j
    has_prev = set(nxt.values())
    out = []
    for i in range(n):
        if i in has_prev:
            continue
        chain = [i]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        base = dict(tracks[chain[0]])
        base["observations"] = [o for k in chain for o in tracks[k]["observations"]]
        base["gaps"] = list(tracks[chain[0]]["gaps"])
        base["flags"] = list(tracks[chain[0]]["flags"])
        base["links"] = []
        base["fragments"] = [tracks[k]["raw_id"] for k in chain]
        for a_i, b_i in zip(chain[:-1], chain[1:]):
            a, b = tracks[a_i], tracks[b_i]
            base["gaps"].append({"after_frame": a["last_frame"], "before_frame": b["first_frame"], "t0": a["t1"],
                                 "t1": b["t0"], "duration_s": round(b["t0"] - a["t1"], 3),
                                 "reason": "fragments re-linked by motion and appearance (hypothesis)"})
            base["gaps"].extend(b["gaps"])
            base["flags"].extend(b["flags"])
            base["links"].append({"from_fragment": a["raw_id"], "to_fragment": b["raw_id"],
                                  "method": "motion+appearance re-identification (machine hypothesis, reviewable)",
                                  **evidence[(a_i, b_i)]})
        base["last_frame"] = tracks[chain[-1]]["last_frame"]
        base["t1"] = tracks[chain[-1]]["t1"]
        votes: dict[str, int] = {}
        for o in base["observations"]:
            votes[o["category"]] = votes.get(o["category"], 0) + 1
        base["category"] = max(votes, key=votes.get)
        base["gaps"].sort(key=lambda g: g["t0"])
        out.append(base)
    out.sort(key=lambda d: d["t0"])
    return out


def renumber(tracks: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    """Stable, readable ids: <video>-<category letter><n>, e.g. v1-P3, v1-V2."""
    counters: dict[str, int] = {}
    for tr in sorted(tracks, key=lambda d: (d["t0"], d["raw_id"])):
        letter = {"person": "P", "vehicle": "V", "two_wheeler": "B", "animal": "A"}.get(tr["group"], "O")
        counters[letter] = counters.get(letter, 0) + 1
        tr["id"] = f"{prefix}-{letter}{counters[letter]}"
    return sorted(tracks, key=lambda d: d["t0"])

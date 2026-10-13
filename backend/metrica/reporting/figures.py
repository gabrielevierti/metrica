"""Report figures (matplotlib + the numpy splat renderer). All figures are generated from
the scene graph, so they show exactly what the analysis contains."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

from ..reconstruction.layers import EVIDENCE, read_layer  # noqa: E402
from .render import overview_camera, sample_mesh, splat  # noqa: E402

EVIDENCE_COLORS = {"OBSERVED_MULTI_VIEW": (46, 160, 90), "OBSERVED_SINGLE_VIEW": (90, 130, 220),
                   "FITTED_PRIMITIVE": (230, 170, 40), "CLASS_PRIOR": (190, 90, 200), "INFERRED": (220, 80, 80),
                   "UNKNOWN": (150, 150, 150)}
TRACK_COLORS = ["#e4572e", "#2e86ab", "#f3a712", "#29335c", "#a8c686", "#669bbc", "#c1121f", "#6a4c93", "#1982c4",
                "#8ac926"]


def _style(ax):
    ax.set_facecolor("#fbfbfd")
    ax.grid(True, color="#e3e3ea", lw=0.6)
    ax.set_axisbelow(True)


def plan_view(scene: dict[str, Any], path: Path, highlight: set[str] | None = None) -> None:
    unit = scene["coordinate_system"]["unit"]
    fig, ax = plt.subplots(figsize=(9, 8), dpi=130)
    _style(ax)
    xs, zs = [], []
    for e in scene["entities"]:
        g = e["geometry"]
        if g["type"] == "ground_plane":
            for p in g["observed_polygons_xz"]:
                P = np.asarray(p)
                ax.add_patch(MplPolygon(P, closed=True, fc="#dfe9df", ec="#9db59d", lw=0.6, zorder=1))
                xs += list(P[:, 0])
                zs += list(P[:, 1])
            for p in g["inferred_polygons_xz"]:
                ax.add_patch(MplPolygon(np.asarray(p), closed=True, fc="none", ec="#d06060", lw=0.6, hatch="////",
                                        zorder=1, alpha=0.6))
            for name, r in e.get("regions", {}).items():
                c = {"road": "#c9c9cf", "sidewalk": "#e7e1d3", "terrain": "#cfe3c4"}.get(name, "#ddd")
                for p in r["polygons_xz"]:
                    ax.add_patch(MplPolygon(np.asarray(p), closed=True, fc=c, ec="none", alpha=0.8, zorder=1.5))
        elif g["type"] == "plane_polygon":
            o, u = np.asarray(g["origin"]), np.asarray(g["u_axis"])
            us = [pt[0] for poly in g["polygons_uv"] for pt in poly]
            a, b = o + min(us) * u, o + max(us) * u
            ax.plot([a[0], b[0]], [a[2], b[2]], color="#8d5a3b", lw=3, solid_capstyle="butt", zorder=4)
            ax.text((a[0] + b[0]) / 2, (a[2] + b[2]) / 2, e["id"].split("-")[-2] + "-" + e["id"].split("-")[-1],
                    fontsize=6, color="#5a3a26", zorder=6)
            xs += [a[0], b[0]]
            zs += [a[2], b[2]]
        elif g["type"] == "cylinder":
            b = g["base"]
            ax.add_patch(plt.Circle((b[0], b[2]), max(g["radius_bottom"], 0.08), color="#d4a017", zorder=5))
            ax.text(b[0] + 0.3, b[2], e["id"].split("-", 2)[-1], fontsize=6, color="#7a5d00", zorder=6)
        elif g["type"] in ("vehicle_box", "box"):
            c = np.asarray(g["center"])
            L, W = (g["length"], g["width"]) if g["type"] == "vehicle_box" else (g["size"][0], g["size"][2])
            yaw = g["yaw_rad"]
            ax_l, ax_w = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
            cc = c[[0, 2]]
            corners = [cc + sl * L / 2 * ax_l + sw * W / 2 * ax_w for sl, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
            col = "#2e86ab" if g["type"] == "vehicle_box" else "#888"
            prior = e.get("evidence", {}).get("class") == "CLASS_PRIOR"
            ax.add_patch(MplPolygon(np.array(corners), closed=True, fc=col, ec="k", lw=0.5, alpha=0.5 if prior else 0.8,
                                    ls="--" if prior else "-", zorder=5))
            ax.text(cc[0], cc[1], e["id"].split("-", 2)[-1], fontsize=6, zorder=6)
    for i, a in enumerate(scene["agents"]):
        if not a.get("smoothed"):
            continue
        P = np.array([s["p"] for s in a["smoothed"]])
        col = TRACK_COLORS[i % len(TRACK_COLORS)]
        lw = 2.2 if (highlight and a["id"] in highlight) else 1.2
        ax.plot(P[:, 0], P[:, 2], color=col, lw=lw, alpha=0.9, zorder=7)
        ax.plot(P[0, 0], P[0, 2], "o", color=col, ms=3, zorder=8)
        ax.annotate(a["id"], (P[-1, 0], P[-1, 2]), fontsize=6, color=col, zorder=9)
        xs += list(P[:, 0])
        zs += list(P[:, 2])
    for c in scene["cameras"]:
        T = np.asarray(c["world_from_cam"])
        C = T[:3, 3]
        f = T[:3, 2]
        ax.plot(C[0], C[2], marker="^", color=c["color"], ms=9, mec="k", zorder=10)
        ax.arrow(C[0], C[2], f[0] * 2, f[2] * 2, color=c["color"], width=0.05, zorder=10)
        ax.text(C[0], C[2] - 1, c["video"], fontsize=7, color=c["color"], zorder=10)
        if c["kind"] == "moving":
            M = np.asarray(c["poses"]["world_from_cam"])
            ax.plot(M[:, 0, 3], M[:, 2, 3], "--", color=c["color"], lw=1, zorder=9)
            xs += list(M[:, 0, 3])
            zs += list(M[:, 2, 3])
        xs.append(C[0])
        zs.append(C[2])
    if xs:
        x0, x1 = np.percentile(xs, [1, 99])
        z0, z1 = np.percentile(zs, [1, 99])
        pad = 0.08 * max(x1 - x0, z1 - z0, 1)
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(z1 + pad, z0 - pad)  # +Z towards the viewer at the bottom
    ax.set_aspect("equal")
    ax.set_xlabel(f"X ({unit})")
    ax.set_ylabel(f"Z ({unit})")
    status = scene["coordinate_system"]["status"]
    ax.set_title(f"Plan view - scale status: {status}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def perspective_views(scene: dict[str, Any], scene_dir: Path, path_color: Path, path_evidence: Path) -> bool:
    meshes = [l for l in scene["layers"] if l["kind"] == "mesh"]
    if not meshes:
        return False
    P, C, E = [], [], []
    for l in meshes:
        meta = {"files": {k: v.replace("layers/", "") for k, v in l["files"].items()}}
        L = read_layer(scene_dir / "layers", meta)
        if "idx" not in L or len(L["idx"]) == 0:
            continue
        if len(L["idx"]) > 400_000:
            sel = np.random.default_rng(0).choice(len(L["idx"]), 400_000, replace=False)
            L["idx"] = L["idx"][sel]
        p, c = sample_mesh(L["pos"], L["idx"], L.get("col", np.full((len(L["pos"]), 3), 180, np.uint8)), 2)
        ev = np.array([EVIDENCE_COLORS[EVIDENCE[i]] for i in L.get("evi", np.zeros(len(L["pos"]), int))], np.uint8)
        _, e = sample_mesh(L["pos"], L["idx"], ev, 2)
        P.append(p)
        C.append(c)
        E.append(e)
    if not P:
        return False
    P, C, E = np.concatenate(P), np.concatenate(C), np.concatenate(E)
    lo, hi = np.percentile(P, 2, 0), np.percentile(P, 98, 0)
    cam = overview_camera(lo, hi, 1200, 760, elevation_deg=32, azimuth_deg=-28)
    import cv2

    cv2.imwrite(str(path_color), splat(P, C, cam)[:, :, ::-1])
    cv2.imwrite(str(path_evidence), splat(P, E, cam)[:, :, ::-1])
    return True


def quality_chart(ingest: dict[str, Any], path: Path) -> None:
    s = ingest["quality"]["samples"]
    t = [x["t"] for x in s]
    fig, ax = plt.subplots(2, 1, figsize=(8, 3.6), dpi=120, sharex=True)
    for a in ax:
        _style(a)
    ax[0].plot(t, [x["sharpness"] for x in s], color="#2e86ab", lw=1)
    ax[0].set_ylabel("sharpness")
    ax[1].plot(t, [x["brightness"] for x in s], color="#f3a712", lw=1, label="brightness")
    ax[1].plot(t, [x["noise_sigma"] * 10 for x in s], color="#c1121f", lw=1, label="noise x10")
    ax[1].set_ylabel("level")
    ax[1].set_xlabel("video time (s)")
    ax[1].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Technical quality over time - {ingest['video_id']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def timeline_chart(scene: dict[str, Any], path: Path) -> None:
    agents = [a for a in scene["agents"] if a["samples"]]
    if not agents:
        return
    fig, ax = plt.subplots(figsize=(9, max(2.2, 0.22 * len(agents) + 1)), dpi=120)
    _style(ax)
    for i, a in enumerate(agents):
        col = TRACK_COLORS[i % len(TRACK_COLORS)]
        ts = np.array([s["t"] for s in a["samples"]])
        # draw observed spans; gaps longer than 1 s are left empty
        breaks = np.where(np.diff(ts) > 1.0)[0]
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, len(ts) - 1]
        for s0, e0 in zip(starts, ends):
            ax.plot([ts[s0], ts[e0]], [i, i], color=col, lw=5, solid_capstyle="butt")
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels([a["id"] for a in agents], fontsize=6)
    ax.set_xlabel("case time (s)")
    ax.set_title("Observation spans per track (gaps = not observed)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

"""Structure from Motion with COLMAP (pycolmap): keyframe poses, sparse points, frame localisation.

* One COLMAP camera per video (shared intrinsics, SIMPLE_RADIAL), refined by bundle adjustment.
* Pairs: sequential overlap (+ quadratic) inside each video, plus a sparse grid of cross-video
  pairs so overlapping recordings can be registered into one reconstruction.
* Non-keyframes are localised by absolute pose (PnP + LO-RANSAC + refinement) against the
  reconstruction's 3D points; frames that cannot be localised are interpolated between their
  neighbouring registered keyframes and flagged as interpolated. Poses are never extrapolated.
"""
from __future__ import annotations

import os
import shutil
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..log import get_logger
from .camera import Intrinsics, Pose, slerp_rotation

log = get_logger("sfm")
DEFAULT_HFOV_PRIOR = 65.0


@dataclass
class SfMResult:
    ok: bool
    model_dir: Path | None = None
    images: dict[tuple[str, int], int] = field(default_factory=dict)       # (vid, frame) -> image_id
    poses: dict[tuple[str, int], Pose] = field(default_factory=dict)       # SfM frame, world_from_cam
    intrinsics: dict[str, Intrinsics] = field(default_factory=dict)
    camera_ids: dict[str, int] = field(default_factory=dict)
    points: np.ndarray | None = None          # (N,3)
    colors: np.ndarray | None = None          # (N,3) uint8
    errors: np.ndarray | None = None          # reprojection error (px)
    track_len: np.ndarray | None = None
    point_ids: np.ndarray | None = None
    observations: dict[tuple[str, int], np.ndarray] = field(default_factory=dict)  # point indices seen per image
    stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _pairs(groups: dict[str, list[str]], overlap: int, cross_every: int = 3, max_cross: int = 600) -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for names in groups.values():
        n = len(names)
        for i in range(n):
            for k in range(1, overlap + 1):
                if i + k < n:
                    pairs.add((names[i], names[i + k]))
            k = 2
            while k <= n:  # quadratic overlap: long-range pairs help loop closure
                if i + k < n:
                    pairs.add((names[i], names[i + k]))
                k *= 2
    vids = list(groups)
    for a in range(len(vids)):
        for b in range(a + 1, len(vids)):
            A = groups[vids[a]][::cross_every]
            B = groups[vids[b]][::cross_every]
            cross = [(x, y) for x in A for y in B]
            if len(cross) > max_cross:
                idx = np.linspace(0, len(cross) - 1, max_cross).astype(int)
                cross = [cross[i] for i in idx]
            pairs.update(cross)
    return sorted(pairs)


def _pose_from_rigid(rig) -> Pose:
    M = np.asarray(rig.matrix())
    return Pose.from_cam_from_world(M[:3, :3], M[:3, 3])


def _intr_from_camera(cam) -> Intrinsics:
    p = np.asarray(cam.params, float)
    name = cam.model.name if hasattr(cam.model, "name") else str(cam.model)
    if "SIMPLE_RADIAL" in name:
        return Intrinsics(p[0], p[0], p[1], p[2], cam.width, cam.height, float(p[3]), "sfm")
    if "SIMPLE_PINHOLE" in name:
        return Intrinsics(p[0], p[0], p[1], p[2], cam.width, cam.height, 0.0, "sfm")
    if name.endswith("PINHOLE"):
        return Intrinsics(p[0], p[1], p[2], p[3], cam.width, cam.height, 0.0, "sfm")
    if "RADIAL" in name:
        return Intrinsics(p[0], p[0], p[1], p[2], cam.width, cam.height, float(p[3]), "sfm")
    return Intrinsics(p[0], p[0], cam.width / 2, cam.height / 2, cam.width, cam.height, 0.0, "sfm")


def run_sfm(work: Path, groups: dict[str, list[tuple[int, Path]]], max_features: int, overlap: int,
            hfov_prior: float | None = None, cancelled: Callable[[], bool] | None = None,
            progress: Callable[[float, str], None] | None = None) -> SfMResult:
    import pycolmap

    progress = progress or (lambda f, m: None)
    work = Path(work)
    if work.exists():
        shutil.rmtree(work)
    imgdir = work / "images"
    names: dict[str, list[str]] = {}
    name_to_key: dict[str, tuple[str, int]] = {}
    for vid, items in groups.items():
        names[vid] = []
        for frame, path in sorted(items):
            nm = f"{vid}/{frame:07d}.jpg"
            _link(Path(path), imgdir / nm)
            names[vid].append(nm)
            name_to_key[nm] = (vid, frame)
    n_images = sum(len(v) for v in names.values())
    if n_images < 3:
        return SfMResult(False, warnings=["Fewer than 3 keyframes: structure from motion is not possible."])
    db = work / "database.db"
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    # An explicit focal prior matters: COLMAP's generic default (1.2 x image size, ~45 deg) is
    # narrower than typical phone/body cameras and can make self-calibration diverge on short
    # sequential captures. The prior is only an initial value; bundle adjustment refines it.
    prior_source = "configured" if hfov_prior else "default prior (typical phone/hand-held camera)"
    hfov_prior = float(hfov_prior or DEFAULT_HFOV_PRIOR)
    first = cv2.imread(str(imgdir / next(iter(names.values()))[0]))
    h0, w0 = first.shape[:2]
    f0 = (w0 / 2) / np.tan(np.radians(hfov_prior) / 2)
    reader.camera_params = f"{f0},{w0 / 2},{h0 / 2},0"
    ext = pycolmap.FeatureExtractionOptions()
    ext.sift.max_num_features = int(max_features)
    ext.num_threads = os.cpu_count() or 2
    progress(0.05, f"extracting features from {n_images} keyframes")
    pycolmap.extract_features(db, imgdir, camera_mode=pycolmap.CameraMode.PER_FOLDER, reader_options=reader,
                              extraction_options=ext, device=pycolmap.Device.cpu)
    if cancelled and cancelled():
        return SfMResult(False, warnings=["cancelled"])
    pairs = _pairs(names, overlap)
    pair_file = work / "pairs.txt"
    pair_file.write_text("\n".join(f"{a} {b}" for a, b in pairs) + "\n")
    progress(0.35, f"matching {len(pairs)} image pairs")
    po = pycolmap.ImportedPairingOptions()
    po.match_list_path = str(pair_file)
    pycolmap.match_image_pairs(db, pairing_options=po, device=pycolmap.Device.cpu)
    if cancelled and cancelled():
        return SfMResult(False, warnings=["cancelled"])
    progress(0.6, "incremental mapping")
    opts = pycolmap.IncrementalPipelineOptions()
    opts.num_threads = os.cpu_count() or 2
    opts.min_model_size = max(3, min(10, n_images // 3))
    opts.random_seed = 0
    out = work / "sparse"
    out.mkdir(parents=True, exist_ok=True)
    recs = pycolmap.incremental_mapping(db, imgdir, out, options=opts)
    if not recs:
        return SfMResult(False, warnings=["COLMAP could not initialise a reconstruction (insufficient parallax, "
                                          "texture or overlap)."])
    best_k = max(recs, key=lambda k: recs[k].num_reg_images())
    rec = recs[best_k]
    res = SfMResult(True, model_dir=out / str(best_k))
    others = [recs[k].num_reg_images() for k in recs if k != best_k]
    if others:
        res.warnings.append(f"COLMAP produced {len(recs)} disconnected models; the largest ({rec.num_reg_images()} "
                            f"images) is used. Images in the others ({sum(others)}) are not registered.")
    vid_of_cam: dict[int, str] = {}
    for iid, im in rec.images.items():
        if not im.has_pose:
            continue
        key = name_to_key.get(im.name)
        if key is None:
            continue
        res.images[key] = iid
        res.poses[key] = _pose_from_rigid(im.cam_from_world())
        vid_of_cam[im.camera_id] = key[0]
    for cid, vid in vid_of_cam.items():
        res.intrinsics[vid] = _intr_from_camera(rec.cameras[cid])
        res.camera_ids[vid] = cid
    pts = list(rec.points3D.items())
    res.point_ids = np.array([k for k, _ in pts], np.int64)
    res.points = np.array([p.xyz for _, p in pts], float).reshape(-1, 3)
    res.colors = np.array([p.color for _, p in pts], np.uint8).reshape(-1, 3)
    res.errors = np.array([p.error for _, p in pts], float)
    res.track_len = np.array([p.track.length() for _, p in pts], int)
    pid_to_idx = {int(pid): k for k, pid in enumerate(res.point_ids)}
    for key, iid in res.images.items():
        im = rec.images[iid]
        res.observations[key] = np.array([pid_to_idx[int(p.point3D_id)] for p in im.points2D if p.has_point3D()
                                          and int(p.point3D_id) in pid_to_idx], np.int32)
    per_video = {vid: {"keyframes": len(names[vid]), "registered": sum(1 for k in res.images if k[0] == vid)}
                 for vid in names}
    angles = _triangulation_angles(rec, res, sample=3000)
    res.stats = {
        "images": n_images, "registered": len(res.images), "models": len(recs),
        "points3D": int(len(res.points)), "mean_reprojection_error_px": float(rec.compute_mean_reprojection_error()),
        "mean_track_length": float(rec.compute_mean_track_length()), "pairs": len(pairs), "per_video": per_video,
        "median_triangulation_angle_deg": angles,
        "focal_prior_hfov_deg": hfov_prior, "focal_prior_source": prior_source,
    }
    for vid, intr in res.intrinsics.items():
        ratio = intr.fx / f0
        res.stats.setdefault("refined_hfov_deg", {})[vid] = intr.hfov_deg
        if not 0.6 < ratio < 1.6 or abs(intr.k1) > 0.4:
            res.warnings.append(f"{vid}: refined focal length ({intr.hfov_deg:.0f} deg HFOV, k1={intr.k1:.2f}) is far "
                                "from the prior; self-calibration may be unstable. Supply the field of view if known.")
    for vid, s in per_video.items():
        if s["registered"] < 0.5 * s["keyframes"]:
            res.warnings.append(f"{vid}: only {s['registered']}/{s['keyframes']} keyframes registered.")
    if angles is not None and angles < 2.0:
        res.warnings.append(f"Low median triangulation angle ({angles:.1f} deg): depth along the viewing direction "
                            "is weakly constrained.")
    log.info(f"SfM: {len(res.images)}/{n_images} images registered, {len(res.points)} points, "
             f"reprojection {res.stats['mean_reprojection_error_px']:.2f} px")
    return res


def _triangulation_angles(rec, res: SfMResult, sample: int) -> float | None:
    centers = {iid: np.asarray(rec.images[iid].projection_center()) for iid in res.images.values()}
    ids = list(rec.points3D.keys())
    if not ids:
        return None
    rng = np.random.default_rng(0)
    ids = rng.choice(ids, min(sample, len(ids)), replace=False)
    angs = []
    for pid in ids:
        p = rec.points3D[int(pid)]
        cs = [centers[e.image_id] for e in p.track.elements if e.image_id in centers]
        if len(cs) < 2:
            continue
        v = np.array(cs) - p.xyz
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        cosm = np.clip(v @ v.T, -1, 1)
        angs.append(np.degrees(np.arccos(cosm.min())))
    return float(np.median(angs)) if angs else None


# ------------------------------------------------------------------------------ localisation
class Localizer:
    """Absolute pose of arbitrary frames against the reconstruction (2D-3D matching)."""

    def __init__(self, model_dir: Path, db_path: Path, res: SfMResult, vid: str, max_features: int = 4096):
        import pycolmap

        self.pycolmap = pycolmap
        self.rec = pycolmap.Reconstruction(str(model_dir))
        self.db = pycolmap.Database.open(str(db_path))
        self.res = res
        self.vid = vid
        self.kf = sorted((f, iid) for (v, f), iid in res.images.items() if v == vid)
        self.cam = self.rec.cameras[res.camera_ids[vid]]
        ext = pycolmap.FeatureExtractionOptions()
        ext.sift.max_num_features = int(max_features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.sift = pycolmap.Sift(ext, device=pycolmap.Device.cpu)
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def _kf_data(self, iid: int) -> tuple[np.ndarray, np.ndarray]:
        if iid not in self._cache:
            im = self.rec.images[iid]
            desc = np.asarray(self.db.read_descriptors(iid).data).astype(np.float32)
            p3 = np.full(len(desc), -1, np.int64)
            for k, p in enumerate(im.points2D):
                if p.has_point3D():
                    p3[k] = p.point3D_id
            self._cache[iid] = (desc, p3)
            if len(self._cache) > 12:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[iid]

    def neighbours(self, frame: int, k: int = 3) -> list[int]:
        if not self.kf:
            return []
        fr = np.array([f for f, _ in self.kf])
        order = np.argsort(np.abs(fr - frame))[:k]
        return [self.kf[i][1] for i in order]

    def localize(self, frame: int, gray: np.ndarray) -> tuple[Pose, dict[str, Any]] | None:
        kps, desc = self.sift.extract(np.ascontiguousarray(gray))
        if len(desc) < 30:
            return None
        d8 = np.minimum(np.round(desc * 512.0), 255).astype(np.float32)
        p2, p3 = [], []
        seen: set[int] = set()
        for iid in self.neighbours(frame):
            kd, kp3 = self._kf_data(iid)
            valid = kp3 >= 0
            if valid.sum() < 20:
                continue
            kd_v, kp3_v = kd[valid], kp3[valid]
            matches = self.matcher.knnMatch(d8, kd_v, k=2)
            for m in matches:
                if len(m) < 2 or m[0].distance > 0.8 * m[1].distance:
                    continue
                pid = int(kp3_v[m[0].trainIdx])
                if pid in seen or pid not in self.rec.points3D:
                    continue
                seen.add(pid)
                p2.append(kps[m[0].queryIdx, :2])
                p3.append(self.rec.points3D[pid].xyz)
        if len(p2) < 25:
            return None
        est = self.pycolmap.AbsolutePoseEstimationOptions()
        est.ransac.max_error = 6.0
        out = self.pycolmap.estimate_and_refine_absolute_pose(np.asarray(p2, float), np.asarray(p3, float),
                                                              self.cam, estimation_options=est)
        if out is None:
            return None
        n_in = int(out["num_inliers"])
        if n_in < 25 or n_in / len(p2) < 0.25:
            return None
        return _pose_from_rigid(out["cam_from_world"]), {"inliers": n_in, "matches": len(p2)}


def interpolate_pose(frame: int, known: dict[int, Pose]) -> Pose | None:
    """Pose between the two nearest known frames (no extrapolation)."""
    if not known:
        return None
    fr = np.array(sorted(known))
    if frame < fr[0] or frame > fr[-1]:
        return None
    j = int(np.searchsorted(fr, frame))
    if fr[min(j, len(fr) - 1)] == frame:
        return known[int(frame)]
    a, b = int(fr[j - 1]), int(fr[j])
    w = (frame - a) / (b - a)
    R = slerp_rotation(known[a].R_wc, known[b].R_wc, w)
    return Pose(R, (1 - w) * known[a].C + w * known[b].C)

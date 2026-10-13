"""Analysis configuration and quality/speed profiles.

All tunable parameters live here, with the reason for each default. The resolved
configuration is stored verbatim in every case manifest, so a result can always be
traced back to the exact parameters that produced it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

Profile = Literal["fast", "standard", "quality"]
# Keys written by earlier development builds; accepted when reading old configs and ignored.
RETIRED_KEYS = {"use_pose"}
CaptureMode = Literal["auto", "static", "rotating", "moving"]


@dataclass
class AnalysisConfig:
    profile: Profile = "standard"

    # ---- sampling ---------------------------------------------------------------------
    # Frames per second analysed by the detector/tracker. Tracking quality degrades below
    # ~5 fps for walking people; above ~12 fps the gain is small on CPU.
    analysis_fps: float = 8.0
    # Process every frame instead of the analysis rate (slow; for short clips/re-runs).
    process_all_frames: bool = False
    # Upper bound on SfM keyframes per video. SIFT matching cost grows ~linearly with
    # sequential matching, so this mainly bounds runtime.
    max_keyframes: int = 120
    # A new keyframe is taken once the median optical flow since the last keyframe exceeds
    # this fraction of the image diagonal (provides parallax for triangulation).
    keyframe_flow_frac: float = 0.035
    # Frames darker/blurrier than these are not used for reconstruction (still analysed).
    min_sharpness: float = 12.0

    # ---- capture mode -----------------------------------------------------------------
    # "auto" classifies each video (static / rotating / moving); can be forced per video.
    capture_mode: CaptureMode = "auto"
    capture_mode_overrides: dict[str, str] = field(default_factory=dict)

    # ---- models -------------------------------------------------------------------------
    detector: str = "yolox_s"
    detector_conf: float = 0.30
    detector_nms_iou: float = 0.55
    depth_input_short_side: int = 518
    use_semantic_segmentation: bool = True
    use_instance_masks: bool = True
    masks_per_track: int = 1
    max_mask_calls: int = 30
    use_faces: bool = True
    use_ocr: bool = True

    # ---- structure from motion ------------------------------------------------------------
    sfm_max_features: int = 4096
    sfm_sequential_overlap: int = 12
    localize_all_frames: bool = True

    # ---- dense reconstruction --------------------------------------------------------------
    # TSDF voxel size as a fraction of the scene's horizontal extent (auto-scaled).
    tsdf_voxel_frac: float = 1.0 / 220.0
    tsdf_max_voxels: int = 18_000_000
    depth_consistency_rel: float = 0.06

    # ---- metric priors (documented assumptions, never silent) ---------------------------------
    # Adult stature: pooled mean ~1.70 m, SD ~0.09 m (sexes combined, varies by population).
    person_height_m: float = 1.70
    person_height_sigma_m: float = 0.09
    # Hand-held camera height (eye/chest level) used only when nothing better is available.
    handheld_camera_height_m: float = 1.50
    handheld_camera_height_sigma_m: float = 0.25
    # Optional known horizontal field of view (degrees) for all videos.
    hfov_deg: float | None = None

    # ---- tracking -------------------------------------------------------------------------
    track_max_gap_s: float = 1.5
    reid_link_max_gap_s: float = 6.0
    min_track_hits: int = 3

    # ---- evidence handling ------------------------------------------------------------------
    copy_originals: bool = True

    # ---- multi-video ------------------------------------------------------------------------
    audio_sync: bool = True
    visual_sync: bool = True
    joint_sfm: bool = True
    manual_offsets: dict[str, float] = field(default_factory=dict)
    manual_align_points: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    # ---- inferred geometry --------------------------------------------------------------------
    infer_hidden_surfaces: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self, keys: list[str] | None = None) -> str:
        d = self.to_dict()
        if keys is not None:
            d = {k: d[k] for k in keys}
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AnalysisConfig:
        data = dict(data or {})
        profile = data.get("profile", "standard")
        cfg = profile_defaults(profile)
        valid = {f.name for f in fields(cls)}
        for k in RETIRED_KEYS & set(data):  # accepted for old case folders, ignored
            data.pop(k)
        unknown = set(data) - valid
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        for k, v in data.items():
            setattr(cfg, k, v)
        return cfg


def profile_defaults(profile: str) -> AnalysisConfig:
    """Return a config preset. 'standard' is the automatic default."""
    if profile == "fast":
        return AnalysisConfig(profile="fast", analysis_fps=5.0, max_keyframes=60, sfm_max_features=2048,
                              use_semantic_segmentation=False, use_instance_masks=False, masks_per_track=0, max_mask_calls=0,
                              tsdf_voxel_frac=1 / 140, use_ocr=False, depth_input_short_side=392)
    if profile == "quality":
        return AnalysisConfig(profile="quality", analysis_fps=12.0, max_keyframes=250, sfm_max_features=8192,
                              detector="yolox_m", masks_per_track=3, max_mask_calls=150,
                              tsdf_voxel_frac=1 / 320, depth_input_short_side=644)
    if profile == "standard":
        return AnalysisConfig()
    raise ValueError(f"Unknown profile '{profile}' (expected fast, standard or quality)")

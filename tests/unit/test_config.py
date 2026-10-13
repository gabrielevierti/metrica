from __future__ import annotations

import pytest

from metrica.config import AnalysisConfig, profile_defaults


def test_profiles_differ_and_are_valid():
    fast, std, q = (profile_defaults(p) for p in ("fast", "standard", "quality"))
    assert fast.analysis_fps < std.analysis_fps < q.analysis_fps
    assert not fast.use_semantic_segmentation and std.use_semantic_segmentation
    assert q.detector == "yolox_m"
    with pytest.raises(ValueError):
        profile_defaults("ultra")


def test_from_dict_rejects_unknown_keys_and_applies_overrides():
    cfg = AnalysisConfig.from_dict({"profile": "fast", "hfov_deg": 70.0})
    assert cfg.profile == "fast" and cfg.hfov_deg == 70.0 and cfg.analysis_fps == 5.0
    with pytest.raises(ValueError):
        AnalysisConfig.from_dict({"not_a_parameter": 1})


def test_fingerprint_is_stable_and_key_scoped():
    a, b = AnalysisConfig(), AnalysisConfig()
    assert a.fingerprint() == b.fingerprint()
    b.detector_conf = 0.5
    assert a.fingerprint() != b.fingerprint()
    # unrelated keys do not change a stage's scoped fingerprint
    assert a.fingerprint(["tsdf_voxel_frac"]) == b.fingerprint(["tsdf_voxel_frac"])

"""Coordinate-system calibration status and scale records.

Every reconstruction carries one of four statuses. Measurements inherit it, and the UI
and reports display it next to every number.

* ARBITRARY          - no gravity, no scale. Only ratios inside one reconstruction are meaningful.
* RELATIVE           - gravity-aligned and internally consistent, but the scale rests on an
                       unverified assumption (e.g. assumed camera height). Units are
                       "su" (scene units). Not real-world distances.
* ESTIMATED_METRIC   - metric scale from statistical evidence with quantified uncertainty
                       (e.g. the statures of many observed people). Metres +/- sigma.
* CALIBRATED_METRIC  - metric scale from analyst-supplied known dimensions or calibrated cameras.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

STATUS_ORDER = ["ARBITRARY", "RELATIVE", "ESTIMATED_METRIC", "CALIBRATED_METRIC"]
STATUS_TEXT = {
    "ARBITRARY": "Arbitrary scale and orientation. Distances are not physical.",
    "RELATIVE": "Gravity-aligned, scale unverified (scene units). Not a real-world measurement.",
    "ESTIMATED_METRIC": "Metric scale estimated from statistical priors; values carry the stated uncertainty.",
    "CALIBRATED_METRIC": "Metric scale calibrated from known reference dimension(s).",
}


@dataclass
class ScaleRecord:
    status: str
    unit: str                     # "m" or "su"
    method: str
    scale_sigma_rel: float | None  # 1-sigma relative uncertainty of the scale
    gravity_aligned: bool
    assumptions: list[str] = field(default_factory=list)
    references: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["explanation"] = STATUS_TEXT[self.status]
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ScaleRecord:
        return cls(**{k: d[k] for k in ("status", "unit", "method", "scale_sigma_rel", "gravity_aligned",
                                        "assumptions", "references", "details") if k in d})


def person_height_scale_sigma(n_people: int, sigma_individual: float = 0.09, height: float = 1.70,
                              sigma_population_mean: float = 0.03, box_bias_rel: float = 0.03) -> float:
    """Relative 1-sigma of a scale derived from the statures of n distinct people.

    Combines (in quadrature) the averaging of individual variation (sigma/sqrt(n)), the
    uncertainty of the population mean itself, and systematic detector-box bias (hair,
    feet, footwear). The last two do not average out.
    """
    n = max(1, int(n_people))
    return float(np.sqrt((sigma_individual / height) ** 2 / n + (sigma_population_mean / height) ** 2
                         + box_bias_rel ** 2))


def known_distance_scale(measured: list[tuple[float, float, float]]) -> tuple[float, float]:
    """Scale factor from analyst references [(reconstructed_length, true_length, true_sigma)].

    Returns (scale, relative sigma). Weighted by the references' own uncertainty plus
    a 1% floor for point-picking error per reference.
    """
    if not measured:
        raise ValueError("no references")
    s = np.array([t / r for r, t, _ in measured])
    rel = np.array([np.hypot(sig / t if t > 0 else 0.0, 0.01) for _, t, sig in measured])
    w = 1 / rel ** 2
    scale = float((s * w).sum() / w.sum())
    sigma_rel = float(np.sqrt(1 / w.sum()))
    if len(s) > 1:  # include the observed spread between references
        spread = float(np.std(s / scale, ddof=1)) / np.sqrt(len(s))
        sigma_rel = float(np.hypot(sigma_rel, spread))
    return scale, sigma_rel


def resolve_global_scale(base: dict[str, Any], references: list[dict[str, Any]] | None) -> tuple[float, dict[str, Any]]:
    """Global scale factor k (displayed = k * base frame) and the resulting scale record.

    ``references`` are analyst-supplied known distances. Their endpoints are stored in the
    *base* (unscaled) world frame so that re-calibrating never compounds earlier corrections.
    """
    refs = [r for r in (references or []) if r.get("active", True)]
    if not refs:
        return 1.0, dict(base)
    meas = []
    for r in refs:
        d = float(np.linalg.norm(np.asarray(r["p2"], float) - np.asarray(r["p1"], float)))
        if d > 1e-9 and r.get("true_length_m", 0) > 0:
            meas.append((d, float(r["true_length_m"]), float(r.get("sigma_m", 0.0))))
    if not meas:
        return 1.0, dict(base)
    k, sig = known_distance_scale(meas)
    rec = ScaleRecord("CALIBRATED_METRIC", "m", f"analyst reference distance(s) ({len(meas)})", sig,
                      bool(base.get("gravity_aligned", True)),
                      assumptions=["Reference points were placed on the correct features in the 3D scene.",
                                   "Reference lengths are correct within their stated uncertainty."],
                      references=[{k2: r.get(k2) for k2 in ("id", "label", "true_length_m", "sigma_m", "created",
                                                            "actor")} for r in refs],
                      details={"factor_vs_previous": k, "previous_status": base.get("status"),
                               "previous_method": base.get("method")})
    return k, rec.to_json()

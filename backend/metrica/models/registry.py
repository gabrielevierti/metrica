"""Registry of pretrained models: source, pinned SHA-256, license, and what uses them.

Weights are never bundled with METRICA. They are downloaded on explicit request
(`metrica models download`, the installer, or the "Models" panel of the app), verified
against the pinned SHA-256, and stored in ``$METRICA_MODELS`` (default
``~/.cache/metrica/models``). A model whose license restricts use (AGPL) is never
downloaded unless the user explicitly accepts its terms.
"""
from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..evidence.hashing import sha256_file
from ..log import get_logger

log = get_logger("models")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    file: str
    url: str
    sha256: str
    size_mb: float
    license: str
    license_url: str
    task: str
    about: str
    used_by: tuple[str, ...]
    default: bool = True          # part of the standard download set
    restricted: bool = False      # copyleft / non-permissive: requires explicit acceptance
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"

MODELS: dict[str, ModelSpec] = {m.key: m for m in [
    ModelSpec(
        key="depth_anything_v2_s", file="depth_anything_v2_vits_dynamic.onnx",
        url="https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits_dynamic.onnx",
        sha256="46c4e8eeda3a27f34701831b6a2ec7753d7b38779b215acb5633424703deed8f", size_mb=95,
        license="Apache-2.0", license_url="https://github.com/DepthAnything/Depth-Anything-V2/blob/main/LICENSE",
        task="monocular relative depth (affine-invariant inverse depth)",
        about="Depth Anything V2 Small, ONNX export by fabio-sim/Depth-Anything-ONNX",
        used_by=("depth",),
        notes="Only the Small variant is Apache-2.0; Base/Large are CC-BY-NC-4.0 and are not used."),
    ModelSpec(
        key="yolox_s", file="yolox_s.onnx",
        url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx",
        sha256="c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063", size_mb=35,
        license="Apache-2.0", license_url="https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE",
        task="object detection, 80 COCO classes", about="YOLOX-S (Megvii), official ONNX release",
        used_by=("detection", "tracking")),
    ModelSpec(
        key="yolox_m", file="yolox_m.onnx",
        url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.onnx",
        sha256="21ff6cfdeb53b013bac2249599e55f00bff3cfdfdab37ed7a4620818c1d15b3f", size_mb=97,
        license="Apache-2.0", license_url="https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE",
        task="object detection, 80 COCO classes (more accurate, ~2.5x slower)",
        about="YOLOX-M (Megvii), official ONNX release", used_by=("detection (quality profile)",),
        default=False),
    ModelSpec(
        key="efficientsam_ti", file="image_segmentation_efficientsam_ti_2025april.onnx",
        url=f"{_ZOO}/image_segmentation_efficientsam/image_segmentation_efficientsam_ti_2025april.onnx",
        sha256="4eb496e0a7259d435b49b66faf1754aa45a5c382a34558ddda9a8c6fe5915d77", size_mb=47,
        license="Apache-2.0", license_url="https://github.com/yformer/EfficientSAM/blob/main/LICENSE",
        task="promptable instance segmentation (box prompts from detections)",
        about="EfficientSAM-Ti via OpenCV Zoo", used_by=("instance masks", "object reconstruction")),
    ModelSpec(
        key="duc_cityscapes", file="ResNet101-DUC-12-int8.onnx",
        url="https://media.githubusercontent.com/media/onnx/models/main/validated/vision/object_detection_segmentation/duc/model/ResNet101-DUC-12-int8.onnx",
        sha256="08a3c24af5e375d064d7a4bb80b46191adf5f0f302fd4b5bee7f632ebdadbf67", size_mb=63,
        license="Apache-2.0", license_url="https://github.com/onnx/models/blob/main/LICENSE",
        task="semantic segmentation, 19 Cityscapes classes (road, building, pole, vegetation, ...)",
        about="ResNet101-DUC-HDC (TuSimple), int8, ONNX Model Zoo", used_by=("scene semantics",),
        notes=("Trained on Cityscapes (street scenes). The weights are distributed under Apache-2.0 by the "
               "ONNX Model Zoo; the Cityscapes dataset itself has its own (non-commercial) terms. Indoor "
               "scenes are outside its training domain: geometry-based labels take over there.")),
    ModelSpec(
        key="yunet", file="face_detection_yunet_2023mar.onnx",
        url=f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", size_mb=0.2,
        license="MIT", license_url="https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE",
        task="face detection with 5 landmarks (detection only, no recognition)",
        about="YuNet (Shiqi Yu) via OpenCV Zoo", used_by=("face observations",)),
    ModelSpec(
        key="ppocr_det_en", file="text_detection_en_ppocrv3_2023may.onnx",
        url=f"{_ZOO}/text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx",
        sha256="03f550c6b406fda8bf54bd8327815f6c7e2edd98cea02348c93d879254366587", size_mb=2.4,
        license="Apache-2.0", license_url="https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE",
        task="scene text detection", about="PP-OCRv3 English detector via OpenCV Zoo",
        used_by=("OCR candidates",)),
    ModelSpec(
        key="crnn_en", file="text_recognition_CRNN_EN_2021sep.onnx",
        url=f"{_ZOO}/text_recognition_crnn/text_recognition_CRNN_EN_2021sep.onnx",
        sha256="a84b1f6e11a65c2d733cb0cc1f014aae3f99051e3f11447dc282faa678eee544", size_mb=33,
        license="Apache-2.0", license_url="https://github.com/opencv/opencv_zoo/blob/main/models/text_recognition_crnn/LICENSE",
        task="scene text recognition (digits + latin letters)", about="CRNN English via OpenCV Zoo",
        used_by=("OCR candidates",)),
    ModelSpec(
        key="youtu_reid", file="person_reid_youtu_2021nov.onnx",
        url=f"{_ZOO}/person_reid_youtureid/person_reid_youtu_2021nov.onnx",
        sha256="0579683334d4b9440221606dcb461656dd0dc64143b18f48faedaced9b4f580d", size_mb=102,
        license="Apache-2.0", license_url="https://github.com/opencv/opencv_zoo/blob/main/models/person_reid_youtureid/LICENSE",
        task="person appearance embedding for track re-linking (not identity recognition)",
        about="Youtu ReID via OpenCV Zoo", used_by=("track re-linking",), default=False),
    ModelSpec(
        key="yolo11n", file="yolo11n.onnx",
        url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx",
        sha256="634279b40c07c6391472c51ad45b81ebc48706a9a1fe72dd3396322acd0c053b", size_mb=10,
        license="AGPL-3.0", license_url="https://github.com/ultralytics/ultralytics/blob/main/LICENSE",
        task="object detection, 80 COCO classes", about="Ultralytics YOLO11n (optional, copyleft)",
        used_by=("detection (optional)",), default=False, restricted=True,
        notes=("AGPL-3.0: distributing software that includes it, or offering it as a network service, "
               "triggers AGPL obligations unless a commercial Ultralytics license is obtained.")),
]}

DEFAULT_SET = [k for k, m in MODELS.items() if m.default]


def models_dir() -> Path:
    d = Path(os.environ.get("METRICA_MODELS", Path.home() / ".cache" / "metrica" / "models")).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_path(key: str) -> Path:
    return models_dir() / MODELS[key].file


def _verify_cache() -> Path:
    return models_dir() / ".verified.json"


def _load_verified() -> dict[str, Any]:
    p = _verify_cache()
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except (OSError, ValueError):
        return {}


def is_verified(key: str) -> bool:
    """True if the file exists and matches its pinned hash (cached by size+mtime)."""
    spec = MODELS[key]
    p = model_path(key)
    if not p.exists():
        return False
    st = p.stat()
    cache = _load_verified()
    tag = f"{st.st_size}:{int(st.st_mtime)}"
    if cache.get(spec.file) == tag:
        return True
    ok = sha256_file(p) == spec.sha256
    if ok:
        cache[spec.file] = tag
        try:
            _verify_cache().write_text(json.dumps(cache))
        except OSError:
            pass
    return ok


def status(key: str) -> dict[str, Any]:
    spec = MODELS[key]
    p = model_path(key)
    present = p.exists()
    verified = present and is_verified(key)
    return {**{k: v for k, v in asdict(spec).items() if k != "extra"}, "path": str(p), "present": present,
            "verified": verified,
            "state": "ready" if verified else ("corrupt" if present else "missing")}


def all_status() -> list[dict[str, Any]]:
    return [status(k) for k in MODELS]


class ModelUnavailable(RuntimeError):
    def __init__(self, key: str, reason: str):
        super().__init__(f"Model '{key}' unavailable: {reason}")
        self.key = key
        self.reason = reason


def require(key: str) -> Path:
    """Path of a verified model or ModelUnavailable (never downloads implicitly)."""
    if key not in MODELS:
        raise ModelUnavailable(key, "unknown model")
    p = model_path(key)
    if not p.exists():
        raise ModelUnavailable(key, f"not downloaded (run `metrica models download {key}`)")
    if not is_verified(key):
        raise ModelUnavailable(key, f"file {p} does not match the pinned SHA-256; delete it and download again")
    return p


def available(key: str) -> bool:
    try:
        require(key)
        return True
    except ModelUnavailable:
        return False


def download(key: str, accept_restricted: bool = False,
             progress: Callable[[str, int, int], None] | None = None) -> Path:
    spec = MODELS[key]
    if spec.restricted and not accept_restricted:
        raise PermissionError(f"{key} is licensed {spec.license}; use --accept-restricted (CLI) or accept_restricted=True (API) after reviewing "
                              f"{spec.license_url}")
    dst = model_path(key)
    if dst.exists() and is_verified(key):
        return dst
    tmp = dst.with_suffix(dst.suffix + ".part")
    log.info(f"Downloading model {key} ({spec.size_mb:.0f} MB) from {spec.url}")
    req = urllib.request.Request(spec.url, headers={"User-Agent": "metrica-model-downloader"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress:
                progress(key, done, total)
    got = sha256_file(tmp)
    if got != spec.sha256:
        tmp.unlink(missing_ok=True)
        raise OSError(f"SHA-256 mismatch for {key}: expected {spec.sha256[:16]}..., got {got[:16]}... "
                      "(download discarded)")
    os.replace(tmp, dst)
    is_verified(key)
    log.info(f"Model {key} verified and stored at {dst}")
    return dst


def model_record(key: str) -> dict[str, Any]:
    """Provenance record for manifests and reports."""
    spec = MODELS[key]
    return {"key": key, "file": spec.file, "sha256": spec.sha256, "license": spec.license,
            "source": spec.url, "about": spec.about, "task": spec.task}

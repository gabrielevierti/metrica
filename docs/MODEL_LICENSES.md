# Models, libraries and licences

METRICA's own code is licensed under Apache-2.0. **No model weights, footage or datasets are
included in the repository or the release archive.** Models are downloaded on explicit request
from their original publishers, verified against the SHA-256 pinned in
`backend/metrica/models/registry.py`, and stored outside the repository
(`~/.cache/metrica/models` or `$METRICA_MODELS`). The table below is the result of a licence
review done for this release; licences can change upstream — check the linked sources before
redistributing anything.

## Pretrained models

| Key | Model | Task in METRICA | Licence of the weights | Default | Source |
|---|---|---|---|---|---|
| `depth_anything_v2_s` | Depth Anything V2 **Small** (ONNX export by fabio-sim/Depth-Anything-ONNX) | monocular relative depth | Apache-2.0 | yes | github.com/DepthAnything/Depth-Anything-V2 |
| `yolox_s` | YOLOX-S (Megvii) | object detection (COCO) | Apache-2.0 | yes | github.com/Megvii-BaseDetection/YOLOX |
| `yolox_m` | YOLOX-M (Megvii) | detection, quality profile | Apache-2.0 | optional | same |
| `efficientsam_ti` | EfficientSAM-Ti (OpenCV Zoo) | box-prompted instance masks | Apache-2.0 | yes | github.com/yformer/EfficientSAM, github.com/opencv/opencv_zoo |
| `duc_cityscapes` | ResNet101-DUC-HDC int8 (ONNX Model Zoo) | street-scene semantic segmentation | Apache-2.0 (ONNX Model Zoo); trained on Cityscapes — see note | yes | github.com/onnx/models |
| `yunet` | YuNet face detector (OpenCV Zoo) | face *detection* only | MIT | yes | github.com/opencv/opencv_zoo |
| `ppocr_det_en` | PP-OCRv3 English text detector (OpenCV Zoo) | text detection for OCR candidates | Apache-2.0 | yes | github.com/PaddlePaddle/PaddleOCR |
| `crnn_en` | CRNN English recogniser (OpenCV Zoo) | text recognition for OCR candidates | Apache-2.0 | yes | github.com/opencv/opencv_zoo |
| `youtu_reid` | Youtu ReID (OpenCV Zoo) | appearance cue for re-linking fragments of the same video | Apache-2.0 | optional | github.com/opencv/opencv_zoo |
| `yolo11n` | Ultralytics YOLO11n | alternative detector | **AGPL-3.0** | opt-in only (`--accept-restricted`) | github.com/ultralytics/ultralytics |

Notes:

* **Depth Anything V2**: only the *Small* checkpoint is Apache-2.0. The Base, Large and Giant
  checkpoints are CC-BY-NC-4.0 (non-commercial) and are deliberately not supported.
* **DUC / Cityscapes**: the ONNX Model Zoo distributes the converted weights under Apache-2.0.
  The Cityscapes dataset used for training has its own terms (non-commercial research use of the
  *dataset*). Whether dataset terms extend to models trained on it is a legal question we do not
  answer here; organisations with commercial use should review it, and can disable dense
  semantics (`use_semantic_segmentation = false`), in which case METRICA uses geometry-only
  labels.
* **YOLO11 (AGPL-3.0)**: offering software that includes it as a network service, or
  distributing it, triggers AGPL obligations unless a commercial Ultralytics licence is obtained.
  METRICA never downloads it without explicit acceptance and does not use it by default. Its
  predecessor Sopralluogo used YOLO11 by default; METRICA replaced it with YOLOX.
* **COCO-trained detectors**: COCO annotations are CC-BY-4.0; images have mixed Flickr licences.
  The detectors' weights are distributed by their authors under the licences above.
* Model weights are third-party works. The SHA-256 pinning verifies that the file is the one
  reviewed for this release, not that it is free of bias or error; see EVALUATION.md for what was
  measured.

## Python libraries (installed by pip, not redistributed by METRICA)

| Package | Version tested | Licence |
|---|---|---|
| numpy | 2.5.4 | BSD-3-Clause (+ bundled MIT/Zlib/0BSD/CC0 parts) |
| scipy | 1.18.1 | BSD-3-Clause |
| opencv-contrib-python-headless | 4.14.0.94 | Apache-2.0 (OpenCV); wheel bundles third-party code, see its LICENSE-3RD-PARTY |
| onnxruntime | 1.31.0 | MIT |
| pycolmap (COLMAP) | 4.2.1 | BSD-3-Clause |
| av (PyAV) | 19.0.1 | BSD-3-Clause — **binary wheels bundle FFmpeg built with GPL components (libx264, libx265)**, see below |
| scikit-image | 0.26.0 | BSD-3-Clause |
| trimesh | 5.1.1 | MIT |
| fastapi / starlette / uvicorn | 0.143.0 / 1.7.0 / 0.54.0 | MIT / BSD-3-Clause / BSD-3-Clause |
| pydantic | 2.14.0 | MIT |
| python-multipart | 0.0.32 | Apache-2.0 |
| jinja2 | 3.1.6 | BSD-3-Clause |
| reportlab | 5.0.1 | BSD |
| matplotlib | 3.11.2 | Matplotlib licence (PSF-based, BSD-compatible) |
| pillow (indirect) | 12.3.0 | MIT-CMU |

**FFmpeg / x264 via PyAV.** The PyAV wheels on PyPI include FFmpeg libraries compiled with
libx264 and libx265, which are GPL-2.0+. METRICA uses them through PyAV's API and does not
redistribute them; users obtain them from PyPI. If you redistribute a packaged METRICA
environment (for example a frozen installer that includes these wheels), GPL obligations apply to
that bundle. Alternatives: build PyAV against an LGPL-only FFmpeg. METRICA's review proxies then
fall back to MPEG-4 Part 2, which many browsers cannot play (the exact-frame viewer still works).

## Frontend libraries (bundled into `frontend/dist`)

| Package | Version | Licence |
|---|---|---|
| react, react-dom | 19.3.0 | MIT |
| three | 0.180.0 | MIT |
| @react-three/fiber | 9.8.1 | MIT |
| @react-three/drei | 10.7.9 | MIT |
| three-mesh-bvh, three-stdlib, camera-controls, troika-three-text (via drei) | 0.8.3, 2.36.1, 3.1.2, 0.52.5 | MIT |
| zustand | 5.0.15 | MIT |

The built bundle in `frontend/dist` contains these MIT-licensed libraries; their notices are
reproduced in [NOTICE](../NOTICE).

## Test data

No real recordings are part of METRICA. Tests and the demo use procedurally rendered scenes
(`metrica.synthetic`, generated at run time, CC0). The public OpenCV sample clip `vtest.avi`
(from the OpenCV samples repository) was used during development for a qualitative check; it is
not included.

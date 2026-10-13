# Evaluation

METRICA claims no accuracy that has not been measured. This document describes how accuracy is
measured, and reports every number that was actually measured for release 0.1.0, together with
the conditions. **These numbers describe the test conditions below, not real footage in
general**: real cameras add compression, motion blur, rolling shutter, lens distortion, low light
and unknown lenses, and real scenes are harder for the neural models than these.

## 1. Procedure

### 1.1 Synthetic scenes with exact ground truth

`metrica.synthetic` ray-traces a procedural street (three buildings with textured, windowed
facades, a kiosk, a bus shelter, planters, a parked car, two poles of 6 m and 7.5 m, a 0.9 m
bollard, road markings, moving boxes) with exact camera poses, intrinsics, depth and object
geometry. Three camera set-ups are available:

| Kind | Camera | Resolution, rate | Notes |
|---|---|---|---|
| `walk` | hand-held, walking ~13 m sideways and forwards while panning, bobbing and rolling | 640×360, 12 fps, 6 s | no anti-aliasing |
| `static` | fixed, 4.2 m high, 22° down, 1.5° roll, 70° HFOV | 640×360, 12 fps, 6 s | 2× supersampled |
| `pan` | fixed centre, yaw sweeping 32° | 640×360, 12 fps, 6 s | 2× supersampled |

The synthetic scene contains **no people** (the procedural movers are boxes, invisible to a COCO
detector), so people-based metric scale is never available: every synthetic case is, correctly,
reported as RELATIVE. Scale is therefore evaluated separately (section 3.4).

`metrica evaluate CASE GROUND_TRUTH.json` maps the reconstruction into the ground-truth frame with
a similarity transform estimated **only from the camera poses** (rotation from the camera
orientations, scale and translation from the camera centres; for a single static camera, the
known camera height and orientation), never from the geometry being evaluated, and reports:

* **trajectory** — absolute trajectory error (RMSE of camera centres) after that alignment;
* **orientation** — static: tilt error (angle between estimated and true gravity in the camera
  frame; the heading of a single camera is a free choice of the world frame); panning: per-frame
  orientation error after one common rotation;
* **surface** — for 6 ground-truth views, every reconstructed vertex is projected and its depth
  compared with the true depth at that pixel: median and 90th percentile relative error, the
  fraction within 5 %, and *completeness* (fraction of true visible pixels with a reconstructed
  vertex within 5 % nearby);
* **poles** — detections matched to true poles within 0.5 m (horizontal), with position, height
  and radius errors; unmatched detections are listed with their distance to the nearest pole;
* **facades** — detected planes matched to true building faces: normal angle and offset errors.

Reproduce:

```bash
metrica demo --kind walk   --out /tmp/eval-walk      # renders, analyses and evaluates
metrica demo --kind static --out /tmp/eval-static
metrica demo --kind pan    --out /tmp/eval-pan
```

### 1.2 Unit and integration tests

`pytest` runs geometry, tracking, measurement, evidence and API tests on generated data with
known answers; `pytest -m slow` adds structure from motion on rendered frames and the complete
pipeline + report on synthetic footage.

### 1.3 Real footage

No real footage with surveyed ground truth was available for this release. The public OpenCV
sample `vtest.avi` (768×576, 10 fps, 79.5 s, a fixed camera over a pedestrian plaza; not
included in the repository) was used for an end-to-end run and for internal-consistency checks
only (section 4). Before relying on METRICA for real cases, calibrate with surveyed distances and
verify an independent distance (see the user guide).

## 2. Test results

Run on Linux x86-64, Python 3.12.3, CPU only (2 vCPU, 8 GB), with the default models.

| Suite | Result |
|---|---|
| Complete suite (`pytest`, including the slow tests) | 55 passed in 86 s |
| Unit tests (`tests/unit`) | 50 passed |
| API integration (`tests/integration/test_api.py`) | 3 passed |
| SfM on rendered frames (`test_sfm_synthetic.py`, slow) | passed in both runs — 16/16 images registered; ATE 49.3 mm and 28.9 mm over a 13.6 m path (centres-only similarity alignment); focal length −1.7 % and −0.8 % (484.5 / 488.9 px vs 492.8 px). COLMAP is not bit-for-bit deterministic across runs (multi-threaded RANSAC), hence the spread |
| Full pipeline + report on synthetic static footage (`test_pipeline_synthetic.py`, slow) | 1 passed (78 s) |
| Frontend type check and production build (`npm run build`) | passed |
| Lint (`ruff check backend tests`) | passed |

## 3. Synthetic accuracy (release 0.1.0, standard profile)

Each value comes from one run. Values that depend on structure from motion vary somewhat between
runs (see the SfM test in section 2); treat the walk numbers as indicative of the order of magnitude.

### 3.1 Camera geometry

| Case | Quantity | Measured |
|---|---|---|
| walk (SfM) | keyframes registered | 19 / 19 |
| walk | camera trajectory ATE (pose-based alignment) | 42.8 mm RMSE, 72.8 mm max, over a 13.4 m path (0.32 %) |
| walk | same, alignment from camera centres only | 16.5 mm RMSE |
| walk | horizontal field of view (SfM, with a 65° prior) | 66.9° vs 66.0° true |
| static | horizontal field of view from orthogonal vanishing points | 70.06° vs 70.0° true (focal 456.5 px vs 457.0 px; reported 1σ 4.0 %) |
| static | tilt (gravity direction) error | 0.19° |
| pan | horizontal field of view from vanishing points | 69.71° vs 70.0° true |
| pan | per-frame orientation error, 12 frames sampled over the 32° sweep | median 0.026°, max 0.067° |

### 3.2 Surfaces

| Case | Median relative depth error | 90th percentile | Vertices within 5 % | Completeness (5 %) |
|---|---|---|---|---|
| walk — multi-view TSDF fusion | 6.0 % | 18.1 % | 41.9 % | 51.9 % |
| static — single view | 0.9 % | 5.9 % | 87.1 % | 89.9 % |
| pan — single reference view | 0.3 % | 5.8 % | 85.5 % | 68.4 % |

The single-view numbers are relative errors *after* the evaluation similarity, i.e. they measure
the shape, not the scale, which for these cases rests on an assumed camera height (section 3.4).
On the static scene the segmentation model labelled 99 % of the image as "building" (procedural
textures are outside its training domain); METRICA detected the contradiction with the calibrated
ground plane, ignored the semantics for that view and recorded the fallback. Before that check
was added, the same scene measured 10 % median error and 7 % completeness.

### 3.3 Objects

| Case | Facades detected / matched | Normal error (median) | Offset error (median) | Poles matched (of 3) | Pole details |
|---|---|---|---|---|---|
| walk | 6 / 3 | 4.6° | 0.31 m | 0 | no pole detected |
| static | 10 / 8 | 3.2° | 0.49 m | 1 | pole_1: position error 0.19 m, radius 0.106 m vs 0.10 m, height ≥ 4.09 m (top outside the frame; reported as a lower bound; true 6.0 m) |
| pan | 7 / 4 | 8.1° | 0.79 m | 0 | one detection 0.67 m from pole_1 (outside the 0.5 m match radius) |

Pole detection is the weakest component: the 0.9 m bollard and the 7.5 m pole at 16–18 m were
not detected in any case. Unmatched facade detections are mostly fronts of the kiosk, shelter and
planters (box-shaped objects that are also vertical planes).

### 3.4 Scale

* **People as a ruler** cannot be evaluated on the synthetic scenes (no people). Its uncertainty
  model is a documented prior (section 4.1 of RECONSTRUCTION_METHODS.md), not a measured accuracy.
* **Assumed camera height (RELATIVE)**: walk 2.9 % from the truth (assumed 1.5 m, true ≈ 1.55 m);
  static/pan 28.6 % (assumed 3.0 su, true 4.2 m). These cases are correctly labelled RELATIVE and
  are not metric claims.
* **Analyst calibration** (`resolve_global_scale`, `known_distance_scale`) is verified by unit
  tests for exactness of the arithmetic; its real-world accuracy depends on the reference.

### 3.5 Multi-camera

On the static + walk pair of the same synthetic street, the shared-landmark alignment found 14
consistent matches out of 44 and was **rejected** by the acceptance rule (≥ 15 inliers and ≥ 40 %
of the matches); the moving unit is reported *unaligned* and the grouping is undetermined. Before
that rule existed, the same pair produced a spurious alignment (scale 1.03 vs 0.74 true, errors of
about 11 m), which is why the rule was introduced. Correct automatic multi-camera alignment was
therefore **not** demonstrated in this release; analyst point pairs remain the reliable route.

## 4. Real footage: vtest.avi (internal consistency, no ground truth)

End-to-end through the HTTP API (upload → analysis job → scene → report), standard profile, all
11 stages `ok`. First full run: 4 min 51 s on 2 vCPUs (detection and tracking 3 min 39 s).

| Check | Result | What it shows |
|---|---|---|
| Focal length from vanishing points | 29.3° horizontal FOV, 1σ 3.3 % (two horizontal directions agree: 1465 px and 1470 px) | an independent geometric estimate |
| People-based fit at that FOV | median reprojection error 3.7 px over 4 382 observations of 23 people; 96 % of observations within 12 % of the person height | the people data agree with the vanishing-point focal length |
| Same fit at the previously assumed 60° | 9.0 px, 11.6 % median relative error | why the focal estimate matters |
| Metric scale | ESTIMATED_METRIC, σ 3.7 % (was 21.4 % with the assumed focal length) | uncertainty of the scale model, not a measured error |
| Walking speeds | median of per-person medians 1.62 m/s (20 moving people, range 0.72–3.77 m/s) | plausibility only: typical adult walking speeds are about 1.2–1.5 m/s; consistent with brisk walking or a scale a few percent too large |
| Track plausibility | 31 of 4 654 person observations flagged as inconsistent with the subject's stature (hidden feet / merged boxes) and excluded from kinematics; 1 track flagged for an implausible speed (4.6 m/s) | the checks fire on real footage |
| Lamp post | detected as a pole, height 4.4 m ± 0.06 m (model σ) | not verified against a survey |

The estimated statures (median 1.69 m) are **not** evidence of accuracy: the same stature prior
(1.70 m) was used to fix the scale.

## 5. What has not been evaluated

* metric accuracy on real footage against surveyed references;
* body-worn, dash and drone footage; night-time and heavily compressed footage;
* rolling-shutter and zoom effects;
* detection, tracking, face-detection and OCR performance (standard benchmarks exist for the
  models used; METRICA's thresholds and filters were not benchmarked);
* audio synchronisation on real multi-camera recordings (unit-tested on synthetic signals only);
* GPU execution providers, macOS and Windows.

Contributions of footage with surveyed ground truth and the permission to use it for evaluation
are the most valuable way to improve this document.

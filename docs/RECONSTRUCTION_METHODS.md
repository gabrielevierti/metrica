# Reconstruction methods

This document describes what METRICA computes, how, under which assumptions, and how each
method fails. It is written for analysts who must explain a result and for developers who
want to improve one. Measured accuracy is in [EVALUATION.md](EVALUATION.md).

Conventions: world frame right-handed, **Y up**, ground plane `y = 0`; camera frame OpenCV
(x right, y down, z forward); `X_world = R_wc · X_cam + C`.

## 1. Ingest

* **Decoding** — PyAV (FFmpeg libraries). Frames are indexed in presentation order with the
  canonical time `t = pts − pts₀` (seconds); frame-rate metadata is not trusted. Variable
  frame rate is detected from the interval statistics (> 2 % irregular intervals) and gaps
  (> 2.5 × median interval) are listed. Display rotation metadata is applied. Packets that
  fail to decode are skipped and counted; decoding continues.
* **Quality** — per sampled frame: sharpness (variance of the Laplacian), brightness,
  contrast, clipped dark/bright fractions, noise σ (Immerkær), compression blocking (8-px
  border/inner gradient ratio), near-duplicates (frozen streams). Flags summarise problems.
* **Camera motion** — consecutive sampled frames are matched with forward-backward-checked
  Lucas–Kanade tracks. A homography (rotation/zoom or planar scene) and a fundamental matrix
  (general motion) are fitted with RANSAC; F must explain clearly more matches than H to call
  the motion *translating* (parallax). The video is classified static / rotating / moving with
  a confidence; time segments are merged (short runs absorbed).
* **Keyframes** — a new keyframe once the accumulated median image motion exceeds 3.5 % of the
  diagonal, choosing the sharpest usable frame in the window; bounded per video.
* **Background plate** — static cameras: per-pixel median of 48 frames spread over the video
  (moving subjects removed); rotating cameras: the sharpest frame.

## 2. Detection, tracking and observations

* **Detector** — YOLOX-S (YOLOX-M in the quality profile), 80 COCO classes mapped to METRICA
  categories (person, car, truck, bus, motorcycle, bicycle, animals, street furniture …).
* **Tracker** — ByteTrack-style: Kalman prediction in (x, y, aspect, height); stage 1 matches
  confident detections on IoU (with a centre-distance fallback for small fast boxes) and an
  HSV appearance distance; stage 2 matches low-confidence detections to the remaining tracks.
  Association never crosses class groups. Tracks shorter than 3 hits are dropped.
* **What is recorded instead of hidden** — gaps (with duration and reason), close interactions
  between same-class subjects (where identity swaps are likely), sudden appearance changes.
* **Re-linking** — fragments are joined across occlusions when the second starts where the
  first's motion predicts, at a similar size, with similar appearance (HSV histograms; plus
  Youtu ReID embeddings of the best crops when that optional model is installed). Each link
  is stored as a *hypothesis* with its evidence and can be reviewed.
* **Crops, masks** — the best crops per track (score = confidence × size × sharpness ×
  truncation); EfficientSAM masks from box prompts (GrabCut fallback) within a budget.
* **Faces** — YuNet face *detection* inside each person's head region (upsampled when small),
  with size, sharpness, detection score and a coarse pose class (frontal / partially turned /
  profile) from the landmarks. There is no recognition, no embedding, no comparison.
* **Vehicles** — dominant body colour (named colour from the mask's chroma, with confidence)
  and OCR *candidates* (PP-OCRv3 detection + CRNN recognition) aggregated over frames. They
  are reported as candidates with their support, never as identifications.

## 3. Depth and semantics

* **Monocular depth** — Depth Anything V2 Small gives affine-invariant inverse depth per
  keyframe (or background plate). It is never used as metric depth directly: it is aligned to
  geometric evidence first (section 5).
* **Semantics** — ResNet101-DUC (Cityscapes, 19 classes: road, sidewalk, building, wall, fence,
  pole, traffic light/sign, vegetation, terrain, sky, person, vehicles …) on up to 12 frames per
  video. Trained on European street scenes: indicative only elsewhere, and absent indoors. When
  unavailable, geometry-only labels are used and the fallback is recorded.

## 4. Camera geometry and scale

### 4.1 Static cameras (single-view metrology)

1. **Vertical vanishing point** — LSD line segments within 20° of image vertical, RANSAC for a
   common vanishing point, least-squares refinement. It fixes pitch and roll for a given focal
   length.
2. **Focal length from orthogonal vanishing points** — non-vertical segments are searched for up
   to two further vanishing points (horizontal building edges, kerbs, windows). For directions
   orthogonal in 3D with finite vanishing points *a*, *b* and principal point *c*:
   `f² = −(a − c)·(b − c)`. Each horizontal direction gives an estimate with the vertical
   direction; the two must agree within 25 %. The uncertainty is the bootstrap spread over both
   the horizontal and the vertical segments. Used only when its 1σ is below 12 %. Assumes the
   principal point is near the image centre and that the edges really are horizontal and
   vertical in the world (built environments).
3. **People as a ruler** — each whole, standing person (aspect ratio 1.8–5, not cut by the image
   border, confidence ≥ 0.5) gives a (feet, head) pixel pair. With an assumed mean stature
   (1.70 m, population SD 0.09 m), the pitch, roll and camera height that make all people stand
   upright on one plane are found by robust least squares (soft-L1), jointly with the vertical
   lines. If the focal length is not fixed by configuration or vanishing points, it is profiled
   over a range of fields of view: when the fit cannot distinguish them, the camera-height
   spread across plausible fields of view becomes a scale-uncertainty term.
4. **Acceptance** — the metric result is rejected when the median relative error exceeds 12 %
   of the person height or fewer than 50 % of observations agree. Then the orientation may still
   be used, but the scale becomes RELATIVE.
5. **Scale uncertainty** — `σ_rel² = (σ_H/H)²/n + (σ_pop/H)² + b²` with n people, the
   uncertainty of the population mean (3 cm) and a detector box bias term (3 %) that does not
   average out; plus the focal-length term in quadrature.
6. **Fallbacks** — without enough people: orientation from the vertical vanishing point, else
   from a ground plane fitted in the relative depth (semantic ground mask when available), else
   an assumed 20° tilt (warned); camera height assumed 3.0 scene units → status RELATIVE.

### 4.2 Rotating cameras (pan / tilt)

When the camera rotates about a fixed centre, images are related by homographies whatever the
scene depth: `x_ref ~ K R_refᵀ R_i K⁻¹ x_i`. Each analysed frame (plus two per second) is
registered to the reference frame with SIFT matches and RANSAC (directly, or chained through the
previous registered frame for frames that no longer overlap the reference, at most 40 hops);
`K⁻¹HK` is normalised and projected onto a rotation. Frames whose normalised matrix deviates from a
rotation by more than 6 % (zoom changes, camera translation, mismatches) get no pose; short gaps
(≤ 1 s) between oriented frames are interpolated and flagged. Person observations of all oriented
frames are mapped into the reference frame through their homographies before the single-view
calibration, which is exact for a pure rotation.

### 4.3 Moving cameras (structure from motion)

* COLMAP incremental mapping through pycolmap on the keyframes: SIFT features, pairs from a
  sequential window plus a sparse quadratic sample (loop closure) and cross-video pairs, one
  SIMPLE_RADIAL camera per video with an explicit focal prior (65° unless configured; COLMAP's
  generic default prior caused focal divergence in our tests), bundle adjustment.
* Warnings when the refined focal length is implausibly far from the prior or the distortion is
  extreme; fewer than 3 registered images → single-view fallback for those videos.
* **Localisation** — other analysis frames are registered against the model by 2D–3D matching
  (SIFT ratio test against neighbouring keyframes' descriptors, PnP-RANSAC + refinement);
  remaining frames between two registered ones get an interpolated pose, flagged as such; no
  extrapolation.
* **Gravity** — the up direction is the mean camera "up" (hand-held and vehicle cameras are
  mostly upright), refined by a RANSAC ground plane on semantic ground points; the ground
  becomes `y = 0`.
* **Scale** — from the statures of ≥ 3 tracked people (median of per-person heights in the SfM
  frame, σ as in 4.1), else from an assumed camera height of 1.5 m above the ground (hand-held),
  status RELATIVE.
* Joint reconstruction of several moving videos is attempted first (`joint_sfm`), so videos of
  the same place share one frame by construction.

### 4.4 Calibration statuses and analyst references

Statuses: ARBITRARY, RELATIVE, ESTIMATED_METRIC, CALIBRATED_METRIC (see the user guide).
Analyst references (two 3D points + true length + σ) are stored in the *base* frame; the global
factor `k` is the weighted mean of `true / reconstructed` with per-reference relative σ
`√((σ_ref/L)² + 1 %²)` (1 % for point picking) and the observed spread between references.

## 5. Dense surfaces

### 5.1 Single view

The relative inverse depth `d` is mapped to metric depth by `d ≈ a/z + b` fitted with RANSAC to
pixels of known depth: the calibrated ground plane under people's feet and on semantic ground
pixels (or, without usable semantics, the lower image part below the horizon). A second pass
refits with every pixel that agrees with the ground plane, which extends the calibration over the
whole depth range of the visible ground. If more than 40 % of the pixels lying on the calibrated
ground plane are labelled as buildings or walls, the segmentation is considered not to fit the
footage (domain mismatch): it is ignored for that view, geometry-based ground/wall labels are
used, and the fallback is recorded. Ground pixels are then snapped to the plane, vertical planar regions (superpixels with
vertical normals) are flattened, and a grid mesh is built that drops faces across depth
discontinuities, at grazing angles, stretched faces and tiny components. Evidence class:
OBSERVED_SINGLE_VIEW; confidence from the fit residual and the angle of incidence. Depth error
grows with distance: far geometry (beyond ~25× the camera height) is cut.

### 5.2 Multi-view (SfM units)

1. Each keyframe's relative depth is aligned to the SfM points visible in it (plus ground
   constraints), with checks on the depth range covered; a second pass adds constraints
   projected from the already aligned neighbours.
2. **Consistency** — each aligned depth sample is checked against the other keyframes:
   consistent, not visible, or conflicting (in front of a surface seen by another view).
3. **TSDF fusion** — narrow-band truncated signed distance fusion with a voxel size adapted to
   the scene extent (default extent/220, voxel budget 18 M); marching cubes (scikit-image).
4. Per-vertex support (number of agreeing views) → OBSERVED_MULTI_VIEW (≥ 2) or
   OBSERVED_SINGLE_VIEW; semantic labels by voting.
5. **Coverage** — a top-down map of what each camera saw (green), what was hidden behind nearer
   surfaces (orange) and what was outside every view.

## 6. Semantic 3D objects

Every object records: category, geometry, dimensions (value, σ, unit, *basis*: measured /
estimated / prior / lower bound), evidence class and confidence, source observations (view,
frame, pixel box), provenance (method and algorithm version), inferred parts, relations, flags.

* **Facades** — sequential RANSAC for vertical planes on building/wall points, coplanar merge,
  extent polygons from an occupancy grid of observed points (holes kept), corners from plane
  intersections (flagged as inferred when the corner itself was not observed), candidate
  openings (dark rectangular regions on the rectified texture — *candidates*, not classified
  windows or doors), inferred patches only where a gap is explained by an occluder seen in
  front of it in every view, never above the highest observed row, and an RGBA rectified texture
  (transparent where unobserved).
* **Poles** — from semantic pole masks and a geometric depth-strip detector (thin vertical
  structures standing out of their background; followed downwards along their two image edges
  where the depth no longer separates them from the ground behind): base on the ground (or from depth when the
  base is hidden — then flagged and the hidden part marked INFERRED), axis along gravity from
  the 3D points, height from the top row (a *lower bound* when the top leaves the frame),
  radius and taper from apparent widths and distance, merged across views (≥ 2 views required in
  multi-view units).
* **Vehicles** — detection → instance mask → 3D points → minimum-area footprint. Dimensions
  that were not observed (typically the far side) are completed with documented class priors
  (car 4.4 × 1.8 × 1.5 m, SD 0.35 / 0.1 / 0.1 …); if the observed footprint is implausible (depth
  noise at range) the position comes from the ground contact point and all dimensions from the
  prior. Evidence CLASS_PRIOR whenever a prior was used. Displayed as a generic template; the
  heading of a stationary vehicle is ambiguous by 180°.
* **Other static objects** — oriented boxes of the observed points (visible extent only).
* **Vegetation / unknown** — clusters of labelled surface vertices, reported as bounding boxes.
* **Ground** — the fitted plane with observed polygons (from the coverage map), inferred
  polygons (hidden ground adjacent to observed ground, assumed to continue at the same level),
  and semantic regions (road / sidewalk / terrain) with areas.

## 7. Trajectories and kinematics

* Ground position of each observation: ray through the bottom-centre of the box intersected
  with the ground plane (vehicles: shifted half a width away from the camera). Per-sample σ from
  a ± 2 % box-height uncertainty of the foot row propagated through the ray geometry.
* Stature from the head row along the vertical through the foot point (people standing).
* Smoothing (Savitzky–Golay) only *within* observed segments; segments are split at gaps longer
  than 1.5 s, which are never bridged.
* Speed by local linear regression over ± 0.6 s, with σ from the position noise and the scale σ;
  only reported when the scale is metric. Stationary subjects: total displacement below 3σ and
  low speed.
* Moving cameras: per-frame poses (keyframe / localised / interpolated) are used; observations
  in frames without a pose get no 3D position.

## 8. Multi-video synchronisation and alignment

* **Time** — audio onset envelopes (log-RMS derivative, 100 Hz) cross-correlated; confidence
  from the peak-to-second-peak ratio. Without audio, scene-activity series (frame differences)
  give weak evidence. Manual offsets override both. Container creation times are reported but
  never used (device clocks are unreliable).
* **Space** (strongest evidence first) — joint SfM; analyst point pairs (≥ 2; scale with ≥ 3);
  shared visual landmarks (SIFT matches between views of two units, lifted to 3D with each
  unit's aligned depth, 4-DOF RANSAC: yaw, ground translation, scale — accepted only with ≥ 15
  inliers forming at least 40 % of the matches); shared tracks (the same subject seen by both
  cameras at the same time; labelled hypothesis). Otherwise the unit is placed beside the
  reference and marked *unaligned*.
* **Conflicts** — subjects of the same class from two aligned cameras are matched over time
  (Hungarian on mean distance); pairs closer than 3 × tolerance but farther than the tolerance,
  or of different classes at the same place, are reported as conflicts.
* **Grouping** — videos are suggested as one scene only on strong evidence (shared
  reconstruction, landmark or analyst alignment, confident synchronisation).

## 9. Events

Machine-generated, neutral: first/last observation, entering/leaving through image borders,
stops (≥ 2 s below 0.3 m/s), proximity of two subjects (< 1.5 m for ≥ 0.4 s), crossing of
another subject's path. Each event has a time, position, the subjects involved, a confidence,
a source frame, and status *unreviewed* until an analyst verifies or rejects it. No event
asserts intent or interaction meaning.

## 10. Measurements

Point uncertainty = class-dependent fraction of the distance to the nearest camera
(multi-view 2 %, single view 6 %, fitted primitive 3 %, class prior 10 %, inferred 15 %, unknown
25 %). Lengths combine the endpoint σ in quadrature and add the scale σ × length. Angles
propagate the endpoint σ through the arm lengths. Track distances interpolate positions only
within observed segments. Every value is stored with its unit, status, σ, evidence of each
point, author and time, and recomputed when the calibration changes.

## Limitations

These are known, documented limitations of version 0.1.0:

* **Single cameras without people or building edges** remain RELATIVE: there is nothing in the
  image to fix the scale. Calibrate with a known length.
* **Monocular depth** is the weakest link for single-view surfaces; errors grow with distance
  and on reflective, transparent, textureless or very dark surfaces.
* **Rotating (pan/tilt) cameras**: the surface is reconstructed from one reference frame; other
  frames are oriented by homography registration. Zoom changes are not modelled: frames whose
  homography is not a pure rotation get no pose and their observations no 3D position.
* **Rolling shutter, strong lens distortion and zoom changes** during a clip are not modelled
  (one SIMPLE_RADIAL camera per video).
* **Poles** are missed when neither the semantic model nor the geometric detector isolates them
  (thin poles at long range, against similar-depth backgrounds); heights are often lower bounds.
* **Vehicles** are boxes with class-prior completion, not shapes; parked-vehicle headings are
  ambiguous by 180°.
* **People** are represented by generic body proxies of estimated stature; no articulated pose
  is estimated in this version.
* **Dense semantics** are trained on European street scenes; indoor scenes get no dense labels.
* **Face and OCR outputs** are low-resolution candidates in typical surveillance footage.
* **Platforms** — tested on Linux x86-64 / CPU only for this release.

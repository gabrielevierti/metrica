# Changelog

## 0.1.0 — 2026-10-11

First release of METRICA, a from-scratch successor of Sopralluogo.

* Evidence intake with verified read-only copies, hash-chained audit log, manifest.
* Exact-timestamp ingest (VFR, rotation, damaged packets), quality metrics, camera-motion
  classification, adaptive keyframes, review proxies.
* YOLOX detection, ByteTrack-style tracking with recorded gaps/flags, fragment re-linking as
  hypotheses, instance masks, face detection (no recognition), vehicle colour and OCR candidates.
* Static cameras: single-view metrology with focal length from orthogonal vanishing points and
  people as a ruler; rotating cameras: per-frame orientation by homography registration; moving
  cameras: COLMAP structure from motion, localisation, gravity alignment.
* Dense surfaces (single-view and TSDF multi-view fusion) with per-vertex evidence classes;
  semantic objects (facades, poles, vehicles, objects, ground) with provenance and uncertainty.
* 3D trajectories, kinematics, neutral event timeline, measurements with propagated uncertainty,
  analyst scale calibration.
* Multi-video synchronisation, alignment, conflicts and grouping suggestions.
* React/three.js viewer; HTML/PDF reports with CSV/JSON exports.
* Synthetic ground-truth scenes, evaluation tools, unit and integration tests.

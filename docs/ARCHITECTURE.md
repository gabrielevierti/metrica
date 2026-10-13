# Architecture

METRICA is one Python package (`backend/metrica`, ~12 k lines) that contains the processing
pipeline, the local HTTP API and the command line, plus a TypeScript web application
(`frontend/`, React + three.js, ~3 k lines) that is served by the same process.

```
            browser (127.0.0.1)                       command line
                  │  HTTP/JSON, binary layers              │
                  ▼                                        ▼
        ┌──────────────────────┐                  ┌────────────────┐
        │ api/app.py (FastAPI) │                  │ cli.py         │
        │  cases, uploads,     │                  │ analyze/report │
        │  scene, measurements │                  │ verify/models  │
        │  annotations, reports│                  └───────┬────────┘
        └─────────┬────────────┘                          │
                  │ api/jobs.py: SQLite job queue         │ in-process
                  ▼ (one worker subprocess at a time)     ▼
        ┌────────────────────────────────────────────────────────────┐
        │ pipeline/runner.py — staged, cached, resumable, cancellable │
        │ ingest → tracking → depth → geometry → reconstruction →     │
        │ objects → trajectories → alignment → analysis → scene →     │
        │ report                                                      │
        └────┬───────────┬───────────┬───────────┬──────────┬────────┘
             ▼           ▼           ▼           ▼          ▼
         ingest/     models/     geometry/  reconstruction/ tracking/ multiview/
         decode,     ONNX        camera,    fusion, meshes, tracker,  sync,
         quality,    adapters,   single     facades, poles, linking,  alignment
         motion,     registry    view, SfM, objects, ground trajectories
         proxy                   gravity                               analysis/
                                                                       reporting/
                          ▼
        ┌─────────────────────────────────────────────┐
        │ evidence/: Case folder, hashing, audit log, │
        │ manifest (integrity of inputs and outputs)  │
        └─────────────────────────────────────────────┘
```

The repository layout groups the Python modules in one installable package rather than in
top-level `reconstruction/`, `models/`, `tracking/`, `reporting/` folders, so that `pip install`
gives a single importable `metrica` package with consistent relative imports; those concerns
are the subpackages of the same names.

```
metrica/
├── backend/metrica/       Python package (pipeline, API, CLI)
│   ├── evidence/          case folders, SHA-256, hash-chained audit log, manifest
│   ├── ingest/            PyAV decoding, probing, quality, motion analysis, proxies
│   ├── models/            ONNX Runtime adapters + model registry (download, SHA-256 verification)
│   ├── semantics/         category vocabulary, COCO / Cityscapes mappings
│   ├── geometry/          camera model, single-view metrology, SfM (pycolmap), gravity, calibration records
│   ├── reconstruction/    layer format, depth alignment, TSDF fusion, meshing, primitives, objects
│   ├── tracking/          Kalman, tracker, fragment linking, 3D trajectories and kinematics
│   ├── multiview/         audio / activity synchronisation, alignment, conflicts
│   ├── analysis/          vehicles (colour, OCR candidates), timeline events, measurements
│   ├── scene/             unit → world transforms, GLB / PLY export
│   ├── reporting/         report data, figures, HTML (Jinja2), PDF (ReportLab), exports
│   ├── pipeline/          stage framework, runner, stages, worker process
│   ├── api/               FastAPI application, job store
│   ├── synthetic/         procedural ray-traced test scenes with ground truth (CC0)
│   └── evaluation/        accuracy evaluation against synthetic ground truth
├── frontend/              React + TypeScript + three.js application (prebuilt in dist/)
├── tests/                 unit and integration tests (pytest)
├── docs/                  this documentation
└── scripts/               installers, demo, release packaging
```

## Data model

### Case folder (`metrica-case/1`)

```
<case>/
  case.json                    identity, title, reference, inventory of inputs (intake hashes)
  evidence/originals/          byte-exact, read-only copies of imported files
  derived/                     everything METRICA computes; can be deleted and recomputed
    config.json                resolved parameters of the last analysis
    status.json                live progress
    stages/<stage>.json        stage records (status, fingerprint, warnings, fallbacks, metrics, outputs)
    videos/<vid>/              ingest.json, frame times, keyframes, proxy, detections, tracks, crops, depth
    geometry/                  per-video calibration/poses, units.json, SfM workspace
    reconstruction/<unit>/     binary layers, coverage, keyframe depth
    objects/<unit>/            objects.json + textures
    trajectories/, alignment/, analysis/
    scene/                     scene.json (metrica-scene/1), layers/, scene.glb, *.ply
    reports/<id>/              report.html, report.pdf, exports/, assets/, zip, report.json
    frames/                    cached exact source frames (PNG) requested in the viewer
  annotations/                 analyst input: measurements, calibration, sync, alignment,
                               verifications, notes, snapshots
  audit/audit.jsonl            hash-chained log of analyst actions and system events
  logs/pipeline.jsonl          structured processing log (+ worker logs)
  manifest.json                inputs and outputs with SHA-256, software and model versions, parameters
```

Evidence, derived data and annotations are separated so that (1) originals are never touched,
(2) all machine output can be regenerated from the originals plus the recorded parameters,
and (3) analyst input is never overwritten by a re-analysis.

### Reconstruction units

A *unit* is a set of videos that share one reconstruction frame: a static camera on its own,
or the moving videos that structure from motion registered together. Every unit is
gravity-aligned (Y up, ground plane y = 0) and carries a scale record. The alignment stage
places the units into the case world frame (the reference video's unit) with similarity
transforms; analyst scale references add a global factor `k`. Unit-frame data on disk is
never rewritten by alignment or calibration; the scene stage applies the transforms when it
assembles the scene graph, so recalibrating never compounds earlier corrections.

### Scene graph (`metrica-scene/1`)

`derived/scene/scene.json` is the single document the viewer and the reports read:

| Key | Contents |
|---|---|
| `coordinate_system` | unit, calibration status and its explanation, scale record (method, σ, assumptions, references), global factor |
| `time` | reference video, per-video offsets, case time range |
| `videos` | hash, proxy, thumbnail, fps, duration, resolution, capture mode, unit, sync record, quality flags |
| `cameras` | intrinsics, static pose or per-frame poses (with source: keyframe / localized / interpolated) |
| `layers` | binary meshes/points (positions, colours, evidence, confidence, labels) and the coverage map |
| `entities` | semantic objects: geometry (plane polygon, cylinder, vehicle box, box, bbox, ground plane), dimensions with basis and σ, evidence, provenance, source observations, inferred parts, relations |
| `agents` | tracked subjects: observed samples (time, position, σ, confidence, flags, source box), smoothed path, speeds with σ, gaps, flags, links, crops, faces, colour and OCR candidates, body proxy |
| `events` | machine-generated timeline |
| `alignment`, `conflicts`, `grouping` | multi-video results |
| `warnings` | everything the pipeline wants an analyst to know |

Binary layers are little-endian flat arrays (`.pos.f32`, `.col.u8`, `.idx.u32`, `.evi.u8`,
`.conf.u8`, `.lab.u8`) that the browser loads without parsing. The same scene is exported as
glTF binary (`scene.glb`) and PLY for use in other tools.

## Pipeline

Each stage is a class with a name, its dependencies, the configuration keys it reads, a
version and the annotation kinds it depends on. Its **fingerprint** is a hash of those keys'
values, its version and its upstream stages' fingerprints. The runner:

* reuses a stage whose record is valid for the current fingerprint and whose outputs exist
  (resume after interruption, and incremental recomputation after a parameter change);
* records `ok`, `degraded` (a documented fallback was used), `failed` (with the traceback),
  `skipped` or `cancelled`; a failed stage fails only the stages that depend on it, except
  ingest, which stops the run;
* checks for cancellation between work units and writes `status.json` for the progress UI;
* writes the manifest at the end, whatever the outcome, and logs the outcome in the audit log.

| Stage | Main inputs | Main outputs |
|---|---|---|
| ingest | original files | probe, exact frame times, quality samples, motion classification, keyframes, background plate, proxy |
| tracking | frames at the analysis rate | detections, tracks with gaps/flags/links, crops, faces, masks |
| depth | keyframes / reference images | relative inverse depth, semantic label maps |
| geometry | keyframes, people observations, depth | calibrated cameras, poses, scale records, units |
| reconstruction | depth, poses, semantics | surface and point layers with evidence and confidence, coverage |
| objects | views (images, aligned depth, semantics, cameras) | facades, poles, vehicles, objects, clusters, ground |
| trajectories | tracks, cameras | 3D trajectories, kinematics, stationary objects |
| alignment | units, audio, activity, tracks | offsets, unit transforms, conflicts, grouping |
| analysis | trajectories, crops | vehicle colours, OCR candidates, faces summary, events |
| scene | all of the above + annotations | scene graph, GLB, PLY |
| report | scene, annotations | the first machine report (users generate more from the app) |

### Job system

The API never runs analysis in its own process. `api/jobs.py` keeps a SQLite queue in the
workspace; one worker subprocess (`python -m metrica.pipeline.worker`) runs at a time, polls
for cancellation, and records its final state. Jobs whose worker died are marked failed on the
next check; jobs that were running when the server stopped are marked interrupted at start-up,
and the next analysis resumes from the last completed stage.

### Models

`models/registry.py` pins every model's URL, SHA-256, licence and purpose. Weights are never
bundled; they are downloaded on request, verified, and cached (with a size/mtime-keyed
verification cache). All inference goes through ONNX Runtime (`models/runtime.py`), which
selects CUDA, CoreML or DirectML when available and falls back to the CPU, recording the
provider. Adapters (`depth.py`, `detection.py`, `segmentation.py`, `faces.py`, `ocr.py`,
`reid.py`) hide model-specific pre/post-processing. Missing optional models trigger
documented fallbacks (for example GrabCut instead of EfficientSAM, geometry-only labels
instead of dense semantics).

### API

FastAPI on `127.0.0.1:8765`. Resources: system/hardware, models (status, download), cases
(create, list, detail), videos (upload, import by path, probe), analysis jobs (start, progress,
cancel), scene and derived files (restricted to `derived/`), exact source frames, measurements,
calibration references, sync offsets, alignment point pairs, verifications, notes, snapshots,
reports, audit log, evidence verification, manifest. Every mutating request records the
analyst (header `X-Metrica-Analyst`) in the audit log.

### Frontend

React 19 + TypeScript (strict) + Vite; three.js through @react-three/fiber and drei; state in
zustand. One shared case clock drives the 3D scene, the video dock and the timeline. Large
meshes use BVH-accelerated picking. The frontend is a client of the documented API only.

## Reliability and resource use

* Videos are decoded in a single streaming pass per stage, never fully loaded into memory.
* Analysis runs at a configurable frame rate; dense steps run on keyframes.
* TSDF fusion uses a narrow band and a voxel budget; the voxel size adapts to the scene extent.
* Expensive models have per-video budgets (masks, faces, semantic frames).
* Every fallback is recorded in the stage record, surfaced in the UI and in the report.

## Appendix: assessment of Sopralluogo

Sopralluogo (github.com/gabrielevierti/sopralluogo, inspected read-only at commit `a6646b5`,
the only commit) was the starting point. Summary of the inspection:

| Area | Sopralluogo | Consequence for METRICA |
|---|---|---|
| Scope | Fixed cameras: `geometry.py` models "a fixed camera looking at a ground plane"; moving footage is out of scope | Added SfM, localisation, multi-view fusion; single view kept for static cameras |
| Calibration | People-as-a-ruler + vertical lines (`calibrate.py`), good idea | Kept and extended: focal identifiability profile, orthogonal-VP focal estimation, explicit statuses and σ |
| Geometry | Per-pixel depth mesh + piecewise planes (`planar.py`) + a content-aware *fill* layer (`fill.py`, honestly labelled as invented) | Replaced by evidence-classified surfaces and parametric objects; inferred parts are explicit, hatched, and never filled with invented texture |
| Models | Depth Anything V2 and **YOLO11 (AGPL-3.0)** | Default detector YOLOX (Apache-2.0); YOLO11 opt-in with explicit licence acceptance |
| Video | FFmpeg subprocesses and frame dumps | PyAV with exact timestamps, VFR, rotation and damaged-packet handling, no system FFmpeg |
| Pipeline | Single `process.py` run | Staged runner with fingerprints, resume, cancellation, job queue, failure records |
| Integrity | Manifest of output hashes; the README correctly warns that estimates are not forensic measurements | Kept the spirit; added intake verification, read-only originals, hash-chained audit log, analyst attribution |
| Viewer | React (JavaScript) viewer with timeline and measurement | Rewritten in TypeScript with evidence colouring, object inspection, camera views, calibration and verification workflow |
| Tests | Three test files (integrity, planar, geometry) | Unit + integration suites, synthetic ground-truth evaluation |
| Repository contents | Commits a sample case including footage (`caso-piazza/media/cam1.mp4`) and `.DS_Store` files; Italian documentation | No footage in the repository (synthetic test data is generated on the fly); build output is produced by the release script, not committed; English throughout |

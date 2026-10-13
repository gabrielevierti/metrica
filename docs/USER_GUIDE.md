# METRICA user guide

This guide follows the normal workflow: create a case, import footage, analyse, explore,
measure and calibrate, review and verify, report. Everything here is available in the web
application (`metrica serve`); the command-line equivalents are listed at the end.

## Core ideas

**Case.** A folder that holds one investigation: the original files (copied read-only and
hashed), everything METRICA derives from them, your annotations, an audit log and a manifest.
Cases live in the workspace (`~/METRICA` by default).

**Calibration status.** Every reconstruction has one of four statuses, shown in the top bar
and next to every number:

| Status | Meaning | Unit |
|---|---|---|
| `ARBITRARY` | no gravity, no scale: only ratios inside the scene mean anything | su |
| `RELATIVE` | gravity-aligned and internally consistent, but the scale rests on an unverified assumption (for example an assumed camera height). **Not real-world distances.** | su (scene units) |
| `ESTIMATED_METRIC` | metric scale estimated from statistical evidence (the statures of many observed people) with a quantified uncertainty | m ± σ |
| `CALIBRATED_METRIC` | metric scale from reference lengths you supplied | m ± σ |

**Evidence class.** Every surface vertex and every object says how it is known:

| Class | Colour | Meaning |
|---|---|---|
| Observed (multi-view) | green | seen consistently from two or more viewpoints |
| Observed (single view) | blue | seen from one viewpoint; depth from a single image, aligned to geometric evidence |
| Fitted primitive | amber | a plane, cylinder or box fitted to observed points; the observed extent is kept separately |
| Class prior | violet | dimensions from typical values for the object class (e.g. a car), positioned by observations — not a scan |
| Inferred | red, hatched | a plausible completion of something not observed (ground behind a parked car, the hidden base of a pole) |
| Unknown | grey | no evidence |

Inferred geometry is drawn hatched in red, can be hidden with one click, and is never used as
if it had been observed.

**Machine-generated vs analyst-verified.** Everything METRICA produces is machine-generated
until an analyst reviews it. Verifications are recorded with your name in the audit log and
reported separately.

## 1. Set your name

Type your name or analyst ID in the **Analyst** field at the top right of the home screen.
It is attached to every action you take (imports, measurements, calibrations, notes,
verifications, reports).

## 2. Create a case and import footage

On the home screen, give the case a title (and optionally a reference number), then drop one or
more video files on the drop zone, or click it to choose files, and press **Create case and
import**. For large files on the same computer use **Large files: import by local path**: the
file is hashed and copied without going through the browser.

Each file is hashed (SHA-256) before copying, copied into `evidence/originals/`, re-hashed and
compared, and made read-only. Importing the same content twice is refused. Supported: anything
FFmpeg decodes (MP4, MOV, MKV, AVI, MTS, WebM, ...), including variable frame rate, rotated
phone footage and files with damaged packets.

In the case view the **Inputs** tab shows each video's hash, format, timing (frame rate, VFR
detection, gaps), the detected capture mode and decode problems.

## 3. Analyse

Press **Analyze…**. The dialog shows each video and the detected capture mode; you can force
*static camera*, *panning / tilting camera* or *moving camera* for a video. Choose a profile:

| Profile | Use it for |
|---|---|
| fast | a quick look: fewer frames and keyframes, no dense semantics or instance masks |
| standard | the default, balanced |
| quality | denser sampling, larger detector, more masks; several times slower |

**Advanced settings** expose the parameters that matter most: analysis frame rate, keyframe
budget, a known horizontal field of view (if you know the lens, enter it: it removes the
largest source of uncertainty for single cameras), the assumed adult stature, the detector,
and switches for semantics, masks, faces, OCR and inferred surfaces.

The progress panel shows every stage. Each stage stores its results: if you cancel, close the
application or the computer restarts, the next analysis resumes from the last completed
stage, and changing a parameter only recomputes the stages it affects. Stages that fall back
to a weaker method are marked *degraded*, failed stages are recorded with their error, and
stages that do not depend on a failed one still run.

For orientation: on the 2-core, CPU-only test machine an 80-second 768×576 clip took about
5 minutes in the standard profile, most of it detection and tracking. A GPU reduces the neural
stages substantially. Videos are analysed at the analysis frame rate (default 8 fps), not at
every frame.

## 4. Explore the scene

The case view has five areas: the left panel (Inputs, Layers, Objects, Tracks, Events,
Measure, Case), the 3D viewport, the inspector on the right, the video dock and the timeline.

### Navigation

| Action | Orbit mode (default) | Walk mode |
|---|---|---|
| rotate / look | left-drag | drag |
| pan | right-drag or Shift+drag | — |
| zoom / move | mouse wheel | W/A/S/D, Q/E down/up, Shift faster |

Toolbar buttons and keys: **overview** (`O`), **top** view (`T`), **view from a camera** (one
button per video; the 3D view takes that camera's position, orientation and field of view, so
you can compare it with the footage), focus on the selection (`F`), photo / evidence /
semantic / confidence colouring (`1`–`4`), wireframe, snapshot (camera icon).

### Layers and filters

**Layers**: the reconstructed surface (mesh), the point cloud, and the coverage map (what each
camera saw — green — and what was hidden behind something — orange). Filter by evidence class
(e.g. hide everything inferred) and by object category. Toggles for objects, subjects,
trajectories, cameras, grid, inferred parts, uncertainty rings, openings and labels.

### Objects

The **Objects** tab lists the semantic objects. Selecting one shows in the inspector its
evidence class, confidence, the views it was seen in, each dimension with its basis
(*measured*, *estimated*, *prior*, *lower bound*) and uncertainty, how it was reconstructed,
its relations (e.g. "meets facade 2 at a corner", "stands on the ground"), its source
observations (each with a **frame** button that opens the exact source frame), and for facades
the rectified observed texture (transparent where nothing was observed).

### Subjects and the timeline

The **Tracks** tab lists every tracked person, vehicle or other subject with its observed
interval, number of observations, gaps and flags (*re-linked* fragments, *close interaction*
where identities could swap, *appearance change*). Selecting a track shows its best crops,
detected faces (with size, sharpness, pose class — detection only, no recognition), vehicle
colour and OCR candidates, observed path length, speed (only when the calibration supports it)
and the stature estimate.

Press **Space** (or the play button) to play: subjects move in 3D in sync with the video dock,
which shows the detector boxes on the review proxy. Arrow keys step 0.1 s (Shift: 1 s). In
the 3D view a subject is drawn solid when observed at that time and faded between nearby
observations; inside a gap it is not drawn at all. Trajectories are solid where observed and
dashed across gaps.

The **Events** tab lists the machine-generated timeline (first/last observation, stops,
proximity between subjects, crossings of other subjects' paths). Events are neutral
descriptions of geometry and time with a confidence; they never assert intent.

> The video dock plays an H.264 review proxy. Some browsers (for example Chromium builds
> without proprietary codecs) cannot play H.264; the exact evidence frames are always
> available through the **frame** buttons, which decode the original file.

## 5. Measure

Open the **Measure** tab or use the toolbar:

| Tool | Key | Clicks | Result |
|---|---|---|---|
| distance | `M` | 2 | 3D distance; the hint bar lets you switch to horizontal distance or height difference |
| height | `H` | 1 | height above the ground plane |
| angle | | 3 | angle at the middle point |
| inspect | | 1 | coordinates, evidence class and uncertainty of a point (not saved) |
| calibrate | `K` | 2 | define a known real length (see below) |

Click on the surface or on objects. Each result shows value ± σ, unit and calibration status;
in RELATIVE/ARBITRARY scenes it is explicitly labelled as not a real-world measurement. The
uncertainty combines the point uncertainty (which depends on the evidence class of the surface
you clicked and on its distance from the camera) and the scale uncertainty. Saved
measurements are listed with their author and time, drawn in the scene, recomputed
automatically when the calibration changes, and can be withdrawn (the withdrawal is logged;
nothing is deleted).

### Calibrating with a known distance

If you know a real length visible in the scene (a measured parking bay, a door width, the
distance between two kerb marks), choose **calibrate**, click its two ends, and enter the true
length and its uncertainty. The reference is stored as an analyst annotation, written to the
audit log, and the scene is recomputed in calibrated metres (status `CALIBRATED_METRIC`).
Several references are combined, weighted by their uncertainty, and their mutual consistency
is included in σ. References can be deactivated again from the Measure tab.

This is the single most effective way to improve accuracy. A reference near the area you want
to measure, and as long as possible, works best.

## 6. Several videos

Videos of the same event are synchronised automatically when they share audio events (onset
cross-correlation) or, more weakly, changes in scene activity. Offsets can be set by hand for
each video in the **Inputs** tab. Moving cameras that see the same structure are reconstructed jointly; other
cameras are aligned through shared visual landmarks, through point pairs you provide, or —
as a labelled hypothesis — through subjects seen by both cameras. Units that cannot be aligned
are placed beside the reference and clearly marked as not comparable. Where two cameras place
the same subject at different positions, a conflict is reported. The **Case** tab and the report
show which videos METRICA suggests belong together and on what evidence (grouping uses only
geometric or synchronisation evidence; container timestamps and colour similarity are shown as
context but never used to group).

## 7. Review, verify and annotate

For any object, track or event, use **Verified**, **Uncertain** or **Rejected** in the
inspector, with a note on the basis of your decision. Free-text notes are added in the **Case**
tab; a note is attached to the current selection, if any. Snapshots of the 3D view (camera icon) are stored in the case with a caption and
appear in the report.

The **Case** tab shows the pipeline stages with their status, warnings and fallbacks, the
integrity check of the evidence files and of the audit chain, and the audit log.

## 8. Report

**Report…** generates a new report each time (earlier ones are kept): HTML and PDF with the
sections listed in the README, plus CSV/JSON exports (objects, observed trajectories, speeds,
events, face observations, measurements, the scene graph and the manifest) and a ZIP with all
of it. The SHA-256 of every output is recorded and the generation is written to the audit log.

## Command line

```bash
metrica serve [--port 8765] [--workspace DIR] [--no-browser]
metrica analyze VIDEO... --case DIR [--title T] [--reference R] [--profile fast|standard|quality]
                [--config '{"hfov_deg": 72}'] [--force] [--no-copy]
metrica report CASE [--analyst NAME] [--purpose TEXT] [--no-pdf]
metrica verify CASE                 # re-hash evidence, check the audit chain and the manifest
metrica models list|download|verify [KEYS] [--all] [--accept-restricted]
metrica doctor                      # environment diagnostics
metrica demo [--kind walk|static] [--out DIR]   # synthetic end-to-end run with evaluation
metrica evaluate CASE GROUND_TRUTH.json
```

`--no-copy` references originals in place instead of copying them (they are re-verified every
time processing starts). Use it only when the originals are stored on write-protected media.

## Troubleshooting

| Symptom | Likely cause and remedy |
|---|---|
| Scale stays `RELATIVE` for a fixed camera | Too few whole, standing people were visible to estimate the camera height. Calibrate with a known length. |
| `ESTIMATED_METRIC` with a large σ (20 % or more) | The lens field of view could not be determined from the footage. Enter the known field of view (Advanced settings) or calibrate. |
| Moving video reconstructed as single view | Not enough parallax or texture for structure from motion (the stage records why). Shoot while walking sideways rather than panning on the spot. |
| Panning camera: subjects missing in some frames | Those frames could not be oriented (no overlap with the reference, zoom change); the warning lists how many. |
| Black video in the dock | The browser cannot play H.264; use the frame buttons or another browser. |
| Very slow analysis | Use the fast profile for a first look, enable GPU acceleration, or reduce the analysis frame rate. |

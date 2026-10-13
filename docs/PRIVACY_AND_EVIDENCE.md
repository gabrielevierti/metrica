# Evidence handling, integrity and privacy

METRICA is built for footage that may become evidence and that shows real people. This
document states what it guarantees, what it records, what it refuses to do, and what remains
the analyst's responsibility.

## Local-first

* All processing happens on the computer running METRICA. The server listens on `127.0.0.1`
  only. Footage, frames, face crops, reports and case data are never uploaded anywhere.
* The only network access is the explicit, user-initiated download of model weights from their
  original publishers (GitHub releases, OpenCV and ONNX model zoos), each verified against a
  pinned SHA-256. After that METRICA runs fully offline.
* There is no telemetry, no update check, no cloud service and no account.

## Originals are never modified

1. On import, the file is hashed (SHA-256) **before** anything else happens.
2. It is copied into `evidence/originals/`, the copy is hashed again and compared; a mismatch
   aborts the import.
3. The copy is made read-only. All processing reads the copy; nothing writes to it.
4. The intake hash, size, original name and path, file modification time, analyst and time are
   recorded in `case.json` and in the audit log.
5. `metrica verify` (or *Re-verify evidence hashes* in the app) re-hashes every original and
   compares it with the intake hash.

With `--no-copy` the original is referenced in place (for example on write-protected media);
it is re-verified every time processing starts and the run stops if it changed.

Importing the same content twice is refused. Derived data (frames, proxies, crops) are separate
files under `derived/`; the review proxy is clearly a transcoded copy for playback, while the
frame viewer decodes the original.

## What a hash proves — and what it does not

> SHA-256 values prove that a file's contents are bit-for-bit identical to the content that was
> hashed. They do not establish how, when, by whom or with which device a recording was made,
> whether it was edited before intake, or the chain of custody before intake: a matching hash is
> file-integrity verification, not proof of authenticity. Those questions require separate
> provenance documentation.

This statement is printed in every manifest and report. METRICA performs no authenticity,
tampering or deep-fake analysis and makes no claim about it. Container metadata (creation
times, encoder tags) is shown as reported by the file and is never treated as verified.

## Audit trail

`audit/audit.jsonl` records every analyst action and system event (case creation, imports,
analyses started and finished with their outcome and manifest hash, measurements, calibration
references, sync offsets, alignment points, verifications, notes, snapshots, reports, evidence
verifications, cancellations) with the analyst name, time, target and details. Each entry
contains the SHA-256 of the previous entry, so editing, deleting or reordering entries breaks
the chain, which `metrica verify` and the app detect.

This makes the log tamper-*evident*, not tamper-*proof*: someone with write access could
rewrite the whole chain. The head hash of the chain is printed in reports; recording it in an
external system (a case management system, a signed e-mail) closes that gap.

## Manifest

`manifest.json` lists the inputs (intake hashes and their verification at manifest time), every
derived and annotation file with its SHA-256, the software environment (METRICA, Python, OS and
library versions), the models used (with their SHA-256 and licences), all parameters, the stage
records (including failed and degraded stages and every fallback) and the audit chain state.
Each report adds the SHA-256 of all of its files.

## Machine findings vs analyst findings

Everything the pipeline produces is labelled machine-generated. Analysts can mark objects,
tracks and events as verified, uncertain or rejected, with a note; these decisions are stored
separately in `annotations/`, logged, and reported in their own section. Re-running the analysis
never overwrites annotations.

## No identification, no profiling

METRICA deliberately does **not**:

* identify people, compare faces with each other or with any database, or compute face
  embeddings. Face *detection* only locates faces and reports their size, sharpness and pose so
  that an analyst can find the best frames for a lawful identification procedure performed
  elsewhere, by people authorised to do it;
* infer age, gender, ethnicity, emotion, health, religion, sexual orientation, political
  opinion, personality, intent, criminality or any other sensitive attribute from appearance;
* infer who owns or drives a vehicle. Vehicle colour and OCR results are *candidates* with
  their support (number of frames, confidence) — a misread plate character is common in
  surveillance footage;
* describe events in terms of intent. Timeline events state geometry and time (first seen,
  stopped, came within 1.5 m of, crossed the path of), with a confidence.

Person re-identification embeddings (optional Youtu ReID model) are used only to decide whether
two track fragments *of the same video* belong together, are kept in memory only during that
step, and are never stored or compared with anything else.

## Personal data in case folders

Case folders contain footage of people, face crops and derived positions — personal data under
most data-protection laws (for example the GDPR and, for law enforcement, Directive (EU)
2016/680). The organisation using METRICA is the controller and is responsible for the lawful
basis, purpose limitation, retention, access control and deletion. Practical measures:

* keep the workspace on an encrypted disk with access restricted to authorised staff;
* do not bind the server to a network interface (`--host`) on shared networks: the API has no
  authentication;
* disable face detection (Advanced settings → `use_faces`) when it is not needed for the
  purpose of the analysis;
* delete case folders when the retention period ends (METRICA keeps nothing elsewhere; model
  files contain no case data);
* do not commit case folders, footage or reports to source control (the repository's
  `.gitignore` excludes the default locations).

## Inferred geometry

Hidden geometry is never presented as observed. Inferred surfaces (ground behind a parked car,
the hidden base of a pole, a wall behind an occluder, a facade corner beyond the observed
extent) have the evidence class INFERRED, are drawn hatched in red, can be hidden, carry the
reason they were inferred, and have the largest point uncertainty in measurements. Class priors
(typical vehicle dimensions) are labelled CLASS_PRIOR and the dimension's basis is "prior".
Missing observations are never interpolated into detections: track gaps are recorded and drawn
dashed, and trajectory smoothing never crosses a gap.

## Accuracy claims

No accuracy figure is claimed unless it was measured with a documented procedure
([EVALUATION.md](EVALUATION.md)). Results on synthetic scenes and on one public clip are not
guarantees for other footage. Every measurement carries its own uncertainty and calibration
status; the safest practice is to calibrate with at least one surveyed distance near the area of
interest and to verify a second, independent distance.

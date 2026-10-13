# Contributing to METRICA

Thank you for helping. METRICA is evidence-oriented software: a change that makes a result look
better without making it more correct, or that hides an uncertainty, is a regression.

## Ground rules

1. **No fabricated evidence.** Never present inferred, interpolated or prior-based geometry or
   observations as observed. New outputs must carry an evidence class, a basis and an uncertainty.
2. **No identification or profiling features.** Face recognition, face search, demographic or
   sensitive-attribute inference, intent or criminality scoring, and vehicle-ownership inference
   are out of scope and will not be merged.
3. **Local-first.** No telemetry, no network calls except explicit model downloads.
4. **Accuracy claims need a procedure.** Any number in the documentation must be reproducible
   with the evaluation tools and recorded in `docs/EVALUATION.md` with the conditions.
5. **No real footage, case data, secrets or model weights in the repository.** Test data is
   generated (`metrica.synthetic`).
6. **Licences.** New dependencies and models must have licences compatible with Apache-2.0
   distribution (or be strictly opt-in, like YOLO11). Record them in `docs/MODEL_LICENSES.md`.

## Development setup

```bash
./scripts/install.sh --dev        # or: python -m venv .venv && pip install -e ".[dev]"
source .venv/bin/activate
metrica models download
cd frontend && npm ci && npm run dev    # frontend dev server on :5173 (proxies /api to :8765)
metrica serve --no-browser              # backend on :8765
```

## Tests

```bash
pytest                       # unit + fast integration tests (~30 s)
pytest -m slow               # also SfM on rendered frames and the full pipeline on synthetic footage
ruff check backend tests     # lint
cd frontend && npm run build # strict TypeScript check + production build
```

Tests must not need network access or real footage. Tests that need the downloaded models are
marked `models` and skip themselves when the models are missing.

## Code structure and conventions

* Python 3.11+, type hints, `from __future__ import annotations`; NumPy for geometry; no PyTorch
  at runtime (all inference via ONNX Runtime).
* Coordinate conventions: world Y up, ground y = 0; OpenCV camera frame; `R_wc`, `C`.
* A pipeline stage declares `depends`, `config_keys`, `annotation_keys` and a `version`. **Bump the
  version whenever its outputs change**, otherwise cached results are reused.
* Every fallback goes through `StageResult.fallback(component, used, implication)` so that it
  reaches the UI, the manifest and the report.
* Analyst actions go through `Case.save_annotations(..., actor, action)` so they are audited.
* Frontend: TypeScript strict mode; the frontend talks only to the documented API.

## Pull requests

Describe what changed, why, how it was tested (with numbers for anything that affects
geometry), and update the documentation. Keep pull requests focused.

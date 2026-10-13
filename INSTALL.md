# Installing METRICA

METRICA is a Python package (backend, pipeline, CLI) with a prebuilt web frontend. It runs
entirely on your computer; after the one-time model download no network access is needed.

## Requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Linux x86-64, macOS 12+ (Apple Silicon or Intel), Windows 10/11 x64 | |
| Python | 3.11, 3.12 or 3.13 (64-bit) | 3.12 |
| RAM | 8 GB | 16 GB or more for long or high-resolution footage |
| Disk | ~1.5 GB for the environment and models, plus ~2-4x the size of the footage per case | SSD |
| GPU | not required | NVIDIA (CUDA 12), Apple Silicon (CoreML) or DirectML on Windows |
| Browser | a current Chrome, Edge, Firefox or Safari with WebGL 2 | |
| Node.js | only to rebuild the frontend from source (20+) | |

No system FFmpeg is needed: video decoding and encoding use the FFmpeg libraries bundled in the
PyAV wheels. COLMAP is used through the `pycolmap` wheels; no separate installation is needed.

> **Tested configuration for this release:** Linux x86-64, Python 3.12, CPU only. The other
> platforms are supported by the dependencies' official wheels but were not tested by the
> author of this release. Please report problems.

## 1. Install

### macOS and Linux

```bash
cd metrica
./scripts/install.sh              # creates .venv, installs METRICA and its dependencies
source .venv/bin/activate
metrica doctor                    # checks the environment
```

### Windows (PowerShell)

```powershell
cd metrica
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
.\.venv\Scripts\Activate.ps1
metrica doctor
```

### Manual installation (any platform)

```bash
python3.12 -m venv .venv
. .venv/bin/activate                       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .                 # add ".[dev]" for tests
```

Exact versions of the configuration that was tested are pinned in
`requirements-lock.txt` (`pip install -r requirements-lock.txt` followed by `pip install -e . --no-deps`
reproduces it on Linux / Python 3.12).

## 2. Download the models

Model weights are not shipped with METRICA (see [docs/MODEL_LICENSES.md](docs/MODEL_LICENSES.md)).
Download them once from their original publishers; each file is verified against a pinned
SHA-256 before it is used:

```bash
metrica models download            # default set, ~280 MB
metrica models download --all      # also the optional models (YOLOX-M, Youtu ReID), ~480 MB total
metrica models list                # status of every model
metrica models verify              # re-hash the files on disk
```

The application's home screen has the same buttons. Models are stored in
`~/.cache/metrica/models` (override with the `METRICA_MODELS` environment variable).

The Ultralytics **YOLO11n** detector is AGPL-3.0 licensed. It is never downloaded unless you
explicitly accept its licence:

```bash
metrica models download yolo11n --accept-restricted
```

### Offline / air-gapped machines

Download the model files on a connected machine (`metrica models download`), copy the models
folder to the offline machine and point `METRICA_MODELS` at it; `metrica models verify` checks
the copies. Install the Python wheels with `pip download -r requirements-lock.txt -d wheels` on
the connected machine and `pip install --no-index --find-links wheels -r requirements-lock.txt`
offline.

## 3. Run

```bash
metrica serve                       # http://127.0.0.1:8765, opens your browser
metrica serve --no-browser --port 9000 --workspace D:\Cases
```

Cases are stored in `~/METRICA` by default (`--workspace` or `METRICA_HOME` to change it). The
server listens on `127.0.0.1` only. Binding to another interface (`--host 0.0.0.0`) exposes the
case data to your network without authentication; do not do this on untrusted networks.

Set your analyst name in the top-right field of the home screen (or `METRICA_ANALYST`); it is
written into the audit log for every action you take.

## GPU acceleration (optional)

METRICA uses ONNX Runtime and picks the best available execution provider automatically,
falling back to the CPU (the fallback is recorded in the case).

| Platform | What to install | Provider |
|---|---|---|
| NVIDIA GPU (Linux/Windows) | `pip uninstall onnxruntime && pip install onnxruntime-gpu` (CUDA 12 + cuDNN 9 runtime) | CUDAExecutionProvider |
| Apple Silicon | nothing: the standard `onnxruntime` wheel includes CoreML | CoreMLExecutionProvider |
| Windows (AMD/Intel/NVIDIA) | `pip uninstall onnxruntime && pip install onnxruntime-directml` | DmlExecutionProvider |

`metrica doctor` shows the providers that were found. To force the CPU, set
`METRICA_DISABLE_PROVIDERS=CUDA,CoreML,Dml`. Structure from motion (pycolmap) runs on the CPU.

## Rebuilding the frontend (developers)

The release contains a prebuilt frontend in `frontend/dist`. To rebuild it:

```bash
cd frontend
npm ci
npm run build        # type-checks (tsc) and builds into frontend/dist
```

`npm run dev` starts a development server on port 5173 that proxies `/api` to a running
`metrica serve`.

## Uninstall

Delete the `metrica` folder (including `.venv`), the models folder (`~/.cache/metrica/models`)
and, if you no longer need them, your cases (`~/METRICA`). METRICA writes nothing else.

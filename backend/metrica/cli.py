"""Command-line interface: `metrica <command>`.

    metrica serve                      start the local application (opens the browser)
    metrica analyze VIDEO... --case DIR [--title T] [--profile standard] [--download-models]
    metrica report CASE                generate a report for a processed case
    metrica verify CASE                re-hash evidence, verify the audit chain and the manifest
    metrica models list|download|verify [--all] [--accept-restricted] [KEY...]
    metrica doctor                     environment diagnostics
    metrica demo [--out DIR]           render a synthetic test video and analyse it end to end
    metrica evaluate CASE GT.json      evaluate a synthetic case against its ground truth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _progress(stage: str, frac: float, msg: str) -> None:
    sys.stderr.write(f"\r[{stage:>14}] {frac * 100:5.1f}%  {msg[:80]:<80}")
    sys.stderr.flush()


def cmd_analyze(a) -> int:
    from .config import AnalysisConfig
    from .evidence.case import Case
    from .pipeline.runner import run_pipeline

    if a.download_models:
        _download(default_only=True, accept=False, keys=[])
    root = Path(a.case)
    case = Case(root) if (root / "case.json").exists() else Case.create(root, a.title or root.name, a.reference)
    have = {v["sha256_intake"] for v in case.videos()}
    from .evidence.hashing import sha256_file

    for v in a.videos:
        if sha256_file(v) in have:
            print(f"{v}: already in case, skipped")
            continue
        rec = case.add_video(Path(v), copy=not a.no_copy, actor=a.actor)
        print(f"Added {rec['id']}: {rec['original_name']} sha256={rec['sha256_intake']}")
    overrides = json.loads(a.config) if a.config else {}
    overrides["profile"] = a.profile
    cfg = AnalysisConfig.from_dict(overrides)
    state = run_pipeline(case, cfg, progress=_progress, force=a.force, actor=a.actor)
    sys.stderr.write("\n")
    print(json.dumps({k: state[k] for k in ("state", "duration_s", "manifest_sha256", "warnings") if k in state},
                     indent=1))
    for s in state["stages"]:
        print(f"  {s['name']:<15} {s['status']:<10} {s.get('duration_s', '')}")
    return 0 if state["state"].startswith("completed") else 1


def cmd_report(a) -> int:
    from .evidence.case import Case
    from .reporting.exports import generate_report

    rec = generate_report(Case(Path(a.case)), {"analyst": a.analyst or "", "purpose": a.purpose or ""},
                          actor=a.analyst or "analyst", pdf=not a.no_pdf)
    print(json.dumps(rec, indent=1))
    return 0


def cmd_verify(a) -> int:
    from .evidence.audit import verify_chain
    from .evidence.case import Case
    from .evidence.hashing import sha256_file

    case = Case(Path(a.case))
    ok = True
    for r in case.verify_originals():
        print(f"evidence {r['id']}: {'OK' if r['ok'] else 'MISMATCH'}  {r.get('sha256', r.get('reason'))}")
        ok &= r["ok"]
    chain = verify_chain(case.audit.entries())
    print(f"audit log: {chain['entries']} entries, chain {'valid' if chain['valid'] else 'BROKEN at ' + str(chain['first_broken_seq'])}")
    ok &= chain["valid"]
    man = case.read_json("manifest.json")
    if man:
        bad = [o["path"] for o in man["outputs"] if not (case.root / o["path"]).exists()
               or sha256_file(case.root / o["path"]) != o["sha256"]]
        print(f"manifest: {len(man['outputs'])} artefacts, {len(bad)} changed or missing since it was written")
        for b in bad[:20]:
            print(f"   changed: {b}")
    return 0 if ok else 2


def _download(default_only: bool, accept: bool, keys: list[str]) -> None:
    from .models.registry import DEFAULT_SET, MODELS, download

    todo = keys or (DEFAULT_SET if default_only else [k for k in MODELS if not MODELS[k].restricted or accept])
    for k in todo:
        try:
            p = download(k, accept_restricted=accept,
                         progress=lambda key, d, t: _progress("download", d / t if t else 0, f"{key} {d / 1e6:.0f} MB"))
            sys.stderr.write("\n")
            print(f"{k}: ready at {p}")
        except Exception as e:  # noqa: BLE001
            sys.stderr.write("\n")
            print(f"{k}: FAILED - {e}")


def cmd_models(a) -> int:
    from .models.registry import all_status

    if a.action == "download":
        _download(default_only=not a.all, accept=a.accept_restricted, keys=a.keys)
        return 0
    for s in all_status():
        flag = "" if s["default"] else " (optional)"
        if s["restricted"]:
            flag += f" [{s['license']} - opt-in]"
        print(f"{s['key']:<22} {s['state']:<8} {s['license']:<11} {s['size_mb']:>6.1f} MB  {s['task']}{flag}")
    return 0


def cmd_doctor(a) -> int:
    import platform
    import shutil

    from .evidence.manifest import software_environment
    from .models.registry import all_status, models_dir

    env = software_environment()
    print(f"METRICA {__version__} on Python {platform.python_version()} ({platform.platform()})")
    for k, v in env["packages"].items():
        print(f"  {k:<32} {v}")
    hw = env["hardware"]
    print(f"Hardware: {hw['cpu_count']} CPU threads, {hw['memory_gb']} GB RAM, ONNX providers: "
          f"{', '.join(hw['onnx_providers'])}")
    if hw["notes"]:
        print("  " + hw["notes"])
    print(f"System FFmpeg: {'found' if shutil.which('ffmpeg') else 'not found (not required: PyAV is used)'}")
    print(f"Models directory: {models_dir()}")
    st = all_status()
    ready = [s for s in st if s["state"] == "ready"]
    print(f"Models ready: {len(ready)}/{len(st)}")
    missing = [s["key"] for s in st if s["default"] and s["state"] != "ready"]
    if missing:
        print(f"  Missing default models: {', '.join(missing)} -> run `metrica models download`")
    from .api.app import frontend_dir

    print(f"Frontend build: {'found' if (frontend_dir() / 'index.html').exists() else 'MISSING'} ({frontend_dir()})")
    return 0


def cmd_serve(a) -> int:
    from .api.app import serve

    serve(host=a.host, port=a.port, open_browser=not a.no_browser, workspace=a.workspace)
    return 0


def cmd_demo(a) -> int:
    from .synthetic.scene import generate

    out = Path(a.out)
    info = generate(out / "input", kind=a.kind, seconds=a.seconds)
    print(f"Synthetic video: {info['video']}\nGround truth: {info['ground_truth']}")
    args = argparse.Namespace(videos=[info["video"]], case=str(out / "case"), title=f"Synthetic {a.kind} demo",
                              reference="DEMO", profile=a.profile, config=None, force=False, no_copy=False,
                              actor="demo", download_models=False)
    rc = cmd_analyze(args)
    if rc == 0:
        from .evaluation.synthetic_eval import evaluate_case

        ev = evaluate_case(out / "case", Path(info["ground_truth"]))
        (out / "evaluation.json").write_text(json.dumps(ev, indent=1, default=float))
        print(f"Evaluation written to {out / 'evaluation.json'}")
    return rc


def cmd_evaluate(a) -> int:
    from .evaluation.synthetic_eval import evaluate_case

    print(json.dumps(evaluate_case(Path(a.case), Path(a.gt)), indent=1, default=float))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="metrica", description="Evidence-oriented video-to-3D reconstruction.")
    p.add_argument("--version", action="version", version=f"metrica {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="start the local application")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")
    s.add_argument("--workspace", default=None, help="folder for cases (default ~/METRICA or $METRICA_HOME)")
    s.set_defaults(fn=cmd_serve)
    s = sub.add_parser("analyze", help="add videos to a case and run the full analysis")
    s.add_argument("videos", nargs="*")
    s.add_argument("--case", required=True)
    s.add_argument("--title")
    s.add_argument("--reference")
    s.add_argument("--profile", default="standard", choices=["fast", "standard", "quality"])
    s.add_argument("--config", help="JSON object with configuration overrides")
    s.add_argument("--force", action="store_true", help="recompute every stage")
    s.add_argument("--no-copy", action="store_true", help="reference originals in place instead of copying")
    s.add_argument("--actor", default="analyst")
    s.add_argument("--download-models", action="store_true")
    s.set_defaults(fn=cmd_analyze)
    s = sub.add_parser("report", help="generate a report")
    s.add_argument("case")
    s.add_argument("--analyst")
    s.add_argument("--purpose")
    s.add_argument("--no-pdf", action="store_true")
    s.set_defaults(fn=cmd_report)
    s = sub.add_parser("verify", help="verify evidence hashes, audit chain and manifest")
    s.add_argument("case")
    s.set_defaults(fn=cmd_verify)
    s = sub.add_parser("models", help="list / download / verify models")
    s.add_argument("action", choices=["list", "download", "verify"], nargs="?", default="list")
    s.add_argument("keys", nargs="*")
    s.add_argument("--all", action="store_true", help="include optional models")
    s.add_argument("--accept-restricted", action="store_true", help="also allow copyleft (AGPL) models")
    s.set_defaults(fn=cmd_models)
    s = sub.add_parser("doctor", help="environment diagnostics")
    s.set_defaults(fn=cmd_doctor)
    s = sub.add_parser("demo", help="synthetic end-to-end demo with evaluation")
    s.add_argument("--out", default="metrica-demo")
    s.add_argument("--kind", default="walk", choices=["walk", "static", "pan"])
    s.add_argument("--seconds", type=float, default=6.0)
    s.add_argument("--profile", default="standard", choices=["fast", "standard", "quality"])
    s.set_defaults(fn=cmd_demo)
    s = sub.add_parser("evaluate", help="evaluate a synthetic case against ground truth")
    s.add_argument("case")
    s.add_argument("gt")
    s.set_defaults(fn=cmd_evaluate)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

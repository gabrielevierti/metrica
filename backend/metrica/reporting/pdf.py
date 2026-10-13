"""PDF report (ReportLab Platypus; pure Python, no system libraries)."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

INK = colors.HexColor("#1d2330")
MUTED = colors.HexColor("#5d6577")
LINE = colors.HexColor("#d9dde6")
SOFT = colors.HexColor("#f4f6fa")
ACCENT = colors.HexColor("#1f5f8b")
WARNBG = colors.HexColor("#fff6e6")


def _styles():
    ss = getSampleStyleSheet()
    base = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=8.8, leading=11.5, textColor=INK)
    return {
        "body": base,
        "small": ParagraphStyle("s", parent=base, fontSize=7.3, leading=9.2, textColor=MUTED),
        "cell": ParagraphStyle("c", parent=base, fontSize=7.2, leading=8.8),
        "mono": ParagraphStyle("m", parent=base, fontName="Courier", fontSize=6.6, leading=8.2),
        "h1": ParagraphStyle("h1", parent=base, fontName="Helvetica-Bold", fontSize=20, leading=24, spaceAfter=4),
        "h2": ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=13, leading=16, spaceBefore=12,
                             spaceAfter=5, textColor=INK),
        "h3": ParagraphStyle("h3", parent=base, fontName="Helvetica-Bold", fontSize=10, leading=13, spaceBefore=8,
                             spaceAfter=3),
        "brand": ParagraphStyle("br", parent=base, fontName="Helvetica-Bold", fontSize=8, textColor=ACCENT),
        "callout": ParagraphStyle("co", parent=base, backColor=WARNBG, borderPadding=5, borderColor=colors.HexColor(
            "#f0d39c"), borderWidth=0.5, spaceBefore=4, spaceAfter=8),
    }


def _p(text: Any, st) -> Paragraph:
    return Paragraph(escape(str(text if text is not None else "—")).replace("\n", "<br/>"), st)


def _table(rows: list[list[Any]], widths: list[float], st, header: bool = True, mono_cols: set[int] | None = None):
    mono_cols = mono_cols or set()
    data = []
    for r, row in enumerate(rows):
        data.append([c if not isinstance(c, (str, int, float, type(None))) else
                     _p(c, st["mono"] if (k in mono_cols and r > 0) else st["cell"]) for k, c in enumerate(row)])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.3, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), SOFT))
    t.setStyle(TableStyle(style))
    return t


def _image(path: Path, max_w: float, max_h: float):
    from reportlab.lib.utils import ImageReader

    if not path.exists():
        return None
    ir = ImageReader(str(path))
    w, h = ir.getSize()
    s = min(max_w / w, max_h / h)
    return Image(str(path), width=w * s, height=h * s)


def render_pdf(d: dict[str, Any], rdir: Path, out: Path, max_events: int = 250) -> Path:
    st = _styles()
    W = A4[0] - 30 * mm

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(15 * mm, 10 * mm, f"METRICA report {d['report_id']} · case {d['case']['id']} · "
                                            f"{d['case'].get('reference') or ''}")
        canvas.drawRightString(A4[0] - 15 * mm, 10 * mm, f"page {doc.page}")
        canvas.drawString(15 * mm, A4[1] - 10 * mm, "Machine-generated findings require verification against the "
                                                   "source footage. No identity, intent or guilt is inferred.")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title=f"METRICA report {d['report_id']}", author="METRICA")
    s: list[Any] = []
    cs = d["coordinate_system"]
    s += [_p("METRICA · VIDEO-TO-3D SCENE ANALYSIS REPORT", st["brand"]), _p(d["case"]["title"], st["h1"])]
    meta = [["Case reference", d["case"].get("reference") or "—"], ["Case ID", d["case"]["id"]],
            ["Report ID", d["report_id"]], ["Generated", d["generated"]],
            ["Software", f"METRICA {d['software_version']}"], ["Scale status", f"{cs['status']} - {cs['status_text']}"]]
    if d.get("analyst"):
        meta.append(["Analyst", d["analyst"]])
    s.append(_table(meta, [40 * mm, W - 40 * mm], st, header=False))
    s.append(_p("This report separates machine-generated findings from analyst-verified findings. Machine findings "
                "are algorithmic observations and must be checked against the source footage.", st["callout"]))
    s.append(_p("1. Executive summary", st["h2"]))
    for l in d["summary"]:
        s.append(_p(l, st["body"]))
        s.append(Spacer(1, 3))
    # 2 inputs
    s.append(_p("2. Input inventory and integrity", st["h2"]))
    rows = [["ID", "File", "SHA-256 at intake", "Size", "Capture"]]
    for i in d["inputs"]:
        rows.append([i["id"], i["name"], i["sha256"], f"{i['size_bytes']:,} B", f"{i['mode']} ({i['mode_confidence']})"])
    s.append(_table(rows, [10 * mm, 38 * mm, 82 * mm, 24 * mm, W - 154 * mm], st, mono_cols={2}))
    rows = [["ID", "Codec", "Resolution", "FPS", "Frames", "Duration", "VFR", "Audio", "Decode err.", "Offset"]]
    for i in d["inputs"]:
        rows.append([i["id"], f"{i['container']}/{i['codec']}", i["resolution"], i["fps"], i["frames"],
                     f"{i['duration_s']} s", "yes" if i["vfr"] else "no", "yes" if i["audio"] else "no",
                     i["decode_errors"], f"{i['offset']:.3f} s"])
    s.append(_table(rows, [10 * mm, 30 * mm, 22 * mm, 14 * mm, 15 * mm, 18 * mm, 10 * mm, 12 * mm, 16 * mm,
                           W - 147 * mm], st))
    s.append(_p(d["integrity"]["statement"], st["small"]))
    s.append(_p(f"Audit log: {d['integrity']['audit_entries']} entries, chain "
                f"{'valid' if d['integrity']['audit_chain_valid'] else 'BROKEN'}, head {d['integrity']['audit_head']}",
                st["small"]))
    # 3 quality
    s.append(_p("3. Technical quality assessment", st["h2"]))
    for i in d["inputs"]:
        q = i["quality"]
        txt = (f"{i['id']}: usable frames {100 * (q.get('usable_fraction') or 0):.0f}%, duplicates "
               f"{100 * (q.get('duplicate_fraction') or 0):.0f}%, timestamp gaps {i['timing'].get('gap_count', 0)}. "
               + ("; ".join(q.get("flags", [])) or "No quality flags."))
        s.append(_p(txt, st["body"]))
        img = _image(rdir / i["quality_chart"], W, 60 * mm)
        if img:
            s.append(img)
    # 4 methodology
    s.append(_p("4. Processing pipeline and methodology", st["h2"]))
    rows = [["Stage", "Status", "Time", "Warnings / fallbacks"]]
    for sg in d["stages"]:
        wf = list(sg["warnings"][:3]) + [f"Fallback: {f['component']} -> {f['used']}. {f['implication']}"
                                         for f in sg["fallbacks"]]
        rows.append([sg["title"], sg["status"], f"{sg['duration_s'] or '—'} s", "\n".join(wf)])
    s.append(_table(rows, [45 * mm, 18 * mm, 14 * mm, W - 77 * mm], st))
    for u in d["units"]:
        s.append(_p(f"Unit {u['id']} ({u['kind']})", st["h3"]))
        s.append(_p(u["method"], st["body"]))
    s.append(_p(d["object_method"], st["body"]))
    rows = [["Evidence class", "Meaning"]] + [[k, v] for k, v in d["evidence_classes"].items()]
    s.append(_table(rows, [45 * mm, W - 45 * mm], st))
    # 5 coordinates
    s.append(_p("5. Coordinate system and calibration", st["h2"]))
    sc = cs["scale"]
    rows = [["Axes", f"Right-handed, {cs['up']} up, ground {cs['ground']}"], ["Unit", cs["unit"]],
            ["Status", f"{cs['status']} - {cs['status_text']}"], ["Scale method", sc.get("method")],
            ["Scale uncertainty (1 sigma)", f"{sc['scale_sigma_rel'] * 100:.1f}%" if sc.get("scale_sigma_rel")
             else "not quantified"],
            ["Assumptions", "\n".join(sc.get("assumptions", []))]]
    s.append(_table(rows, [45 * mm, W - 45 * mm], st, header=False))
    # 6/7 scene
    s.append(PageBreak())
    s.append(_p("6-7. Reconstruction quality and scene overview", st["h2"]))
    for u in d["units"]:
        rows = [["Layer", "Kind", "Vertices", "Faces", "Evidence (vertices)"]]
        for l in u["layers"]:
            rows.append([l["name"], l["kind"], l["vertices"], l["faces"],
                         ", ".join(f"{k}: {v}" for k, v in (l.get("evidence_vertices") or {}).items())])
        s.append(_p(u["id"], st["h3"]))
        s.append(_table(rows, [25 * mm, 18 * mm, 20 * mm, 20 * mm, W - 83 * mm], st))
        if u.get("coverage"):
            s.append(_p(f"Ground coverage: {100 * u['coverage']['seen_fraction']:.0f}% seen, "
                        f"{100 * u['coverage']['occluded_fraction']:.0f}% in view but hidden.", st["small"]))
    for key, cap in (("plan_view", "Plan view (observed ground green, inferred red hatching, facades brown, poles "
                                   "yellow, vehicles blue, trajectories, cameras)."),
                     ("view_color", "Observed surface."), ("view_evidence", "Evidence classes: green multi-view, "
                                                                            "blue single-view.")):
        if d["figures"].get(key):
            img = _image(rdir / d["figures"][key], W, 120 * mm)
            if img:
                s.append(KeepTogether([img, _p(cap, st["small"])]))
    # 8 objects
    s.append(_p("8. Reconstructed object inventory", st["h2"]))
    rows = [["ID", "Category", "Evidence", "Conf.", "Dimensions (value ± sigma, basis)", "Notes"]]
    for o in d["objects"]:
        notes = []
        if o["openings"]:
            notes.append(f"{o['openings']} candidate opening(s)")
        if o["inferred"]:
            notes.append(f"{o['inferred']} inferred part(s)")
        notes += o["flags"]
        rows.append([o["id"], o["display"], o["evidence"], f"{o['confidence']:.2f}" if o["confidence"] else "—",
                     "\n".join(f"{k}: {v}" for k, v in o["dimensions"].items()), "\n".join(notes)])
    s.append(_table(rows, [30 * mm, 20 * mm, 28 * mm, 10 * mm, 58 * mm, W - 146 * mm], st, mono_cols={0}))
    # 9 people
    s.append(_p("9. People and face observations", st["h2"]))
    s.append(_p("Face crops are unmodified original pixels. Detection only: no recognition, no database comparison, "
                "no inference of personal attributes.", st["callout"]))
    for p in d["people"]:
        txt = (f"<b>{escape(p['id'])}</b> (machine-generated): observed {p['t0']:.2f}-{p['t1']:.2f} s in {p['video']}, "
               f"{p['observations']} observations, {p['gaps']} gap(s), {p['links']} re-link(s).")
        if p.get("stature"):
            txt += f" Estimated stature {p['stature']['median']:.2f} {d['unit']} (n={p['stature']['n']})."
        imgs = []
        for c in p["crops"][:2] + p.get("faces", [])[:3]:
            im = _image(rdir / c["image"], 32 * mm, 40 * mm)
            if im:
                imgs.append([im, _p(f"f{c['frame']} · {c['t']:.2f} s" + (f" · {c['pose']} · {c['height_px']} px"
                                                                          if "pose" in c else ""), st["small"])])
        block = [Paragraph(txt, st["body"])]
        if imgs:
            t = Table([[x[0] for x in imgs], [x[1] for x in imgs]], colWidths=[34 * mm] * len(imgs), hAlign="LEFT")
            t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "BOTTOM")]))
            block.append(t)
        s.append(KeepTogether(block + [Spacer(1, 4)]))
    # 10 vehicles
    s.append(_p("10. Vehicle observations", st["h2"]))
    for v in d["vehicles"]:
        txt = f"<b>{escape(v['id'])}</b> ({v['category']}): {v['t0']:.2f}-{v['t1']:.2f} s in {v['video']}."
        if v.get("color"):
            txt += f" Apparent colour {v['color']['name']} ({'reliable' if v['color']['reliable'] else 'not reliable'})."
        for o in v.get("ocr", [])[:3]:
            txt += f" Candidate text '{escape(o['text'])}' (confidence {o['confidence']}; requires verification)."
        s.append(Paragraph(txt, st["body"]))
    # 11 timeline
    s.append(PageBreak())
    s.append(_p("11. Event timeline", st["h2"]))
    if d["figures"].get("timeline"):
        img = _image(rdir / d["figures"]["timeline"], W, 90 * mm)
        if img:
            s.append(img)
    rows = [["ID", "Time (s)", "Type", "Description", "Source", "Conf."]]
    for e in d["events"][:max_events]:
        rows.append([e["id"], f"{e['t']:.2f}" + (f"-{e['t_end']:.2f}" if e.get("t_end") else ""), e["type"],
                     e["description"], f"{e['video']}" + (f" f{e['frame']}" if e.get("frame") is not None else ""),
                     f"{e['confidence']:.2f}"])
    s.append(_table(rows, [13 * mm, 20 * mm, 24 * mm, W - 85 * mm, 18 * mm, 10 * mm], st))
    if len(d["events"]) > max_events:
        s.append(_p(f"{len(d['events']) - max_events} further events are listed in exports/events.csv.", st["small"]))
    # 12 trajectories + measurements
    s.append(_p("12. Trajectories and measurements", st["h2"]))
    rows = [["Track", "Observed path", "Displacement", "Median speed", "Max speed", "Position sigma"]]
    for p in d["people"] + d["vehicles"]:
        sm = p.get("summary") or {}
        if sm.get("path_length_observed") is None:
            continue
        u = d["unit"]
        rows.append([p["id"], f"{sm['path_length_observed']:.1f} {u}", f"{sm['displacement']:.1f} {u}",
                     f"{sm['speed_median']:.2f} {u}/s" if sm.get("speed_median") is not None else "—",
                     f"{sm['speed_max']:.2f} {u}/s" if sm.get("speed_max") is not None else "—",
                     f"{sm['position_sigma_median']:.2f} {u}"])
    s.append(_table(rows, [25 * mm] + [(W - 25 * mm) / 5] * 5, st))
    if d["measurements"]:
        rows = [["ID", "Type", "Value", "Status", "By"]]
        for m in d["measurements"]:
            rows.append([m["id"], m["kind"], m["display"] + (f" - WARNING: {m['stale']}" if m.get("stale") else ""),
                         m["status"], f"{m['actor']} {m['created']}"])
        s.append(_table(rows, [20 * mm, 25 * mm, 50 * mm, 35 * mm, W - 130 * mm], st))
    else:
        s.append(_p("No analyst measurements recorded.", st["small"]))
    # 13 multi-video
    s.append(_p("13. Multi-video synchronisation and alignment", st["h2"]))
    rows = [["Video", "Offset", "Method", "Confidence"]]
    for i in d["inputs"]:
        sy = i.get("sync") or {}
        rows.append([i["id"], f"{i['offset']:.3f} s", sy.get("method", "—"), f"{sy.get('confidence', 0):.2f}"])
    s.append(_table(rows, [20 * mm, 25 * mm, W - 75 * mm, 30 * mm], st))
    rows = [["Unit", "Alignment method", "Status", "Residual"]]
    for u in d["units"]:
        al = u.get("alignment") or {}
        rows.append([u["id"], al.get("method", "—"), al.get("status", "—"),
                     f"{al['rms']:.2f}" if isinstance(al.get("rms"), (int, float)) else "—"])
    s.append(_table(rows, [25 * mm, W - 85 * mm, 30 * mm, 30 * mm], st))
    g = d.get("grouping") or {}
    if g.get("pairs"):
        s.append(_p("Grouping suggestion: " + g.get("note", "") + " Suggested groups: " +
                    ", ".join("[" + ", ".join(x) + "]" for x in g.get("groups", [])) + ".", st["small"]))
        rows = [["Pair", "Suggestion", "Strong evidence", "Weak context (not used)"]]
        for pr in g["pairs"]:
            rows.append([f"{pr['a']} / {pr['b']}", pr["suggestion"], "; ".join(pr["strong_evidence"]) or "—",
                         "; ".join(pr["weak_context"]) or "—"])
        s.append(_table(rows, [22 * mm, 25 * mm, (W - 47 * mm) / 2, (W - 47 * mm) / 2], st))
    conf = [c for c in d.get("conflicts", [])]
    if conf:
        rows = [["Tracks", "Interval", "Median distance", "Conflict"]]
        for c in conf:
            rows.append([" / ".join(c["tracks"]), f"{c['t0']:.1f}-{c['t1']:.1f} s", f"{c['median_distance']:.2f}",
                         c.get("conflict") or "consistent"])
        s.append(_table(rows, [45 * mm, 30 * mm, 30 * mm, W - 105 * mm], st))
    # 14 analyst
    s.append(_p("14. Analyst-verified findings and notes", st["h2"]))
    if d["verifications"]:
        rows = [["Item", "Verdict", "Note", "Analyst / time"]]
        for v in d["verifications"]:
            rows.append([v["target"], v["verdict"], v.get("note", ""), f"{v['actor']} {v['created']}"])
        s.append(_table(rows, [35 * mm, 20 * mm, W - 95 * mm, 40 * mm], st))
    else:
        s.append(_p("No findings have been verified by an analyst; all findings are machine-generated.", st["body"]))
    # 15 limitations
    s.append(_p("15. Limitations and warnings", st["h2"]))
    for l in d["limitations"]:
        s.append(_p("• " + l, st["body"]))
    if d["warnings"]:
        rows = [["Source", "Message"]] + [[w["source"], w["message"]] for w in d["warnings"]]
        s.append(_table(rows, [25 * mm, W - 25 * mm], st))
    # 16 appendix
    s.append(_p("16. Appendix: models and parameters", st["h2"]))
    rows = [["Model", "License", "SHA-256"]] + [[m.get("about") or m.get("key"), m.get("license", "—"),
                                                  m.get("sha256", "—")] for m in d["models"]]
    s.append(_table(rows, [60 * mm, 22 * mm, W - 82 * mm], st, mono_cols={2}))
    rows = [["Parameter", "Value"]] + [[k, str(v)] for k, v in d["config"].items()]
    s.append(_table(rows, [55 * mm, W - 55 * mm], st, mono_cols={0, 1}))
    doc.build(s, onFirstPage=on_page, onLaterPages=on_page)
    return out

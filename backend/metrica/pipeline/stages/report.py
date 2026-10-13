"""Stage 10 - default report (HTML + PDF + exports) at the end of an analysis."""
from __future__ import annotations

from ...reporting.exports import generate_report
from ..base import Context, Stage, StageResult


class ReportStage(Stage):
    name = "report"
    title = "Report generation"
    depends = ("scene",)
    config_keys = ()
    version = "1"

    def run(self, ctx: Context) -> StageResult:
        res = StageResult()
        ctx.progress(0.1, "rendering figures and report")
        rec = generate_report(ctx.case, {"kind": "automatic end-of-analysis report"})
        res.metrics = {"report_id": rec["report_id"], "files": list(rec["files"])}
        for k in ("html", "pdf", "zip"):
            if k in rec["files"]:
                res.outputs.append(ctx.case.root / rec["files"][k])
        return res

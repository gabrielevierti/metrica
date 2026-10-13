"""HTML report rendering (Jinja2, autoescaped)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

_ENV = Environment(loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
                   autoescape=select_autoescape(["html", "j2"]), trim_blocks=True, lstrip_blocks=True)


def render_html(data: dict[str, Any], out: Path) -> Path:
    html = _ENV.get_template("report.html.j2").render(d=data)
    out.write_text(html, encoding="utf-8")
    return out

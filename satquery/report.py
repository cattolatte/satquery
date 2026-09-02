"""Downloadable analysis reports.

The problem statement lists "visual evidence, confidence information, execution
summaries, and downloadable reports" among the things the solution should
include, so a report is a deliverable in its own right rather than a convenience.

The HTML report embeds its images as data URIs and carries no external
references, so it stays readable after being emailed, archived, or opened on a
machine that has never run this project.
"""
from __future__ import annotations

import base64
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CSS = """
body{font:14px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
 max-width:840px;margin:36px auto;padding:0 22px;color:#1a1f26}
h1{font-size:21px;margin:0 0 4px} h2{font-size:12px;text-transform:uppercase;
 letter-spacing:.09em;color:#667; margin:30px 0 10px;font-weight:600}
.meta{color:#667;font-size:12px;margin-bottom:26px}
.answer{background:#f5f7fa;border:1px solid #dde3ea;border-radius:9px;
 padding:15px;white-space:pre-wrap;font-size:15px}
table{width:100%;border-collapse:collapse;font:12px ui-monospace,Menlo,monospace}
th,td{text-align:left;padding:7px 6px;border-bottom:1px solid #e6eaef;vertical-align:top}
th{color:#667;font-weight:500}
.note{border-left:3px solid #d29922;background:#fdf7e6;padding:7px 11px;margin:7px 0;font-size:13px}
.stage{position:relative;display:inline-block;max-width:100%;margin:9px 9px 0 0}
.stage img{max-width:390px;border-radius:7px;border:1px solid #dde3ea;display:block}
.box{position:absolute;border:2px solid #1f6feb;border-radius:3px}
.box b{position:absolute;top:-18px;left:-2px;background:#1f6feb;color:#fff;
 font:10px/1.5 monospace;padding:1px 5px;border-radius:3px;white-space:nowrap}
.ok{color:#1a7f37} .no{color:#c0392b}
footer{margin-top:34px;padding-top:14px;border-top:1px solid #e6eaef;color:#889;font-size:11px}
"""


def _data_uri(path: str) -> str | None:
    """Inline an image so the report survives being moved off this machine."""
    p = Path(path)
    if not p.is_file():
        return None
    suffix = p.suffix.lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(suffix)
    if mime is None:
        # GeoTIFF has no browser-native rendering; convert if Pillow can read it.
        try:
            import io
            from PIL import Image
            buf = io.BytesIO()
            Image.open(p).convert("RGB").save(buf, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:                                      # noqa: BLE001
            return None
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def build_html(payload: dict[str, Any], image_paths: list[str]) -> str:
    """Render one answer, with its evidence and trace, as standalone HTML."""
    e = html.escape
    trace = payload.get("trace") or {}
    conf = payload.get("confidence") or 0.0
    boxes = [ev for ev in payload.get("evidence", []) if ev.get("kind") == "bbox"]

    stages = ""
    for path in image_paths[:2]:
        uri = _data_uri(path)
        if not uri:
            continue
        overlay = "".join(
            f'<div class="box" style="left:{b["data"][0]*100}%;top:{b["data"][1]*100}%;'
            f'width:{(b["data"][2]-b["data"][0])*100}%;height:{(b["data"][3]-b["data"][1])*100}%">'
            f'<b>{e(str(b.get("label","")))}</b></div>'
            for b in boxes) if path == image_paths[0] else ""
        stages += f'<div class="stage"><img src="{uri}">{overlay}</div>'

    calls = "".join(
        f"<tr><td>{e(c['tool'])}</td><td>{c['ms']} ms</td>"
        f"<td class=\"{'ok' if c['ok'] else 'no'}\">{'ok' if c['ok'] else 'failed'}</td>"
        f"<td>{e(json.dumps(c.get('params', {})))}</td></tr>"
        + (f'<tr><td colspan="4" class="no">{e(str(c["error"]))}</td></tr>' if c.get("error") else "")
        for c in trace.get("calls", []))

    images_meta = "".join(
        f"<tr><td>{e(m['fmt'])}</td><td>{m['width']}x{m['height']}</td>"
        f"<td>{m['bands']}</td><td>{e(m['modality'])}</td>"
        f"<td>{'yes' if m.get('georeferenced') else 'no'}</td></tr>"
        for m in trace.get("images", []))

    notes = "".join(f'<div class="note">{e(n)}</div>' for n in trace.get("rejected", []))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>SatQuery AI — analysis report</title><style>{_CSS}</style></head><body>
<h1>SatQuery AI — analysis report</h1>
<div class="meta">{stamp} · task <b>{e(str(trace.get('task','—')))}</b>
 · inputs {e(str(trace.get('input_kind','—')))} · {trace.get('total_ms','—')} ms total</div>

<h2>Query</h2><div class="answer">{e(payload.get('query',''))}</div>
<h2>Answer</h2><div class="answer">{e(payload.get('text',''))}</div>
<h2>Confidence</h2><p>{conf:.0%} — the minimum across the executed chain, so it
reflects the weakest step rather than an average that could hide one.</p>
{f'<h2>Input validation</h2>{notes}' if notes else ''}
{f'<h2>Visual evidence</h2>{stages}' if stages else ''}
<h2>Execution trace</h2>
<table><tr><th>tool</th><th>time</th><th>status</th><th>permitted parameters</th></tr>
{calls or '<tr><td colspan="4">no tools ran</td></tr>'}</table>
{f'<h2>Inputs</h2><table><tr><th>format</th><th>size</th><th>bands</th><th>modality</th><th>georeferenced</th></tr>{images_meta}</table>' if images_meta else ''}
<footer>Generated by SatQuery AI. Evidence coordinates are normalised to the
image extent. This report is self-contained: images are embedded, nothing is
fetched when it is opened.</footer></body></html>"""

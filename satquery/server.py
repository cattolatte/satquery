"""HTTP API and static host for the SatQuery web application.

The problem statement asks for "an interactive GUI or web application with an
agentic remote-sensing AI backend", and for the system to "present results
along with an auditable summary of the tools used". So the API returns the full
trace on every request, not only on failure -- the trace is a product surface
here, not a debug channel.

Uploads are held in a per-process temporary directory and served back read-only
so the browser can draw evidence over the exact pixels the model saw.
"""
from __future__ import annotations

import shutil
import tempfile
import uuid
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .registry import build_controller, build_registry

WEB = Path(__file__).resolve().parent.parent / "web"
UPLOADS = Path(tempfile.gettempdir()) / "satquery-uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

# Two images is the documented maximum: the statement's multi-image tasks are
# bi-temporal pairs and optical/SAR pairs, both of which are exactly two.
MAX_IMAGES = 2
MAX_BYTES = 64 * 1024 * 1024
ALLOWED_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

app = FastAPI(title="SatQuery AI", version="0.1.0")
_controller = None


def controller():
    """Built lazily so the process starts instantly and the first request pays
    the model load. Importing torch at module scope would make `--reload`
    unusable."""
    global _controller
    if _controller is None:
        _controller = build_controller()
    return _controller


def _plain(obj: Any) -> Any:
    """Dataclasses and enums to JSON-safe primitives."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


@app.get("/api/tools")
def tools() -> JSONResponse:
    """What the system can do, and what is currently loadable.

    The UI shows unavailable tools rather than hiding them, so a missing
    dependency reads as a known gap instead of a silently narrower system.
    """
    registry = build_registry()
    out = []
    for entry in registry.describe():
        tool = registry.get(entry["name"])
        ok, why = tool.available()
        out.append({**_plain(entry), "available": ok, "reason": why})
    return JSONResponse(out)


@app.post("/api/query")
async def query(q: str = Form(...), images: list[UploadFile] = File(default=[])):
    """Answer one question about one or two images."""
    if not q.strip():
        raise HTTPException(422, "query is empty")
    if not images:
        raise HTTPException(422, "at least one image is required")
    if len(images) > MAX_IMAGES:
        raise HTTPException(422, f"at most {MAX_IMAGES} images")

    batch = UPLOADS / uuid.uuid4().hex
    batch.mkdir(parents=True)
    saved: list[str] = []
    for upload in images:
        suffix = Path(upload.filename or "image.png").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(422, f"unsupported file type: {suffix}")
        target = batch / f"{len(saved)}{suffix}"
        with target.open("wb") as fh:
            shutil.copyfileobj(upload.file, fh, length=1024 * 1024)
        if target.stat().st_size > MAX_BYTES:
            target.unlink()
            raise HTTPException(413, f"{upload.filename} exceeds {MAX_BYTES // 2**20} MB")
        saved.append(str(target))

    answer = controller().run(q, saved)
    payload = _plain(asdict(answer))
    # Hand back URLs so the browser can draw boxes over the same pixels.
    payload["images"] = [f"/api/image/{batch.name}/{Path(p).name}" for p in saved]
    return JSONResponse(payload)


@app.get("/api/image/{batch}/{name}")
def image(batch: str, name: str) -> FileResponse:
    """Serve an uploaded image back, with the path confined to the upload root."""
    target = (UPLOADS / batch / name).resolve()
    if not target.is_file() or UPLOADS.resolve() not in target.parents:
        raise HTTPException(404, "not found")
    return FileResponse(target)


@app.get("/api/health")
def health() -> dict:
    from .tools.backbone import load
    bb = load()
    return {"ok": True,
            "backbone": bb.name if bb else None,
            "device": bb.device if bb else None}


if WEB.is_dir():
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")

"""Gradio front end for Hugging Face Spaces.

Docker Spaces became a paid feature, so Gradio is the only free way to host a
Python app. That turns out to be the better option anyway: ZeroGPU is
Gradio-only, so this same file can run on a free GPU.

Nothing about the system changes here. This is an interface layer over the
existing `Controller.run(query, image_paths)`; routing, tool selection, fusion,
confidence and the trace are all untouched, so what the Space demonstrates is
the same code path the benchmarks were measured on.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr

# `spaces` only exists on Hugging Face. The decorator is a no-op elsewhere, so
# the same file runs locally without the import guard leaking into the app.
try:
    import spaces
    GPU = spaces.GPU(duration=120)
except Exception:                                    # local, or CPU Space
    def GPU(fn):
        return fn

MAX_IMAGES = 2                                       # the documented maximum

# The adapters are not in the Space's git history -- 600 MB of weights do not
# belong there -- so they are fetched Hub-to-Hub at startup instead.
#
# The destination is a *relative* path on purpose. The tools resolve
# checkpoints/rs_clip and checkpoints/rs_vlm relative to the working directory,
# so writing to the same relative root is what guarantees the download and the
# lookup cannot disagree about where the weights are.
WEIGHTS_REPO = os.environ.get("WEIGHTS_REPO", "").strip()
CHECKPOINTS = Path("checkpoints")


def _fetch_weights() -> str:
    """Pull the serving adapters, and say plainly what the outcome was.

    Running unadapted is a supported state -- confidence is capped and the
    trace records it -- so a failure here degrades the answer rather than
    stopping the Space. What it must never do is degrade *silently*: an
    unadapted backbone looks like a modelling problem rather than a missing
    environment variable, so the reason is surfaced in the interface.
    """
    if (CHECKPOINTS / "rs_clip" / "config.json").is_file():
        return ""                                    # already present, local run
    if not WEIGHTS_REPO:
        return ("**WEIGHTS_REPO is not set.** Running unadapted, so confidence "
                "is capped by design. Set it in Settings -> Variables and "
                "secrets and restart.")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=WEIGHTS_REPO,
            repo_type="model",
            local_dir=str(CHECKPOINTS),
            allow_patterns=["rs_clip/*", "rs_vlm/*"],
        )
        return ""
    except Exception as exc:                         # noqa: BLE001
        return (f"**Could not fetch `{WEIGHTS_REPO}`** ({exc}). Running "
                "unadapted, confidence capped. Check the name is exactly "
                "`<username>/satquery-weights` and that the repo is public.")


_weights_note = _fetch_weights()

# Built at module level, not lazily.
#
# ZeroGPU runs a CUDA emulation mode outside @spaces.GPU functions specifically
# so that models can be placed on cuda during startup; loading them inside the
# decorated function instead is documented as significantly less efficient,
# because the real CUDA transfer then happens on every call rather than once.
# The FastAPI server loads lazily for a different reason -- keeping --reload
# usable -- which does not apply here.
#
# A failure is held rather than raised so the Space still boots and can say what
# went wrong, instead of showing only a stack trace in the build log.
_controller = None
_load_error = None
try:
    from satquery.registry import build_controller
    _controller = build_controller()
except Exception as exc:                             # noqa: BLE001
    _load_error = exc


def controller():
    if _controller is None:
        raise RuntimeError(f"models failed to load: {_load_error}")
    return _controller


@GPU
def answer(query: str, image_a, image_b):
    if not query or not query.strip():
        return "Enter a question.", "", ""

    images = [im for im in (image_a, image_b) if im is not None]
    if not images:
        return "Attach one or two images.", "", ""
    if len(images) > MAX_IMAGES:
        return f"At most {MAX_IMAGES} images.", "", ""

    # gr.File hands back either a path string or an object carrying one,
    # depending on Gradio version; normalise before touching the filesystem.
    tmp = Path(tempfile.mkdtemp(prefix="satquery-"))
    paths = []
    for i, im in enumerate(images):
        src = Path(getattr(im, "name", None) or str(im))
        dst = tmp / f"image_{i}{src.suffix or '.png'}"
        dst.write_bytes(src.read_bytes())
        paths.append(str(dst))

    result = controller().run(query.strip(), paths)

    # Confidence is reported, never hidden: it is capped when a component runs
    # without its adapter, and a reader needs to see that rather than infer it.
    header = f"**{result.text}**\n\n*Confidence {result.confidence:.2f}*"

    ev = "\n".join(
        f"- `{getattr(e, 'kind', 'evidence')}` {getattr(e, 'label', '')} "
        f"{getattr(e, 'box', '')}".rstrip()
        for e in (result.evidence or [])
    ) or "_No spatial evidence for this task._"

    trace = "_No trace._"
    if result.trace:
        d = result.trace.to_dict()
        rows = [
            f"| task | `{d.get('task', '—')}` |",
            f"| input kind | `{d.get('input_kind', '—')}` |",
            f"| tools | `{', '.join(d.get('tools', [])) or '—'}` |",
        ]
        if d.get("notes"):
            rows.append(f"| notes | {d['notes']} |")
        trace = "| | |\n|---|---|\n" + "\n".join(rows)

    return header, ev, trace


DESCRIPTION = """
# 🛰️ SatQuery AI

**Smart India Hackathon 2026 · Problem Statement `SIH26167` · ISRO**

Ask satellite imagery a question in plain English. A controller classifies the
task, checks the imagery can actually answer it, selects specialist models,
runs them, fuses the outputs and returns the answer with its evidence and a
full execution trace.

**One image** → visual question answering, captioning, referring grounding.
**Two dates of the same place** → change description and change VQA.
**Optical + SAR of the same area** → joint cross-sensor extraction.

A question the inputs cannot support is *downgraded and the reason recorded*,
rather than answered anyway.
"""

NOTES = """
### Measured results

Held-out data, through the real serving path, each beside the reference it beats.

| benchmark | result | reference |
|---|---|---|
| VRSBench VQA | **55.2%** | 7.6% scene-level only |
| Captioning, ROUGE-L | **0.306** | 0.026 |
| Referring grounding, Acc@0.5 | **28.0%** | 0.2% patch-token |
| CDVQA change VQA | **64.3%** | 47.3% hand-written heuristic |
| RSVQA-LR | **51.7%** | 34.9% |
| Optical–SAR fused, P@3 | **37.6%** | 18.9% better single sensor |

Twelve of twelve VRSBench question types beat their own majority baseline.

### Honest limitations

- **Counting is weak** at 23.8% — at 10 m ground sample distance the difference
  between six and seven objects may not be present in the data at all.
- **Comparison questions regressed** to 43.8% from 54.5%: comparing needs two
  counts, and the generative specialist answers in one shot without counting.
- Confidence is **capped when a component runs unadapted**, by design.
- The first query loads the backbone and is slow; later ones are faster.
"""

with gr.Blocks(title="SatQuery AI") as demo:
    gr.Markdown(DESCRIPTION)
    # Shown only when something is actually wrong, so it stays meaningful.
    if _weights_note or _load_error:
        gr.Markdown("> " + (_weights_note or f"**Models failed to load:** {_load_error}"))
    with gr.Row():
        with gr.Column(scale=1):
            img_a = gr.File(label="Image 1  (GeoTIFF / TIFF / PNG / JPEG)",
                            file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"])
            img_b = gr.File(label="Image 2 — optional (second date, or SAR)",
                            file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"])
            q = gr.Textbox(
                label="Question",
                placeholder="What changed between these two dates, and where?",
                lines=2)
            go = gr.Button("Ask", variant="primary")
            gr.Examples(
                examples=[
                    ["Describe the land cover and major objects visible in this image."],
                    ["Highlight the water body referred to in the query."],
                    ["What changed between these two dates, and where did the change occur?"],
                    ["Use the optical and SAR images together to identify built-up and water-covered regions."],
                    ["Has the built-up area increased, decreased, or remained unchanged?"],
                ],
                inputs=[q],
                label="Representative queries from the problem statement",
            )
        with gr.Column(scale=1):
            out = gr.Markdown(label="Answer")
            gr.Markdown("### Evidence")
            ev = gr.Markdown()
            gr.Markdown("### Execution trace")
            tr = gr.Markdown()

    gr.Markdown(NOTES)
    go.click(answer, inputs=[q, img_a, img_b], outputs=[out, ev, tr])

if __name__ == "__main__":
    demo.queue(max_size=8).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )

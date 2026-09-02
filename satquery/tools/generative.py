"""The generative specialist: free-form answers, captions and grounded boxes.

The fixed-vocabulary head has a measured hard ceiling. Of VRSBench's 37,409 VQA
answers, only 65.5% are expressible by ranking a 21-class land-cover
vocabulary at all -- "Body of water", "residential", "grayscale" and "high" are
simply not in its output space, at any confidence. Captioning is the sharper
case: references average 48 words against a five-class template.

A generative model removes the ceiling rather than raising it, and fine-tuning
on VRSBench's own training split closes the distance to it. Both were needed:
the ceiling was 65.5% and the score was 11.5%, so output space alone explained
under half the gap.

This does not replace the other specialists, and the controller still chooses.
The land-cover head remains better calibrated on the vocabulary it was trained
for, and the detector remains the only component that produces instance boxes
with scores. The statement asks for an agentic system that selects among
specialists -- adding a stronger one is the intended direction, not a shortcut
around it.
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import Any

from ..schema import Evidence, ImageMeta, Task
from .base import Tool, ToolSpec
from .specialists import _open

BASE = "HuggingFaceTB/SmolVLM-500M-Instruct"
ADAPTER = "checkpoints/rs_vlm"

# Must match training exactly. The processor's own default differs, so leaving
# it unset resized images differently at inference than the weights were
# trained on -- a silent accuracy loss that looks like a modelling problem.
# Defined here and imported by the trainer so the two cannot drift apart.
IMAGE_SIZE = 512

PROMPTS = {
    "vqa": "{q}\nAnswer in as few words as possible.",
    "caption": "{q}",
    "refer": "{q}\nRespond with a bounding box as {{<x0><y0><x1><y1>}} on a 0-99 scale.",
}

# VRSBench writes boxes as {<x0><y0><x1><y1>} on a 0-99 grid.
_BOX = re.compile(r"<(\d{1,2})><(\d{1,2})><(\d{1,2})><(\d{1,2})>")


@dataclass
class Generative:
    model: object
    processor: object
    device: str
    adapted: bool


@functools.lru_cache(maxsize=1)
def load() -> Generative | None:
    """Load the fine-tuned adapter if present, else the base model, else None."""
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError:
        return None

    from pathlib import Path
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    try:
        adapter = Path(ADAPTER)
        if (adapter / "adapter_config.json").exists():
            from peft import PeftModel
            processor = AutoProcessor.from_pretrained(str(adapter), do_image_splitting=False)
            processor.image_processor.size = {"longest_edge": IMAGE_SIZE}
            base = AutoModelForImageTextToText.from_pretrained(BASE, torch_dtype=torch.float32)
            model = PeftModel.from_pretrained(base, str(adapter)).to(device).eval()
            return Generative(model=model, processor=processor, device=device, adapted=True)

        processor = AutoProcessor.from_pretrained(BASE, do_image_splitting=False)
        processor.image_processor.size = {"longest_edge": IMAGE_SIZE}
        model = AutoModelForImageTextToText.from_pretrained(
            BASE, torch_dtype=torch.float32).to(device).eval()
        return Generative(model=model, processor=processor, device=device, adapted=False)
    except Exception:                                          # noqa: BLE001
        return None


def generate(gen: Generative, image, question: str, kind: str = "vqa",
             max_new_tokens: int = 64) -> str:
    import torch

    prompt = PROMPTS.get(kind, "{q}").format(q=question)
    messages = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": prompt}]}]
    text = gen.processor.apply_chat_template(messages, add_generation_prompt=True)
    enc = gen.processor(text=text, images=[image], return_tensors="pt").to(gen.device)
    with torch.no_grad():
        out = gen.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    answer = gen.processor.batch_decode(
        out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return answer.strip()


def parse_box(text: str) -> list[float] | None:
    """A normalised box from the model's 0-99 output, if it emitted one."""
    m = _BOX.search(text)
    if not m:
        return None
    x0, y0, x1, y1 = (int(v) / 99.0 for v in m.groups())
    if x1 <= x0 or y1 <= y0:
        return None
    return [round(min(max(v, 0.0), 1.0), 4) for v in (x0, y0, x1, y1)]


class GenerativeTool(Tool):
    """Free-form VQA, captioning and referring grounding from one model."""

    spec = ToolSpec(
        name="rs_vlm",
        tasks={Task.VQA, Task.CAPTION, Task.GROUNDING},
        accepts={"kind", "max_new_tokens"},
        needs_images=1,
        description="Generative vision-language specialist fine-tuned on VRSBench.",
        requires=["torch", "transformers"],
    )

    def available(self) -> tuple[bool, str]:
        ok, why = super().available()
        if not ok:
            return ok, why
        return (True, "") if load() is not None else (False, "SmolVLM weights unavailable")

    def run(self, images: list[ImageMeta], query: str,
            params: dict[str, Any]) -> tuple[str, list[Evidence], float]:
        gen = load()
        if gen is None:
            return "[generative specialist unavailable]", [], 0.0

        kind = str(params.get("kind", "vqa"))
        answer = generate(gen, _open(images[0]), query, kind,
                          int(params.get("max_new_tokens", 64)))

        evidence: list[Evidence] = []
        box = parse_box(answer)
        if box is not None:
            evidence.append(Evidence("bbox", box, query[:60], None))

        # Untuned weights answer fluently and wrongly on overhead imagery, so
        # confidence is capped until the adapter is present rather than letting
        # fluency read as certainty.
        conf = 0.65 if gen.adapted else 0.3
        if not answer:
            conf = 0.1
        return answer, evidence, conf

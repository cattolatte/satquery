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
import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

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
    # Image 1 is the earlier date and image 2 the later one; saying so in the
    # prompt is what makes "increased" and "decreased" answerable at all.
    "change": "Image 1 is the earlier date and image 2 the later date. {q}\n"
              "Answer in as few words as possible.",
}

# VRSBench writes boxes as {<x0><y0><x1><y1>} on a 0-99 grid.
_BOX = re.compile(r"<(\d{1,2})><(\d{1,2})><(\d{1,2})><(\d{1,2})>")

# VRSBench's question taxonomy, defined here on the serving side and imported by
# the data preparation, so the two cannot drift the way the prompts once did.
_QTYPE = [
    ("object quantity", r"^\s*how many|number of"),
    ("object color", r"\bcolou?r\b"),
    ("object shape", r"\bshape\b"),
    ("object size", r"\bhow (large|big|small)\b|\bsize\b"),
    ("object direction", r"\bdirection\b|\bfacing\b|\boriented\b"),
    ("object position", r"\bwhere\b|\bposition\b|\blocated\b|\bside\b"),
    ("scene type", r"\bscene\b|\btype of (area|scene|land)\b|\bprimary\b"),
    ("rural or urban", r"\brural\b|\burban\b"),
    ("image", r"\bimage (quality|resolution|source|taken)\b|\bgrayscale\b"),
    ("object existence", r"^\s*(is|are|does|do)\b"),
]


def question_type(query: str) -> str:
    """Which of VRSBench's question families a query belongs to."""
    for name, pattern in _QTYPE:
        if re.search(pattern, query, re.I):
            return name
    return "other"


@functools.lru_cache(maxsize=1)
def answer_vocab() -> dict[str, list[str]]:
    """Candidate answers per question type, learned from the training split.

    Empty if the file is absent, which disables constrained decoding rather
    than failing -- a base model with no adapter has no vocabulary to constrain
    to, and free generation is the right fallback.
    """
    from pathlib import Path
    path = Path(ADAPTER) / "answer_vocab.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text()).get("vocab", {})
    except Exception:                                          # noqa: BLE001
        return {}


def score_candidates(gen: Generative, images, question: str,
                     candidates: list[str], kind: str = "vqa") -> list[tuple[str, float]]:
    """Rank candidate answers by the model's own likelihood, best first.

    Free generation is scored on exact string match, so an answer that is right
    but differently worded earns nothing -- object category was 28% exact
    against 50% when near-misses are allowed. Choosing among the answers the
    corpus actually uses converts that difference into score without changing
    what the model knows.

    Length-normalised, otherwise short answers win on token count alone.
    """
    import torch

    if not isinstance(images, (list, tuple)):
        images = [images]
    prompt = PROMPTS.get(kind, "{q}").format(q=question)
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt})
    text = gen.processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True)

    # The prompt's length must be measured AFTER the processor expands the
    # image placeholder into its token run. Tokenising the text alone
    # undercounts by that whole span, which masks the wrong positions and
    # scores image tokens instead of the answer -- every candidate then comes
    # back at roughly the same implausible logprob.
    prompt_len = gen.processor(
        text=[text], images=[list(images)], return_tensors="pt"
    )["input_ids"].shape[1]

    scored: list[tuple[str, float]] = []
    for start in range(0, len(candidates), 8):
        batch = candidates[start:start + 8]
        texts = [text + " " + c for c in batch]
        enc = gen.processor(text=texts, images=[list(images)] * len(batch),
                            return_tensors="pt", padding=True).to(gen.device)
        base = prompt_len
        with torch.no_grad():
            logits = gen.model(**enc).logits.float()
        logprobs = torch.log_softmax(logits[:, :-1], dim=-1)
        target = enc["input_ids"][:, 1:]
        token_lp = logprobs.gather(2, target.unsqueeze(-1)).squeeze(-1)
        mask = enc["attention_mask"][:, 1:].clone()
        mask[:, : max(base - 1, 0)] = 0            # score the answer only
        total = (token_lp * mask).sum(dim=1)
        length = mask.sum(dim=1).clamp(min=1)
        for c, lp in zip(batch, (total / length).tolist()):
            scored.append((c, lp))
    return sorted(scored, key=lambda kv: -kv[1])


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


def generate(gen: Generative, images, question: str, kind: str = "vqa",
             max_new_tokens: int = 64) -> str:
    """Answer a question about one image or a bi-temporal pair."""
    import torch

    if not isinstance(images, (list, tuple)):
        images = [images]
    prompt = PROMPTS.get(kind, "{q}").format(q=question)
    # One placeholder per image, in order: the processor aligns them
    # positionally, so a missing placeholder silently drops the second frame.
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = gen.processor.apply_chat_template(messages, add_generation_prompt=True)
    enc = gen.processor(text=text, images=list(images), return_tensors="pt").to(gen.device)
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
        tasks={Task.VQA, Task.CAPTION, Task.GROUNDING,
               Task.CHANGE_VQA, Task.CHANGE_DESCRIPTION},
        accepts={"kind", "max_new_tokens", "constrain"},
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

        kind = str(params.get("kind", "change" if len(images) > 1 else "vqa"))
        frames = [_open(m) for m in images[:2]]

        # Constrained decoding: choose among the answers the corpus uses
        # instead of generating freely. Off by default, because it measured
        # worse -- 52.3% against 55.2% on the same 1,200 questions, with object
        # shape down fourteen points. The premise was that the exact/lenient
        # gap was recoverable format loss; it is not. The model was fine-tuned
        # on this corpus and already generates in its vocabulary, so forcing a
        # choice from a fixed top-24 list trades a contextually right answer
        # that is outside the list for a frequent one inside it.
        #
        # Kept as a permitted parameter rather than deleted: the machinery is
        # sound and would help a model that had not been fine-tuned on the
        # answer distribution.
        vocab = answer_vocab() if params.get("constrain", False) else {}
        candidates = vocab.get(question_type(query), []) if kind == "vqa" else []
        if len(candidates) >= 2:
            ranked = score_candidates(gen, frames, query, candidates, kind)
            best, margin = ranked[0][0], ranked[0][1] - ranked[1][1]
            evidence = [Evidence("label", c, "candidate", float(lp))
                        for c, lp in ranked[:5]]
            # Margin between the top two, squashed: a near-tie among candidates
            # is a guess however confident the decoder looks.
            conf = float(np.clip(0.35 + 2.0 * margin, 0.15, 0.95))
            return best, evidence, conf

        answer = generate(gen, frames, query, kind,
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

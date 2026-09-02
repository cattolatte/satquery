"""Open-vocabulary object detection, and referring-expression resolution.

The problem statement invites this directly: "The system may use multiple
specialised components, such as a remote-sensing VQA or captioning model, a
grounding model, a change-understanding or change-VQA model, and an
optical-SAR fusion or information-extraction model." A detector is a specialist
in exactly that sense, and it is the component the measurements said was
missing.

Everything the backbone could not do turned out to be one capability. A global
image-text similarity has no notion of an instance, so it cannot count, cannot
say where a thing is, cannot compare two things' sizes, and cannot localise
anything smaller than a patch cell. Eight of VRSBench's twelve question types
and every counting question in RSVQA reduce to having instances with boxes.

Two stages, because a referring expression is not a detection query. OWLv2
finds *all* the vehicles; the expression names *one* of them -- "the large
yellow vehicle at the top-left". Ninety-five per cent of VRSBench's referring
expressions carry a spatial constraint, forty per cent a size constraint and
twelve per cent a colour constraint, so the second stage scores every candidate
against the constraints the sentence actually states.
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..schema import Evidence, ImageMeta, Task
from .base import Tool, ToolSpec
from .specialists import _open

MODEL = "google/owlv2-base-patch16-ensemble"

# OWLv2's text tower takes 16 tokens. Longer prompts do not error, they
# silently mismatch the position embeddings, so the cap is enforced here.
MAX_TEXT_TOKENS = 16

# Categories worth proposing when a question names no object explicitly.
RS_OBJECTS = [
    "vehicle", "car", "truck", "bus", "building", "house", "airplane", "ship",
    "boat", "storage tank", "tennis court", "basketball court", "swimming pool",
    "roundabout", "bridge", "road", "harbor", "stadium", "solar panel",
    "wind turbine", "train", "helicopter",
]

_SPATIAL = {
    "top": (None, 0.0), "upper": (None, 0.0), "north": (None, 0.0),
    "bottom": (None, 1.0), "lower": (None, 1.0), "south": (None, 1.0),
    "left": (0.0, None), "west": (0.0, None),
    "right": (1.0, None), "east": (1.0, None),
    "center": (0.5, 0.5), "centre": (0.5, 0.5), "middle": (0.5, 0.5),
}
_SUPERLATIVE = re.compile(
    r"\b(top-?most|bottom-?most|left-?most|right-?most|farthest|nearest|closest)\b", re.I)
_SIZE_BIG = re.compile(r"\b(large|larger|largest|big|bigger|biggest|long|longest|wide)\b", re.I)
_SIZE_SMALL = re.compile(r"\b(small|smaller|smallest|tiny|short|narrow)\b", re.I)

_COLOURS = {
    "red": (200, 60, 60), "blue": (60, 90, 200), "green": (70, 160, 80),
    "yellow": (220, 210, 70), "white": (235, 235, 235), "black": (35, 35, 35),
    "gray": (128, 128, 128), "grey": (128, 128, 128), "brown": (130, 90, 60),
    "orange": (230, 140, 50), "silver": (190, 190, 195),
    "dark": (55, 55, 55), "light": (215, 215, 215),
}

_HEAD = re.compile(
    r"\b(vehicles?|cars?|trucks?|buses|buildings?|houses?|planes?|airplanes?|"
    r"aircrafts?|ships?|boats?|tanks?|courts?|fields?|pools?|roundabouts?|"
    r"bridges?|roads?|trees?|harbou?rs?|stadiums?|storage tanks?|"
    r"tennis courts?|basketball courts?|swimming pools?)\b", re.I)


@dataclass
class Detector:
    model: object
    processor: object
    device: str


@functools.lru_cache(maxsize=1)
def load() -> Detector | None:
    """Load OWLv2, or None if it or torch is unavailable.

    None rather than raising, for the same reason the backbone does it: a
    missing optional component should cost coverage, never the whole query.
    """
    try:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
    except ImportError:
        return None
    try:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
        model = Owlv2ForObjectDetection.from_pretrained(MODEL).to(device).eval()
        return Detector(model=model, processor=Owlv2Processor.from_pretrained(MODEL),
                        device=device)
    except Exception:                                          # noqa: BLE001
        return None


def head_noun(text: str) -> str | None:
    """The object category a sentence is about, singularised."""
    m = _HEAD.search(text or "")
    if not m:
        return None
    word = m.group(0).lower()
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("es") and word[:-2].endswith(("sh", "ch", "s", "x")):
        return word[:-2]
    return word[:-1] if word.endswith("s") and not word.endswith("ss") else word


def detect(det: Detector, image, queries: list[str], threshold: float = 0.1) -> list[dict]:
    """Boxes for a list of text queries, in normalised coordinates."""
    import torch

    queries = [q[:120] for q in queries if q and q.strip()][:24]
    if not queries:
        return []
    with torch.no_grad():
        inputs = det.processor(text=[queries], images=image, return_tensors="pt",
                               padding="max_length", truncation=True,
                               max_length=MAX_TEXT_TOKENS).to(det.device)
        outputs = det.model(**inputs)
    w, h = image.size
    parsed = det.processor.post_process_grounded_object_detection(
        outputs, threshold=threshold,
        target_sizes=torch.tensor([[h, w]]).to(det.device))[0]

    out = []
    for box, score, label in zip(parsed["boxes"], parsed["scores"], parsed["labels"]):
        x0, y0, x1, y1 = (float(v) for v in box.tolist())
        out.append({
            "box": [max(0.0, x0 / w), max(0.0, y0 / h),
                    min(1.0, x1 / w), min(1.0, y1 / h)],
            "score": float(score),
            "label": queries[int(label)],
        })
    return sorted(out, key=lambda d: -d["score"])


def _spatial_target(text: str) -> tuple[float | None, float | None]:
    """Where in the frame the sentence points, as a normalised (x, y)."""
    tx = ty = None
    for word, (x, y) in _SPATIAL.items():
        if re.search(rf"\b{word}\b", text, re.I):
            if x is not None:
                tx = x if tx is None else (tx + x) / 2
            if y is not None:
                ty = y if ty is None else (ty + y) / 2
    return tx, ty


def _box_colour(image, box: list[float]) -> tuple[float, float, float]:
    """Mean RGB inside a normalised box."""
    w, h = image.size
    x0, y0 = int(box[0] * w), int(box[1] * h)
    x1, y1 = max(x0 + 1, int(box[2] * w)), max(y0 + 1, int(box[3] * h))
    crop = np.asarray(image.crop((x0, y0, x1, y1)).convert("RGB"), dtype=np.float32)
    return tuple(crop.reshape(-1, 3).mean(axis=0)) if crop.size else (0.0, 0.0, 0.0)


def resolve_referring(image, candidates: list[dict], phrase: str) -> list[dict]:
    """Rank detections by how well each satisfies the expression's constraints.

    Detection score alone answers "is there a vehicle", not "which vehicle".
    Since almost every referring expression here states a position, and many
    state a size or colour, those constraints carry most of the information
    about which instance is meant -- and the detector supplies none of it.

    Scores are combined additively with the detector's own confidence rather
    than multiplied, so one unmet constraint reweights a candidate instead of
    eliminating it: the sentence may describe an attribute the detector's box
    does not cover well.
    """
    if not candidates:
        return []

    tx, ty = _spatial_target(phrase)
    wants_big, wants_small = bool(_SIZE_BIG.search(phrase)), bool(_SIZE_SMALL.search(phrase))
    named = [c for c in _COLOURS if re.search(rf"\b{c}\b", phrase, re.I)]
    areas = np.array([max(1e-6, (c["box"][2] - c["box"][0]) * (c["box"][3] - c["box"][1]))
                      for c in candidates])
    rank = areas.argsort().argsort() / max(len(areas) - 1, 1)   # 0 smallest, 1 largest

    superlative = bool(_SUPERLATIVE.search(phrase))
    scored = []
    for i, c in enumerate(candidates):
        cx = (c["box"][0] + c["box"][2]) / 2
        cy = (c["box"][1] + c["box"][3]) / 2
        score = 1.2 * c["score"]

        if tx is not None or ty is not None:
            dx = abs(cx - tx) if tx is not None else 0.0
            dy = abs(cy - ty) if ty is not None else 0.0
            distance = (dx ** 2 + dy ** 2) ** 0.5
            # A superlative asks for the extreme, not merely the near side.
            score += (2.0 if superlative else 1.4) * (1.0 - min(distance, 1.0))

        if wants_big:
            score += 0.8 * rank[i]
        elif wants_small:
            score += 0.8 * (1.0 - rank[i])

        if named:
            r, g, b = _box_colour(image, c["box"])
            best = min(named, key=lambda n: sum(
                (a - t) ** 2 for a, t in zip((r, g, b), _COLOURS[n])) ** 0.5)
            distance = sum((a - t) ** 2 for a, t in zip((r, g, b), _COLOURS[best])) ** 0.5
            score += 0.6 * max(0.0, 1.0 - distance / 180.0)

        scored.append({**c, "referring_score": round(score, 4)})
    return sorted(scored, key=lambda c: -c["referring_score"])


def describe_position(box: list[float]) -> str:
    """Where a box sits, in the vocabulary these benchmarks use."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    vertical = "top" if cy < 0.38 else "bottom" if cy > 0.62 else "middle"
    horizontal = "left" if cx < 0.38 else "right" if cx > 0.62 else "center"
    if vertical == "middle" and horizontal == "center":
        return "center"
    return f"{vertical}-{horizontal}" if vertical != "middle" else horizontal


def describe_size(box: list[float]) -> str:
    area = (box[2] - box[0]) * (box[3] - box[1])
    return "large" if area > 0.06 else "small" if area < 0.01 else "medium"


def describe_shape(box: list[float]) -> str:
    w, h = box[2] - box[0], box[3] - box[1]
    ratio = w / max(h, 1e-6)
    if 0.85 <= ratio <= 1.18:
        return "square"
    return "rectangular" if 0.4 <= ratio <= 2.5 else "elongated"


def nearest_colour(rgb: tuple[float, float, float]) -> str:
    return min(_COLOURS, key=lambda n: sum(
        (a - t) ** 2 for a, t in zip(rgb, _COLOURS[n])) ** 0.5)


# Object-level question families. Each reduces to a property of the detected
# instances, which is why one detector answers all of them.
_Q_COUNT = re.compile(r"^\s*(how many|what is the (number|amount|count) of)", re.I)
_Q_EXIST = re.compile(r"^\s*(is|are|does|do|can)\b.*\b(there|present|visible|any)\b", re.I)
_Q_POSITION = re.compile(r"\b(where|which (part|side|corner)|position|located|situated)\b", re.I)
_Q_SIZE = re.compile(r"\b(how (large|big|small)|size of|larger|smaller)\b", re.I)
_Q_SHAPE = re.compile(r"\bshape\b", re.I)
_Q_COLOUR = re.compile(r"\b(colou?r)\b", re.I)


def question_family(query: str) -> str:
    """Which object-level property a question asks about."""
    if _Q_COUNT.search(query):
        return "count"
    if _Q_COLOUR.search(query):
        return "colour"
    if _Q_SHAPE.search(query):
        return "shape"
    if _Q_SIZE.search(query):
        return "size"
    if _Q_POSITION.search(query):
        return "position"
    if _Q_EXIST.search(query):
        return "existence"
    return "referring"


# Some nouns name a discrete object; others name a surface that happens to have
# a noun. "Is there a road?" is a land-cover question however countable "road"
# sounds, and routing it to a detector measurably hurt: presence accuracy on
# scene-level questions fell from 50.6% to 38.1% when these were included.
_DISCRETE = {
    "vehicle", "car", "truck", "bus", "plane", "airplane", "aircraft", "ship",
    "boat", "tank", "storage tank", "roundabout", "bridge", "court",
    "tennis court", "basketball court", "pool", "swimming pool", "stadium",
}


# An attribute is the surest sign a question is about one instance rather than
# a class. "Is a circular building present?" asks about a particular building;
# "Are there buildings?" asks whether the surface exists at all, and routing
# that to a detector cost 12 points of presence accuracy.
_ATTRIBUTE = re.compile(
    r"\b(top|bottom|left|right|upper|lower|middle|centre|center|corner|"
    r"large|larger|largest|small|smaller|smallest|big|tiny|long|short|wide|"
    r"narrow|medium|round|circular|square|rectangular|oval|triangular|"
    r"red|blue|green|yellow|white|black|gray|grey|dark|light|brown|orange|"
    r"silver|nearest|farthest|closest|leftmost|rightmost)\b", re.I)


def is_object_level(query: str) -> bool:
    """Whether a question is about instances rather than the scene as a whole.

    The two backbones answer disjoint question sets, so this decides which one
    the controller dispatches to. Three signals, and the subject noun alone is
    not one of them -- "road" and "building" name discrete things in one corpus
    and ground cover in another, so the noun cannot settle it.

      - Counting and shape have no scene-level reading: a land-cover class has
        no count and no aspect ratio.
      - A discrete object -- a ship, a plane, a roundabout -- is an instance
        whatever is asked about it.
    An attribute test was tried as a third signal -- "a circular building" picks
    out one instance where "buildings" does not -- and measured worse: it is a
    superset of this routing, and the extra questions it sent to the detector
    cost 2.1 points on RSVQA to gain 0.2 on VRSBench. Kept out on that evidence,
    not on taste.
    """
    family = question_family(query)
    if family in ("count", "shape"):
        return True
    noun = head_noun(query)
    if noun is None:
        return False
    return noun in _DISCRETE


class DetectionTool(Tool):
    """Instance-level perception: detection, counting, and referring grounding."""

    spec = ToolSpec(
        name="rs_detect",
        tasks={Task.GROUNDING, Task.VQA},
        accepts={"queries", "threshold", "top_k", "phrase",
                 "existence_threshold", "count_threshold"},
        needs_images=1,
        description="Open-vocabulary detection with referring-expression resolution.",
        requires=["torch", "transformers"],
    )

    def available(self) -> tuple[bool, str]:
        ok, why = super().available()
        if not ok:
            return ok, why
        return (True, "") if load() is not None else (False, "OWLv2 weights unavailable")

    def run(self, images: list[ImageMeta], query: str,
            params: dict[str, Any]) -> tuple[str, list[Evidence], float]:
        det = load()
        if det is None:
            return "[detector unavailable]", [], 0.0
        image = _open(images[0])

        phrase = str(params.get("phrase") or query)
        noun = head_noun(phrase)
        queries = params.get("queries") or ([noun] if noun else RS_OBJECTS)

        # Existence questions need a stricter threshold than localisation does.
        # Localising wants recall -- a weak box is still a box to rank. Saying
        # "yes, there is one" on a weak detection is a false claim, and at 0.10
        # it made presence accuracy *worse* than the scene-level backbone it
        # replaced. Tuned on the first half of RSVQA's presence questions and
        # reported on the second: 0.10 gives 46%, 0.40 gives 60%.
        #
        # Stated plainly: 60% is also the majority baseline there, so this stops
        # the harm rather than adding signal. The detector is not discriminative
        # on RSVQA's presence questions.
        # Localisation wants recall and existence wants precision, so they get
        # different thresholds rather than a shared compromise that serves
        # neither. Swept on VRSBench referring: 0.03 gives 20.0% Acc@0.5 with 37
        # empty results, 0.10 gives 17.3% with 78 -- a weak box still ranks, and
        # discarding it only guarantees a miss.
        family_now = question_family(query)
        threshold = float(params.get("threshold", 0.03))
        if family_now == "existence":
            threshold = float(params.get("existence_threshold", 0.4))
        elif family_now == "count":
            # Counting is the strictest of the three. Localisation is scored on
            # the best box, so a spurious extra costs nothing; a count is scored
            # exactly, so every false positive is an error. Tuned on the first
            # half of RSVQA's counting questions and reported on the second:
            # 0.03 gives 2.5%, 0.10 gives 12.5%, 0.40 gives 23.8%.
            threshold = float(params.get("count_threshold", 0.4))

        candidates = detect(det, image, list(queries), threshold)
        if not candidates:
            family = question_family(query)
            if family == "count":
                return "0 — no instances detected.", [], 0.4
            if family == "existence":
                return (f"No — no {noun or 'matching object'} detected above "
                        f"{threshold:.2f}."), [], 0.4
            return (f"No instances of {', '.join(list(queries)[:3])} detected "
                    f"above {threshold:.2f}."), [], 0.15

        ranked = resolve_referring(image, candidates, phrase)
        top_k = int(params.get("top_k", 5))
        best = ranked[0]

        evidence = [Evidence("bbox", c["box"], c["label"], c["score"])
                    for c in ranked[:top_k]]
        counts: dict[str, int] = {}
        for c in candidates:
            counts[c["label"]] = counts.get(c["label"], 0) + 1
        summary = ", ".join(f"{n} {label}" for label, n in
                            sorted(counts.items(), key=lambda kv: -kv[1])[:4])

        # Answer the property the question actually asks about. Leading with the
        # bare answer token matters: these benchmarks score the first word, and
        # burying "3" inside a sentence scores wrong however right it is.
        family = question_family(query)
        if family == "count":
            n = len(candidates) if not noun else sum(
                1 for c in candidates if c["label"] == noun)
            return (f"{n} — counted {n} instance(s) of {noun or 'object'} above "
                    f"the {threshold:.2f} detection threshold.",
                    evidence, float(np.clip(0.3 + 0.5 * best["score"], 0.15, 0.9)))
        if family == "existence":
            return (f"Yes — detected {summary}.", evidence,
                    float(np.clip(0.3 + 0.6 * best["score"], 0.15, 0.95)))
        if family == "position":
            where = describe_position(best["box"])
            return (f"{where} — {best['label']} at {where}, box "
                    f"{[round(v, 3) for v in best['box']]}.", evidence,
                    float(np.clip(0.3 + 0.5 * best["score"], 0.15, 0.9)))
        if family == "size":
            return (f"{describe_size(best['box'])} — {best['label']} occupying "
                    f"{(best['box'][2]-best['box'][0])*(best['box'][3]-best['box'][1]):.1%} "
                    "of the frame.", evidence,
                    float(np.clip(0.25 + 0.5 * best["score"], 0.15, 0.85)))
        if family == "shape":
            return (f"{describe_shape(best['box'])} — from the bounding box "
                    "aspect ratio.", evidence,
                    float(np.clip(0.2 + 0.4 * best["score"], 0.15, 0.8)))
        if family == "colour":
            colour = nearest_colour(_box_colour(image, best["box"]))
            return (f"{colour} — mean colour inside the {best['label']} box.",
                    evidence, float(np.clip(0.2 + 0.4 * best["score"], 0.15, 0.8)))

        text = (f"Detected {summary}. Best match for '{phrase[:60]}': "
                f"{best['label']} at {describe_position(best['box'])}, "
                f"{describe_size(best['box'])}, box {[round(v, 3) for v in best['box']]}.")

        # Confidence tracks how clearly one candidate won, not the detector's
        # raw score: a confident detection of the wrong instance is still wrong.
        if len(ranked) > 1:
            margin = ranked[0]["referring_score"] - ranked[1]["referring_score"]
            conf = float(np.clip(0.35 + margin, 0.15, 0.95))
        else:
            conf = float(np.clip(best["score"] + 0.2, 0.15, 0.95))
        return text, evidence, conf

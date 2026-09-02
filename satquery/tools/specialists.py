"""
The specialist tools.

One backbone, several small heads. The problem statement asks for "specialist
tools for VQA, captioning or grounding, change understanding, and optical-SAR
analysis", and for the controller to pick between them — which only works if
each is genuinely separable and declares what it does.

Every tool degrades rather than raises. A missing backbone makes a tool
unavailable and the controller routes around it; a failure inside one is
captured in its ToolCall and the chain continues.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import Evidence, ImageMeta, Task
from .backbone import Backbone, embed_images, embed_texts, load, patch_tokens
from .base import Tool, ToolSpec

# CORINE-derived vocabulary. BigEarthNet is labelled with these, so they are the
# classes the adapted backbone has actually seen described in text.
LAND_COVER = [
    "urban fabric", "industrial or commercial units", "arable land", "permanent crops",
    "pastures", "complex cultivation patterns", "broad-leaved forest",
    "coniferous forest", "mixed forest", "natural grassland", "moors and heathland",
    "transitional woodland shrub", "beaches dunes sands", "bare rock",
    "sparsely vegetated areas", "inland wetlands", "coastal wetlands",
    "inland waters", "marine waters", "airport runways", "road network",
]


def _open(meta: ImageMeta):
    """Load an image as RGB for the backbone.

    Sentinel-2 GeoTIFFs carry 12-13 bands; CLIP wants three. Bands 4/3/2 are
    red/green/blue in Sentinel-2 order, so the natural-colour composite is the
    right default — it is also what the text descriptions were written against.
    """
    from PIL import Image
    p = Path(meta.path)
    if meta.fmt == "GeoTIFF":
        import rasterio
        with rasterio.open(p) as src:
            if src.count >= 4:
                arr = np.stack([src.read(4), src.read(3), src.read(2)], axis=-1)
            elif src.count >= 3:
                arr = np.stack([src.read(1), src.read(2), src.read(3)], axis=-1)
            else:
                band = src.read(1)
                arr = np.stack([band] * 3, axis=-1)            # SAR: grey to RGB
        # Percentile stretch. Satellite reflectance has a long tail and a raw
        # min-max stretch leaves everything dark grey.
        lo, hi = np.percentile(arr, [2, 98])
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))
    return Image.open(p).convert("RGB")


# Everyday words map onto CORINE class names that never contain them: nobody
# asks "is there inland waters in this image". Without this, every binary
# question falls through to the land-cover listing and answers nothing.
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "water": ("inland waters", "marine waters"),
    "sea": ("marine waters",), "ocean": ("marine waters",),
    "lake": ("inland waters",), "river": ("inland waters",),
    "forest": ("broad-leaved forest", "coniferous forest", "mixed forest"),
    "trees": ("broad-leaved forest", "coniferous forest", "mixed forest"),
    "woodland": ("transitional woodland shrub", "mixed forest"),
    "urban": ("urban fabric",), "city": ("urban fabric",),
    "buildings": ("urban fabric", "industrial or commercial units"),
    "building": ("urban fabric", "industrial or commercial units"),
    "residential": ("urban fabric",), "commercial": ("industrial or commercial units",),
    "housing": ("urban fabric",), "settlement": ("urban fabric",),
    "industry": ("industrial or commercial units",),
    "factory": ("industrial or commercial units",),
    "crops": ("arable land", "permanent crops", "complex cultivation patterns"),
    "farmland": ("arable land", "complex cultivation patterns"),
    "agriculture": ("arable land", "permanent crops", "complex cultivation patterns"),
    "farming": ("arable land", "complex cultivation patterns"),
    "fields": ("arable land", "complex cultivation patterns"),
    "vineyard": ("permanent crops",), "orchard": ("permanent crops",),
    "grass": ("pastures", "natural grassland"),
    "grassland": ("natural grassland",), "pasture": ("pastures",),
    "meadow": ("pastures", "natural grassland"),
    "wetland": ("inland wetlands", "coastal wetlands"),
    "marsh": ("inland wetlands",), "bog": ("inland wetlands",),
    "beach": ("beaches dunes sands",), "sand": ("beaches dunes sands",),
    "dunes": ("beaches dunes sands",), "coast": ("coastal wetlands", "marine waters"),
    "rock": ("bare rock",), "bare": ("bare rock", "sparsely vegetated areas"),
    "road": ("road network",), "roads": ("road network",),
    "highway": ("road network",), "airport": ("airport runways",),
    "runway": ("airport runways",), "airfield": ("airport runways",),
    "shrub": ("transitional woodland shrub", "moors and heathland"),
    "heath": ("moors and heathland",), "moor": ("moors and heathland",),
    "vegetation": ("natural grassland", "sparsely vegetated areas"),
}

_POLAR = ("is ", "are ", "does ", "do ", "would ", "has ", "have ", "can ",
          "was ", "were ", "any ", "did ")

# "Is it a rural or an urban area" is not a yes/no question despite its opener:
# the answer is one of the two alternatives it names. Scoring those directly is
# both more accurate and more honest than answering "yes".
_EITHER_OR = re.compile(
    r"\b(?:is|are|was|were)\s+(?:it|this|the\s+\w+|there)?\s*"
    r"an?\s+(\w[\w\s-]*?)\s+or\s+an?\s+(\w[\w\s-]*?)\s*(?:area|region|scene|zone)?\s*\??$",
    re.I)


def _either_or(query: str) -> list[str] | None:
    """The two alternatives an either/or question offers, if it is one."""
    m = _EITHER_OR.search(query.strip())
    if not m:
        return None
    options = [re.sub(r"\s+", " ", g).strip().lower() for g in m.groups()]
    options = [o for o in options if o and len(o) < 40]
    return options if len(options) == 2 and options[0] != options[1] else None


def _resolve_target(query: str, vocab: list[str]) -> list[str]:
    """Which vocabulary classes a question is asking about.

    Three routes, most specific first: the class name appears verbatim, a
    synonym maps to it, or a content word of the class appears. Returning a
    list rather than one class matters -- "is there forest" is a question about
    three CORINE classes at once.
    """
    q = f" {query.lower()} "
    hits: list[str] = []

    for v in vocab:                                   # verbatim class name
        if f" {v.lower()} " in q or v.lower() in q:
            hits.append(v)
    for word, classes in _SYNONYMS.items():           # everyday synonym
        if re.search(rf"\b{re.escape(word)}\b", q):
            hits += [c for c in classes if c in vocab]
    if not hits:                                      # any content word
        for v in vocab:
            for w in v.lower().split():
                if len(w) > 3 and re.search(rf"\b{re.escape(w)}\b", q):
                    hits.append(v)
                    break

    seen: set[str] = set()
    return [h for h in hits if not (h in seen or seen.add(h))]


def _score_vocab(bb: Backbone, image, vocab: list[str]) -> list[tuple[str, float]]:
    """Rank a vocabulary against one image by cosine similarity."""
    img = embed_images(bb, [image])[0]
    txt = embed_texts(bb, [f"a satellite image of {v}" for v in vocab])
    sims = txt @ img
    order = np.argsort(-sims)
    return [(vocab[i], float(sims[i])) for i in order]


def _softmax_conf(scores: np.ndarray, temp: float = 100.0) -> float:
    """Confidence as the softmax margin of the top choice.

    Raw cosine similarity is a poor confidence signal: CLIP similarities sit in
    a narrow band and an absolute value of 0.28 means nothing on its own. How
    far the best option is ahead of the rest is the informative quantity.
    """
    e = np.exp(temp * (scores - scores.max()))
    p = e / e.sum()
    return float(p.max())


class _BackboneTool(Tool):
    """Shared loading so the model is read once across every specialist."""

    def _bb(self) -> Backbone | None:
        return load()

    def available(self) -> tuple[bool, str]:
        ok, why = super().available()
        if not ok:
            return ok, why
        return (True, "") if self._bb() else (False, "backbone unavailable")


class VQATool(_BackboneTool):
    spec = ToolSpec(
        name="rs_vqa",
        tasks={Task.VQA},
        accepts={"vocab", "top_k", "temperature"},
        needs_images=1,
        description="Remote-sensing visual question answering over a single image.",
        requires=["torch", "transformers"],
    )

    def run(self, images, query, params):
        bb = self._bb()
        image = _open(images[0])
        vocab = params.get("vocab") or LAND_COVER
        top_k = int(params.get("top_k", 5))

        ranked = _score_vocab(bb, image, vocab)
        scores = np.array([s for _, s in ranked])
        conf = _softmax_conf(scores, float(params.get("temperature", 100.0)))
        top = ranked[:top_k]

        # Either/or questions name their own answer space, so score exactly
        # that rather than falling through to the land-cover vocabulary.
        options = _either_or(query)
        if options:
            noun = "area"
            ranked_opt = _score_vocab(bb, image, [f"{o} {noun}" for o in options])
            opt_scores = np.array([s for _, s in ranked_opt])
            best = ranked_opt[0][0].rsplit(" ", 1)[0]
            text = (f"{best.capitalize()} — scored "
                    + " vs ".join(f"{v.rsplit(' ',1)[0]} {s:.3f}" for v, s in ranked_opt)
                    + ".")
            return (text, [Evidence("label", v.rsplit(" ", 1)[0], v, s) for v, s in ranked_opt],
                    _softmax_conf(opt_scores))

        # Binary questions are the dominant type in BigEarthNet.txt, and they
        # are answerable from whether the referenced classes rank for the scene.
        targets = _resolve_target(query, vocab)
        polar = query.strip().lower().startswith(_POLAR)
        if targets and polar:
            order = [v for v, _ in ranked]
            sims = dict(ranked)
            best = min(targets, key=order.index)
            rank = order.index(best)

            # Softmax mass over the asked-about classes, against the mass a
            # uniform model would give them. Rank alone cannot distinguish
            # "clearly present" from "least unlikely of twenty-one".
            probs = np.exp(100.0 * (scores - scores.max()))
            probs = probs / probs.sum()
            mass = float(sum(probs[order.index(t)] for t in targets))
            prior = len(targets) / len(vocab)
            yes = rank < 3 and mass > prior

            named = best if len(targets) == 1 else f"{best} (of {len(targets)} matching classes)"
            text = (f"{'Yes' if yes else 'No'} — '{named}' ranks {rank + 1} of {len(vocab)} "
                    f"for this scene (similarity {sims[best]:.3f}, "
                    f"probability mass {mass:.2f} against a {prior:.2f} prior).")
            decision_conf = float(np.clip(mass if yes else 1.0 - mass, 0.05, 0.99))
            ev = [Evidence("label", t, t, float(sims[t])) for t in targets]
            return text, ev, decision_conf

        text = ("Most likely land cover: "
                + ", ".join(f"{v} ({s:.3f})" for v, s in top))
        ev = [Evidence("label", v, v, s) for v, s in top]
        return text, ev, conf


class CaptionTool(_BackboneTool):
    spec = ToolSpec(
        name="rs_caption",
        tasks={Task.CAPTION},
        accepts={"vocab", "max_classes", "threshold"},
        needs_images=1,
        description="Scene description from ranked remote-sensing land-cover classes.",
        requires=["torch", "transformers"],
    )

    def run(self, images, query, params):
        bb = self._bb()
        image = _open(images[0])
        vocab = params.get("vocab") or LAND_COVER
        ranked = _score_vocab(bb, image, vocab)
        scores = np.array([s for _, s in ranked])

        # Keep classes within a margin of the best rather than a fixed count:
        # a scene that is unambiguously one thing should not be described as
        # five, and a mixed scene should not be truncated to three.
        cutoff = scores.max() - float(params.get("threshold", 0.02))
        keep = [(v, s) for v, s in ranked if s >= cutoff][: int(params.get("max_classes", 5))]

        meta = images[0]
        where = f" over {meta.width}x{meta.height} px" if meta.width else ""
        modality = meta.modality.value
        article = "An" if modality[:1].lower() in "aeiou" else "A"
        text = (f"{article} {modality} remote-sensing scene{where} showing "
                + ", ".join(v for v, _ in keep) + ".")
        return text, [Evidence("label", v, v, s) for v, s in keep], _softmax_conf(scores)


class GroundingTool(_BackboneTool):
    spec = ToolSpec(
        name="rs_grounding",
        tasks={Task.GROUNDING},
        accepts={"threshold", "top_k", "min_area"},
        needs_images=1,
        description="Text-guided region localisation from dense patch-text similarity.",
        requires=["torch", "transformers"],
    )

    def run(self, images, query, params):
        bb = self._bb()
        image = _open(images[0])
        tokens, grid = patch_tokens(bb, image)
        phrase = _referring_phrase(query)
        txt = embed_texts(bb, [f"a satellite image of {phrase}"])[0]

        raw = (tokens @ txt).reshape(grid, grid)
        # Min-max only to pick regions. It must never feed the confidence: the
        # normalised maximum is 1.0 by construction, so reporting it would make
        # every successful localisation look certain regardless of the evidence.
        heat = (raw - raw.min()) / max(np.ptp(raw), 1e-6)
        thr = float(params.get("threshold", 0.6))
        boxes = _boxes_from_heat(heat, thr, int(params.get("min_area", 2)))
        boxes = boxes[: int(params.get("top_k", 3))]

        if not boxes:
            return (f"No region matching '{phrase}' passed the {thr:.2f} threshold.",
                    [Evidence("heatmap", heat.tolist(), phrase)], 0.2)

        ev = [Evidence("bbox", b["box"], phrase, b["score"]) for b in boxes]
        ev.append(Evidence("heatmap", heat.tolist(), phrase))
        text = (f"Localised '{phrase}' to {len(boxes)} region(s); "
                f"strongest at {boxes[0]['box']} (normalised coordinates).")
        return text, ev, _grounding_confidence(raw, heat >= thr)


def _referring_phrase(query: str) -> str:
    """Pull the thing being asked about out of a grounding query.

    BigEarthNet.txt marks these explicitly with <ref>...</ref>, so honour that
    when present and fall back to stripping the imperative otherwise.
    """
    import re
    m = re.search(r"<ref>(.*?)</ref>", query, re.S)
    if m:
        return m.group(1).strip()
    q = re.sub(r"^\s*(please\s+)?(highlight|locate|find|show me|point to|mark|outline|"
               r"segment|identify|where (is|are))\s*", "", query, flags=re.I)
    # "the location of the runway in this image" -> "runway". Feeding the whole
    # imperative to the text encoder buries the noun among filler tokens and
    # measurably weakens the heat map.
    q = re.sub(r"^\s*(the\s+)?(location|position|extent|area|region)s?\s+of\s+", "", q, flags=re.I)
    q = re.sub(r"\s+(in|on|within|from)\s+(this|the)\s+(image|scene|picture|photo)\b.*$", "", q, flags=re.I)
    # "the water body referred to in the query" -> "water body". The statement
    # phrases one of its representative queries exactly this way.
    q = re.sub(r"\s+(referred to|mentioned|described|asked about)\b.*$", "", q, flags=re.I)
    q = re.sub(r"^\s*(the|a|an|any|all)\s+", "", q, flags=re.I)
    return re.sub(r"[.?!]+$", "", q).strip() or query


def _grounding_confidence(raw: np.ndarray, selected: np.ndarray) -> float:
    """How much the selected patches actually stand out, in [0, 1].

    Two independent ways a localisation can be worthless, so both are penalised:

      - No separation. If the selected patches score barely above the rest, the
        model has not found anything; the z-score of their mean carries this.
      - No localisation. A mask covering most of the scene is not an answer to
        "where is it", however peaked the similarity looks, so confidence falls
        away as coverage grows past half the image.
    """
    if not selected.any() or selected.all():
        return 0.15
    spread = float(raw.std())
    if spread < 1e-6:
        return 0.15
    z = (float(raw[selected].mean()) - float(raw.mean())) / spread
    separation = float(np.clip(z / 2.0, 0.0, 1.0))     # 2 sigma reads as certain
    coverage = float(selected.mean())
    focus = float(np.clip((1.0 - coverage) / 0.5, 0.0, 1.0))
    return round(float(np.clip(separation * focus, 0.05, 0.99)), 3)


def _boxes_from_heat(heat: np.ndarray, thr: float, min_area: int) -> list[dict]:
    """Connected components above threshold, as normalised boxes.

    Flood fill rather than one box per cell: a dozen adjacent 1/7th-of-image
    rectangles over one lake is both wrong to look at and useless as evidence.
    """
    g = heat.shape[0]
    on = heat >= thr
    seen = np.zeros_like(on)
    out: list[dict] = []
    for y in range(g):
        for x in range(g):
            if not on[y, x] or seen[y, x]:
                continue
            stack, cells = [(y, x)], []
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)):
                    if 0 <= ny < g and 0 <= nx < g and on[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(cells) < min_area:
                continue
            ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
            out.append({
                "box": [round(min(xs) / g, 3), round(min(ys) / g, 3),
                        round((max(xs) + 1) / g, 3), round((max(ys) + 1) / g, 3)],
                "score": float(heat[min(ys):max(ys)+1, min(xs):max(xs)+1].max()),
                "cells": len(cells),
            })
    return sorted(out, key=lambda b: -b["score"])

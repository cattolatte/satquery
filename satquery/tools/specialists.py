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

import numpy as np

from ..schema import Evidence, ImageMeta, Modality, Task
from .backbone import (
    Backbone, embed_images, embed_texts, embed_texts_cached, load, patch_tokens,
)
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


def _stretch(channel: np.ndarray) -> np.ndarray:
    """Percentile stretch one channel to [0, 1].

    Per channel, not over the stack. Satellite reflectance has a long tail, and
    for SAR the polarisation channels differ in dynamic range by enough that a
    shared stretch flattens one of them into noise.
    """
    channel = np.asarray(channel, dtype=np.float32)
    mask = np.isfinite(channel)
    if not mask.any():
        return np.zeros_like(channel, dtype=np.float32)
    lo, hi = np.percentile(channel[mask], [2, 98])
    # Non-finite pixels (SAR no-data, division artefacts) are floored rather
    # than propagated: a single NaN would otherwise poison the whole embedding.
    out = np.where(mask, channel, lo)
    return np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1)


def _sar_composite(bands: list[np.ndarray]) -> np.ndarray:
    """Render SAR as three channels the way SAR is conventionally read.

    Dual-pol products carry VV and VH, which describe different scattering:
    surfaces and double bounce in VV, volume scattering in VH. Reading only the
    first band -- as this did -- discarded VH entirely, which is most of what
    distinguishes vegetation from built-up in radar.

    The conventional false-colour composite is (VV, VH, VV/VH); the ratio is
    what separates urban from vegetated. Amplitude is also strongly
    right-skewed, so it goes to dB before stretching rather than after.
    """
    def db(x: np.ndarray) -> np.ndarray:
        """To decibels, unless the product is already in decibels.

        BigEarthNet's Sentinel-1 patches ship as dB (roughly -35..0), while raw
        GRD amplitude is non-negative. Taking the log of dB values clamps every
        negative sample to the floor and flattens the channel to a constant --
        which is exactly what was happening: SAR scored at chance not because
        the encoder was unadapted but because the input had been destroyed.
        """
        x = np.asarray(x, dtype=np.float32)
        finite = x[np.isfinite(x)]
        if finite.size and finite.min() < 0:
            return x                                   # already logarithmic
        return 10.0 * np.log10(np.maximum(x, 1e-6))

    if len(bands) >= 2:
        vv, vh = db(bands[0]), db(bands[1])
        ratio = vv - vh                        # a dB difference is the ratio
        return np.stack([_stretch(vv), _stretch(vh), _stretch(ratio)], axis=-1)
    grey = _stretch(db(bands[0]))
    return np.stack([grey] * 3, axis=-1)


def _open(meta: ImageMeta):
    """Load an image as three channels for the backbone.

    Optical: bands 4/3/2 are red/green/blue in Sentinel-2 order, so the
    natural-colour composite is the right default -- it is also what the text
    descriptions were written against.

    SAR: natural colour is meaningless, so a polarimetric composite is built
    instead. See `_sar_composite`.
    """
    from PIL import Image
    p = Path(meta.path)
    if meta.fmt not in ("GeoTIFF", "TIFF"):
        return Image.open(p).convert("RGB")

    import rasterio
    with rasterio.open(p) as src:
        bands = [src.read(i + 1) for i in range(src.count)]

    if meta.modality is Modality.SAR or (len(bands) <= 2 and meta.modality is not Modality.OPTICAL):
        arr = _sar_composite(bands)
    elif len(bands) >= 4:
        # Which indices carry red, green and blue depends on how the product is
        # packed. A full Sentinel-2 product starts at B01, so B04/B03/B02 are
        # indices 3/2/1; BigEarthNet drops B01 and starts at B02, putting them
        # at 2/1/0. Reading the wrong three bands silently returns a plausible
        # false-colour image, so the count decides rather than an assumption.
        r, g, b = (2, 1, 0) if len(bands) in (10, 11) else (3, 2, 1)
        arr = np.stack([_stretch(bands[r]), _stretch(bands[g]), _stretch(bands[b])], axis=-1)
    elif len(bands) == 3:
        arr = np.stack([_stretch(b) for b in bands], axis=-1)
    else:
        arr = np.stack([_stretch(bands[0])] * 3, axis=-1)

    return Image.fromarray((arr * 255).astype(np.uint8))


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
    r"\b(?:is|are|was|were|does|do|did|show|shows|depicts?)\b[^?]*?"
    # The second article is optional: "a rural or urban area" is at least as
    # common as "a rural or an urban area", and the trailing head noun belongs
    # to both alternatives rather than to the second one.
    r"\ban?\s+(\w[\w\s-]*?)\s+or\s+(?:an?\s+)?(\w[\w\s-]*?)\s*"
    r"(?:area|region|scene|zone|setting|image|photo)?\s*\??$", re.I)


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
        # Allow a plural: the map is keyed on singulars, and questions ask
        # about "farmlands" and "buildings" as readily as the singular. Without
        # this, "are there less farmlands than water areas" resolved only one
        # side of the comparison and fell through entirely.
        if re.search(rf"\b{re.escape(word)}e?s?\b", q):
            hits += [c for c in classes if c in vocab]
    if not hits:                                      # any content word
        for v in vocab:
            for w in v.lower().split():
                if len(w) > 3 and re.search(rf"\b{re.escape(w)}\b", q):
                    hits.append(v)
                    break

    seen: set[str] = set()
    return [h for h in hits if not (h in seen or seen.add(h))]


# Comparing the extent of two classes ("are there more farmlands than water
# areas") was implemented and removed. Assigning patches to classes and
# comparing the counts is the right model of the question, and it measured
# 39.3% over the full vocabulary and 42.9% restricted to the two classes named,
# against a 55.4% majority baseline -- worse than guessing, in both
# formulations. Inverting the comparison would have scored 59%, which is a
# quirk of how this vocabulary assigns built-up patches rather than a model of
# anything, so it was not adopted. See docs/adr/0007.


def _score_vocab(bb: Backbone, image, vocab: list[str]) -> list[tuple[str, float]]:
    """Rank a vocabulary against one image by cosine similarity."""
    img = embed_images(bb, [image])[0]
    txt = embed_texts_cached(bb, [f"a satellite image of {v}" for v in vocab])
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
        accepts={"threshold", "top_k", "min_area", "tiles"},
        needs_images=1,
        description="Text-guided region localisation from dense patch-text similarity.",
        requires=["torch", "transformers"],
    )

    def run(self, images, query, params):
        bb = self._bb()
        image = _open(images[0])
        phrase = _referring_phrase(query)
        txt = embed_texts(bb, [f"a satellite image of {phrase}"])[0]

        # Tiling raises the localisation resolution. One pass gives a 7x7 grid,
        # which over a 512 px image is 73 px per cell -- larger than many of the
        # objects a referring expression names, so the smallest expressible box
        # is bigger than the target. Running the encoder over a T x T grid of
        # crops and stitching the maps gives 7T x 7T cells for T^2 passes.
        tiles = max(1, int(params.get("tiles", 1)))
        if tiles > 1:
            raw = _tiled_heat(bb, image, txt, tiles)
        else:
            tokens, grid = patch_tokens(bb, image)
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
    # The imperative is not always at the start: "Use the optical and SAR
    # images together to identify built-up regions" buries it mid-sentence, and
    # feeding the whole clause to the text encoder grounds on the instruction
    # rather than on the thing being asked for.
    q = re.sub(r"^.*?\bto\s+(?:identify|locate|find|highlight|show|detect|mark)\s+",
               "", query, flags=re.I)
    q = re.sub(r"^\s*(please\s+)?(highlight|locate|find|show me|point to|mark|outline|"
               r"segment|identify|detect|where (is|are))\s*", "", q, flags=re.I)
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


def _tiled_heat(bb: Backbone, image, txt: np.ndarray, tiles: int) -> np.ndarray:
    """Similarity map stitched from a T x T grid of overlapping crops.

    Overlapping by half a tile so an object straddling a tile boundary is whole
    in at least one crop; the maximum is taken where crops overlap, since a
    target seen clearly in one crop should not be diluted by a crop that only
    caught its edge.
    """
    w, h = image.size
    step_x, step_y = w / tiles, h / tiles
    # Half-tile margin, clamped to the image.
    mx, my = step_x / 2, step_y / 2

    acc: np.ndarray | None = None
    counts: np.ndarray | None = None
    for ty in range(tiles):
        for tx in range(tiles):
            left = max(0, int(tx * step_x - mx))
            top = max(0, int(ty * step_y - my))
            right = min(w, int((tx + 1) * step_x + mx))
            bottom = min(h, int((ty + 1) * step_y + my))
            crop = image.crop((left, top, right, bottom))
            tokens, grid = patch_tokens(bb, crop)
            local = (tokens @ txt).reshape(grid, grid)

            if acc is None:
                size = grid * tiles
                acc = np.full((size, size), -np.inf, dtype=np.float32)
                counts = np.zeros((size, size), dtype=np.float32)
            size = acc.shape[0]
            y0 = int(round(top / h * size)); y1 = max(y0 + 1, int(round(bottom / h * size)))
            x0 = int(round(left / w * size)); x1 = max(x0 + 1, int(round(right / w * size)))

            # Nearest-neighbour resize of the local map onto its footprint.
            yi = np.clip((np.arange(y1 - y0) * grid) // max(y1 - y0, 1), 0, grid - 1)
            xi = np.clip((np.arange(x1 - x0) * grid) // max(x1 - x0, 1), 0, grid - 1)
            patch = local[np.ix_(yi, xi)]
            acc[y0:y1, x0:x1] = np.maximum(acc[y0:y1, x0:x1], patch)
            counts[y0:y1, x0:x1] += 1

    acc[~np.isfinite(acc)] = float(np.nanmin(acc[np.isfinite(acc)])) if np.isfinite(acc).any() else 0.0
    return acc


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

"""
The two-image specialists: change analysis and optical-SAR fusion.

Both are mandatory:

    "Multi-image change analysis: Change description or change-based visual
     question answering from a bi-temporal image pair shall be mandatory."

    "Cross-modal pair analysis: The system must extract complementary
     information from a co-registered optical/multispectral and SAR image pair."

The word doing the work in the second is *complementary*. Stacking two images
and running one model over them is not extraction of complementary information;
it is extraction of whatever survives the stacking. The two sensors are good at
different things, so each is read for what it is good at and the disagreement
between them is itself a signal.
"""
from __future__ import annotations

import numpy as np

from ..schema import Evidence, Modality, Task
from .backbone import embed_images, embed_texts, load, patch_tokens
from .base import Tool, ToolSpec
from .specialists import LAND_COVER, _BackboneTool, _open, _score_vocab, _boxes_from_heat

# What each sensor is actually reliable for. SAR is specular over smooth water,
# so open water is close to black and unusually easy; it also sees structure
# through cloud, which is why it is the sensor of record for built-up extent.
# Optical carries the spectral information that separates vegetation types.
SAR_STRENGTHS = ["open water", "flooded area", "built-up structures",
                 "bare surface", "smooth surface", "rough terrain"]
OPTICAL_STRENGTHS = ["broad-leaved forest", "coniferous forest", "arable land",
                     "pastures", "urban fabric", "inland waters", "bare rock"]


def _peakedness(scores: np.ndarray, temp: float = 100.0) -> float:
    """How decisively a sensor picked one class, in (0, 1].

    The softmax maximum over the vocabulary. A sensor that scores everything
    alike carries no information about which class is present, and this is the
    quantity that says so.
    """
    e = np.exp(temp * (scores - scores.max()))
    return float((e / e.sum()).max())


class ChangeTool(_BackboneTool):
    """Bi-temporal change from patch-embedding divergence.

    Two images of one place, embedded patch-wise by the same encoder, differ
    where the ground differs. Cosine distance between corresponding patches is
    a change score that needs no change-detection training at all — the
    adaptation is carried entirely by the backbone.

    Direction comes separately, by ranking the land-cover vocabulary on each
    date and differencing the rankings. That answers "what changed" rather
    than merely "where", which is what the statement asks for.
    """

    spec = ToolSpec(
        name="rs_change",
        tasks={Task.CHANGE_DESCRIPTION, Task.CHANGE_VQA},
        accepts={"threshold", "top_k", "vocab", "min_area", "delta", "max_classes"},
        needs_images=2,
        description="Bi-temporal change map and description from patch-embedding divergence.",
        requires=["torch", "transformers"],
    )

    def run(self, images, query, params):
        bb = self._bb()
        before, after = _open(images[0]), _open(images[1])

        # Where did it change?
        t0, grid = patch_tokens(bb, before)
        t1, _ = patch_tokens(bb, after)
        n = min(len(t0), len(t1))
        divergence = 1.0 - np.sum(t0[:n] * t1[:n], axis=1)      # cosine distance
        heat = divergence.reshape(grid, grid)
        heat = (heat - heat.min()) / max(np.ptp(heat), 1e-6)

        thr = float(params.get("threshold", 0.6))
        regions = _boxes_from_heat(heat, thr, int(params.get("min_area", 2)))
        regions = regions[: int(params.get("top_k", 4))]

        # What changed, and in which direction?
        vocab = params.get("vocab") or LAND_COVER

        # Per-class AREA change, from patch-level assignment on each date.
        #
        # Measured against the scene-level score difference below, this is the
        # better signal for "did the area of X change": on CDVQA's polar change
        # questions it scores 58.1% on a held-out shard against 56.5% for the
        # scene-level delta and a 54.8% majority baseline. Both effects are
        # small and neither is presented as more than that -- see
        # docs/adr/0004-change-signal.md.
        txt = embed_texts(bb, [f"a satellite image of {c}" for c in vocab])
        areas = []
        for toks in (t0, t1):
            assign = (toks @ txt.T).argmax(axis=1)
            areas.append(np.bincount(assign, minlength=len(vocab)) / max(len(assign), 1))
        area_delta = {c: float(areas[1][i] - areas[0][i]) for i, c in enumerate(vocab)}

        r0 = dict(_score_vocab(bb, before, vocab))
        r1 = dict(_score_vocab(bb, after, vocab))
        delta = sorted(((c, r1[c] - r0[c]) for c in vocab), key=lambda kv: -abs(kv[1]))
        min_delta = float(params.get("delta", 0.01))
        # How many classes to report. Configurable because a caller asking
        # about one specific class needs that class's delta even when three
        # others moved more; the default keeps prose readable.
        moved = [(c, d) for c, d in delta
                 if abs(d) >= min_delta][:int(params.get("max_classes", 4))]

        changed_frac = float((heat >= thr).mean())
        if not moved and changed_frac < 0.02:
            return ("No substantive change detected between the two dates.",
                    [Evidence("change_map", heat.tolist(), "divergence")], 0.7)

        parts = []
        for cls, d in moved:
            parts.append(f"{cls} {'increased' if d > 0 else 'decreased'} ({d:+.3f})")
        where = (f" Change is concentrated in {len(regions)} region(s), "
                 f"strongest at {regions[0]['box']}." if regions else "")
        text = (f"{changed_frac:.0%} of the scene changed. " + "; ".join(parts) + "." + where)

        ev: list[Evidence] = [Evidence("change_map", heat.tolist(), "divergence")]
        ev += [Evidence("bbox", r["box"], "changed region", r["score"]) for r in regions]
        ev += [Evidence("delta", d, cls, abs(d)) for cls, d in moved]
        # Area deltas for every class, not just the movers: a caller asking
        # about one class needs its value even when others moved more.
        ev += [Evidence("area_delta", d, cls, abs(d)) for cls, d in area_delta.items()]

        # Confidence tracks how decisively the scene separates into changed and
        # unchanged. A heatmap that is uniformly middling means the encoder
        # could not tell, and should not be reported as if it could.
        conf = float(np.clip(heat.std() * 3.0, 0.15, 0.95))
        return text, ev, conf


class CrossModalTool(_BackboneTool):
    """Optical + SAR joint extraction.

    Each sensor is scored against the vocabulary it is actually reliable for,
    then the two readings are combined. Where they agree, confidence rises;
    where they disagree, that is reported rather than averaged away, because
    disagreement between an optical and a SAR reading of the same ground is
    usually informative — cloud over the optical, or water that is bright in
    one and black in the other.
    """

    spec = ToolSpec(
        name="rs_cross_modal",
        tasks={Task.CROSS_MODAL},
        accepts={"top_k", "optical_vocab", "sar_vocab", "vocab", "fusion"},
        needs_images=2,
        description="Complementary information extraction from a co-registered optical-SAR pair.",
        requires=["torch", "transformers"],
    )

    def run(self, images, query, params):
        bb = self._bb()
        # Order by modality rather than by argument position: the controller
        # does not promise which came first.
        optical_meta = next((m for m in images if m.modality is Modality.OPTICAL), images[0])
        sar_meta = next((m for m in images if m.modality is Modality.SAR), images[-1])

        opt_img, sar_img = _open(optical_meta), _open(sar_meta)
        k = int(params.get("top_k", 4))

        opt = _score_vocab(bb, opt_img, params.get("optical_vocab") or OPTICAL_STRENGTHS)[:k]
        sar = _score_vocab(bb, sar_img, params.get("sar_vocab") or SAR_STRENGTHS)[:k]

        # Do the two sensors describe the same scene? Low similarity between the
        # pooled embeddings suggests they are not the same ground, or that one
        # is obscured.
        e_opt = embed_images(bb, [opt_img])[0]
        e_sar = embed_images(bb, [sar_img])[0]
        agreement = float(e_opt @ e_sar)

        # A shared-vocabulary fused reading. The statement asks the system to
        # "extract complementary information" from the pair and combine the
        # outputs; reporting the two sensors side by side describes them but
        # never actually combines them, so neither reading is improved by the
        # other. Scoring one vocabulary through both sensors and fusing gives a
        # single answer that either sensor alone could not produce.
        fused: list[tuple[str, float]] = []
        shared = params.get("vocab")
        if shared:
            o = dict(_score_vocab(bb, opt_img, shared))
            r = dict(_score_vocab(bb, sar_img, shared))
            # Standardise before combining: the two sensors' similarities sit in
            # different ranges, so a raw sum is dominated by whichever spreads
            # wider rather than by whichever is more confident.
            def z(d: dict) -> dict:
                v = np.array([d[c] for c in shared])
                sd = v.std() or 1.0
                return {c: (d[c] - v.mean()) / sd for c in shared}

            zo, zr = z(o), z(r)
            how = str(params.get("fusion", "weighted"))
            if how == "max":
                combined = {c: max(zo[c], zr[c]) for c in shared}
            elif how == "optical":
                combined = zo
            elif how == "sar":
                combined = zr
            elif how == "mean":
                combined = {c: 0.5 * (zo[c] + zr[c]) for c in shared}
            else:
                # Weight each sensor by how decisively it read the scene.
                #
                # An equal-weight mean assumes both sensors are equally
                # informative, and when one is not, fusion scores *below* the
                # better sensor alone -- measurably so here, because the
                # backbone is adapted on optical and has never seen SAR.
                # Weighting by the peakedness of each sensor's own score
                # distribution means an uninformative sensor contributes
                # proportionally little, so fusion degrades toward the better
                # reading instead of away from it.
                wo = _peakedness(np.array([o[c] for c in shared]))
                wr = _peakedness(np.array([r[c] for c in shared]))
                total = wo + wr or 1.0
                combined = {c: (wo * zo[c] + wr * zr[c]) / total for c in shared}
            fused = sorted(combined.items(), key=lambda kv: -kv[1])[:k]

        lines = [
            "Optical (spectral, land-cover discrimination): "
            + ", ".join(f"{c} ({s:.3f})" for c, s in opt),
            "SAR (structure, water, all-weather): "
            + ", ".join(f"{c} ({s:.3f})" for c, s in sar),
        ]
        water_sar = next((s for c, s in sar if "water" in c or "flood" in c), None)
        built_sar = next((s for c, s in sar if "built" in c), None)
        if water_sar is not None:
            lines.append(f"SAR supports a water reading ({water_sar:.3f}); SAR is specular "
                         "over calm water, so this is the more reliable sensor for extent.")
        if built_sar is not None:
            lines.append(f"SAR supports built-up structure ({built_sar:.3f}), which persists "
                         "through cloud where the optical reading may not.")
        lines.append(f"Cross-sensor agreement: {agreement:.3f}"
                     + (" — consistent." if agreement > 0.5 else
                        " — low; the two readings may not describe the same conditions."))

        if fused:
            lines.insert(0, "Combined reading: "
                         + ", ".join(f"{c} ({v:+.2f})" for c, v in fused))

        ev = ([Evidence("label", c, f"fused: {c}", float(v)) for c, v in fused]
              + [Evidence("label", c, f"optical: {c}", s) for c, s in opt]
              + [Evidence("label", c, f"sar: {c}", s) for c, s in sar]
              + [Evidence("agreement", agreement, "cross-sensor agreement", agreement)])

        conf = float(np.clip(0.5 + 0.5 * agreement, 0.2, 0.95))
        return "\n".join(lines), ev, conf

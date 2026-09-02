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
        accepts={"threshold", "top_k", "vocab", "min_area", "delta"},
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
        r0 = dict(_score_vocab(bb, before, vocab))
        r1 = dict(_score_vocab(bb, after, vocab))
        delta = sorted(((c, r1[c] - r0[c]) for c in vocab), key=lambda kv: -abs(kv[1]))
        min_delta = float(params.get("delta", 0.01))
        moved = [(c, d) for c, d in delta if abs(d) >= min_delta][:4]

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
        accepts={"top_k", "optical_vocab", "sar_vocab"},
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

        ev = ([Evidence("label", c, f"optical: {c}", s) for c, s in opt]
              + [Evidence("label", c, f"sar: {c}", s) for c, s in sar]
              + [Evidence("agreement", agreement, "cross-sensor agreement", agreement)])

        conf = float(np.clip(0.5 + 0.5 * agreement, 0.2, 0.95))
        return "\n".join(lines), ev, conf

"""Convert an open_clip CLIP checkpoint into HuggingFace ``CLIPModel`` layout.

Every published remote-sensing CLIP (RemoteCLIP, GeoRSCLIP, SkyCLIP) is
distributed in ``open_clip`` naming.  Loading one through
``CLIPModel.from_pretrained`` does *not* fail -- transformers silently reports
every tensor as missing and hands back a **randomly initialised** model.  That
is the worst possible failure mode for us: training would appear to run and the
resulting numbers would be meaningless.  This module does the rename explicitly
and refuses to return a model unless every expected tensor was populated.

See docs/adr/0002-remote-sensing-backbone.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

# open_clip packs Q, K and V into one ``in_proj`` tensor; HF keeps them apart.
_ATTN = {"q_proj": 0, "k_proj": 1, "v_proj": 2}

_TOP_LEVEL = {
    "logit_scale": "logit_scale",
    "positional_embedding": "text_model.embeddings.position_embedding.weight",
    "token_embedding.weight": "text_model.embeddings.token_embedding.weight",
    "ln_final.weight": "text_model.final_layer_norm.weight",
    "ln_final.bias": "text_model.final_layer_norm.bias",
    "visual.class_embedding": "vision_model.embeddings.class_embedding",
    "visual.conv1.weight": "vision_model.embeddings.patch_embedding.weight",
    "visual.positional_embedding": "vision_model.embeddings.position_embedding.weight",
    # NOTE: "layrnorm" is a real typo in the transformers source, not ours.
    "visual.ln_pre.weight": "vision_model.pre_layrnorm.weight",
    "visual.ln_pre.bias": "vision_model.pre_layrnorm.bias",
    "visual.ln_post.weight": "vision_model.post_layernorm.weight",
    "visual.ln_post.bias": "vision_model.post_layernorm.bias",
}

_BLOCK = {
    "ln_1": "layer_norm1",
    "ln_2": "layer_norm2",
    "mlp.c_fc": "mlp.fc1",
    "mlp.c_proj": "mlp.fc2",
    "attn.out_proj": "self_attn.out_proj",
}

_RESBLOCK = re.compile(r"^(visual\.)?transformer\.resblocks\.(\d+)\.(.+)$")


def _strip_prefix(state: dict) -> dict:
    """Drop a leading ``model.`` / ``module.`` wrapper if the checkpoint has one."""
    for prefix in ("model.", "module."):
        if all(k.startswith(prefix) for k in state):
            return {k[len(prefix):]: v for k, v in state.items()}
    return state


def convert_state_dict(state: dict) -> dict:
    """Rename an open_clip state dict into HF ``CLIPModel`` keys."""
    state = _strip_prefix(state)
    out: dict[str, torch.Tensor] = {}

    for key, value in state.items():
        if key in _TOP_LEVEL:
            out[_TOP_LEVEL[key]] = value
            continue

        # The two projection heads are stored transposed relative to nn.Linear.
        if key == "text_projection":
            out["text_projection.weight"] = value.t().contiguous()
            continue
        if key == "visual.proj":
            out["visual_projection.weight"] = value.t().contiguous()
            continue

        match = _RESBLOCK.match(key)
        if not match:
            continue  # attention masks and other buffers HF rebuilds itself
        is_visual, layer, rest = match.groups()
        stem = "vision_model" if is_visual else "text_model"
        prefix = f"{stem}.encoder.layers.{layer}"

        # in_proj holds Q|K|V stacked on dim 0 -- split it into three tensors.
        if rest in ("attn.in_proj_weight", "attn.in_proj_bias"):
            kind = "weight" if rest.endswith("weight") else "bias"
            size = value.shape[0] // 3
            for name, index in _ATTN.items():
                chunk = value[index * size:(index + 1) * size]
                out[f"{prefix}.self_attn.{name}.{kind}"] = chunk.contiguous()
            continue

        for src, dst in _BLOCK.items():
            if rest.startswith(src + "."):
                out[f"{prefix}.{dst}{rest[len(src):]}"] = value
                break

    return out


def load_openclip_as_hf(checkpoint: str | Path, architecture: str):
    """Build an HF ``CLIPModel`` carrying the weights from ``checkpoint``.

    Raises ``RuntimeError`` rather than returning a partly-random model, since a
    silent fallback here would invalidate every number measured downstream.
    """
    from transformers import CLIPModel

    path = Path(checkpoint)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        raw = load_file(str(path))
    else:
        raw = torch.load(str(path), map_location="cpu", weights_only=False)
        for field in ("state_dict", "model"):
            if isinstance(raw, dict) and field in raw and isinstance(raw[field], dict):
                raw = raw[field]
                break

    model = CLIPModel.from_pretrained(architecture)
    converted = convert_state_dict(raw)
    missing, unexpected = model.load_state_dict(converted, strict=False)

    # position_ids are non-persistent buffers in recent transformers; ignore them.
    missing = [k for k in missing if not k.endswith("position_ids")]
    if missing or unexpected:
        raise RuntimeError(
            f"conversion incomplete: {len(missing)} missing, "
            f"{len(unexpected)} unexpected (first missing: {missing[:3]})"
        )
    return model.eval()

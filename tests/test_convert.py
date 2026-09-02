"""Round-trip tests for the open_clip -> HuggingFace weight conversion.

This conversion guards a failure that is silent and total: loading an
open_clip checkpoint with ``CLIPModel.from_pretrained`` raises nothing and
returns a fully random model. Every number measured downstream would then be
meaningless, so the conversion is tested rather than trusted.

The tests build a tiny CLIP, push its weights *into* open_clip layout, and
check the converter brings them back bit-for-bit. That runs offline in a second
and is a stronger check than eyeballing one real checkpoint.
"""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import CLIPConfig, CLIPModel

from satquery.adapt.convert_openclip import convert_state_dict


def tiny_model():
    """A structurally faithful CLIP small enough to build in-process."""
    cfg = CLIPConfig(
        text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, max_position_embeddings=77,
                         vocab_size=100),
        vision_config=dict(hidden_size=48, intermediate_size=96, num_hidden_layers=2,
                           num_attention_heads=4, image_size=32, patch_size=16),
        projection_dim=16,
    )
    torch.manual_seed(0)
    return CLIPModel(cfg)


def to_openclip(model):
    """Inverse of the converter: HF state dict -> open_clip naming."""
    sd = model.state_dict()
    out = {
        "logit_scale": sd["logit_scale"],
        "positional_embedding": sd["text_model.embeddings.position_embedding.weight"],
        "token_embedding.weight": sd["text_model.embeddings.token_embedding.weight"],
        "ln_final.weight": sd["text_model.final_layer_norm.weight"],
        "ln_final.bias": sd["text_model.final_layer_norm.bias"],
        "text_projection": sd["text_projection.weight"].t().contiguous(),
        "visual.class_embedding": sd["vision_model.embeddings.class_embedding"],
        "visual.conv1.weight": sd["vision_model.embeddings.patch_embedding.weight"],
        "visual.positional_embedding": sd["vision_model.embeddings.position_embedding.weight"],
        "visual.ln_pre.weight": sd["vision_model.pre_layrnorm.weight"],
        "visual.ln_pre.bias": sd["vision_model.pre_layrnorm.bias"],
        "visual.ln_post.weight": sd["vision_model.post_layernorm.weight"],
        "visual.ln_post.bias": sd["vision_model.post_layernorm.bias"],
        "visual.proj": sd["visual_projection.weight"].t().contiguous(),
    }
    blocks = {"layer_norm1": "ln_1", "layer_norm2": "ln_2",
              "mlp.fc1": "mlp.c_fc", "mlp.fc2": "mlp.c_proj",
              "self_attn.out_proj": "attn.out_proj"}
    for stem, prefix in (("text_model", ""), ("vision_model", "visual.")):
        layers = model.config.text_config.num_hidden_layers if stem == "text_model" \
            else model.config.vision_config.num_hidden_layers
        for i in range(layers):
            src = f"{stem}.encoder.layers.{i}"
            dst = f"{prefix}transformer.resblocks.{i}"
            for hf, oc in blocks.items():
                for suffix in ("weight", "bias"):
                    out[f"{dst}.{oc}.{suffix}"] = sd[f"{src}.{hf}.{suffix}"]
            for suffix in ("weight", "bias"):
                out[f"{dst}.attn.in_proj_{suffix}"] = torch.cat([
                    sd[f"{src}.self_attn.q_proj.{suffix}"],
                    sd[f"{src}.self_attn.k_proj.{suffix}"],
                    sd[f"{src}.self_attn.v_proj.{suffix}"]])
    return out


def test_round_trip_recovers_every_tensor():
    model = tiny_model()
    original = model.state_dict()
    recovered = convert_state_dict(to_openclip(model))

    expected = {k for k in original if not k.endswith("position_ids")}
    assert set(recovered) == expected, (
        f"missing {expected - set(recovered)}, extra {set(recovered) - expected}")
    for key in expected:
        assert torch.equal(recovered[key], original[key]), f"tensor differs: {key}"


def test_qkv_split_is_not_permuted():
    """A q/k/v mix-up still loads cleanly and silently ruins attention."""
    model = tiny_model()
    oc = to_openclip(model)
    recovered = convert_state_dict(oc)
    src = "text_model.encoder.layers.0.self_attn"
    packed = oc["transformer.resblocks.0.attn.in_proj_weight"]
    size = packed.shape[0] // 3
    assert torch.equal(recovered[f"{src}.q_proj.weight"], packed[:size])
    assert torch.equal(recovered[f"{src}.k_proj.weight"], packed[size:2 * size])
    assert torch.equal(recovered[f"{src}.v_proj.weight"], packed[2 * size:])


def test_projections_are_transposed():
    """open_clip stores the projections transposed relative to nn.Linear."""
    model = tiny_model()
    oc = to_openclip(model)
    recovered = convert_state_dict(oc)
    assert torch.equal(recovered["text_projection.weight"], oc["text_projection"].t())
    assert torch.equal(recovered["visual_projection.weight"], oc["visual.proj"].t())


def test_model_prefix_is_stripped():
    """Real checkpoints wrap everything in `model.` -- ours does."""
    model = tiny_model()
    oc = to_openclip(model)
    plain = convert_state_dict(oc)
    prefixed = convert_state_dict({f"model.{k}": v for k, v in oc.items()})
    assert set(plain) == set(prefixed)
    for key in plain:
        assert torch.equal(plain[key], prefixed[key])


def test_incomplete_checkpoint_is_rejected_not_silently_random():
    """The whole point: refuse to hand back a partly-initialised model."""
    from satquery.adapt.convert_openclip import load_openclip_as_hf
    import tempfile, os
    from safetensors.torch import save_file

    model = tiny_model()
    oc = to_openclip(model)
    del oc["visual.proj"]                       # simulate a truncated download
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.safetensors")
        save_file({k: v.contiguous() for k, v in oc.items()}, path)
        model.config.save_pretrained(d)
        with pytest.raises(RuntimeError, match="conversion incomplete"):
            load_openclip_as_hf(path, d)

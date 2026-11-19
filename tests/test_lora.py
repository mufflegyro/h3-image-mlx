"""LoRA reading and the fused-projection arithmetic, on small fabricated adapters."""

import json

import mlx.core as mx
import numpy as np
import pytest

from h3_image_mlx import lora

HIDDEN = 4
FF = 8  # fc1 writes [gate; up], so its output is 2 * FF
RANK = 2


def pair(module, inp, out, rank=RANK, seed=0):
    rng = np.random.default_rng(seed)
    return {
        f"{module}.lora_A.weight": mx.array(rng.normal(size=(rank, inp)).astype(np.float32)),
        f"{module}.lora_B.weight": mx.array(rng.normal(size=(out, rank)).astype(np.float32)),
    }


def block_tensors(prefix="transformer_blocks.0"):
    tensors = {}
    for index, name in enumerate(("to_q", "to_k", "to_v")):
        tensors.update(pair(f"{prefix}.attn.{name}", HIDDEN, HIDDEN, seed=index))
    tensors.update(pair(f"{prefix}.attn.to_out.0", HIDDEN, HIDDEN, seed=3))
    tensors.update(pair(f"{prefix}.ff.net.0.proj", HIDDEN, 2 * FF, seed=4))
    tensors.update(pair(f"{prefix}.ff.net.2", FF, HIDDEN, seed=5))
    return tensors


def test_blocks_map_onto_the_engines_fused_projections():
    adapter = lora.read(block_tensors())
    assert sorted(adapter.modules) == [
        "blocks.0.attn.out_proj",
        "blocks.0.attn.qkv_proj",
        "blocks.0.mlp.fc1",
        "blocks.0.mlp.fc2",
    ]
    assert len(adapter.parts["blocks.0.attn.qkv_proj"]) == 3
    assert adapter.parts["blocks.0.mlp.fc1"][0].halves is True
    assert adapter.parts["blocks.0.attn.out_proj"][0].halves is False


def test_q_k_v_are_concatenated_in_order_with_their_scaling():
    tim = mx.array(np.ones((2, HIDDEN), dtype=np.float32))
    adapter = lora.read(block_tensors())
    parts = adapter.parts["blocks.0.attn.qkv_proj"]
    expected = np.concatenate([
        np.asarray((tim @ part.a.T @ part.b.T) * part.scaling) for part in parts
    ], axis=-1)
    fused = np.asarray(adapter.delta("blocks.0.attn.qkv_proj", tim))
    assert np.allclose(fused, expected, atol=1e-5)


def test_fc1_moves_the_diffusers_halves_into_the_engines_gate_first_order():
    # diffusers writes [value; gate]; this engine's swiglu reads [gate; up].
    tensors = {
        "transformer_blocks.0.ff.net.0.proj.lora_A.weight": mx.array(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)),
        "transformer_blocks.0.ff.net.0.proj.lora_B.weight": mx.array(
            np.arange(1, 2 * FF + 1, dtype=np.float32).reshape(2 * FF, 1)),
    }
    adapter = lora.read(tensors)
    x = mx.array(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    delta = np.asarray(adapter.delta("blocks.0.mlp.fc1", x)).reshape(-1)
    expected = np.concatenate([np.arange(FF + 1, 2 * FF + 1), np.arange(1, FF + 1)])
    assert np.array_equal(delta, expected.astype(np.float32))


def test_alpha_tensor_and_scale_fold_into_the_step():
    tensors = pair("transformer_blocks.0.attn.to_out.0", HIDDEN, HIDDEN, rank=2)
    tensors["transformer_blocks.0.attn.to_out.0.alpha"] = mx.array(
        np.array(4.0, dtype=np.float32))
    adapter = lora.read(tensors, scale=0.5)
    assert adapter.parts["blocks.0.attn.out_proj"][0].scaling == pytest.approx(0.5 * 4 / 2)


def test_peft_default_infix_and_the_transformer_prefix_are_read():
    tensors = {}
    for name, value in pair("transformer_blocks.0.attn.to_out.0", HIDDEN, HIDDEN).items():
        tensors["transformer." + name.replace(".lora_A.", ".lora_A.default.")] = value
    adapter = lora.read(tensors)
    assert "blocks.0.attn.out_proj" in adapter.parts


def test_the_forks_own_fused_keys_pass_through():
    tensors = {
        "diffusion_model.blocks.0.attn.out_proj.lora_A.weight": mx.array(
            np.ones((RANK, HIDDEN), dtype=np.float32)),
        "diffusion_model.blocks.0.attn.out_proj.lora_B.weight": mx.array(
            np.ones((HIDDEN, RANK), dtype=np.float32)),
    }
    adapter = lora.read(tensors)
    assert "blocks.0.attn.out_proj" in adapter.parts


def test_a_qkv_adapter_missing_a_part_is_refused_rather_than_written_into_the_wrong_slots():
    tensors = {}
    tensors.update(pair("transformer_blocks.0.attn.to_q", HIDDEN, HIDDEN))
    tensors.update(pair("transformer_blocks.0.attn.to_k", HIDDEN, HIDDEN))
    with pytest.raises(ValueError, match="missing v"):
        lora.read(tensors)


def test_original_checkpoint_names_are_refused_with_the_conversion_named():
    tensors = {
        "lora_unet_blocks_0_attn_q.lora_down.weight": mx.array(np.zeros((2, 4), np.float32)),
        "lora_unet_blocks_0_attn_q.lora_up.weight": mx.array(np.zeros((4, 2), np.float32)),
    }
    with pytest.raises(NotImplementedError, match="diffusers names"):
        lora.read(tensors)


def test_dora_is_refused():
    tensors = pair("transformer_blocks.0.attn.to_out.0", HIDDEN, HIDDEN)
    tensors["transformer_blocks.0.attn.to_out.0.dora_scale"] = mx.array(
        np.ones(HIDDEN, dtype=np.float32))
    with pytest.raises(NotImplementedError, match="DoRA"):
        lora.read(tensors)


def test_a_module_outside_the_block_linears_is_refused():
    tensors = pair("transformer_blocks.0.norm1", HIDDEN, HIDDEN)
    with pytest.raises(ValueError, match="not one of the block Linears"):
        lora.read(tensors)


def test_a_scale_must_be_finite():
    tensors = pair("transformer_blocks.0.attn.to_out.0", HIDDEN, HIDDEN)
    with pytest.raises(ValueError, match="finite"):
        lora.read(tensors, scale=float("nan"))


def test_the_file_metadata_carries_r_and_alpha(tmp_path):
    path = tmp_path / "adapter.safetensors"
    mx.save_safetensors(
        str(path),
        pair("transformer_blocks.0.attn.to_out.0", HIDDEN, HIDDEN),
        metadata={"lora_adapter_metadata": json.dumps({"r": RANK, "lora_alpha": 4})},
    )
    adapter = lora.load(path, scale=2.0)
    assert adapter.parts["blocks.0.attn.out_proj"][0].scaling == pytest.approx(2.0 * 4 / RANK)


def test_a_metadata_flag_this_port_does_not_implement_is_refused(tmp_path):
    path = tmp_path / "adapter.safetensors"
    mx.save_safetensors(
        str(path),
        pair("transformer_blocks.0.attn.to_out.0", HIDDEN, HIDDEN),
        metadata={"lora_adapter_metadata": json.dumps({"r": RANK, "lora_alpha": 4,
                                                       "use_rslora": True})},
    )
    with pytest.raises(NotImplementedError, match="use_rslora"):
        lora.load(path)


def test_a_wrong_input_width_names_the_projection():
    adapter = lora.read(block_tensors())
    with pytest.raises(ValueError, match="takes 4-wide input"):
        adapter.delta("blocks.0.attn.qkv_proj", mx.zeros((1, HIDDEN + 1)))
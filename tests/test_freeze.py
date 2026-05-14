"""Tests for :func:`critnet.model.freeze_neurons`."""

from __future__ import annotations

import pytest
import torch

from critnet import CriticalNeuronConfig, freeze_neurons


def _make_loss(model, batch):
    out = model(input_ids=batch["input_ids"], labels=batch["labels"])
    return out.loss


def test_grad_zeroed_on_frozen_neuron(tiny_lm, tiny_loader):
    """Backward hook on q_proj must zero rows 0 and 2."""
    cfg = CriticalNeuronConfig()
    handle = freeze_neurons(
        tiny_lm,
        {"layers.0.self_attn.q_proj": [0, 2]},
        cfg,
    )

    _make_loss(tiny_lm, tiny_loader[0]).backward()
    g = tiny_lm.layers[0].self_attn.q_proj.weight.grad
    assert g is not None
    assert torch.all(g[0] == 0)
    assert torch.all(g[2] == 0)
    # other rows should typically have nonzero gradients
    assert torch.any(g[1] != 0)
    handle.remove()


def test_remove_drops_hook(tiny_lm, tiny_loader):
    cfg = CriticalNeuronConfig()
    handle = freeze_neurons(
        tiny_lm,
        {"layers.0.self_attn.q_proj": [0]},
        cfg,
    )
    handle.remove()
    _make_loss(tiny_lm, tiny_loader[0]).backward()
    g = tiny_lm.layers[0].self_attn.q_proj.weight.grad
    # without the hook, row 0 should now have nonzero gradient
    assert torch.any(g[0] != 0)


def test_restore_frozen_weights_undoes_changes(tiny_lm, tiny_loader):
    cfg = CriticalNeuronConfig()
    saved = tiny_lm.layers[0].self_attn.q_proj.weight.data[0].clone()
    handle = freeze_neurons(
        tiny_lm,
        {"layers.0.self_attn.q_proj": [0]},
        cfg,
    )

    # Simulate a weight-decay style drift by mutating the supposedly-frozen row.
    tiny_lm.layers[0].self_attn.q_proj.weight.data[0] += 0.5

    handle.restore_frozen_weights()
    assert torch.allclose(tiny_lm.layers[0].self_attn.q_proj.weight.data[0], saved)
    handle.remove()


def test_column_module_axis_handling(tiny_lm, tiny_loader):
    cfg = CriticalNeuronConfig()
    handle = freeze_neurons(
        tiny_lm,
        {"layers.0.self_attn.o_proj": [1, 3]},
        cfg,
    )
    _make_loss(tiny_lm, tiny_loader[0]).backward()
    g = tiny_lm.layers[0].self_attn.o_proj.weight.grad
    assert g is not None
    assert torch.all(g[:, 1] == 0)
    assert torch.all(g[:, 3] == 0)
    handle.remove()


def test_norm_auto_fallback_classifies_unlisted_norm(tiny_lm, tiny_loader):
    """`q_norm`-style modules that are 1-D norms but not in config should still freeze."""
    # The default config has no `q_norm`, but a structural norm should be auto-typed.
    # Use the model's `norm` (final RMSNorm) which has a 1-D weight.
    cfg = CriticalNeuronConfig()
    handle = freeze_neurons(tiny_lm, {"norm": [0, 1]}, cfg)
    _make_loss(tiny_lm, tiny_loader[0]).backward()
    g = tiny_lm.norm.weight.grad
    assert g is not None
    assert g[0] == 0 and g[1] == 0
    handle.remove()


def test_unclassifiable_linear_raises(tiny_lm):
    """If a Linear module is not in the config, freezing must raise (no row guess)."""
    cfg = CriticalNeuronConfig(row_modules=["k_proj"], column_modules=[])  # q_proj absent
    with pytest.raises(ValueError, match="Cannot classify"):
        freeze_neurons(tiny_lm, {"layers.0.self_attn.q_proj": [0]}, cfg)


def test_summary_format(tiny_lm):
    cfg = CriticalNeuronConfig()
    handle = freeze_neurons(
        tiny_lm,
        {"layers.0.self_attn.q_proj": [0, 1]},
        cfg,
    )
    s = handle.summary()
    assert "frozen params" in s
    assert "trainable params" in s
    handle.remove()

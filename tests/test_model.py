"""Tests for :mod:`critnet.model`: delta wrappers + :func:`get_neuron_model`."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from critnet import (
    CriticalNeuronConfig,
    CriticalNeuronModel,
    EmbeddingDeltaSubspace,
    LinearDeltaSubspace,
    NormDeltaSubspace,
    get_neuron_model,
)


# ---------------------------------------------------------------------------
# LinearDeltaSubspace
# ---------------------------------------------------------------------------


def test_linear_row_zero_dw_is_identity():
    torch.manual_seed(0)
    base = nn.Linear(6, 4, bias=False)
    wrapped = LinearDeltaSubspace(base, indices=[0, 3], mode="row")
    x = torch.randn(2, 6)
    out_base = nn.functional.linear(x, base.weight)
    assert torch.allclose(wrapped(x), out_base)


def test_linear_column_zero_dw_is_identity():
    torch.manual_seed(0)
    base = nn.Linear(6, 4, bias=False)
    wrapped = LinearDeltaSubspace(base, indices=[1, 5], mode="column")
    x = torch.randn(2, 6)
    out_base = nn.functional.linear(x, base.weight)
    assert torch.allclose(wrapped(x), out_base)


def test_linear_row_dw_matches_full_update():
    torch.manual_seed(1)
    base = nn.Linear(6, 4, bias=False)
    base_W_before = base.weight.data.clone()
    idx = [0, 3]
    wrapped = LinearDeltaSubspace(base, indices=idx, mode="row")
    wrapped.dW.data = torch.randn_like(wrapped.dW)

    # Forward pass should equal: base output + (dW @ x) inserted at idx
    x = torch.randn(7, 6)
    out_wrapped = wrapped(x)

    expected_W = base_W_before.clone()
    expected_W[idx] = base_W_before[idx] + wrapped.dW.detach()
    expected = nn.functional.linear(x, expected_W)
    assert torch.allclose(out_wrapped, expected, atol=1e-5)


def test_linear_column_dw_matches_full_update():
    torch.manual_seed(2)
    base = nn.Linear(6, 4, bias=False)
    base_W_before = base.weight.data.clone()
    idx = [1, 5]
    wrapped = LinearDeltaSubspace(base, indices=idx, mode="column")
    wrapped.dW.data = torch.randn_like(wrapped.dW)

    x = torch.randn(7, 6)
    out_wrapped = wrapped(x)

    expected_W = base_W_before.clone()
    expected_W[:, idx] = base_W_before[:, idx] + wrapped.dW.detach()
    expected = nn.functional.linear(x, expected_W)
    assert torch.allclose(out_wrapped, expected, atol=1e-5)


def test_linear_merge_is_post_merge_identity():
    torch.manual_seed(3)
    base = nn.Linear(6, 4, bias=False)
    wrapped = LinearDeltaSubspace(base, indices=[0, 3], mode="row")
    wrapped.dW.data = torch.randn_like(wrapped.dW)

    x = torch.randn(7, 6)
    out_before = wrapped(x).detach()
    merged = wrapped.merge_to_linear_()
    out_after = merged(x).detach()
    assert torch.allclose(out_before, out_after, atol=1e-5)


def test_linear_freezes_base_weight_and_bias():
    base = nn.Linear(6, 4, bias=True)
    w = LinearDeltaSubspace(base, indices=[0, 2], mode="row")
    assert not w.base.weight.requires_grad
    assert not w.base.bias.requires_grad
    assert w.dW.requires_grad


# ---------------------------------------------------------------------------
# NormDeltaSubspace
# ---------------------------------------------------------------------------


def _make_norm(dim: int) -> nn.Module:
    norm = nn.LayerNorm(dim, elementwise_affine=True)
    return norm


def test_norm_zero_dw_is_identity():
    torch.manual_seed(0)
    norm = _make_norm(6)
    wrapped = NormDeltaSubspace(norm, indices=[1, 4])
    x = torch.randn(2, 6)
    out_base = norm(x)
    assert torch.allclose(wrapped(x), out_base, atol=1e-5)


def test_norm_merge_is_post_merge_identity():
    torch.manual_seed(1)
    norm = _make_norm(6)
    wrapped = NormDeltaSubspace(norm, indices=[1, 4])
    wrapped.dW.data = torch.randn(2)

    x = torch.randn(2, 6)
    out_before = wrapped(x).detach()
    merged = wrapped.merge_to_norm_()
    out_after = merged(x).detach()
    assert torch.allclose(out_before, out_after, atol=1e-5)


# ---------------------------------------------------------------------------
# EmbeddingDeltaSubspace
# ---------------------------------------------------------------------------


def test_embedding_zero_dw_is_identity():
    emb = nn.Embedding(8, 4)
    wrapped = EmbeddingDeltaSubspace(emb, indices=[2, 5])
    ids = torch.tensor([0, 2, 3, 5, 7])
    assert torch.allclose(wrapped(ids), emb(ids))


def test_embedding_merge_is_post_merge_identity():
    torch.manual_seed(0)
    emb = nn.Embedding(8, 4)
    wrapped = EmbeddingDeltaSubspace(emb, indices=[2, 5])
    wrapped.dW.data = torch.randn_like(wrapped.dW)

    ids = torch.tensor([0, 2, 3, 5, 7])
    out_before = wrapped(ids).detach()
    merged = wrapped.merge_to_embedding_()
    out_after = merged(ids).detach()
    assert torch.allclose(out_before, out_after, atol=1e-5)


# ---------------------------------------------------------------------------
# get_neuron_model / CriticalNeuronModel
# ---------------------------------------------------------------------------


def _q_indices(tiny_lm):
    out = {}
    for i in range(tiny_lm.N_LAYERS):
        out[f"layers.{i}.self_attn.q_proj"] = [0, 1]
        out[f"layers.{i}.input_layernorm"] = [0]
    return out


def test_get_neuron_model_wraps_only_targeted_modules(tiny_lm):
    cfg = CriticalNeuronConfig()
    wrapped = get_neuron_model(tiny_lm, cfg, _q_indices(tiny_lm))

    assert isinstance(wrapped, CriticalNeuronModel)
    # targeted q_proj is wrapped
    assert isinstance(wrapped.model.layers[0].self_attn.q_proj, LinearDeltaSubspace)
    # non-targeted v_proj is untouched
    assert isinstance(wrapped.model.layers[0].self_attn.v_proj, nn.Linear)
    # targeted norm is wrapped
    assert isinstance(wrapped.model.layers[0].input_layernorm, NormDeltaSubspace)


def test_get_neuron_model_only_dw_trainable(tiny_lm):
    cfg = CriticalNeuronConfig()
    wrapped = get_neuron_model(tiny_lm, cfg, _q_indices(tiny_lm))
    trainable_names = [
        n for n, p in wrapped.model.named_parameters() if p.requires_grad
    ]
    assert all(name.endswith(".dW") for name in trainable_names)
    assert len(trainable_names) == 2 * tiny_lm.N_LAYERS  # q_proj + norm per layer


def test_get_neuron_model_empty_indices_raises(tiny_lm):
    cfg = CriticalNeuronConfig()
    with pytest.raises(ValueError, match="empty"):
        get_neuron_model(tiny_lm, cfg, {})


def test_get_neuron_model_unclassified_module_raises(tiny_lm):
    """The model's final `norm` is structurally a 1-D RMSNorm but the
    default config's ``norm_modules`` does not include the leaf ``"norm"``.
    Strict classifier must refuse to wrap it instead of guessing."""
    cfg = CriticalNeuronConfig()
    with pytest.raises(ValueError, match="not classified"):
        get_neuron_model(tiny_lm, cfg, {"norm": [0]})


def test_get_neuron_model_missing_path_raises(tiny_lm):
    cfg = CriticalNeuronConfig()
    with pytest.raises(ValueError, match="not found in model"):
        get_neuron_model(tiny_lm, cfg, {"layers.99.self_attn.q_proj": [0]})


def test_neuron_indices_property_reads_wrappers(tiny_lm):
    cfg = CriticalNeuronConfig()
    indices = _q_indices(tiny_lm)
    wrapped = get_neuron_model(tiny_lm, cfg, indices)
    live = wrapped.neuron_indices
    assert set(live.keys()) == set(indices.keys())
    for k in indices:
        assert live[k] == sorted(indices[k])


def test_peft_config_alias_points_to_config(tiny_lm):
    cfg = CriticalNeuronConfig()
    wrapped = get_neuron_model(tiny_lm, cfg, _q_indices(tiny_lm))
    assert wrapped.peft_config is wrapped.config


def test_save_and_load_roundtrip(tmp_path, tiny_lm):
    cfg = CriticalNeuronConfig()
    wrapped = get_neuron_model(tiny_lm, cfg, _q_indices(tiny_lm))
    # randomise dWs
    for n, p in wrapped.model.named_parameters():
        if n.endswith(".dW"):
            p.data = torch.randn_like(p)

    saved_dw = {
        n: p.detach().clone() for n, p in wrapped.model.named_parameters()
        if n.endswith(".dW")
    }
    wrapped.save_pretrained(str(tmp_path))

    # Files written?
    assert (tmp_path / "critical_neuron_config.json").exists()
    assert (tmp_path / "neuron_indices.json").exists()
    assert (tmp_path / "adapter_model.safetensors").exists() or \
           (tmp_path / "adapter_model.pt").exists()

    # Re-wrap a fresh base model and load only the adapter state.
    from tests.conftest import TinyLM
    torch.manual_seed(0)
    fresh = TinyLM()
    new_wrapped = get_neuron_model(fresh, cfg, _q_indices(fresh))
    sf_path = tmp_path / "adapter_model.safetensors"
    if sf_path.exists():
        from safetensors.torch import load_file
        state = load_file(str(sf_path))
    else:
        state = torch.load(str(tmp_path / "adapter_model.pt"), map_location="cpu")
    for n, p in new_wrapped.model.named_parameters():
        if n in state:
            p.data.copy_(state[n])
    for n, p in new_wrapped.model.named_parameters():
        if n.endswith(".dW"):
            assert torch.allclose(p.data, saved_dw[n])


def test_merge_and_unload_restores_plain_module(tiny_lm):
    cfg = CriticalNeuronConfig()
    wrapped = get_neuron_model(tiny_lm, cfg, _q_indices(tiny_lm))
    inner = wrapped.merge_and_unload()
    # after merge, q_proj should be a plain nn.Linear again
    assert isinstance(inner.layers[0].self_attn.q_proj, nn.Linear)
    # and norm should be a plain norm (no _DeltaModule)
    from critnet.model import NormDeltaSubspace as _ND
    assert not isinstance(inner.layers[0].input_layernorm, _ND)

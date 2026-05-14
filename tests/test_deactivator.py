"""Tests for :mod:`critnet.deactivator`."""

from __future__ import annotations

import pytest
import torch

from critnet import CriticalNeuronConfig, NeuronDeactivator


def test_row_module_zeros_full_rows(tiny_lm):
    cfg = CriticalNeuronConfig()
    deact = NeuronDeactivator(tiny_lm, cfg)
    weight = tiny_lm.layers[0].self_attn.q_proj.weight
    weight.data.fill_(1.0)

    res = deact.deactivate({"layers.0.self_attn.q_proj": [0, 2]})

    assert torch.all(weight.data[0] == 0)
    assert torch.all(weight.data[2] == 0)
    assert torch.all(weight.data[1] != 0)
    assert res.modules_affected == 1
    assert res.neurons_zeroed == 2
    assert res.total_weights_zeroed == 2 * weight.size(1)
    assert res.per_module["layers.0.self_attn.q_proj"]["module_type"] == "row"


def test_column_module_zeros_full_columns(tiny_lm):
    cfg = CriticalNeuronConfig()
    deact = NeuronDeactivator(tiny_lm, cfg)
    weight = tiny_lm.layers[0].self_attn.o_proj.weight
    weight.data.fill_(1.0)

    res = deact.deactivate({"layers.0.self_attn.o_proj": [3, 5]})

    assert torch.all(weight.data[:, 3] == 0)
    assert torch.all(weight.data[:, 5] == 0)
    assert torch.all(weight.data[:, 0] != 0)
    assert res.total_weights_zeroed == weight.size(0) * 2


def test_norm_module_zeros_individual_scalars(tiny_lm):
    cfg = CriticalNeuronConfig()
    deact = NeuronDeactivator(tiny_lm, cfg)
    weight = tiny_lm.layers[0].input_layernorm.weight
    weight.data.fill_(2.0)

    res = deact.deactivate({"layers.0.input_layernorm": [1, 4]})

    assert weight.data[1] == 0 and weight.data[4] == 0
    assert weight.data[0] == 2.0
    assert res.total_weights_zeroed == 2


def test_embedding_module_zeros_rows(tiny_lm):
    cfg = CriticalNeuronConfig(embedding_modules=["embed_tokens"])
    deact = NeuronDeactivator(tiny_lm, cfg)
    weight = tiny_lm.embed_tokens.weight
    weight.data.fill_(1.0)

    res = deact.deactivate({"embed_tokens": [0, 3]})
    assert torch.all(weight.data[0] == 0)
    assert torch.all(weight.data[3] == 0)
    assert torch.all(weight.data[1] != 0)
    assert res.per_module["embed_tokens"]["module_type"] == "embedding"


def test_empty_indices_returns_empty_result(tiny_lm):
    res = NeuronDeactivator(tiny_lm, CriticalNeuronConfig()).deactivate({})
    assert res.modules_affected == 0
    assert res.neurons_zeroed == 0


def test_unknown_module_raises(tiny_lm):
    cfg = CriticalNeuronConfig()
    deact = NeuronDeactivator(tiny_lm, cfg)
    with pytest.raises(ValueError, match="Cannot classify"):
        deact.deactivate({"layers.0.self_attn.q_norm": [0, 1]})


def test_save_pretrained_writes_config_and_indices(tmp_path, tiny_lm):
    cfg = CriticalNeuronConfig()
    deact = NeuronDeactivator(tiny_lm, cfg)
    indices = {"layers.0.self_attn.q_proj": [0]}
    # tiny model has no save_pretrained -> patch it for the test
    tiny_lm.save_pretrained = lambda d: open(f"{d}/marker.bin", "w").close()
    deact.deactivate(indices)
    deact.save_pretrained(str(tmp_path), indices=indices)
    assert (tmp_path / "marker.bin").exists()
    assert (tmp_path / "critical_neuron_config.json").exists()
    assert (tmp_path / "neuron_indices.json").exists()


def test_summary_string_present(tiny_lm):
    cfg = CriticalNeuronConfig()
    res = NeuronDeactivator(tiny_lm, cfg).deactivate(
        {"layers.0.self_attn.q_proj": [0]}
    )
    assert "Deactivation summary" in res.summary()

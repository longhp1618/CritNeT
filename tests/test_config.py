"""Tests for :mod:`critnet.config`."""

from __future__ import annotations

import json
import os

import pytest

from critnet.config import (
    DEFAULT_COLUMN_MODULES,
    DEFAULT_GATE_COMBINES_WITH,
    DEFAULT_NORM_MODULES,
    DEFAULT_ROW_MODULES,
    CriticalNeuronConfig,
    load_neuron_indices,
    save_neuron_indices,
)


class TestDefaults:
    def test_empty_init_fills_defaults(self):
        c = CriticalNeuronConfig()
        assert c.row_modules == DEFAULT_ROW_MODULES
        assert c.column_modules == DEFAULT_COLUMN_MODULES
        assert c.norm_modules == DEFAULT_NORM_MODULES
        assert c.embedding_modules is None
        assert c.gate_combines_with == DEFAULT_GATE_COMBINES_WITH

    def test_empty_list_overrides_default(self):
        c = CriticalNeuronConfig(norm_modules=[])
        assert c.norm_modules == []
        # other categories still defaulted
        assert c.row_modules == DEFAULT_ROW_MODULES

    def test_gate_default_drops_if_partner_absent(self):
        c = CriticalNeuronConfig(row_modules=["q_proj"])
        # up_proj is not in row/column anymore -> default gate map cleaned up
        assert c.gate_combines_with == {}


class TestValidation:
    def test_overlap_raises(self):
        with pytest.raises(ValueError, match="appears in both"):
            CriticalNeuronConfig(
                row_modules=["q_proj"],
                column_modules=["q_proj"],
            )

    def test_unknown_gate_raises(self):
        with pytest.raises(ValueError, match="gate_combines_with"):
            CriticalNeuronConfig(gate_combines_with={"phantom": "up_proj"})

    def test_unknown_partner_raises(self):
        with pytest.raises(ValueError, match="gate_combines_with"):
            CriticalNeuronConfig(gate_combines_with={"gate_proj": "phantom"})


class TestClassify:
    def test_row(self):
        c = CriticalNeuronConfig()
        assert c.classify("model.layers.0.self_attn.q_proj") == "row"
        assert c.classify("model.layers.5.mlp.up_proj") == "row"

    def test_column(self):
        c = CriticalNeuronConfig()
        assert c.classify("model.layers.0.self_attn.o_proj") == "column"
        assert c.classify("model.layers.5.mlp.down_proj") == "column"

    def test_norm(self):
        c = CriticalNeuronConfig()
        assert c.classify("model.layers.0.input_layernorm") == "norm"

    def test_unknown_returns_none(self):
        c = CriticalNeuronConfig()
        assert c.classify("model.layers.0.self_attn.q_norm") is None
        assert c.classify("totally_unrelated") is None

    def test_uses_leaf_only(self):
        c = CriticalNeuronConfig()
        assert c.classify("q_proj") == "row"
        assert c.classify("deeply.nested.path.q_proj") == "row"


class TestComputedViews:
    def test_target_modules_unions_categories(self):
        c = CriticalNeuronConfig(embedding_modules=["embed_tokens"])
        targets = set(c.target_modules)
        assert "q_proj" in targets
        assert "down_proj" in targets
        assert "input_layernorm" in targets
        assert "embed_tokens" in targets

    def test_linear_modules_excludes_norms_and_embeddings(self):
        c = CriticalNeuronConfig(embedding_modules=["embed_tokens"])
        linear = set(c.linear_modules)
        assert "q_proj" in linear
        assert "down_proj" in linear
        assert "input_layernorm" not in linear
        assert "embed_tokens" not in linear


class TestSerialisation:
    def test_save_load_roundtrip(self, tmp_path):
        c1 = CriticalNeuronConfig(
            embedding_modules=["embed_tokens"],
            base_model_name_or_path="meta-llama/foo",
        )
        c1.save_pretrained(str(tmp_path))
        c2 = CriticalNeuronConfig.from_pretrained(str(tmp_path))
        assert c2.row_modules == c1.row_modules
        assert c2.column_modules == c1.column_modules
        assert c2.norm_modules == c1.norm_modules
        assert c2.embedding_modules == c1.embedding_modules
        assert c2.gate_combines_with == c1.gate_combines_with
        assert c2.base_model_name_or_path == c1.base_model_name_or_path

    def test_save_with_indices_creates_indices_file(self, tmp_path):
        c = CriticalNeuronConfig()
        indices = {"model.layers.0.self_attn.q_proj": [0, 3, 7]}
        c.save_pretrained(str(tmp_path), indices=indices)
        assert os.path.isfile(tmp_path / "neuron_indices.json")
        loaded = load_neuron_indices(str(tmp_path))
        assert loaded == indices

    def test_save_without_indices_omits_file(self, tmp_path):
        CriticalNeuronConfig().save_pretrained(str(tmp_path))
        assert not (tmp_path / "neuron_indices.json").exists()

    def test_load_indices_missing_file_raises(self, tmp_path):
        CriticalNeuronConfig().save_pretrained(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            load_neuron_indices(str(tmp_path))

    def test_save_neuron_indices_creates_dir(self, tmp_path):
        target = tmp_path / "deep" / "deeper"
        save_neuron_indices(str(target), {"a": [1, 2]})
        with open(target / "neuron_indices.json") as f:
            assert json.load(f) == {"a": [1, 2]}

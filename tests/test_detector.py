"""Tests for :mod:`critnet.detector`."""

from __future__ import annotations

import os

import pytest
import torch

from critnet import (
    CriticalNeuronConfig,
    DetectionResult,
    NeuronDetector,
    select_neurons_from_cache,
)
from critnet.detector import (
    _apply_gate_combination,
    _global_topk,
)


# ---------------------------------------------------------------------------
# DetectionResult plain-data behaviour
# ---------------------------------------------------------------------------


def test_detection_result_totals():
    r = DetectionResult(
        indices={"a": [0, 1, 2], "b": [4]}, sparsity_ratio=0.05,
    )
    assert r.total_selected == 4
    assert r.n_modules == 2
    assert "DetectionResult" in r.summary()
    assert "0.0500" in r.summary()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_none_model():
    cfg = CriticalNeuronConfig()
    with pytest.raises(TypeError, match="select_neurons_from_cache"):
        NeuronDetector(None, cfg)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, "0.1"])
def test_detect_rejects_bad_sparsity_ratio(tiny_lm, tiny_loader, bad):
    cfg = CriticalNeuronConfig()
    det = NeuronDetector(tiny_lm, cfg)
    with pytest.raises((ValueError, TypeError)):
        det.detect(tiny_loader, sparsity_ratio=bad)


# ---------------------------------------------------------------------------
# End-to-end detection
# ---------------------------------------------------------------------------


def _count_target_neurons(lm) -> int:
    cfg = CriticalNeuronConfig()
    total = 0
    for name, module in lm.named_modules():
        kind = cfg.classify(name)
        if kind is None or getattr(module, "weight", None) is None:
            continue
        if kind in ("row", "embedding"):
            total += module.weight.size(0)
        elif kind == "column":
            total += module.weight.size(1)
        elif kind == "norm":
            total += module.weight.size(0)
    return total


def test_detect_returns_detection_result_with_expected_total(tiny_lm, tiny_loader):
    cfg = CriticalNeuronConfig()
    det = NeuronDetector(tiny_lm, cfg)
    result = det.detect(tiny_loader, sparsity_ratio=0.10)

    assert isinstance(result, DetectionResult)
    assert result.sparsity_ratio == 0.10

    # With gate combination, the pool drops gate_proj rows into up_proj.
    # Total selected count should equal int(remaining_pool * 0.10)
    # plus the gate-mirroring extra (gate gets a *copy* of partner's indices).
    # We just check it's > 0 and respects 0 < total <= total_targeted.
    total = _count_target_neurons(tiny_lm)
    assert 0 < result.total_selected <= total


def test_detect_indices_are_sorted_and_in_range(tiny_lm, tiny_loader):
    cfg = CriticalNeuronConfig()
    det = NeuronDetector(tiny_lm, cfg)
    result = det.detect(tiny_loader, sparsity_ratio=0.20)
    named = dict(tiny_lm.named_modules())
    for mod_path, idx in result.indices.items():
        assert idx == sorted(idx)
        mod_type = cfg.classify(mod_path)
        w = named[mod_path].weight
        if mod_type == "row" or mod_type == "embedding":
            assert max(idx) < w.size(0)
        elif mod_type == "column":
            assert max(idx) < w.size(1)
        else:
            assert max(idx) < w.size(0)


def test_detect_restores_training_and_requires_grad(tiny_lm, tiny_loader):
    cfg = CriticalNeuronConfig()
    tiny_lm.eval()
    for p in tiny_lm.parameters():
        p.requires_grad_(False)

    det = NeuronDetector(tiny_lm, cfg)
    det.detect(tiny_loader, sparsity_ratio=0.05)

    assert not tiny_lm.training, "model.train() leaked out of detect()"
    for p in tiny_lm.parameters():
        assert p.requires_grad is False, "requires_grad leaked out of detect()"


def test_detect_zeros_grad_on_exit(tiny_lm, tiny_loader):
    cfg = CriticalNeuronConfig()
    det = NeuronDetector(tiny_lm, cfg)
    det.detect(tiny_loader, sparsity_ratio=0.05)
    for p in tiny_lm.parameters():
        assert p.grad is None


def test_detect_gate_and_partner_share_indices(tiny_lm, tiny_loader):
    cfg = CriticalNeuronConfig()
    det = NeuronDetector(tiny_lm, cfg)
    result = det.detect(tiny_loader, sparsity_ratio=0.20)
    for gate, partner in result.gate_to_partner_path.items():
        assert gate in result.indices and partner in result.indices
        assert result.indices[gate] == result.indices[partner]


def test_detect_with_no_targets_raises(tiny_lm, tiny_loader):
    cfg = CriticalNeuronConfig(
        row_modules=[], column_modules=[], norm_modules=[], embedding_modules=[],
    )
    det = NeuronDetector(tiny_lm, cfg)
    with pytest.raises(ValueError, match="No target parameters"):
        det.detect(tiny_loader, sparsity_ratio=0.05)


# ---------------------------------------------------------------------------
# Importance cache round-trip
# ---------------------------------------------------------------------------


def test_importance_cache_roundtrip_matches(tiny_lm, tiny_loader, tmp_path):
    cfg = CriticalNeuronConfig()
    cache = tmp_path / "imp.pt"
    det = NeuronDetector(tiny_lm, cfg)
    r1 = det.detect(
        tiny_loader, sparsity_ratio=0.05,
        save_importance_cache_path=str(cache),
    )
    assert cache.exists()
    r2 = select_neurons_from_cache(str(cache), cfg, sparsity_ratio=0.05)
    assert r1.indices == r2.indices


def test_importance_cache_supports_ratio_sweep(tiny_lm, tiny_loader, tmp_path):
    cfg = CriticalNeuronConfig()
    cache = tmp_path / "imp.pt"
    NeuronDetector(tiny_lm, cfg).detect(
        tiny_loader, sparsity_ratio=0.10,
        save_importance_cache_path=str(cache),
    )
    low = select_neurons_from_cache(str(cache), cfg, sparsity_ratio=0.05)
    high = select_neurons_from_cache(str(cache), cfg, sparsity_ratio=0.30)
    # higher ratio => strictly more (or equal) neurons selected
    assert high.total_selected >= low.total_selected


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def test_apply_gate_combination_sums_into_partner():
    cfg = CriticalNeuronConfig()
    scores = {
        "m.0.mlp.gate_proj": torch.tensor([1.0, 2.0, 3.0]),
        "m.0.mlp.up_proj":   torch.tensor([10.0, 20.0, 30.0]),
        "m.0.self_attn.q_proj": torch.tensor([5.0, 5.0, 5.0]),
    }
    combined, gate_map = _apply_gate_combination(scores, cfg)
    # gate folded into partner
    assert "m.0.mlp.gate_proj" not in combined
    assert combined["m.0.mlp.up_proj"].tolist() == [11.0, 22.0, 33.0]
    assert combined["m.0.self_attn.q_proj"].tolist() == [5.0, 5.0, 5.0]
    assert gate_map == {"m.0.mlp.gate_proj": "m.0.mlp.up_proj"}


def test_apply_gate_combination_handles_shape_mismatch_gracefully():
    cfg = CriticalNeuronConfig()
    scores = {
        "m.0.mlp.gate_proj": torch.tensor([1.0, 2.0]),
        "m.0.mlp.up_proj":   torch.tensor([10.0, 20.0, 30.0]),  # different length
    }
    combined, gate_map = _apply_gate_combination(scores, cfg)
    # shape mismatch -> not combined, gate still in scores
    assert "m.0.mlp.gate_proj" in combined
    assert combined["m.0.mlp.up_proj"].tolist() == [10.0, 20.0, 30.0]


def test_global_topk_count_matches_request():
    scores = {
        "a": torch.tensor([0.1, 0.9, 0.2, 0.8]),
        "b": torch.tensor([0.5, 0.5]),
    }
    out = _global_topk(scores, sparsity_ratio=0.5, gate_to_partner_path={})
    total = sum(len(v) for v in out.values())
    assert total == int(6 * 0.5) == 3


def test_global_topk_mirrors_gate_indices():
    scores = {
        "gate": torch.tensor([0.0, 9.9, 0.0, 9.9]),
        "up":   torch.tensor([0.0, 9.9, 0.0, 9.9]),
    }
    # mark them as paired (real call removes gate from scores; here we simulate
    # the post-_apply_gate_combination state)
    out = _global_topk(
        {"up": scores["up"]},
        sparsity_ratio=0.5,
        gate_to_partner_path={"gate": "up"},
    )
    assert out["gate"] == out["up"]

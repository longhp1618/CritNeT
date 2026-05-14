"""Tests for :mod:`critnet.statistician`."""

from __future__ import annotations

import csv
import json
import os

import pytest

from critnet import CriticalNeuronConfig, NeuronStatistician, StatisticsResult


def _task_indices_two():
    return {
        "A": {
            "layers.0.self_attn.q_proj": [0, 1, 2],
            "layers.0.mlp.up_proj":      [4, 5],
        },
        "B": {
            "layers.0.self_attn.q_proj": [1, 2, 3],
            "layers.0.mlp.up_proj":      [5, 6],
        },
    }


def test_analyze_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        NeuronStatistician().analyze({})


def test_set_algebra_two_tasks():
    res = NeuronStatistician().analyze(_task_indices_two())
    q = "layers.0.self_attn.q_proj"
    u = "layers.0.mlp.up_proj"
    assert res.union[q] == [0, 1, 2, 3]
    assert res.intersection[q] == [1, 2]
    assert res.exclusive["A"][q] == [0]
    assert res.exclusive["B"][q] == [3]
    assert res.non_shared[q] == [0, 3]
    assert res.union[u] == [4, 5, 6]
    assert res.intersection[u] == [5]
    assert res.exclusive["A"][u] == [4]
    assert res.exclusive["B"][u] == [6]
    assert res.non_shared[u] == [4, 6]


def test_set_algebra_three_tasks_intersection():
    tasks = {
        "A": {"m.q_proj": [1, 2, 3]},
        "B": {"m.q_proj": [2, 3, 4]},
        "C": {"m.q_proj": [3, 4, 5]},
    }
    res = NeuronStatistician().analyze(tasks)
    assert res.intersection["m.q_proj"] == [3]
    assert res.union["m.q_proj"] == [1, 2, 3, 4, 5]
    # exclusive[task] = task indices minus the global intersection
    assert res.exclusive["A"]["m.q_proj"] == [1, 2]
    assert res.exclusive["B"]["m.q_proj"] == [2, 4]
    assert res.exclusive["C"]["m.q_proj"] == [4, 5]


def test_counts_property_helpers():
    res = NeuronStatistician().analyze(_task_indices_two())
    assert res.union_count == 4 + 3       # q union + up union
    assert res.intersection_count == 2 + 1
    assert res.non_shared_count == 2 + 2
    assert res.task_count("A") == 3 + 2
    assert res.exclusive_count("A") == 1 + 1


def test_param_coverage_with_model(tiny_lm):
    cfg = CriticalNeuronConfig()
    stats = NeuronStatistician(model=tiny_lm, config=cfg)
    res = stats.analyze(_task_indices_two())
    # totals come from real weight shapes (hidden=8)
    q_total = tiny_lm.layers[0].self_attn.q_proj.weight.size(0)
    up_total = tiny_lm.layers[0].mlp.up_proj.weight.size(0)
    assert res.total_neurons_per_module["layers.0.self_attn.q_proj"] == q_total
    assert res.total_neurons_per_module["layers.0.mlp.up_proj"] == up_total
    # params_per_neuron for a row module is in_features = hidden
    assert res.params_per_neuron["layers.0.self_attn.q_proj"] == q_total
    # union coverage = (4 q rows * hidden + 3 up rows * hidden) / total_model_params
    cov = res.param_coverage(res.union)
    assert 0.0 < cov < 100.0


def test_param_coverage_without_model_returns_zero():
    res = NeuronStatistician().analyze(_task_indices_two())
    assert res.param_coverage(res.union) == 0.0
    assert res.total_model_params == 0


def test_summary_string_mentions_every_task():
    res = NeuronStatistician().analyze(_task_indices_two())
    s = res.summary()
    assert "A" in s and "B" in s
    assert "union" in s
    assert "intersection" in s


def test_save_report_writes_expected_files(tmp_path):
    res = NeuronStatistician().analyze(_task_indices_two())
    res.save_report(str(tmp_path))

    expected = {
        "union_neurons.json",
        "shared_neurons.json",
        "non_shared_neurons.json",
        "exclusive_A_neurons.json",
        "exclusive_B_neurons.json",
        "statistics.csv",
    }
    assert expected.issubset(set(os.listdir(tmp_path)))

    with open(tmp_path / "union_neurons.json") as f:
        union = json.load(f)
    assert union["layers.0.self_attn.q_proj"] == [0, 1, 2, 3]

    with open(tmp_path / "statistics.csv") as f:
        rows = list(csv.reader(f))
    # header + at least 5 metric rows
    assert rows[0] == ["metric", "count", "neuron_pct", "param_pct"]
    metric_names = {row[0] for row in rows[1:]}
    assert {"union", "shared", "non_shared", "task_A_total", "task_A_exclusive"}.issubset(metric_names)


def test_save_report_runs_with_model(tmp_path, tiny_lm):
    cfg = CriticalNeuronConfig()
    res = NeuronStatistician(tiny_lm, cfg).analyze(_task_indices_two())
    res.save_report(str(tmp_path))
    with open(tmp_path / "statistics.csv") as f:
        rows = list(csv.reader(f))
    # total_model_params should appear
    metric_names = [r[0] for r in rows[1:]]
    assert "total_model_params" in metric_names

"""Set-algebra over neuron index dicts produced by detection runs.

Given two or more detection results (typically: per-task or per-language),
:class:`NeuronStatistician` reports their union, intersection, per-task
exclusive sets, and the union-minus-intersection (non-shared) partition,
together with parameter-coverage statistics derived from the real model
weight shapes.

Every statistic lives on :class:`StatisticsResult`; the analyzer class
itself is a thin builder so users do not depend on its internal state.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import torch.nn as nn

from .config import CriticalNeuronConfig

logger = logging.getLogger(__name__)


# =====================================================================
# StatisticsResult
# =====================================================================


@dataclass
class StatisticsResult:
    """Outcome of :meth:`NeuronStatistician.analyze`.

    All counts are derived deterministically from
    :attr:`task_indices`; this object owns enough state to re-serialise
    itself without any reference back to the analyzer.

    Attributes
    ----------
    union
        Per-module neurons present in **any** task.
    intersection
        Per-module neurons present in **every** task (shared).
    exclusive
        ``task -> module_path -> indices`` -- the per-task neurons left
        after removing the cross-task intersection.  For two tasks this
        coincides with the pairwise set-difference; for three or more it
        is "task minus shared".
    non_shared
        Per-module ``union - intersection``.
    task_indices
        The original ``task_name -> module_path -> indices`` mapping.
    task_names
        Ordered list of task names (insertion order of ``task_indices``).
    total_neurons_per_module
        Per-module neuron count derived from real weight shapes
        (zero-filled when no model was provided).
    params_per_neuron
        Per-module scalar-weight count per neuron index
        (e.g. ``in_features`` for a row module).
    total_model_params
        Total scalar parameter count of the model
        (``0`` when no model was provided).
    """

    union: Dict[str, List[int]] = field(default_factory=dict)
    intersection: Dict[str, List[int]] = field(default_factory=dict)
    exclusive: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)
    non_shared: Dict[str, List[int]] = field(default_factory=dict)
    task_indices: Mapping[str, Mapping[str, List[int]]] = field(default_factory=dict)
    task_names: List[str] = field(default_factory=list)
    total_neurons_per_module: Dict[str, int] = field(default_factory=dict)
    params_per_neuron: Dict[str, int] = field(default_factory=dict)
    total_model_params: int = 0

    # ------------------------------------------------------------------
    # Aggregate counts
    # ------------------------------------------------------------------

    @property
    def union_count(self) -> int:
        return sum(len(v) for v in self.union.values())

    @property
    def intersection_count(self) -> int:
        return sum(len(v) for v in self.intersection.values())

    @property
    def non_shared_count(self) -> int:
        return sum(len(v) for v in self.non_shared.values())

    @property
    def total_neurons(self) -> int:
        return sum(self.total_neurons_per_module.values())

    def exclusive_count(self, task: str) -> int:
        return sum(len(v) for v in self.exclusive.get(task, {}).values())

    def task_count(self, task: str) -> int:
        """Total selected neurons for *task* (across all modules)."""
        return sum(len(v) for v in self.task_indices.get(task, {}).values())

    # ------------------------------------------------------------------
    # Parameter-coverage helpers
    # ------------------------------------------------------------------

    def _scalar_params(self, indices: Mapping[str, List[int]]) -> int:
        return sum(
            len(idx) * self.params_per_neuron.get(mp, 0)
            for mp, idx in indices.items()
        )

    def param_coverage(self, indices: Mapping[str, List[int]]) -> float:
        """Percentage (0--100) of model parameters covered by *indices*."""
        if self.total_model_params == 0:
            return 0.0
        return 100.0 * self._scalar_params(indices) / self.total_model_params

    # ------------------------------------------------------------------
    # Text summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary table."""
        total_n = self.total_neurons or 1

        def _pct_n(count: int) -> str:
            return f"{100 * count / total_n:.2f}%"

        def _pct_p(indices: Mapping[str, List[int]]) -> str:
            return f"{self.param_coverage(indices):.2f}%"

        lines = [
            f"Total neurons across targeted modules: {self.total_neurons:,}",
            f"Total model parameters: {self.total_model_params:,}",
            f"{'':36s} {'count':>12s}  {'neuron%':>10s}  {'param%':>10s}",
            f"  union (any task)        {self.union_count:>12,}  "
            f"{_pct_n(self.union_count):>10s}  {_pct_p(self.union):>10s}",
            f"  intersection (shared)   {self.intersection_count:>12,}  "
            f"{_pct_n(self.intersection_count):>10s}  {_pct_p(self.intersection):>10s}",
            f"  non-shared              {self.non_shared_count:>12,}  "
            f"{_pct_n(self.non_shared_count):>10s}  {_pct_p(self.non_shared):>10s}",
        ]
        for task in self.task_names:
            tc = self.task_count(task)
            ec = self.exclusive_count(task)
            lines.append(
                f"  {task}: {tc:>10,} ({_pct_n(tc)} / {_pct_p(self.task_indices[task])} param), "
                f"{ec:,} exclusive ({_pct_n(ec)} / {_pct_p(self.exclusive[task])} param)"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # On-disk report
    # ------------------------------------------------------------------

    def save_report(self, save_directory: str) -> None:
        """Persist this result to *save_directory*.

        Writes:

        * ``union_neurons.json``, ``shared_neurons.json``,
          ``non_shared_neurons.json``
        * ``exclusive_<task>_neurons.json`` -- one file per task
        * ``statistics.csv`` -- aggregate counts and coverages
        """
        os.makedirs(save_directory, exist_ok=True)

        _dump_json(self.union, os.path.join(save_directory, "union_neurons.json"))
        _dump_json(
            self.intersection,
            os.path.join(save_directory, "shared_neurons.json"),
        )
        _dump_json(
            self.non_shared,
            os.path.join(save_directory, "non_shared_neurons.json"),
        )
        for task in self.task_names:
            _dump_json(
                self.exclusive[task],
                os.path.join(save_directory, f"exclusive_{task}_neurons.json"),
            )

        csv_path = os.path.join(save_directory, "statistics.csv")
        total_n = self.total_neurons or 1
        total_p = self.total_model_params or 1
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "count", "neuron_pct", "param_pct"])

            total_scalar = sum(
                n * self.params_per_neuron.get(mp, 0)
                for mp, n in self.total_neurons_per_module.items()
            )
            writer.writerow([
                "total_neurons", self.total_neurons,
                "100.00%", f"{100 * total_scalar / total_p:.4f}%",
            ])
            writer.writerow([
                "total_model_params", self.total_model_params, "", "100.00%",
            ])
            for label, count, idx_dict in (
                ("union", self.union_count, self.union),
                ("shared", self.intersection_count, self.intersection),
                ("non_shared", self.non_shared_count, self.non_shared),
            ):
                writer.writerow([
                    label, count,
                    f"{100 * count / total_n:.4f}%",
                    f"{self.param_coverage(idx_dict):.4f}%",
                ])
            for task in self.task_names:
                tc = self.task_count(task)
                ec = self.exclusive_count(task)
                writer.writerow([
                    f"task_{task}_total", tc,
                    f"{100 * tc / total_n:.4f}%",
                    f"{self.param_coverage(self.task_indices[task]):.4f}%",
                ])
                writer.writerow([
                    f"task_{task}_exclusive", ec,
                    f"{100 * ec / total_n:.4f}%",
                    f"{self.param_coverage(self.exclusive[task]):.4f}%",
                ])

        logger.info("Statistics saved to %s", save_directory)


# =====================================================================
# NeuronStatistician
# =====================================================================


class NeuronStatistician:
    """Compute set-algebra over multiple detection runs.

    The class is a thin builder: it caches the optional *model* and
    *config* (so weight shapes do not have to be re-walked for every
    ``analyze`` call) and returns a freshly-constructed
    :class:`StatisticsResult` per call.  No state is retained between
    calls.

    Parameters
    ----------
    model
        If provided, used to derive per-module neuron totals from real
        weight shapes.  When ``None``, count-based percentages become
        relative-to-selected only and param coverage is ``0``.
    config
        Used to classify modules (row vs column vs norm vs embedding).
        Without it the analyzer falls back to a "row" assumption for
        2-D weights.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[CriticalNeuronConfig] = None,
    ) -> None:
        self.model = model
        self.config = config

    def analyze(
        self, task_indices: Mapping[str, Mapping[str, List[int]]]
    ) -> StatisticsResult:
        """Compute union, intersection, exclusive, and non-shared sets.

        Parameters
        ----------
        task_indices
            Outer key: task / language name.  Inner dict:
            ``{module_path: [neuron_idx, ...]}`` as produced by
            :class:`~critnet.detector.NeuronDetector`.

        Returns
        -------
        StatisticsResult
        """
        if not task_indices:
            raise ValueError("task_indices is empty -- nothing to analyse.")

        task_names = list(task_indices.keys())
        all_modules: Set[str] = set()
        for inner in task_indices.values():
            all_modules.update(inner.keys())
        module_paths = sorted(all_modules)

        per_task_sets: Dict[str, Dict[str, Set[int]]] = {
            t: {mp: set(task_indices[t].get(mp, [])) for mp in module_paths}
            for t in task_names
        }

        union: Dict[str, List[int]] = {}
        intersection: Dict[str, List[int]] = {}
        for mp in module_paths:
            sets = [per_task_sets[t][mp] for t in task_names]
            union[mp] = sorted(set.union(*sets)) if sets else []
            intersection[mp] = sorted(set.intersection(*sets)) if sets else []

        exclusive: Dict[str, Dict[str, List[int]]] = {}
        for t in task_names:
            shared_inv: Dict[str, Set[int]] = {
                mp: set(intersection[mp]) for mp in module_paths
            }
            exclusive[t] = {
                mp: sorted(per_task_sets[t][mp] - shared_inv[mp])
                for mp in module_paths
            }

        non_shared: Dict[str, List[int]] = {
            mp: sorted(set(union[mp]) - set(intersection[mp]))
            for mp in module_paths
        }

        totals, ppn = _module_totals(self.model, self.config, module_paths)
        total_model_params = (
            sum(p.numel() for p in self.model.parameters())
            if self.model is not None
            else 0
        )

        return StatisticsResult(
            union=union,
            intersection=intersection,
            exclusive=exclusive,
            non_shared=non_shared,
            task_indices={t: dict(task_indices[t]) for t in task_names},
            task_names=task_names,
            total_neurons_per_module=totals,
            params_per_neuron=ppn,
            total_model_params=total_model_params,
        )


# =====================================================================
# Internals
# =====================================================================


def _module_totals(
    model: Optional[nn.Module],
    config: Optional[CriticalNeuronConfig],
    module_paths: List[str],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Walk the model to derive (total_neurons_per_module, params_per_neuron)."""
    totals: Dict[str, int] = {}
    ppn: Dict[str, int] = {}
    if model is None:
        return totals, ppn

    named = dict(model.named_modules())
    for mp in module_paths:
        module = named.get(mp)
        if module is None or getattr(module, "weight", None) is None:
            continue
        w = module.weight
        mod_type = "row"
        if config is not None:
            mod_type = config.classify(mp) or "row"

        if mod_type == "column" and w.dim() == 2:
            totals[mp] = w.size(1)
            ppn[mp] = w.size(0)
        elif mod_type == "norm" and w.dim() == 1:
            totals[mp] = w.size(0)
            ppn[mp] = 1
        elif mod_type == "embedding" and w.dim() == 2:
            totals[mp] = w.size(0)
            ppn[mp] = w.size(1)
        elif w.dim() == 2:
            totals[mp] = w.size(0)
            ppn[mp] = w.size(1)
        else:
            totals[mp] = w.size(0)
            ppn[mp] = 1
    return totals, ppn


def _dump_json(data: Dict[str, Any], path: str) -> None:
    payload = {k: list(v) if isinstance(v, set) else v for k, v in data.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

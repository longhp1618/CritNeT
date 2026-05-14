"""Cross-task / cross-language analysis of detected critical neurons.

Given neuron index sets from multiple tasks (or languages), the
:class:`NeuronStatistician` computes structural partitions (union,
intersection, exclusive sets) and reports statistics.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import torch.nn as nn

from .config import CriticalNeuronConfig

logger = logging.getLogger(__name__)


@dataclass
class StatisticsResult:
    """Container returned by :meth:`NeuronStatistician.analyze`.

    Attributes
    ----------
    union : dict[str, list[int]]
        Global critical neurons -- union across all tasks.
    intersection : dict[str, list[int]]
        Shared neurons -- intersection across all tasks.
    exclusive : dict[str, dict[str, list[int]]]
        Per-task exclusive neurons (task -> module_path -> indices).
        Exclusive = task set minus shared intersection.
    non_shared : dict[str, list[int]]
        Non-shared critical neurons: union minus intersection.
    total_neurons_per_module : dict[str, int]
        Total neuron count for each module path (from model weights).
    params_per_neuron : dict[str, int]
        Scalar weight count that one neuron corresponds to, per module.
    total_model_params : int
        Total scalar parameters in the entire model.
    task_names : list[str]
        Ordered list of task/language names.
    """

    union: Dict[str, List[int]] = field(default_factory=dict)
    intersection: Dict[str, List[int]] = field(default_factory=dict)
    exclusive: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)
    non_shared: Dict[str, List[int]] = field(default_factory=dict)
    total_neurons_per_module: Dict[str, int] = field(default_factory=dict)
    params_per_neuron: Dict[str, int] = field(default_factory=dict)
    total_model_params: int = 0
    task_names: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience counts
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

    def task_count(self, task: str, task_indices: Dict[str, Dict[str, List[int]]]) -> int:
        return sum(len(v) for v in task_indices.get(task, {}).values())

    # ------------------------------------------------------------------
    # Parameter-level coverage
    # ------------------------------------------------------------------

    def _params_for_indices(self, indices: Dict[str, List[int]]) -> int:
        """Sum the scalar weights covered by a set of neuron indices."""
        total = 0
        for mp, idxs in indices.items():
            ppn = self.params_per_neuron.get(mp, 0)
            total += len(idxs) * ppn
        return total

    def param_coverage(self, indices: Dict[str, List[int]]) -> float:
        """Percentage of total model parameters covered by *indices*."""
        if self.total_model_params == 0:
            return 0.0
        return 100.0 * self._params_for_indices(indices) / self.total_model_params

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, task_indices: Optional[Dict[str, Dict[str, List[int]]]] = None) -> str:
        """Return a human-readable summary string."""
        total_n = self.total_neurons or 1
        total_p = self.total_model_params or 1

        def _pct_n(count: int) -> str:
            return f"{100 * count / total_n:.2f}%"

        def _pct_p(indices: Dict[str, List[int]]) -> str:
            return f"{self.param_coverage(indices):.2f}%"

        lines = [
            f"Total neurons across targeted modules: {self.total_neurons:,}",
            f"Total model parameters: {self.total_model_params:,}",
            f"{'':36s} {'neuron%':>10s}  {'param%':>10s}",
            f"Union (global critical): {self.union_count:>10,}  {_pct_n(self.union_count):>10s}  {_pct_p(self.union):>10s}",
            f"Intersection (shared):   {self.intersection_count:>10,}  {_pct_n(self.intersection_count):>10s}  {_pct_p(self.intersection):>10s}",
            f"Non-shared (exclusive):  {self.non_shared_count:>10,}  {_pct_n(self.non_shared_count):>10s}  {_pct_p(self.non_shared):>10s}",
        ]
        if task_indices:
            for task in self.task_names:
                tc = self.task_count(task, task_indices)
                ec = self.exclusive_count(task)
                t_pp = _pct_p(task_indices[task])
                e_pp = _pct_p(self.exclusive[task])
                lines.append(
                    f"  {task}: {tc:,} total ({_pct_n(tc)} / {t_pp} param), "
                    f"{ec:,} exclusive ({_pct_n(ec)} / {e_pp} param)"
                )
        return "\n".join(lines)


class NeuronStatistician:
    """Analyse neuron index sets across multiple tasks or languages.

    Parameters
    ----------
    model : nn.Module or None
        If provided, used to derive the total neuron count per module
        from the actual weight shapes.  When ``None``, total counts are
        unavailable and percentages will be relative counts only.
    config : CriticalNeuronConfig or None
        Used to determine module types (row vs column) for correct
        neuron-count extraction from weight shapes.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[CriticalNeuronConfig] = None,
    ) -> None:
        self.model = model
        self.config = config
        self._result: Optional[StatisticsResult] = None
        self._task_indices: Optional[Dict[str, Dict[str, List[int]]]] = None

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(
        self, task_indices: Dict[str, Dict[str, List[int]]]
    ) -> StatisticsResult:
        """Compute union, intersection, exclusive, and non-shared sets.

        Parameters
        ----------
        task_indices : dict[str, dict[str, list[int]]]
            Outer key = task / language name.
            Inner dict = ``{module_path: [neuron_idx, ...]}``
            as produced by :meth:`NeuronDetector.detect`.

        Returns
        -------
        StatisticsResult
        """
        if len(task_indices) == 0:
            raise ValueError("task_indices is empty -- nothing to analyse.")

        task_names = list(task_indices.keys())
        all_module_paths: Set[str] = set()
        for indices in task_indices.values():
            all_module_paths.update(indices.keys())
        module_paths = sorted(all_module_paths)

        # Convert to per-module sets
        per_task_sets: Dict[str, Dict[str, Set[int]]] = {}
        for task, indices in task_indices.items():
            per_task_sets[task] = {
                mp: set(indices.get(mp, [])) for mp in module_paths
            }

        # Union & intersection per module
        union: Dict[str, List[int]] = {}
        intersection: Dict[str, List[int]] = {}
        for mp in module_paths:
            sets = [per_task_sets[t][mp] for t in task_names]
            union[mp] = sorted(set.union(*sets)) if sets else []
            intersection[mp] = sorted(set.intersection(*sets)) if sets else []

        # Exclusive per task: task set minus intersection
        exclusive: Dict[str, Dict[str, List[int]]] = {}
        for task in task_names:
            exclusive[task] = {}
            for mp in module_paths:
                excl = per_task_sets[task][mp] - set(intersection[mp])
                exclusive[task][mp] = sorted(excl)

        # Non-shared: union minus intersection
        non_shared: Dict[str, List[int]] = {}
        for mp in module_paths:
            ns = set(union[mp]) - set(intersection[mp])
            non_shared[mp] = sorted(ns)

        # Total neuron counts and per-neuron param sizes from model
        total_per_module, ppn = self._get_total_neurons(module_paths)
        total_model_params = (
            sum(p.numel() for p in self.model.parameters())
            if self.model is not None else 0
        )

        result = StatisticsResult(
            union=union,
            intersection=intersection,
            exclusive=exclusive,
            non_shared=non_shared,
            total_neurons_per_module=total_per_module,
            params_per_neuron=ppn,
            total_model_params=total_model_params,
            task_names=task_names,
        )
        self._result = result
        self._task_indices = task_indices
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_report(self, save_directory: str) -> None:
        """Save analysis results: CSVs and JSON index files.

        Writes to *save_directory*:

        * ``statistics.csv`` -- summary table.
        * ``union_neurons.json`` -- global critical neurons.
        * ``shared_neurons.json`` -- intersection neurons.
        * ``exclusive_{task}_neurons.json`` -- per-task exclusive neurons.
        * ``non_shared_neurons.json`` -- union minus intersection.
        """
        if self._result is None:
            raise RuntimeError("Call .analyze() before .save_report().")

        os.makedirs(save_directory, exist_ok=True)
        r = self._result

        # JSON dumps
        self._save_json(r.union, os.path.join(save_directory, "union_neurons.json"))
        self._save_json(r.intersection, os.path.join(save_directory, "shared_neurons.json"))
        self._save_json(r.non_shared, os.path.join(save_directory, "non_shared_neurons.json"))
        for task in r.task_names:
            self._save_json(
                r.exclusive[task],
                os.path.join(save_directory, f"exclusive_{task}_neurons.json"),
            )

        # CSV summary
        csv_path = os.path.join(save_directory, "statistics.csv")
        total_n = r.total_neurons or 1
        total_p = r.total_model_params or 1
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["metric", "count", "neuron_pct", "param_pct"]
            writer.writerow(header)
            writer.writerow([
                "total_neurons", r.total_neurons, "100.00%",
                f"{100 * sum(n * r.params_per_neuron.get(mp, 0) for mp, n in r.total_neurons_per_module.items()) / total_p:.4f}%",
            ])
            writer.writerow([
                "total_model_params", r.total_model_params, "", "100.00%",
            ])
            writer.writerow([
                "union", r.union_count,
                f"{100 * r.union_count / total_n:.4f}%",
                f"{r.param_coverage(r.union):.4f}%",
            ])
            writer.writerow([
                "shared", r.intersection_count,
                f"{100 * r.intersection_count / total_n:.4f}%",
                f"{r.param_coverage(r.intersection):.4f}%",
            ])
            writer.writerow([
                "non_shared", r.non_shared_count,
                f"{100 * r.non_shared_count / total_n:.4f}%",
                f"{r.param_coverage(r.non_shared):.4f}%",
            ])
            if self._task_indices:
                for task in r.task_names:
                    tc = r.task_count(task, self._task_indices)
                    ec = r.exclusive_count(task)
                    writer.writerow([
                        f"task_{task}_total", tc,
                        f"{100 * tc / total_n:.4f}%",
                        f"{r.param_coverage(self._task_indices[task]):.4f}%",
                    ])
                    writer.writerow([
                        f"task_{task}_exclusive", ec,
                        f"{100 * ec / total_n:.4f}%",
                        f"{r.param_coverage(r.exclusive[task]):.4f}%",
                    ])

        logger.info("Statistics saved to %s", save_directory)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_total_neurons(
        self, module_paths: List[str]
    ) -> tuple:
        """Derive total neuron count and params-per-neuron per module.

        Returns ``(total_neurons_per_module, params_per_neuron)`` where
        both are ``dict[str, int]``.  ``params_per_neuron[mp]`` is the
        number of scalar weights that a single neuron index in module
        *mp* corresponds to (e.g. ``in_features`` for a row module).
        """
        totals: Dict[str, int] = {}
        ppn: Dict[str, int] = {}
        if self.model is None:
            return totals, ppn

        named_modules = dict(self.model.named_modules())
        for mp in module_paths:
            module = named_modules.get(mp)
            if module is None or not hasattr(module, "weight") or module.weight is None:
                continue

            w = module.weight
            mod_type = "row"
            if self.config is not None:
                try:
                    mod_type = self.config.get_module_type(mp)
                except ValueError:
                    mod_type = "row"

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

    @staticmethod
    def _save_json(data: Dict[str, Any], path: str) -> None:
        serializable = {k: list(v) if isinstance(v, set) else v for k, v in data.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

"""First-order Taylor importance detection.

For a parameter tensor :math:`W` and a calibration loss :math:`\\mathcal{L}`,
each candidate "neuron" -- a row of :math:`W`, a column of :math:`W`, or a
single scalar of a 1-D norm weight -- receives a scalar importance score

.. math::

    \\mathcal{I}(w_i) \\;\\approx\\; \\bigl|w_i^{\\top}\\,\\nabla_{w_i}\\mathcal{L}\\bigr|.

Scores from every targeted module are pooled into a single list and the
top ``sparsity_ratio`` fraction is kept.  SwiGLU gate/up partners share
selected indices (see :class:`~critnet.config.CriticalNeuronConfig`).

Public surface
--------------
* :class:`NeuronDetector` -- runs the detection from a model + dataloader.
* :class:`DetectionResult` -- immutable record returned by both
  :meth:`NeuronDetector.detect` and :func:`select_neurons_from_cache`.
* :func:`select_neurons_from_cache` -- replay top-k from a saved importance
  cache without loading a model.
"""

from __future__ import annotations

import inspect
import logging
import os
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import CriticalNeuronConfig

logger = logging.getLogger(__name__)

# Schema version for files written by :func:`_save_importance_cache`.
# Bumped only on breaking changes; readers refuse mismatching majors.
_IMPORTANCE_CACHE_VERSION = 1


# =====================================================================
# DetectionResult
# =====================================================================


@dataclass
class DetectionResult:
    """Immutable record of one detection run.

    Attributes
    ----------
    indices
        Mapping from full module path to the **sorted** list of selected
        neuron indices.
    sparsity_ratio
        The fraction of pooled neurons retained.
    gate_to_partner_path
        Mapping from each gate module path to its partner module path
        (for SwiGLU-style sharing).  Empty when ``gate_combines_with``
        was disabled or no gate/partner pairs were targeted.

    Notes
    -----
    The result deliberately does **not** mutate any config -- to attach
    these indices to a config for downstream use, do so explicitly::

        result = detector.detect(loader, sparsity_ratio=0.05)
        config.save_pretrained("./neurons", indices=result.indices)
    """

    indices: Dict[str, List[int]]
    sparsity_ratio: float
    gate_to_partner_path: Dict[str, str] = field(default_factory=dict)

    @property
    def total_selected(self) -> int:
        """Total number of selected neurons across all modules."""
        return sum(len(v) for v in self.indices.values())

    @property
    def n_modules(self) -> int:
        """Number of modules that received at least one selected neuron."""
        return len(self.indices)

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"DetectionResult(sparsity_ratio={self.sparsity_ratio:.4f}, "
            f"total_selected={self.total_selected:,}, "
            f"n_modules={self.n_modules})"
        )


# =====================================================================
# NeuronDetector
# =====================================================================


class NeuronDetector:
    """Detect critical neurons from a calibration corpus.

    Parameters
    ----------
    model
        A pretrained ``transformers`` model **before** any wrapping.
    config
        Architectural description of which modules carry neurons.

    Example
    -------
    >>> detector = NeuronDetector(model, config)
    >>> result = detector.detect(loader, sparsity_ratio=0.05)
    >>> config.save_pretrained("./neurons", indices=result.indices)

    For a cache-only ratio sweep (no model load), use the module-level
    :func:`select_neurons_from_cache` instead.
    """

    def __init__(self, model: nn.Module, config: CriticalNeuronConfig) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(
                "NeuronDetector requires a model. For cache-only selection use "
                "critnet.select_neurons_from_cache(cache_path, config, sparsity_ratio=...)."
            )
        self.model = model
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        dataloader: DataLoader,
        *,
        sparsity_ratio: float,
        save_importance_cache_path: Optional[str] = None,
    ) -> DetectionResult:
        """Accumulate gradients over *dataloader* and return the top-k neurons.

        Parameters
        ----------
        dataloader
            Each batch must be a ``dict`` containing at least
            ``input_ids``, ``attention_mask``, and ``labels``.  The
            collator is responsible for preparing ``labels``: mask prompt
            tokens to ``-100`` for chat SFT, or set ``labels`` equal to
            ``input_ids`` for pre-training.
        sparsity_ratio
            Fraction of pooled neurons (across every targeted module) to
            retain.  Must be strictly in :math:`(0, 1]`.
        save_importance_cache_path
            Optional path; when given, the **post-gate-combined** per-module
            score tensors and gate metadata are pickled with
            :func:`torch.save` so a later run can call
            :func:`select_neurons_from_cache` with a different
            ``sparsity_ratio`` without another backward pass.

        Returns
        -------
        DetectionResult

        Notes
        -----
        ``detect`` is non-mutating to the caller's perspective: the
        model's training mode and the ``requires_grad`` flags of target
        parameters are restored on exit, and ``model.zero_grad()`` is
        called both before and after the backward loop.
        """
        _validate_ratio(sparsity_ratio)

        scores, gate_map = self._compute_combined_scores(dataloader)

        if save_importance_cache_path is not None:
            _save_importance_cache(save_importance_cache_path, scores, gate_map)

        indices = _global_topk(scores, sparsity_ratio, gate_map)

        return DetectionResult(
            indices=indices,
            sparsity_ratio=sparsity_ratio,
            gate_to_partner_path=dict(gate_map),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _find_target_params(self) -> List[Tuple[str, nn.Parameter]]:
        """Return ``(full_param_name, Parameter)`` for every targeted weight."""
        out: List[Tuple[str, nn.Parameter]] = []
        for mod_name, module in self.model.named_modules():
            if self.config.classify(mod_name) is None:
                continue
            if getattr(module, "weight", None) is not None:
                out.append((f"{mod_name}.weight", module.weight))
        return out

    def _compute_combined_scores(
        self, dataloader: DataLoader
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
        """Run the backward pass and return (gate-merged scores, gate map)."""
        device = next(self.model.parameters()).device
        target_params = self._find_target_params()

        if not target_params:
            raise ValueError(
                "No target parameters found. Check that the config's module "
                "categories match the model's module names."
            )

        with _detect_state(self.model, [p for _, p in target_params]):
            self.model.zero_grad()

            total_loss = 0.0
            num_batches = 0
            for batch in tqdm(
                dataloader, desc="NeuronDetector: accumulating gradients"
            ):
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**inputs)
                loss = outputs.loss
                loss.backward()
                total_loss += loss.item()
                num_batches += 1

            if num_batches > 0:
                logger.info(
                    "Average loss over %d batches: %.4f",
                    num_batches, total_loss / num_batches,
                )

            raw_scores = self._reduce_scores(target_params)

        scores, gate_map = _apply_gate_combination(raw_scores, self.config)
        return scores, gate_map

    def _reduce_scores(
        self, target_params: List[Tuple[str, nn.Parameter]]
    ) -> Dict[str, torch.Tensor]:
        """Reduce :math:`|W \\odot \\nabla W|` to one score per neuron."""
        scores: Dict[str, torch.Tensor] = {}
        for param_name, param in target_params:
            if param.grad is None:
                continue
            mod_name = param_name.rsplit(".weight", 1)[0]
            mod_type = self.config.classify(mod_name)
            if mod_type is None:
                continue

            importance = torch.abs(param.data * param.grad)
            if mod_type == "row":
                neuron_scores = (
                    importance.sum(dim=1) if importance.dim() == 2 else importance
                )
            elif mod_type == "column":
                neuron_scores = (
                    importance.sum(dim=0) if importance.dim() == 2 else importance
                )
            elif mod_type == "norm":
                neuron_scores = importance
            elif mod_type == "embedding":
                neuron_scores = (
                    importance.sum(dim=1) if importance.dim() == 2 else importance
                )
            else:  # pragma: no cover -- exhaustive over classify() output
                continue

            scores[mod_name] = neuron_scores.detach().float().cpu()
        return scores


# =====================================================================
# Cache-only sweep entry point
# =====================================================================


def select_neurons_from_cache(
    cache_path: str,
    config: CriticalNeuronConfig,
    *,
    sparsity_ratio: float,
) -> DetectionResult:
    """Replay global top-k from a previously saved importance cache.

    The companion of :meth:`NeuronDetector.detect`'s
    ``save_importance_cache_path`` argument.  Reads the cache (no model
    needed), runs top-k at the requested *sparsity_ratio*, and returns a
    fresh :class:`DetectionResult`.  Cheap; safe to call in a tight loop
    when sweeping ratios.

    Parameters
    ----------
    cache_path
        Path to a file written by :meth:`NeuronDetector.detect` via
        ``save_importance_cache_path``.
    config
        Used only to attach :class:`DetectionResult` metadata; the cache
        already carries the per-module scores and gate map.
    sparsity_ratio
        Same semantics as :meth:`NeuronDetector.detect`.

    Returns
    -------
    DetectionResult
    """
    _validate_ratio(sparsity_ratio)
    del config  # accepted for API symmetry; cache is self-sufficient.

    scores, gate_map = _load_importance_cache(cache_path)
    indices = _global_topk(scores, sparsity_ratio, gate_map)
    return DetectionResult(
        indices=indices,
        sparsity_ratio=sparsity_ratio,
        gate_to_partner_path=dict(gate_map),
    )


# =====================================================================
# Module-level helpers
# =====================================================================


def _validate_ratio(sparsity_ratio: float) -> None:
    if not isinstance(sparsity_ratio, (int, float)):
        raise TypeError(f"sparsity_ratio must be a number, got {type(sparsity_ratio).__name__}")
    if not (0.0 < float(sparsity_ratio) <= 1.0):
        raise ValueError(
            f"sparsity_ratio must be in (0, 1], got {sparsity_ratio!r}."
        )


@contextmanager
def _detect_state(
    model: nn.Module, target_params: List[nn.Parameter]
) -> Iterator[None]:
    """Temporarily enable grads on *target_params* and ``train()`` mode.

    Restores ``training`` mode and the previous ``requires_grad`` flags
    on exit, then calls ``model.zero_grad()`` so callers do not inherit
    stale gradients.
    """
    was_training = model.training
    saved_flags = [p.requires_grad for p in target_params]
    try:
        model.train()
        for p in target_params:
            p.requires_grad_(True)
        yield
    finally:
        for p, flag in zip(target_params, saved_flags):
            p.requires_grad_(flag)
        if not was_training:
            model.eval()
        model.zero_grad(set_to_none=True)


def _apply_gate_combination(
    scores: Dict[str, torch.Tensor],
    config: CriticalNeuronConfig,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """Fold gate scores into partner scores; return (scores, gate->partner map).

    For each ``(gate_suffix, partner_suffix)`` in
    ``config.gate_combines_with``, every concrete ``gate_path`` whose
    leaf is ``gate_suffix`` has its scores added element-wise into the
    sibling ``partner_path`` (same parent module, leaf ``partner_suffix``).
    The gate entry is then removed from ``scores`` and recorded in the
    returned map so :func:`_global_topk` can mirror the partner's
    selected indices back onto the gate.
    """
    out = dict(scores)
    gate_suffix_to_partner = config.gate_combines_with or {}
    if not gate_suffix_to_partner:
        return out, {}

    gate_to_partner_path: Dict[str, str] = {}
    for mod_path in list(out.keys()):
        leaf = mod_path.rsplit(".", 1)[-1]
        partner_suffix = gate_suffix_to_partner.get(leaf)
        if partner_suffix is None:
            continue
        partner_path = mod_path.rsplit(".", 1)[0] + "." + partner_suffix
        gate_to_partner_path[mod_path] = partner_path

    for gate_path, partner_path in gate_to_partner_path.items():
        if gate_path in out and partner_path in out:
            if out[gate_path].shape == out[partner_path].shape:
                out[partner_path] = out[partner_path] + out[gate_path]
                del out[gate_path]
            else:
                logger.warning(
                    "Cannot combine gate '%s' (%s) with partner '%s' (%s): "
                    "shape mismatch. Keeping them separate.",
                    gate_path, tuple(out[gate_path].shape),
                    partner_path, tuple(out[partner_path].shape),
                )

    return out, gate_to_partner_path


def _global_topk(
    scores: Dict[str, torch.Tensor],
    sparsity_ratio: float,
    gate_to_partner_path: Dict[str, str],
) -> Dict[str, List[int]]:
    """Pool every per-neuron score and keep the top *sparsity_ratio* fraction.

    After top-k, gates listed in *gate_to_partner_path* receive a copy
    of their partner's selected indices.  Output is sorted per module.
    """
    if not scores:
        logger.warning("No importance scores were computed -- returning empty indices.")
        return {}

    flat: List[float] = []
    meta: List[Tuple[str, int]] = []
    for mod_path, s in scores.items():
        s_flat = s.reshape(-1)
        for idx in range(s_flat.numel()):
            flat.append(s_flat[idx].item())
            meta.append((mod_path, idx))

    total = len(flat)
    k = max(1, int(total * sparsity_ratio))
    scores_tensor = torch.tensor(flat)
    top_vals, top_idx = torch.topk(scores_tensor, k, largest=True, sorted=True)

    selected: Dict[str, List[int]] = defaultdict(list)
    for flat_idx in top_idx.tolist():
        mod_path, neuron_idx = meta[flat_idx]
        selected[mod_path].append(neuron_idx)

    for gate_path, partner_path in gate_to_partner_path.items():
        if partner_path in selected:
            selected[gate_path] = list(selected[partner_path])

    for mod_path in selected:
        selected[mod_path].sort()

    logger.info(
        "Global top-k: %d / %d neurons (score range %.6g .. %.6g)",
        k, total, top_vals[-1].item(), top_vals[0].item(),
    )
    return dict(selected)


# ---------------------------------------------------------------------------
# Importance cache I/O
# ---------------------------------------------------------------------------


def _save_importance_cache(
    path: str,
    scores: Dict[str, torch.Tensor],
    gate_to_partner_path: Dict[str, str],
) -> None:
    """Write the combined-score blob to *path* (``torch.save``)."""
    payload = {
        "version": _IMPORTANCE_CACHE_VERSION,
        "scores": {k: v.contiguous().clone() for k, v in scores.items()},
        "gate_to_partner_path": dict(gate_to_partner_path),
    }
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(payload, path)
    logger.info(
        "Wrote importance cache (v%d, %d modules) to %s",
        _IMPORTANCE_CACHE_VERSION, len(scores), path,
    )


def _load_importance_cache(
    path: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """Read an importance cache and return ``(scores, gate_to_partner_path)``."""
    load_kw: Dict[str, object] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kw["weights_only"] = False
    payload = torch.load(path, **load_kw)  # type: ignore[arg-type]

    if not isinstance(payload, dict) or "scores" not in payload:
        # Tolerate raw ``dict[str, Tensor]`` blobs from older / manual saves.
        logger.warning(
            "Importance cache at %s has no metadata header; gate modules "
            "may miss mirrored indices.", path,
        )
        return dict(payload), {}  # type: ignore[arg-type]

    version = int(payload.get("version", 0))
    if version != _IMPORTANCE_CACHE_VERSION:
        raise ValueError(
            f"Importance cache at {path} has version {version}, but this "
            f"build of critnet expects version {_IMPORTANCE_CACHE_VERSION}."
        )
    scores: Dict[str, torch.Tensor] = payload["scores"]
    gate_map: Dict[str, str] = dict(payload.get("gate_to_partner_path") or {})
    return scores, gate_map

"""Neuron importance detection via first-order Taylor expansion.

Given a model and a dataset the :class:`NeuronDetector` computes
per-neuron impact scores:

.. math::

    \\mathcal{I}_t(w_i) \\approx \\left| w_i^\\top
    \\nabla_{w_i}\\mathcal{L}(\\Theta, \\mathcal{D}_t) \\right|

and selects the global top-k most critical neurons across all target
modules.
"""

from __future__ import annotations

import inspect
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import CriticalNeuronConfig
from .model import _is_norm

logger = logging.getLogger(__name__)

# torch.save payload written by :meth:`NeuronDetector.save_importance_cache`.
_IMPORTANCE_CACHE_VERSION = 1


class NeuronDetector:
    """Detect critical neurons using first-order Taylor importance scores.

    Parameters
    ----------
    model : nn.Module or None
        A pretrained ``transformers`` model **before** any wrapping for
        :meth:`detect`.  May be ``None`` when only using
        :meth:`select_from_importance_cache`.
    config : CriticalNeuronConfig
        Configuration specifying which modules to target and the
        sparsity ratio.

    Example
    -------
    >>> config = CriticalNeuronConfig(sparsity_ratio=0.05)
    >>> detector = NeuronDetector(model, config)
    >>> indices = detector.detect(dataloader, mode="chat")
    >>> detector.save("./detected_neurons")
    """

    def __init__(self, model: Optional[nn.Module], config: CriticalNeuronConfig) -> None:
        self.model = model
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        dataloader: DataLoader,
        mode: str = "chat",
        save_importance_cache_path: Optional[str] = None,
    ) -> Dict[str, List[int]]:
        """Run importance detection and return neuron indices.

        Parameters
        ----------
        dataloader : DataLoader
            Must yield dicts with at least ``input_ids``,
            ``attention_mask``, and ``labels``.

            * ``mode="chat"``: ``labels`` should already have prompt
              tokens masked to ``-100`` (only completion tokens
              contribute to the loss).
            * ``mode="pre-train"``: ``labels`` should equal
              ``input_ids`` (standard next-token prediction on all
              tokens).
        mode : str
            ``"chat"`` or ``"pre-train"``.  Controls how the loss is
            expected to be prepared -- the actual masking is done
            upstream in the dataset / collator.
        save_importance_cache_path : str, optional
            If set, persist the **post–gate-combined** per-module importance
            tensors (CPU floats) plus gate metadata so a later run can call
            :meth:`select_from_importance_cache` with a different
            ``sparsity_ratio`` without another backward pass.

        Returns
        -------
        dict[str, list[int]]
            Mapping from full module path to sorted list of selected
            neuron indices.  Also stored in ``self.config.neuron_indices``.
        """
        if mode not in ("chat", "pre-train"):
            raise ValueError(f"mode must be 'chat' or 'pre-train', got '{mode}'")
        if self.model is None:
            raise ValueError("detect() requires a model; use select_from_importance_cache() for cached scores.")

        device = next(self.model.parameters()).device
        self.model.train()

        target_params = self._find_target_params()
        for _, param in target_params:
            param.requires_grad_(True)

        self.model.zero_grad()

        total_loss = 0.0
        num_batches = 0
        for batch in tqdm(dataloader, desc="NeuronDetector: accumulating gradients"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = self.model(**inputs)
            loss = outputs.loss
            loss.backward()
            total_loss += loss.item()
            num_batches += 1

        if num_batches > 0:
            logger.info("Average loss over %d batches: %.4f", num_batches, total_loss / num_batches)

        neuron_scores = self._compute_scores(target_params)

        neuron_scores = self._apply_gate_combination(neuron_scores)

        if save_importance_cache_path is not None:
            self.save_importance_cache(neuron_scores, save_importance_cache_path)

        neuron_indices = self._global_topk(neuron_scores)

        self.config.neuron_indices = neuron_indices
        return neuron_indices

    def save(self, save_path: str) -> None:
        """Persist detected indices and config to *save_path*."""
        self.config.save_pretrained(save_path)

    def save_importance_cache(
        self,
        combined_scores: Dict[str, torch.Tensor],
        path: str,
    ) -> None:
        """Write combined per-module importance tensors (after gate merge) to disk.

        The file is a ``torch.save`` blob with keys ``version``, ``scores``,
        and ``gate_to_partner_path``.  Load with
        :meth:`select_from_importance_cache` using a config whose module
        lists and ``gate_combines_with`` match the run that produced the
        cache.
        """
        payload = {
            "version": _IMPORTANCE_CACHE_VERSION,
            "scores": {k: v.contiguous().clone() for k, v in combined_scores.items()},
            "gate_to_partner_path": dict(
                getattr(self, "_gate_to_partner_path", {})
            ),
        }
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(payload, path)
        logger.info("Wrote importance cache (%d modules) to %s", len(combined_scores), path)

    def select_from_importance_cache(self, path: str) -> Dict[str, List[int]]:
        """Load a cache from :meth:`save_importance_cache` and run global top-k.

        Uses ``self.config.sparsity_ratio`` (and gate metadata in the file)
        to produce ``self.config.neuron_indices``.
        """
        load_kw: Dict[str, object] = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kw["weights_only"] = False
        payload = torch.load(path, **load_kw)  # type: ignore[arg-type]
        if isinstance(payload, dict) and "scores" in payload:
            scores: Dict[str, torch.Tensor] = payload["scores"]
            self._gate_to_partner_path = dict(payload.get("gate_to_partner_path") or {})
        else:
            # Raw dict[str, Tensor] from older or manual saves — gate mirroring may be incomplete.
            scores = payload  # type: ignore[assignment]
            self._gate_to_partner_path = {}
            logger.warning(
                "Importance cache at %s has no gate_to_partner_path metadata; "
                "gate modules may miss mirrored indices.",
                path,
            )

        neuron_indices = self._global_topk(scores)
        self.config.neuron_indices = neuron_indices
        return neuron_indices

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_target_params(self) -> List[tuple]:
        """Return ``(full_param_name, Parameter)`` for all target weight tensors."""
        if self.model is None:
            raise ValueError("_find_target_params requires a model.")
        target_params: List[tuple] = []
        for mod_name, module in self.model.named_modules():
            if not self.config.matches_target(mod_name):
                continue
            if hasattr(module, "weight") and module.weight is not None:
                param_name = f"{mod_name}.weight"
                target_params.append((param_name, module.weight))
        return target_params

    def _compute_scores(
        self, target_params: List[tuple]
    ) -> Dict[str, torch.Tensor]:
        """Compute per-neuron importance ``|W * grad|`` reduced along the appropriate axis."""
        scores: Dict[str, torch.Tensor] = {}

        for param_name, param in target_params:
            if param.grad is None:
                continue

            mod_name = param_name.rsplit(".weight", 1)[0]

            importance = torch.abs(param.data * param.grad)

            mod_type = self.config.get_module_type(mod_name)

            if mod_type == "row":
                if importance.dim() == 2:
                    neuron_scores = importance.sum(dim=1).float().cpu()
                else:
                    neuron_scores = importance.float().cpu()
            elif mod_type == "column":
                if importance.dim() == 2:
                    neuron_scores = importance.sum(dim=0).float().cpu()
                else:
                    neuron_scores = importance.float().cpu()
            elif mod_type == "norm":
                neuron_scores = importance.float().cpu()
            elif mod_type == "embedding":
                if importance.dim() == 2:
                    neuron_scores = importance.sum(dim=1).float().cpu()
                else:
                    neuron_scores = importance.float().cpu()
            else:
                continue

            scores[mod_name] = neuron_scores

        return scores

    def _apply_gate_combination(
        self, scores: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Combine gate/partner importance scores and assign shared indices.

        For each ``(gate, partner)`` pair in ``config.gate_combines_with``,
        add the gate's scores element-wise into the partner's scores, then
        remove the gate entry so both modules later share the partner's
        selected indices.
        """
        if not self.config.gate_combines_with:
            return scores

        gate_suffix_to_partner: Dict[str, str] = self.config.gate_combines_with

        gate_to_partner_path: Dict[str, str] = {}
        for mod_path in list(scores.keys()):
            leaf = mod_path.rsplit(".", 1)[-1]
            if leaf in gate_suffix_to_partner:
                partner_suffix = gate_suffix_to_partner[leaf]
                partner_path = mod_path.rsplit(".", 1)[0] + "." + partner_suffix
                gate_to_partner_path[mod_path] = partner_path

        for gate_path, partner_path in gate_to_partner_path.items():
            if gate_path in scores and partner_path in scores:
                if scores[gate_path].shape == scores[partner_path].shape:
                    scores[partner_path] = scores[partner_path] + scores[gate_path]
                    del scores[gate_path]
                else:
                    logger.warning(
                        "Cannot combine gate '%s' (%s) with partner '%s' (%s): shape mismatch. Keeping separate.",
                        gate_path, scores[gate_path].shape,
                        partner_path, scores[partner_path].shape,
                    )

        self._gate_to_partner_path = gate_to_partner_path
        return scores

    def _global_topk(
        self, scores: Dict[str, torch.Tensor]
    ) -> Dict[str, List[int]]:
        """Pool all neuron scores and select global top-k."""
        all_scores: List[float] = []
        score_meta: List[tuple] = []

        for mod_path, s in scores.items():
            for idx in range(s.numel()):
                all_scores.append(s[idx].item())
                score_meta.append((mod_path, idx))

        if not all_scores:
            logger.warning("No importance scores computed -- returning empty indices.")
            return {}

        total = len(all_scores)
        k = max(1, int(total * self.config.sparsity_ratio))

        scores_tensor = torch.tensor(all_scores)
        top_scores, top_indices = torch.topk(scores_tensor, k, largest=True, sorted=True)

        selected: Dict[str, List[int]] = defaultdict(list)
        for flat_idx in top_indices.tolist():
            mod_path, neuron_idx = score_meta[flat_idx]
            selected[mod_path].append(neuron_idx)

        # Assign gate modules the same indices as their partner
        gate_to_partner = getattr(self, "_gate_to_partner_path", {})
        for gate_path, partner_path in gate_to_partner.items():
            if partner_path in selected:
                selected[gate_path] = list(selected[partner_path])

        for mod_path in selected:
            selected[mod_path].sort()

        logger.info(
            "Global top-k: %d / %d neurons (score range %.6f .. %.6f)",
            k, total, top_scores[-1].item(), top_scores[0].item(),
        )

        return dict(selected)

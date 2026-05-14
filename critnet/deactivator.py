"""Neuron deactivation -- zero out detected neurons from a model's weights.

The :class:`NeuronDeactivator` sets the weight vectors of specified
neurons to zero, effectively "pruning" them while keeping the model
architecture intact.  This is useful for ablation studies (measuring the
effect of removing language-specific or shared neurons) and for building
deactivated-model baselines.

Example
-------
>>> from critnet import CriticalNeuronConfig, NeuronDeactivator
>>> config = CriticalNeuronConfig.from_pretrained("./detected_neurons/en")
>>> deactivator = NeuronDeactivator(model, config)
>>> stats = deactivator.deactivate()
>>> deactivator.save_pretrained("./deactivated_model")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .config import CriticalNeuronConfig

logger = logging.getLogger(__name__)


@dataclass
class DeactivationResult:
    """Summary returned by :meth:`NeuronDeactivator.deactivate`.

    Attributes
    ----------
    modules_affected : int
        Number of modules that had neurons zeroed out.
    neurons_zeroed : int
        Total number of individual neurons set to zero.
    total_weights_zeroed : int
        Total number of scalar weight values set to zero.
    per_module : dict[str, dict]
        Per-module breakdown with keys ``"neurons"`` (count),
        ``"weights"`` (scalar count), and ``"module_type"``.
    """

    modules_affected: int = 0
    neurons_zeroed: int = 0
    total_weights_zeroed: int = 0
    per_module: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Deactivation summary:",
            f"  Modules affected  : {self.modules_affected}",
            f"  Neurons zeroed    : {self.neurons_zeroed:,}",
            f"  Weights zeroed    : {self.total_weights_zeroed:,}",
        ]
        return "\n".join(lines)


class NeuronDeactivator:
    """Zero out detected neurons in a model's weight tensors.

    Given a set of neuron indices (from :class:`NeuronDetector` or
    :class:`NeuronStatistician`), this class sets the corresponding
    weight vectors to zero **in-place**, producing a model where those
    neurons contribute nothing to the forward pass.

    The zeroing dimension is determined automatically from the config's
    module categories:

    * **row** modules (``q_proj``, ``up_proj``, ...): ``W[idx, :] = 0``
    * **column** modules (``o_proj``, ``down_proj``): ``W[:, idx] = 0``
    * **norm** modules: ``w[idx] = 0``
    * **embedding** modules: ``E[idx, :] = 0``

    Parameters
    ----------
    model : nn.Module
        A pretrained ``transformers`` model (unmodified, **not** wrapped
        with ``CriticalNeuronModel``).
    config : CriticalNeuronConfig
        Must have module categories configured (``row_modules``,
        ``column_modules``, etc.) so that the zeroing dimension can be
        resolved.  ``neuron_indices`` may be set on the config, or
        passed directly to :meth:`deactivate`.

    Example
    -------
    >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    >>> config = CriticalNeuronConfig.from_pretrained("./detected_neurons/en")
    >>> deactivator = NeuronDeactivator(model, config)
    >>> result = deactivator.deactivate()
    >>> print(result.summary())
    >>> deactivator.save_pretrained("./deactivated_model")
    """

    def __init__(
        self,
        model: nn.Module,
        config: CriticalNeuronConfig,
    ) -> None:
        self.model = model
        self.config = config

    @torch.no_grad()
    def deactivate(
        self,
        neuron_indices: Optional[Dict[str, List[int]]] = None,
    ) -> DeactivationResult:
        """Zero out specified neurons in the model's weights **in-place**.

        Parameters
        ----------
        neuron_indices : dict[str, list[int]] or None
            Mapping from full module path to neuron indices to deactivate.
            If ``None``, uses ``self.config.neuron_indices``.

        Returns
        -------
        DeactivationResult
            Summary of what was zeroed.

        Raises
        ------
        ValueError
            If no neuron indices are available.
        """
        indices = neuron_indices or self.config.neuron_indices
        if indices is None:
            raise ValueError(
                "No neuron indices provided. Pass neuron_indices to "
                "deactivate() or set config.neuron_indices."
            )

        named_modules = dict(self.model.named_modules())
        result = DeactivationResult()

        for mod_path, neuron_idxs in indices.items():
            if not neuron_idxs:
                continue

            module = named_modules.get(mod_path)
            if module is None:
                logger.warning("Module '%s' not found in model -- skipping.", mod_path)
                continue

            if not hasattr(module, "weight") or module.weight is None:
                logger.warning("Module '%s' has no weight tensor -- skipping.", mod_path)
                continue

            try:
                mod_type = self.config.get_module_type(mod_path)
            except ValueError:
                logger.warning(
                    "Module '%s' does not match any configured category -- skipping.",
                    mod_path,
                )
                continue

            idx_tensor = torch.tensor(neuron_idxs, dtype=torch.long, device=module.weight.device)
            weights_zeroed = self._zero_neurons(module, idx_tensor, mod_type)

            result.modules_affected += 1
            result.neurons_zeroed += len(neuron_idxs)
            result.total_weights_zeroed += weights_zeroed
            result.per_module[mod_path] = {
                "neurons": len(neuron_idxs),
                "weights": weights_zeroed,
                "module_type": mod_type,
            }

        logger.info(
            "Deactivated %d neurons across %d modules (%d scalar weights zeroed).",
            result.neurons_zeroed,
            result.modules_affected,
            result.total_weights_zeroed,
        )
        return result

    def save_pretrained(
        self,
        save_directory: str,
        tokenizer: Optional[Any] = None,
    ) -> None:
        """Save the deactivated model as a standard HuggingFace checkpoint.

        Parameters
        ----------
        save_directory : str
            Output directory for the full model checkpoint.
        tokenizer : optional
            If provided, the tokenizer is saved alongside the model.
        """
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_directory)
        self.config.save_pretrained(save_directory)
        logger.info("Deactivated model saved to %s", save_directory)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_neurons(
        module: nn.Module,
        idx: torch.Tensor,
        mod_type: str,
    ) -> int:
        """Zero out neuron weights and return the count of scalar weights set to zero."""
        w = module.weight
        count = 0

        if mod_type == "row":
            if w.dim() == 2:
                w.data[idx, :] = 0
                count = idx.numel() * w.size(1)
            else:
                w.data[idx] = 0
                count = idx.numel()

        elif mod_type == "column":
            if w.dim() == 2:
                w.data[:, idx] = 0
                count = w.size(0) * idx.numel()
            else:
                w.data[idx] = 0
                count = idx.numel()

        elif mod_type == "norm":
            w.data[idx] = 0
            count = idx.numel()
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data[idx] = 0
                count += idx.numel()

        elif mod_type == "embedding":
            if w.dim() == 2:
                w.data[idx, :] = 0
                count = idx.numel() * w.size(1)
            else:
                w.data[idx] = 0
                count = idx.numel()

        return count

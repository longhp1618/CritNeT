"""In-place neuron deactivation -- zero out detected neurons in the weights.

The :class:`NeuronDeactivator` sets the weight slice of every supplied
neuron to zero, effectively pruning the neuron while keeping the model's
architecture intact.  The resulting checkpoint is a vanilla HuggingFace
model and can be reloaded with ``AutoModelForCausalLM.from_pretrained``.

Example
-------
>>> from critnet import (
...     CriticalNeuronConfig, NeuronDeactivator, load_neuron_indices,
... )
>>> config = CriticalNeuronConfig.from_pretrained("./neurons")
>>> indices = load_neuron_indices("./neurons")
>>> deactivator = NeuronDeactivator(model, config)
>>> result = deactivator.deactivate(indices)
>>> deactivator.save_pretrained("./deactivated", tokenizer=tokenizer)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn

from .config import CriticalNeuronConfig

logger = logging.getLogger(__name__)


# =====================================================================
# DeactivationResult
# =====================================================================


@dataclass
class DeactivationResult:
    """Summary returned by :meth:`NeuronDeactivator.deactivate`.

    Attributes
    ----------
    modules_affected
        Number of modules that had at least one neuron zeroed.
    neurons_zeroed
        Total neuron indices processed.
    total_weights_zeroed
        Total scalar weight elements set to zero.
    per_module
        ``module_path -> {"neurons", "weights", "module_type"}`` for
        every module that was touched.
    """

    modules_affected: int = 0
    neurons_zeroed: int = 0
    total_weights_zeroed: int = 0
    per_module: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            "Deactivation summary:\n"
            f"  Modules affected : {self.modules_affected}\n"
            f"  Neurons zeroed   : {self.neurons_zeroed:,}\n"
            f"  Weights zeroed   : {self.total_weights_zeroed:,}"
        )


# =====================================================================
# NeuronDeactivator
# =====================================================================


class NeuronDeactivator:
    """Zero out chosen neurons in a model's weight tensors **in place**.

    The model must be a plain HF-style stack (not wrapped by
    :class:`~critnet.model.CriticalNeuronModel`).  The config supplies
    module categories so the deactivator knows **which axis** to zero
    per module.

    Per module type::

        row       W[idx, :] = 0
        column    W[:, idx] = 0
        norm      w[idx]    = 0          (and bias[idx] when present)
        embedding E[idx, :] = 0
    """

    def __init__(self, model: nn.Module, config: CriticalNeuronConfig) -> None:
        self.model = model
        self.config = config

    @torch.no_grad()
    def deactivate(
        self, indices: Mapping[str, List[int]]
    ) -> DeactivationResult:
        """Zero out neurons listed in *indices* (in place).

        Raises
        ------
        ValueError
            If any module path appears in *indices* but cannot be
            classified by ``config.classify(...)``.  Silent
            mis-classification was the old default and would zero the
            wrong axis -- the strict check is intentional.
        """
        if not indices:
            return DeactivationResult()

        named = dict(self.model.named_modules())
        unknown = [mp for mp in indices if self.config.classify(mp) is None]
        if unknown:
            raise ValueError(
                f"Cannot classify {len(unknown)} module(s) supplied to "
                f"deactivate(): {unknown[:5]}{'...' if len(unknown) > 5 else ''}. "
                "Add the missing leaf name(s) to row_modules / column_modules / "
                "norm_modules / embedding_modules on the config, or drop them "
                "from the indices dict."
            )

        result = DeactivationResult()
        for mod_path, neuron_idxs in indices.items():
            if not neuron_idxs:
                continue
            module = named.get(mod_path)
            if module is None:
                logger.warning("Module '%s' not found in model -- skipping.", mod_path)
                continue
            if getattr(module, "weight", None) is None:
                logger.warning("Module '%s' has no .weight -- skipping.", mod_path)
                continue

            mod_type = self.config.classify(mod_path)
            assert mod_type is not None  # checked above

            idx_tensor = torch.as_tensor(
                sorted(neuron_idxs), dtype=torch.long, device=module.weight.device
            )
            weights_zeroed = _zero_neurons(module, idx_tensor, mod_type)

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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_pretrained(
        self,
        save_directory: str,
        *,
        tokenizer: Optional[Any] = None,
        indices: Optional[Mapping[str, List[int]]] = None,
    ) -> None:
        """Save the (now-deactivated) model, the config, and optional extras.

        Convenience wrapper around::

            model.save_pretrained(save_directory)
            tokenizer.save_pretrained(save_directory)        # if provided
            config.save_pretrained(save_directory, indices=indices)
        """
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_directory)
        self.config.save_pretrained(
            save_directory,
            indices=dict(indices) if indices is not None else None,
        )
        logger.info("Deactivated model saved to %s", save_directory)


# =====================================================================
# Internals
# =====================================================================


@torch.no_grad()
def _zero_neurons(
    module: nn.Module,
    idx: torch.Tensor,
    mod_type: str,
) -> int:
    """Zero out *idx* on *module* along the axis implied by *mod_type*.

    Returns the count of scalar weight elements that were set to zero.
    """
    w = module.weight
    if mod_type == "row":
        if w.dim() == 2:
            w.data[idx, :] = 0
            return idx.numel() * w.size(1)
        w.data[idx] = 0
        return idx.numel()

    if mod_type == "column":
        if w.dim() == 2:
            w.data[:, idx] = 0
            return w.size(0) * idx.numel()
        w.data[idx] = 0
        return idx.numel()

    if mod_type == "norm":
        w.data[idx] = 0
        count = idx.numel()
        if getattr(module, "bias", None) is not None:
            module.bias.data[idx] = 0
            count += idx.numel()
        return count

    if mod_type == "embedding":
        if w.dim() == 2:
            w.data[idx, :] = 0
            return idx.numel() * w.size(1)
        w.data[idx] = 0
        return idx.numel()

    raise ValueError(f"Unknown module type {mod_type!r}.")  # defensive

"""Delta-subspace wrappers, model wrapping, and checkpoint logic.

This module provides:

* :class:`LinearDeltaSubspace` -- wraps ``nn.Linear`` to train only a
  sparse set of neuron rows or columns while freezing the rest.
* :class:`NormDeltaSubspace` -- wraps a normalisation layer (RMSNorm /
  LayerNorm) to train only selected elements of the 1-D weight.
* :class:`EmbeddingDeltaSubspace` -- wraps ``nn.Embedding`` to train
  only selected vocabulary rows.
* :func:`get_neuron_model` -- factory that wraps a HuggingFace model
  into a :class:`CriticalNeuronModel`.
* :class:`CriticalNeuronModel` -- thin wrapper providing
  ``save_pretrained`` / ``from_pretrained`` / ``merge_and_unload``.
* :func:`freeze_neurons` -- freeze specific neurons during full
  fine-tuning via gradient hooks (zero forward overhead).
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set

import torch
import torch.nn as nn

from .config import CriticalNeuronConfig, _CONFIG_FILENAME, _INDICES_FILENAME

logger = logging.getLogger(__name__)

# ======================================================================
# Delta-subspace wrappers
# ======================================================================


class LinearDeltaSubspace(nn.Module):
    """Sparse adapter for ``nn.Linear``: ``y = base(x) + delta(x)``.

    Only a small ``dW`` parameter covering the selected neuron indices
    is trainable; the full base weight stays frozen.

    Parameters
    ----------
    base_linear : nn.Linear
        The original linear layer (will be frozen in-place).
    indices : sequence of int
        Neuron indices to make trainable.
    mode : ``"row"`` or ``"column"``
        * ``"row"``: each selected index is a row of W (output neuron).
          ``dW`` shape ``[k, in_features]``.
        * ``"column"``: each selected index is a column of W (input neuron).
          ``dW`` shape ``[out_features, k]``.
    train_bias : bool
        If ``True``, the bias (if present) remains trainable.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        indices: Sequence[int],
        mode: str = "row",
        train_bias: bool = False,
    ) -> None:
        super().__init__()
        if mode not in ("column", "row"):
            raise ValueError(f"mode must be 'row' or 'column', got '{mode}'")

        self.base = base_linear
        self.mode = mode
        self.register_buffer("idx", torch.tensor(sorted(indices), dtype=torch.long))

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None and not train_bias:
            self.base.bias.requires_grad_(False)

        W = self.base.weight.detach()
        if mode == "column":
            self.dW = nn.Parameter(
                torch.zeros(W.size(0), self.idx.numel(), device=W.device, dtype=W.dtype)
            )
        else:
            self.dW = nn.Parameter(
                torch.zeros(self.idx.numel(), W.size(1), device=W.device, dtype=W.dtype)
            )

    # -- nn.Linear-compatible properties for code that reads module.weight --

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.mode == "column":
            return y + (x[..., self.idx] @ self.dW.t())
        upd = x @ self.dW.t()
        return y.index_add(-1, self.idx, upd)

    @torch.no_grad()
    def merge_to_linear_(self) -> nn.Linear:
        """Merge ``dW`` into the base weight in-place and return the ``nn.Linear``."""
        W = self.base.weight
        if self.mode == "column":
            W[:, self.idx] += self.dW
        else:
            W[self.idx, :] += self.dW
        return self.base


class NormDeltaSubspace(nn.Module):
    """Sparse adapter for normalisation layers with a 1-D weight.

    Only selected elements of the weight vector are trainable.

    Parameters
    ----------
    base_norm : nn.Module
        The original norm layer (e.g. ``RMSNorm``, ``LayerNorm``).
        Must have a ``.weight`` attribute of shape ``[dim]``.
    indices : sequence of int
        Elements of the weight to make trainable.
    """

    def __init__(self, base_norm: nn.Module, indices: Sequence[int]) -> None:
        super().__init__()
        self.base = base_norm
        self.register_buffer("idx", torch.tensor(sorted(indices), dtype=torch.long))

        self.base.weight.requires_grad_(False)
        if hasattr(self.base, "bias") and self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.dW = nn.Parameter(
            torch.zeros(self.idx.numel(), device=self.base.weight.device, dtype=self.base.weight.dtype)
        )
        self.register_buffer(
            "_inv_base_w",
            1.0 / (self.base.weight.detach()[self.idx] + 1e-12),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        scale = self.dW * self._inv_base_w
        correction = out[..., self.idx] * scale
        return out.index_add(-1, self.idx, correction)

    @torch.no_grad()
    def merge_to_norm_(self) -> nn.Module:
        """Merge ``dW`` into the base weight in-place and return the base module."""
        self.base.weight[self.idx] += self.dW
        return self.base


class EmbeddingDeltaSubspace(nn.Module):
    """Sparse adapter for ``nn.Embedding``: only selected vocabulary rows are trainable.

    Parameters
    ----------
    base_embedding : nn.Embedding
        The original embedding layer (frozen in-place).
    indices : sequence of int
        Row indices (vocabulary entries) to make trainable.
    """

    def __init__(self, base_embedding: nn.Embedding, indices: Sequence[int]) -> None:
        super().__init__()
        self.base = base_embedding
        self.register_buffer("idx", torch.tensor(sorted(indices), dtype=torch.long))

        self.base.weight.requires_grad_(False)

        self.dW = nn.Parameter(
            torch.zeros(self.idx.numel(), self.base.embedding_dim, device=self.base.weight.device, dtype=self.base.weight.dtype)
        )
        self._idx_to_local = {int(v): i for i, v in enumerate(self.idx.tolist())}

        _lookup = torch.zeros(base_embedding.num_embeddings, dtype=torch.long)
        for g, l in self._idx_to_local.items():
            _lookup[g] = l
        self.register_buffer("_lookup", _lookup)

        _hit_mask = torch.zeros(base_embedding.num_embeddings, dtype=torch.bool)
        _hit_mask[self.idx] = True
        self.register_buffer("_hit_mask", _hit_mask)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.base(input_ids)
        hit = self._hit_mask[input_ids]
        if hit.any():
            local = self._lookup[input_ids]
            delta = self.dW[local]
            out = out + delta * hit.unsqueeze(-1).to(out.dtype)
        return out

    @torch.no_grad()
    def merge_to_embedding_(self) -> nn.Embedding:
        """Merge ``dW`` into the base weight in-place and return the base module."""
        self.base.weight[self.idx] += self.dW
        return self.base


# Helper type for delta wrappers
_DeltaModule = (LinearDeltaSubspace, NormDeltaSubspace, EmbeddingDeltaSubspace)

# ======================================================================
# Module replacement helpers
# ======================================================================


def _set_submodule(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace a named submodule in *model* (supports dotted paths)."""
    parts = name.rsplit(".", 1)
    if len(parts) == 2:
        parent = model.get_submodule(parts[0])
        setattr(parent, parts[1], new_module)
    else:
        setattr(model, parts[0], new_module)


def _is_norm(module: nn.Module) -> bool:
    """Heuristic: module is a normalisation layer with a 1-D weight."""
    cls_name = type(module).__name__.lower()
    has_weight = hasattr(module, "weight") and module.weight is not None
    if not has_weight:
        return False
    is_1d = module.weight.dim() == 1
    looks_like_norm = any(tok in cls_name for tok in ("norm", "layernorm", "rmsnorm"))
    return is_1d and looks_like_norm


# ======================================================================
# get_neuron_model
# ======================================================================


DEFAULT_SKIP_MODULES: FrozenSet[str] = frozenset({"lm_head", "embed_tokens"})
"""Module leaf-names that are skipped during delta-wrapping by default.

``lm_head`` is excluded because fused-kernel libraries (e.g. Liger)
replace the model's forward to access ``lm_head.weight`` directly,
bypassing the wrapper's ``forward()`` and breaking autograd for the
``dW`` delta.  ``embed_tokens`` is excluded for the same reason
(weight-tying with ``lm_head`` and incompatibility with fused kernels).

Users who explicitly need to wrap these can pass
``modules_to_skip=set()`` to :func:`get_neuron_model`.
"""


def get_neuron_model(
    model: nn.Module,
    config: CriticalNeuronConfig,
    modules_to_skip: Optional[Set[str]] = None,
) -> "CriticalNeuronModel":
    """Wrap a HuggingFace model for critical-neuron fine-tuning.

    1. Iterates all named modules and replaces each target module with its
       delta-subspace wrapper.
    2. Freezes all parameters, then unfreezes only the ``dW`` parameters.
    3. Calls ``enable_input_require_grads`` for gradient flow.

    Parameters
    ----------
    model : nn.Module
        A pretrained ``transformers`` model (e.g. from
        ``AutoModelForCausalLM.from_pretrained``).
    config : CriticalNeuronConfig
        Must have ``neuron_indices`` populated (not ``None``).
    modules_to_skip : set of str, optional
        Module leaf-names (e.g. ``{"lm_head"}``) to exclude from
        delta-wrapping.  These modules stay as plain ``nn.Linear``
        with frozen weights.  Defaults to :data:`DEFAULT_SKIP_MODULES`.
        Pass an empty set to wrap everything.

    Returns
    -------
    CriticalNeuronModel
    """
    if modules_to_skip is None:
        modules_to_skip = DEFAULT_SKIP_MODULES

    if config.neuron_indices is None:
        raise ValueError(
            "config.neuron_indices is None. Run NeuronDetector.detect() "
            "or load indices via CriticalNeuronConfig.from_pretrained() first."
        )

    modules_to_wrap: List[tuple] = []
    skipped: List[str] = []
    for name, module in model.named_modules():
        if not config.matches_target(name):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in modules_to_skip:
            skipped.append(name)
            continue
        if name not in config.neuron_indices:
            continue
        indices = config.neuron_indices[name]
        if len(indices) == 0:
            continue
        modules_to_wrap.append((name, module, indices))

    if skipped:
        logger.warning(
            "Skipped delta-wrapping for %d module(s) (incompatible with "
            "fused kernels): %s. Pass modules_to_skip=set() to override.",
            len(skipped),
            skipped[:5],
        )

    for name, module, indices in modules_to_wrap:
        mod_type = config.get_module_type(name)

        if mod_type in ("row", "column"):
            if not isinstance(module, nn.Linear):
                raise TypeError(
                    f"Module '{name}' is categorised as '{mod_type}' "
                    f"but is {type(module).__name__}, not nn.Linear."
                )
            wrapper = LinearDeltaSubspace(module, indices, mode=mod_type)

        elif mod_type == "norm":
            if not _is_norm(module):
                raise TypeError(
                    f"Module '{name}' is categorised as 'norm' but "
                    f"does not look like a normalisation layer "
                    f"({type(module).__name__})."
                )
            wrapper = NormDeltaSubspace(module, indices)

        elif mod_type == "embedding":
            if not isinstance(module, nn.Embedding):
                raise TypeError(
                    f"Module '{name}' is categorised as 'embedding' "
                    f"but is {type(module).__name__}, not nn.Embedding."
                )
            wrapper = EmbeddingDeltaSubspace(module, indices)
        else:
            raise ValueError(f"Unknown module type '{mod_type}' for '{name}'")

        _set_submodule(model, name, wrapper)

    # Freeze everything, then unfreeze only delta params
    for p in model.parameters():
        p.requires_grad_(False)
    for _, module in model.named_modules():
        if isinstance(module, _DeltaModule):
            module.dW.requires_grad_(True)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    wrapped = CriticalNeuronModel(model, config)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "CriticalNeuronModel: %d / %d params trainable (%.4f%%)",
        n_trainable, n_total, 100.0 * n_trainable / n_total if n_total else 0,
    )

    return wrapped


# ======================================================================
# freeze_neurons  (safety-preserving full fine-tuning)
# ======================================================================


class FrozenNeuronHandle:
    """Handle returned by :func:`freeze_neurons`.

    Gradient hooks zero out frozen-neuron gradients during backward,
    preventing the optimizer from updating them.  When the optimizer
    uses weight decay (``weight_decay > 0``), call
    :meth:`restore_frozen_weights` after each optimizer step to undo
    the decay on frozen neurons.
    """

    def __init__(
        self,
        hooks: List[torch.utils.hooks.RemovableHandle],
        frozen_snapshots: List[tuple],
        n_frozen: int,
        n_total: int,
    ) -> None:
        self._hooks = hooks
        self._frozen = frozen_snapshots
        self.n_frozen = n_frozen
        self.n_total = n_total

    @torch.no_grad()
    def restore_frozen_weights(self) -> None:
        """Restore frozen neurons to their original pre-training values.

        Call after each optimizer step to counteract weight-decay drift.
        If ``weight_decay == 0``, this is unnecessary (gradient hooks
        already prevent all updates).
        """
        for weight, mode, idx, saved in self._frozen:
            dev = weight.device
            idx_d = idx.to(dev)
            saved_d = saved.to(device=dev, dtype=weight.dtype)
            if mode == "column":
                weight.data[:, idx_d] = saved_d
            else:
                weight.data[idx_d] = saved_d

    def remove(self) -> None:
        """Remove all gradient hooks (unfreezes the neurons)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._frozen.clear()

    def make_trainer_callback(self) -> Any:
        """Return a HuggingFace ``TrainerCallback`` that restores frozen
        weights after each optimizer step.

        Usage::

            frozen = freeze_neurons(model, indices, config)
            trainer = Trainer(..., callbacks=[frozen.make_trainer_callback()])
        """
        from transformers import TrainerCallback

        handle = self

        class _FreezeCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                handle.restore_frozen_weights()

        return _FreezeCallback()

    def print_frozen_summary(self) -> None:
        """Print summary of frozen vs total parameters."""
        pct = 100.0 * self.n_frozen / self.n_total if self.n_total else 0
        n_trainable = self.n_total - self.n_frozen
        print(
            f"frozen params: {self.n_frozen:,} || "
            f"trainable params: {n_trainable:,} || "
            f"all params: {self.n_total:,} || "
            f"frozen%: {pct:.4f}%"
        )


def freeze_neurons(
    model: nn.Module,
    neuron_indices: Dict[str, List[int]],
    config: Optional[CriticalNeuronConfig] = None,
) -> FrozenNeuronHandle:
    """Freeze specific neurons during full fine-tuning.

    Registers backward hooks that zero out gradients for the specified
    neurons.  The model structure is **unchanged** -- no wrappers are
    inserted, so the forward pass is identical to the base model (zero
    overhead).

    Use case: train the full model while *preserving* safety-critical
    (or otherwise important) neurons.

    Parameters
    ----------
    model : nn.Module
        A pretrained model (all parameters should have
        ``requires_grad=True``).
    neuron_indices : dict[str, list[int]]
        Mapping from fully-qualified module path (e.g.
        ``"model.layers.0.self_attn.q_proj"``) to the neuron indices
        to **freeze**.
    config : CriticalNeuronConfig, optional
        Used to determine module types (row / column / norm).
        A default config is created if not provided.

    Returns
    -------
    FrozenNeuronHandle
        Call ``handle.restore_frozen_weights()`` after each optimizer
        step when ``weight_decay > 0``, or pass
        ``handle.make_trainer_callback()`` to the HF Trainer.
    """
    if config is None:
        config = CriticalNeuronConfig()

    hooks: List[torch.utils.hooks.RemovableHandle] = []
    frozen_snapshots: List[tuple] = []
    n_frozen = 0

    def _infer_module_type(name: str, module: nn.Module) -> Optional[str]:
        """Try config first; fall back to heuristic for modules not in config targets."""
        if config.matches_target(name):
            return config.get_module_type(name)
        if _is_norm(module):
            return "norm"
        if isinstance(module, nn.Linear):
            leaf = name.rsplit(".", 1)[-1]
            if leaf in (config.column_modules or []):
                return "column"
            return "row"
        return None

    for name, module in model.named_modules():
        if name not in neuron_indices:
            continue
        indices = neuron_indices[name]
        if not indices:
            continue

        mod_type = _infer_module_type(name, module)
        if mod_type is None:
            logger.warning(
                "freeze_neurons: '%s' (%s) not recognised; skipping.",
                name, type(module).__name__,
            )
            continue

        idx = torch.tensor(sorted(indices), dtype=torch.long)

        if isinstance(module, nn.Linear):
            if mod_type == "row":
                saved = module.weight.data[idx].clone()
                hook = module.weight.register_hook(
                    lambda g, i=idx: g.index_fill_(0, i.to(g.device), 0)
                )
                frozen_snapshots.append((module.weight, "row", idx, saved))
                n_frozen += len(indices) * module.weight.size(1)
            elif mod_type == "column":
                saved = module.weight.data[:, idx].clone()
                hook = module.weight.register_hook(
                    lambda g, i=idx: g.index_fill_(1, i.to(g.device), 0)
                )
                frozen_snapshots.append((module.weight, "column", idx, saved))
                n_frozen += module.weight.size(0) * len(indices)
            else:
                continue
            hooks.append(hook)

        elif _is_norm(module):
            saved = module.weight.data[idx].clone()
            hook = module.weight.register_hook(
                lambda g, i=idx: g.index_fill_(0, i.to(g.device), 0)
            )
            frozen_snapshots.append((module.weight, "norm", idx, saved))
            hooks.append(hook)
            n_frozen += len(indices)

    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "freeze_neurons: %d frozen weight elements out of %d total (%.4f%%)",
        n_frozen,
        n_total,
        100.0 * n_frozen / n_total if n_total else 0,
    )

    return FrozenNeuronHandle(hooks, frozen_snapshots, n_frozen, n_total)


# ======================================================================
# CriticalNeuronModel
# ======================================================================


class CriticalNeuronModel(nn.Module):
    """Thin wrapper around a ``transformers`` model for critical-neuron PEFT.

    Delegates ``forward`` and most attribute access to the inner model
    while adding ``save_pretrained``, ``from_pretrained``, and
    ``merge_and_unload``.
    """

    _ADAPTER_FILENAME = "adapter_model.safetensors"
    _ADAPTER_FILENAME_PT = "adapter_model.pt"

    def __init__(self, model: nn.Module, config: CriticalNeuronConfig) -> None:
        super().__init__()
        self.model = model
        self.peft_config = config

    # ------------------------------------------------------------------
    # Forward delegation
    # ------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.generate(*args, **kwargs)

    # ------------------------------------------------------------------
    # Attribute proxy (Trainer compatibility)
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    # ------------------------------------------------------------------
    # State dict helpers
    # ------------------------------------------------------------------

    def get_adapter_state_dict(self) -> OrderedDict:
        """Collect all ``dW`` (and ``idx``) tensors from delta wrappers."""
        out = OrderedDict()
        for name, module in self.model.named_modules():
            if isinstance(module, _DeltaModule):
                out[f"{name}.dW"] = module.dW.detach().cpu()
                out[f"{name}.idx"] = module.idx.cpu()
        return out

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None:
        """Save the adapter weights and config.

        Only the sparse ``dW`` parameters are saved -- **not** the full
        base-model weights.

        Files written to *save_directory*:

        * ``adapter_model.safetensors`` -- delta weights.
        * ``critical_neuron_config.json`` -- config.
        * ``neuron_indices.json`` -- neuron index mapping.
        """
        os.makedirs(save_directory, exist_ok=True)

        state = self.get_adapter_state_dict()

        try:
            from safetensors.torch import save_file
            save_file(state, os.path.join(save_directory, self._ADAPTER_FILENAME))
        except ImportError:
            logger.warning(
                "safetensors not installed; falling back to torch.save (.pt)."
            )
            torch.save(state, os.path.join(save_directory, self._ADAPTER_FILENAME_PT))

        self.peft_config.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        adapter_path: str,
        base_model_name_or_path: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        modules_to_skip: Optional[Set[str]] = None,
    ) -> "CriticalNeuronModel":
        """Load a base model and inject saved adapter weights.

        Parameters
        ----------
        adapter_path : str
            Directory containing the adapter files (``adapter_model.*``,
            ``critical_neuron_config.json``, ``neuron_indices.json``).
        base_model_name_or_path : str, optional
            HuggingFace model id or local path.  If ``None``, falls back
            to the ``base_model_name_or_path`` recorded inside
            ``adapter_path/critical_neuron_config.json``.
        model_kwargs : dict, optional
            Extra keyword arguments forwarded to
            ``AutoModelForCausalLM.from_pretrained`` (e.g.
            ``torch_dtype``, ``device_map``).
        modules_to_skip : set of str, optional
            Forwarded to :func:`get_neuron_model`.

        Returns
        -------
        CriticalNeuronModel
        """
        from transformers import AutoModelForCausalLM

        config = CriticalNeuronConfig.from_pretrained(adapter_path)
        if base_model_name_or_path is None:
            base_model_name_or_path = config.base_model_name_or_path
        if not base_model_name_or_path:
            raise ValueError(
                "base_model_name_or_path was not provided and is not recorded in "
                f"{adapter_path}/critical_neuron_config.json. Pass it explicitly."
            )
        config.base_model_name_or_path = base_model_name_or_path

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path, **(model_kwargs or {})
        )
        wrapped = get_neuron_model(model, config, modules_to_skip=modules_to_skip)

        # Load adapter state dict
        sf_path = os.path.join(adapter_path, cls._ADAPTER_FILENAME)
        pt_path = os.path.join(adapter_path, cls._ADAPTER_FILENAME_PT)
        if os.path.isfile(sf_path):
            from safetensors.torch import load_file
            state = load_file(sf_path)
        elif os.path.isfile(pt_path):
            state = torch.load(pt_path, map_location="cpu")
        else:
            logger.warning("No adapter weights found in %s; wrappers have zero dW.", adapter_path)
            return wrapped

        # Inject dW values
        for name, module in wrapped.model.named_modules():
            dw_key = f"{name}.dW"
            if isinstance(module, _DeltaModule) and dw_key in state:
                module.dW.data.copy_(state[dw_key])

        return wrapped

    # ------------------------------------------------------------------
    # Merge & unload
    # ------------------------------------------------------------------

    def merge_and_unload(self) -> nn.Module:
        """Merge all adapters into the base weights and return the plain model.

        After calling this method the ``CriticalNeuronModel`` wrapper is
        no longer needed; the returned model is a standard
        ``transformers`` model with the neuron updates baked in.
        """
        modules = list(self.model.named_modules())
        for name, module in modules:
            if isinstance(module, LinearDeltaSubspace):
                merged = module.merge_to_linear_()
                _set_submodule(self.model, name, merged)
            elif isinstance(module, NormDeltaSubspace):
                merged = module.merge_to_norm_()
                _set_submodule(self.model, name, merged)
            elif isinstance(module, EmbeddingDeltaSubspace):
                merged = module.merge_to_embedding_()
                _set_submodule(self.model, name, merged)
        return self.model

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def print_trainable_parameters(self) -> None:
        """Print the number and percentage of trainable parameters."""
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        pct = 100.0 * n_trainable / n_total if n_total else 0
        print(
            f"trainable params: {n_trainable:,} || "
            f"all params: {n_total:,} || "
            f"trainable%: {pct:.4f}%"
        )

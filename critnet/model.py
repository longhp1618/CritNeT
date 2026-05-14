"""Delta-subspace adapters, model wrapping, and full-FT freezing helpers.

This module hosts both sides of "doing something with a neuron set":

* **Sparse PEFT** -- :func:`get_neuron_model` rewires a HuggingFace model so
  that only a small dense ``dW`` parameter per targeted module is
  trainable, and :class:`CriticalNeuronModel` wraps the result with
  save / load / merge helpers compatible with ``transformers.Trainer``.
* **Frozen full FT** -- :func:`freeze_neurons` registers backward hooks
  that zero out gradients on a chosen neuron set during otherwise normal
  fine-tuning, returning a :class:`FrozenNeuronHandle` for cleanup and
  weight-decay-aware restoration.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set

import torch
import torch.nn as nn

from .config import (
    CriticalNeuronConfig,
    _CONFIG_FILENAME,
    _INDICES_FILENAME,
    load_neuron_indices,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Delta-subspace wrappers
# =====================================================================


class LinearDeltaSubspace(nn.Module):
    """Sparse adapter for ``nn.Linear``: ``y = base(x) + delta(x)``.

    Only a small ``dW`` parameter -- a slice of the weight matrix
    indexed by *indices* -- is trainable.  The base weight (and the
    bias, unless ``train_bias=True``) is frozen in place.

    Parameters
    ----------
    base_linear
        The original linear layer; will be frozen in place.
    indices
        Neuron indices to make trainable.
    mode
        ``"row"`` or ``"column"``.

        * ``"row"``: each index is a row of :math:`W` (output neuron),
          ``dW`` shape ``[k, in_features]``.
        * ``"column"``: each index is a column of :math:`W` (input
          neuron), ``dW`` shape ``[out_features, k]``.
    train_bias
        If ``True``, leaves the bias trainable.  Default ``False``.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        indices: Sequence[int],
        mode: str = "row",
        train_bias: bool = False,
    ) -> None:
        super().__init__()
        if mode not in ("row", "column"):
            raise ValueError(f"mode must be 'row' or 'column', got {mode!r}.")

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

    # ``nn.Linear``-shaped surface so external code that reads
    # ``module.weight`` / ``module.bias`` still works.
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
        return y.index_add(-1, self.idx, x @ self.dW.t())

    @torch.no_grad()
    def merge_to_linear_(self) -> nn.Linear:
        """Merge ``dW`` into the base weight in place and return the ``nn.Linear``."""
        W = self.base.weight
        if self.mode == "column":
            W[:, self.idx] += self.dW
        else:
            W[self.idx, :] += self.dW
        return self.base


class NormDeltaSubspace(nn.Module):
    """Sparse adapter for normalisation layers with a 1-D weight.

    Only selected positions of the weight vector are trainable.

    Parameters
    ----------
    base_norm
        Any module with a 1-D ``.weight`` attribute (``RMSNorm`` /
        ``LayerNorm`` / etc.).  Frozen in place.
    indices
        Positions of the weight vector to make trainable.
    """

    def __init__(self, base_norm: nn.Module, indices: Sequence[int]) -> None:
        super().__init__()
        self.base = base_norm
        self.register_buffer("idx", torch.tensor(sorted(indices), dtype=torch.long))

        self.base.weight.requires_grad_(False)
        if getattr(self.base, "bias", None) is not None:
            self.base.bias.requires_grad_(False)

        self.dW = nn.Parameter(
            torch.zeros(
                self.idx.numel(),
                device=self.base.weight.device,
                dtype=self.base.weight.dtype,
            )
        )
        self.register_buffer(
            "_inv_base_w",
            1.0 / (self.base.weight.detach()[self.idx] + 1e-12),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        scale = self.dW * self._inv_base_w
        return out.index_add(-1, self.idx, out[..., self.idx] * scale)

    @torch.no_grad()
    def merge_to_norm_(self) -> nn.Module:
        """Merge ``dW`` into the base weight in place and return the base module."""
        self.base.weight[self.idx] += self.dW
        return self.base


class EmbeddingDeltaSubspace(nn.Module):
    """Sparse adapter for ``nn.Embedding``: only selected rows are trainable.

    Parameters
    ----------
    base_embedding
        The original embedding layer (frozen in place).
    indices
        Vocabulary row indices to make trainable.
    """

    def __init__(self, base_embedding: nn.Embedding, indices: Sequence[int]) -> None:
        super().__init__()
        self.base = base_embedding
        self.register_buffer("idx", torch.tensor(sorted(indices), dtype=torch.long))
        self.base.weight.requires_grad_(False)

        self.dW = nn.Parameter(
            torch.zeros(
                self.idx.numel(),
                self.base.embedding_dim,
                device=self.base.weight.device,
                dtype=self.base.weight.dtype,
            )
        )

        # Build O(1) lookup tables for the forward path.
        lookup = torch.zeros(base_embedding.num_embeddings, dtype=torch.long)
        for local, global_idx in enumerate(self.idx.tolist()):
            lookup[global_idx] = local
        self.register_buffer("_lookup", lookup)

        hit_mask = torch.zeros(base_embedding.num_embeddings, dtype=torch.bool)
        hit_mask[self.idx] = True
        self.register_buffer("_hit_mask", hit_mask)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.base(input_ids)
        hit = self._hit_mask[input_ids]
        if hit.any():
            local = self._lookup[input_ids]
            out = out + self.dW[local] * hit.unsqueeze(-1).to(out.dtype)
        return out

    @torch.no_grad()
    def merge_to_embedding_(self) -> nn.Embedding:
        """Merge ``dW`` into the base weight in place and return the base module."""
        self.base.weight[self.idx] += self.dW
        return self.base


# Tuple used for ``isinstance`` checks across this module.
_DeltaModule = (LinearDeltaSubspace, NormDeltaSubspace, EmbeddingDeltaSubspace)


# =====================================================================
# Module replacement helpers
# =====================================================================


def _set_submodule(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace a named submodule in *model* (supports dotted paths)."""
    parts = name.rsplit(".", 1)
    if len(parts) == 2:
        parent = model.get_submodule(parts[0])
        setattr(parent, parts[1], new_module)
    else:
        setattr(model, parts[0], new_module)


def _is_norm(module: nn.Module) -> bool:
    """Heuristic: *module* is a normalisation layer with a 1-D weight."""
    cls_name = type(module).__name__.lower()
    has_weight = getattr(module, "weight", None) is not None
    if not has_weight:
        return False
    is_1d = module.weight.dim() == 1
    looks_like_norm = any(tok in cls_name for tok in ("norm", "layernorm", "rmsnorm"))
    return is_1d and looks_like_norm


# =====================================================================
# get_neuron_model
# =====================================================================


DEFAULT_SKIP_MODULES: FrozenSet[str] = frozenset({"lm_head", "embed_tokens"})
"""Module leaf-names that are skipped during delta-wrapping by default.

``lm_head`` is excluded because fused-kernel libraries (e.g. Liger)
replace the model's forward to read ``lm_head.weight`` directly, which
bypasses any wrapper and breaks autograd through ``dW``.
``embed_tokens`` is excluded for the same reason (weight-tying with
``lm_head`` + fused-kernel incompatibility).  Pass
``modules_to_skip=set()`` to override.
"""


def get_neuron_model(
    model: nn.Module,
    config: CriticalNeuronConfig,
    indices: Dict[str, List[int]],
    *,
    modules_to_skip: Optional[Set[str]] = None,
) -> "CriticalNeuronModel":
    """Wrap *model* for critical-neuron sparse PEFT.

    The function:

    1. Replaces every targeted module that appears in *indices* with a
       delta-subspace wrapper of the appropriate type
       (:class:`LinearDeltaSubspace` / :class:`NormDeltaSubspace` /
       :class:`EmbeddingDeltaSubspace`).
    2. Freezes every base parameter; unfreezes only the ``dW`` of each
       wrapper.
    3. Calls ``enable_input_require_grads`` if the model exposes it
       (so gradient checkpointing keeps working).

    Parameters
    ----------
    model
        A pretrained ``transformers`` model.  Mutated in place.
    config
        Provides ``classify(...)`` and the skip / target sets.
    indices
        ``module_path -> [neuron_idx, ...]`` -- the modules to wrap.
    modules_to_skip
        Module leaf-names to exclude.  Defaults to
        :data:`DEFAULT_SKIP_MODULES` (``{"lm_head", "embed_tokens"}``).

    Returns
    -------
    CriticalNeuronModel
    """
    if not indices:
        raise ValueError(
            "indices is empty: nothing to wrap. "
            "Run NeuronDetector.detect(...) and pass `result.indices`, "
            "or load with critnet.load_neuron_indices(...)."
        )
    if modules_to_skip is None:
        modules_to_skip = DEFAULT_SKIP_MODULES

    named = dict(model.named_modules())
    missing = [k for k in indices if k not in named]
    if missing:
        raise ValueError(
            f"{len(missing)} module path(s) not found in model: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}."
        )

    to_wrap: List[tuple] = []
    skipped: List[str] = []
    for name, idx_list in indices.items():
        if not idx_list:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in modules_to_skip:
            skipped.append(name)
            continue
        mod_type = config.classify(name)
        if mod_type is None:
            raise ValueError(
                f"Module '{name}' appears in indices but is not classified by the "
                f"config (leaf {leaf!r}). Add it to row_modules / column_modules / "
                f"norm_modules / embedding_modules on the config."
            )
        to_wrap.append((name, named[name], mod_type, idx_list))

    if skipped:
        logger.warning(
            "Skipped delta-wrapping for %d module(s) incompatible with fused "
            "kernels: %s. Pass modules_to_skip=set() to override.",
            len(skipped), skipped[:5],
        )

    for name, module, mod_type, idx_list in to_wrap:
        if mod_type in ("row", "column"):
            if not isinstance(module, nn.Linear):
                raise TypeError(
                    f"Module {name!r} is categorised as {mod_type!r} but is "
                    f"{type(module).__name__}, not nn.Linear."
                )
            wrapper: nn.Module = LinearDeltaSubspace(module, idx_list, mode=mod_type)
        elif mod_type == "norm":
            if not _is_norm(module):
                raise TypeError(
                    f"Module {name!r} is categorised as 'norm' but does not "
                    f"look like a 1-D norm layer ({type(module).__name__})."
                )
            wrapper = NormDeltaSubspace(module, idx_list)
        elif mod_type == "embedding":
            if not isinstance(module, nn.Embedding):
                raise TypeError(
                    f"Module {name!r} is categorised as 'embedding' but is "
                    f"{type(module).__name__}, not nn.Embedding."
                )
            wrapper = EmbeddingDeltaSubspace(module, idx_list)
        else:  # pragma: no cover -- classify() never returns anything else
            raise ValueError(f"Unsupported module type {mod_type!r} for {name!r}.")

        _set_submodule(model, name, wrapper)

    for p in model.parameters():
        p.requires_grad_(False)
    for _, m in model.named_modules():
        if isinstance(m, _DeltaModule):
            m.dW.requires_grad_(True)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    wrapped = CriticalNeuronModel(model, config)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "CriticalNeuronModel: %d / %d params trainable (%.4f%%)",
        n_trainable, n_total, 100.0 * n_trainable / n_total if n_total else 0.0,
    )
    return wrapped


# =====================================================================
# CriticalNeuronModel
# =====================================================================


class CriticalNeuronModel(nn.Module):
    """Thin wrapper around a ``transformers`` model for critical-neuron PEFT.

    Forward, ``generate``, and attribute access are delegated to the inner
    model.  This object adds:

    * :attr:`neuron_indices` -- live view of the wrapped indices.
    * :meth:`save_pretrained` / :meth:`from_pretrained` -- adapter-only
      checkpoint I/O.
    * :meth:`merge_and_unload` -- in-place delta merge, returning a
      plain ``nn.Module``.

    ``self.peft_config`` is exposed as a HuggingFace-Trainer compatibility
    alias for :attr:`config`; do not rely on it elsewhere.
    """

    _ADAPTER_FILENAME = "adapter_model.safetensors"
    _ADAPTER_FILENAME_PT = "adapter_model.pt"

    def __init__(self, model: nn.Module, config: CriticalNeuronConfig) -> None:
        super().__init__()
        self.model = model
        self.config = config

    # HF Trainer treats objects with `peft_config` as PEFT models -- alias it.
    @property
    def peft_config(self) -> CriticalNeuronConfig:
        return self.config

    # ------------------------------------------------------------------
    # Forward delegation
    # ------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.generate(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    # ------------------------------------------------------------------
    # Live state
    # ------------------------------------------------------------------

    @property
    def neuron_indices(self) -> Dict[str, List[int]]:
        """Live ``module_path -> [neuron_idx, ...]`` view of the wrapped indices."""
        out: Dict[str, List[int]] = {}
        for name, module in self.model.named_modules():
            if isinstance(module, _DeltaModule):
                out[name] = module.idx.tolist()
        return out

    def get_adapter_state_dict(self) -> "OrderedDict[str, torch.Tensor]":
        """Collect all ``dW`` and ``idx`` tensors from every delta wrapper."""
        out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for name, module in self.model.named_modules():
            if isinstance(module, _DeltaModule):
                out[f"{name}.dW"] = module.dW.detach().cpu()
                out[f"{name}.idx"] = module.idx.cpu()
        return out

    def print_trainable_parameters(self) -> None:
        """Print trainable / total parameter counts (PEFT-compatible API)."""
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        pct = 100.0 * n_trainable / n_total if n_total else 0.0
        print(
            f"trainable params: {n_trainable:,} || "
            f"all params: {n_total:,} || "
            f"trainable%: {pct:.4f}%"
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None:
        """Save adapter weights, config, and indices.

        Files written to *save_directory*:

        * ``adapter_model.safetensors`` (or ``adapter_model.pt`` when
          ``safetensors`` is unavailable) -- ``dW`` + ``idx`` per module.
        * ``critical_neuron_config.json`` -- the config.
        * ``neuron_indices.json`` -- the live neuron index map.

        Extra kwargs are accepted (for compatibility with HF Trainer's
        ``save_pretrained`` call site) and ignored.
        """
        del kwargs
        os.makedirs(save_directory, exist_ok=True)

        state = self.get_adapter_state_dict()
        try:
            from safetensors.torch import save_file
            save_file(state, os.path.join(save_directory, self._ADAPTER_FILENAME))
        except ImportError:
            logger.warning("safetensors not installed; falling back to torch.save (.pt).")
            torch.save(state, os.path.join(save_directory, self._ADAPTER_FILENAME_PT))

        self.config.save_pretrained(save_directory, indices=self.neuron_indices)

    @classmethod
    def from_pretrained(
        cls,
        adapter_path: str,
        *,
        base_model_name_or_path: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        modules_to_skip: Optional[Set[str]] = None,
    ) -> "CriticalNeuronModel":
        """Load a base model and inject saved adapter weights.

        Parameters
        ----------
        adapter_path
            Directory containing the adapter files
            (``adapter_model.*``, ``critical_neuron_config.json``,
            ``neuron_indices.json``).
        base_model_name_or_path
            HuggingFace id or local path of the *base* model.  When
            ``None``, falls back to the value recorded in
            ``adapter_path/critical_neuron_config.json``.
        model_kwargs
            Extra kwargs forwarded to ``AutoModelForCausalLM.from_pretrained``.
        modules_to_skip
            Forwarded to :func:`get_neuron_model`.
        """
        from transformers import AutoModelForCausalLM

        config = CriticalNeuronConfig.from_pretrained(adapter_path)
        indices = load_neuron_indices(adapter_path)

        if base_model_name_or_path is None:
            base_model_name_or_path = config.base_model_name_or_path
        if not base_model_name_or_path:
            raise ValueError(
                "base_model_name_or_path was not provided and is not recorded in "
                f"{os.path.join(adapter_path, _CONFIG_FILENAME)}. Pass it explicitly."
            )
        config.base_model_name_or_path = base_model_name_or_path

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path, **(model_kwargs or {})
        )
        wrapped = get_neuron_model(
            model, config, indices, modules_to_skip=modules_to_skip
        )

        sf_path = os.path.join(adapter_path, cls._ADAPTER_FILENAME)
        pt_path = os.path.join(adapter_path, cls._ADAPTER_FILENAME_PT)
        if os.path.isfile(sf_path):
            from safetensors.torch import load_file
            state = load_file(sf_path)
        elif os.path.isfile(pt_path):
            state = torch.load(pt_path, map_location="cpu")
        else:
            logger.warning(
                "No adapter weights at %s or %s; wrappers retain zero dW.",
                sf_path, pt_path,
            )
            return wrapped

        for name, module in wrapped.model.named_modules():
            dw_key = f"{name}.dW"
            if isinstance(module, _DeltaModule) and dw_key in state:
                module.dW.data.copy_(state[dw_key])

        return wrapped

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge_and_unload(self) -> nn.Module:
        """Merge every delta into the base weights and return the plain model.

        After this call the wrapper is no longer needed; the returned
        model is a vanilla ``transformers`` model with the neuron
        updates baked into the weights.
        """
        for name, module in list(self.model.named_modules()):
            if isinstance(module, LinearDeltaSubspace):
                _set_submodule(self.model, name, module.merge_to_linear_())
            elif isinstance(module, NormDeltaSubspace):
                _set_submodule(self.model, name, module.merge_to_norm_())
            elif isinstance(module, EmbeddingDeltaSubspace):
                _set_submodule(self.model, name, module.merge_to_embedding_())
        return self.model


# =====================================================================
# freeze_neurons  (safety-preserving full fine-tuning)
# =====================================================================


class FrozenNeuronHandle:
    """Resource handle returned by :func:`freeze_neurons`.

    The forward pass of the wrapped model is unchanged: only backward
    hooks are inserted, which zero out the gradient slices of the
    frozen neurons each step.  When the optimiser uses weight decay,
    call :meth:`restore_frozen_weights` after every optimiser step to
    counteract decay drift -- or pass :meth:`make_trainer_callback` to
    the HF ``Trainer`` to do it for you.
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
        """Restore the frozen weight slices to their pre-training values.

        Call after every optimiser step when ``weight_decay > 0``.
        Unnecessary when ``weight_decay == 0`` (gradient hooks already
        block every update).
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
        """Detach every gradient hook (effectively unfreezes the neurons)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._frozen.clear()

    def make_trainer_callback(self) -> Any:
        """Return a ``transformers.TrainerCallback`` for HF Trainer.

        The callback invokes :meth:`restore_frozen_weights` at the end of
        every optimiser step.
        """
        from transformers import TrainerCallback

        handle = self

        class _FreezeCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):  # noqa: D401
                handle.restore_frozen_weights()

        return _FreezeCallback()

    def summary(self) -> str:
        """One-line summary of frozen vs trainable parameter counts."""
        pct = 100.0 * self.n_frozen / self.n_total if self.n_total else 0.0
        n_trainable = self.n_total - self.n_frozen
        return (
            f"frozen params: {self.n_frozen:,} || "
            f"trainable params: {n_trainable:,} || "
            f"all params: {self.n_total:,} || "
            f"frozen%: {pct:.4f}%"
        )


def freeze_neurons(
    model: nn.Module,
    indices: Dict[str, List[int]],
    config: Optional[CriticalNeuronConfig] = None,
) -> FrozenNeuronHandle:
    """Freeze the chosen neurons during full fine-tuning.

    The model is left structurally unchanged -- only backward hooks are
    registered, so the forward pass has **zero overhead**.  Each hook
    zeroes the gradient slice corresponding to one frozen neuron.

    Strictness
    ----------
    The classifier path is:

    1. If ``config.classify(name)`` returns a known type, use it.
    2. Otherwise, if the module is structurally a 1-D norm
       (:func:`_is_norm`), treat it as ``"norm"`` -- this is safe
       because norms have no axis ambiguity.
    3. Otherwise, raise ``ValueError``.  Wrong-axis fallbacks are not
       attempted: silently freezing the wrong axis of an ``nn.Linear``
       would be undetectable from the outside.

    Parameters
    ----------
    model
        A trainable model.  ``requires_grad`` is not changed here.
    indices
        ``module_path -> [neuron_idx, ...]`` to freeze.
    config
        Used for classification.  A default config is constructed when
        ``None``.

    Returns
    -------
    FrozenNeuronHandle
    """
    if config is None:
        config = CriticalNeuronConfig()

    named = dict(model.named_modules())
    plan: List[tuple] = []
    bad: List[str] = []
    for name, idx_list in indices.items():
        module = named.get(name)
        if module is None or not idx_list:
            continue
        mod_type = config.classify(name)
        if mod_type is None:
            if _is_norm(module):
                mod_type = "norm"
            else:
                bad.append(name)
                continue
        plan.append((name, module, mod_type, idx_list))
    if bad:
        raise ValueError(
            f"Cannot classify {len(bad)} module(s) for freezing: "
            f"{bad[:5]}{'...' if len(bad) > 5 else ''}. Either add the leaf "
            "names to the appropriate category on the config or remove them "
            "from the indices dict."
        )

    hooks: List[torch.utils.hooks.RemovableHandle] = []
    frozen_snapshots: List[tuple] = []
    n_frozen = 0

    for name, module, mod_type, idx_list in plan:
        idx = torch.tensor(sorted(idx_list), dtype=torch.long)

        if mod_type in ("row", "column") and isinstance(module, nn.Linear):
            axis = 0 if mod_type == "row" else 1
            saved = (
                module.weight.data[idx].clone()
                if axis == 0
                else module.weight.data[:, idx].clone()
            )
            hook = module.weight.register_hook(
                lambda g, _i=idx, _a=axis: g.index_fill_(_a, _i.to(g.device), 0)
            )
            frozen_snapshots.append((module.weight, mod_type, idx, saved))
            hooks.append(hook)
            scalar_per_neuron = module.weight.size(1 - axis)
            n_frozen += len(idx_list) * scalar_per_neuron

        elif mod_type == "norm" and _is_norm(module):
            saved = module.weight.data[idx].clone()
            hook = module.weight.register_hook(
                lambda g, _i=idx: g.index_fill_(0, _i.to(g.device), 0)
            )
            frozen_snapshots.append((module.weight, "norm", idx, saved))
            hooks.append(hook)
            n_frozen += len(idx_list)

        else:
            # Should be unreachable due to strict classification above.
            logger.warning(
                "Skipping module %s (%s, %s): no freeze rule.",
                name, mod_type, type(module).__name__,
            )

    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "freeze_neurons: %d frozen scalar weights out of %d total (%.4f%%)",
        n_frozen, n_total, 100.0 * n_frozen / n_total if n_total else 0.0,
    )
    return FrozenNeuronHandle(hooks, frozen_snapshots, n_frozen, n_total)

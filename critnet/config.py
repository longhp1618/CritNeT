"""Configuration for CriticalNeuronToolkit -- a PEFT method based on critical-neuron tuning.

This module defines :class:`CriticalNeuronConfig`, the central configuration
object that governs which modules are targeted, how neurons are categorised
(row / column / norm / embedding), and where detected neuron indices are
persisted.

Architectural defaults
----------------------
LLaMA, Mistral, and Qwen model families share an identical module-naming
convention for their linear projections and layer norms.  When a category
field is left as ``None``, the config automatically fills it with sensible
defaults so that users can get started with a single line:

>>> config = CriticalNeuronConfig(sparsity_ratio=0.05)

To customise -- for example to include the LM head or Qwen-3 QK norms --
simply pass the relevant list explicitly:

>>> config = CriticalNeuronConfig(
...     row_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
...     norm_modules=["input_layernorm", "post_attention_layernorm", "q_norm", "k_norm"],
... )
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Architecture defaults -- shared by LLaMA / Mistral / Qwen
# ---------------------------------------------------------------------------

DEFAULT_ROW_MODULES: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "gate_proj",
    "up_proj",
]
"""Linear modules whose neurons correspond to **rows** of the weight matrix
(i.e. output-dimension neurons).  Importance is computed as
``|W[i, :] * grad[i, :]|.sum()`` with reduction over ``dim=1``."""

DEFAULT_COLUMN_MODULES: List[str] = [
    "o_proj",
    "down_proj",
]
"""Linear modules whose neurons correspond to **columns** of the weight matrix
(i.e. input-dimension neurons).  Importance is computed as
``|W[:, j] * grad[:, j]|.sum()`` with reduction over ``dim=0``."""

DEFAULT_NORM_MODULES: List[str] = [
    "input_layernorm",
    "post_attention_layernorm",
]
"""Normalization layers with a 1-D learnable weight.  Each element is treated
as an individual "neuron".  Importance is simply ``|w[i] * grad[i]|``."""

DEFAULT_GATE_COMBINES_WITH: Dict[str, str] = {
    "gate_proj": "up_proj",
}
"""Default pairing for SwiGLU architectures: ``gate_proj`` importance scores
are added element-wise to ``up_proj`` scores before global top-k selection,
and both modules share the resulting selected indices."""

_CONFIG_FILENAME = "critical_neuron_config.json"
_INDICES_FILENAME = "neuron_indices.json"


@dataclass
class CriticalNeuronConfig:
    """Configuration for critical-neuron detection and fine-tuning.

    Module categories
    -----------------
    Modules are split into three categories that determine how "neurons" are
    defined and how the delta-subspace wrapper operates:

    * **row_modules** -- ``nn.Linear`` layers where each row of the weight
      matrix is one neuron (e.g. ``q_proj``, ``up_proj``).
    * **column_modules** -- ``nn.Linear`` layers where each column is one
      neuron (e.g. ``o_proj``, ``down_proj``).
    * **norm_modules** -- normalisation layers (``RMSNorm`` / ``LayerNorm``)
      with a 1-D weight vector; each scalar element is one neuron.

    ``embed_tokens`` and ``lm_head`` are deliberately excluded: wrapping
    them is incompatible with fused-kernel libraries (e.g. Liger) and
    weight-tying, and they are skipped by default in
    :func:`get_neuron_model`.

    Smart defaults
    --------------
    Any category left as ``None`` is automatically populated with the
    standard module names shared by LLaMA, Mistral, and Qwen.  Pass an
    explicit list (even an empty ``[]``) to override a default.

    Parameters
    ----------
    row_modules : list[str] or None
        Module-name suffixes for row-neuron linear layers.
        *Default:* ``["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]``.
    column_modules : list[str] or None
        Module-name suffixes for column-neuron linear layers.
        *Default:* ``["o_proj", "down_proj"]``.
    norm_modules : list[str] or None
        Module-name suffixes for normalisation layers.
        *Default:* ``["input_layernorm", "post_attention_layernorm"]``.
        Add ``"q_norm"``, ``"k_norm"`` for Qwen-3 or ``"norm"`` for the
        final RMSNorm.
    embedding_modules : list[str] or None
        Module-name suffixes for embedding layers.  *Default:* ``None``
        (not included).  Advanced use only.
    sparsity_ratio : float
        Fraction of total neurons to keep as "critical" during global
        top-k selection.  Must be in (0, 1).  *Default:* ``0.05`` (5 %).
    gate_combines_with : dict[str, str] or None
        Mapping from a "gate" module suffix to its partner, so their
        importance scores are summed before selection and they share the
        same neuron indices.
        *Default:* ``{"gate_proj": "up_proj"}``.
    neuron_indices : dict[str, list[int]] or None
        Detected neuron indices keyed by **full module path** (e.g.
        ``"model.layers.0.self_attn.q_proj"``).  Populated by
        :class:`NeuronDetector` or loaded from disk.
    base_model_name_or_path : str or None
        HuggingFace model identifier or local path for the base model.
    """

    row_modules: Optional[List[str]] = None
    column_modules: Optional[List[str]] = None
    norm_modules: Optional[List[str]] = None
    embedding_modules: Optional[List[str]] = None

    sparsity_ratio: float = 0.05

    gate_combines_with: Optional[Dict[str, str]] = None

    neuron_indices: Optional[Dict[str, List[int]]] = field(default=None, repr=False)

    base_model_name_or_path: Optional[str] = None

    def __post_init__(self) -> None:
        # ---- smart defaults ------------------------------------------------
        if self.row_modules is None:
            self.row_modules = list(DEFAULT_ROW_MODULES)
        if self.column_modules is None:
            self.column_modules = list(DEFAULT_COLUMN_MODULES)
        if self.norm_modules is None:
            self.norm_modules = list(DEFAULT_NORM_MODULES)
        # embedding_modules stays None unless the user explicitly provides it

        if self.gate_combines_with is None:
            linear = set(self.row_modules or []) | set(self.column_modules or [])
            self.gate_combines_with = {
                k: v for k, v in DEFAULT_GATE_COMBINES_WITH.items()
                if k in linear and v in linear
            }

        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Run consistency checks and raise ``ValueError`` on problems."""
        if not (0.0 < self.sparsity_ratio < 1.0):
            raise ValueError(
                f"sparsity_ratio must be in (0, 1), got {self.sparsity_ratio}"
            )

        # Collect all suffixes per category and check for overlaps
        categories: Dict[str, List[str]] = {
            "row_modules": self.row_modules or [],
            "column_modules": self.column_modules or [],
            "norm_modules": self.norm_modules or [],
            "embedding_modules": self.embedding_modules or [],
        }
        seen: Dict[str, str] = {}
        for cat_name, suffixes in categories.items():
            for s in suffixes:
                if s in seen:
                    raise ValueError(
                        f"Module suffix '{s}' appears in both "
                        f"'{seen[s]}' and '{cat_name}'. "
                        f"Each suffix must belong to exactly one category."
                    )
                seen[s] = cat_name

        # gate_combines_with keys/values must reference linear categories
        linear_suffixes = set(categories["row_modules"]) | set(categories["column_modules"])
        if self.gate_combines_with:
            for gate, partner in self.gate_combines_with.items():
                if gate not in linear_suffixes:
                    raise ValueError(
                        f"gate_combines_with key '{gate}' is not in "
                        f"row_modules or column_modules: {sorted(linear_suffixes)}"
                    )
                if partner not in linear_suffixes:
                    raise ValueError(
                        f"gate_combines_with value '{partner}' is not in "
                        f"row_modules or column_modules: {sorted(linear_suffixes)}"
                    )

    # ------------------------------------------------------------------
    # Computed helpers
    # ------------------------------------------------------------------

    @property
    def target_modules(self) -> List[str]:
        """Union of all module-category suffixes.

        Returns a flat list of every module-name suffix that this config
        targets, across all four categories.  Useful for iterating over
        a model's named modules and checking membership.
        """
        modules: List[str] = []
        modules.extend(self.row_modules or [])
        modules.extend(self.column_modules or [])
        modules.extend(self.norm_modules or [])
        modules.extend(self.embedding_modules or [])
        return modules

    @property
    def linear_modules(self) -> List[str]:
        """Combined list of row and column module suffixes (all ``nn.Linear`` targets)."""
        return list(self.row_modules or []) + list(self.column_modules or [])

    def get_module_type(self, module_name: str) -> str:
        """Determine the category of a module from its name.

        The *leaf* component of ``module_name`` (the part after the last
        ``'.'``) is matched against each category list.

        Parameters
        ----------
        module_name : str
            Fully-qualified module path, e.g.
            ``"model.layers.0.self_attn.q_proj"``.

        Returns
        -------
        str
            One of ``"row"``, ``"column"``, ``"norm"``, or ``"embedding"``.

        Raises
        ------
        ValueError
            If the leaf name does not match any category.
        """
        leaf = module_name.rsplit(".", 1)[-1]
        if leaf in (self.row_modules or []):
            return "row"
        if leaf in (self.column_modules or []):
            return "column"
        if leaf in (self.norm_modules or []):
            return "norm"
        if leaf in (self.embedding_modules or []):
            return "embedding"
        raise ValueError(
            f"Module '{module_name}' (leaf='{leaf}') does not match any "
            f"configured category.  target_modules={self.target_modules}"
        )

    def matches_target(self, module_name: str) -> bool:
        """Return ``True`` if *module_name*'s leaf matches any target suffix."""
        leaf = module_name.rsplit(".", 1)[-1]
        return leaf in self.target_modules

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str) -> None:
        """Persist the config and neuron indices to *save_directory*.

        Two files are written:

        * ``critical_neuron_config.json`` -- all fields except
          ``neuron_indices``.
        * ``neuron_indices.json`` -- the neuron index mapping (only when
          ``neuron_indices`` is not ``None``).
        """
        os.makedirs(save_directory, exist_ok=True)

        config_dict = {
            "row_modules": self.row_modules,
            "column_modules": self.column_modules,
            "norm_modules": self.norm_modules,
            "embedding_modules": self.embedding_modules,
            "sparsity_ratio": self.sparsity_ratio,
            "gate_combines_with": self.gate_combines_with,
            "base_model_name_or_path": self.base_model_name_or_path,
        }

        config_path = os.path.join(save_directory, _CONFIG_FILENAME)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        if self.neuron_indices is not None:
            indices_path = os.path.join(save_directory, _INDICES_FILENAME)
            with open(indices_path, "w", encoding="utf-8") as f:
                json.dump(self.neuron_indices, f, indent=2)

    @classmethod
    def from_pretrained(cls, load_directory: str) -> "CriticalNeuronConfig":
        """Load a config (and optional neuron indices) from *load_directory*.

        Parameters
        ----------
        load_directory : str
            Directory containing ``critical_neuron_config.json`` and
            optionally ``neuron_indices.json``.

        Returns
        -------
        CriticalNeuronConfig
        """
        config_path = os.path.join(load_directory, _CONFIG_FILENAME)
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        indices_path = os.path.join(load_directory, _INDICES_FILENAME)
        neuron_indices = None
        if os.path.isfile(indices_path):
            with open(indices_path, "r", encoding="utf-8") as f:
                neuron_indices = json.load(f)

        return cls(
            row_modules=config_dict.get("row_modules"),
            column_modules=config_dict.get("column_modules"),
            norm_modules=config_dict.get("norm_modules"),
            embedding_modules=config_dict.get("embedding_modules"),
            sparsity_ratio=config_dict.get("sparsity_ratio", 0.05),
            gate_combines_with=config_dict.get("gate_combines_with"),
            neuron_indices=neuron_indices,
            base_model_name_or_path=config_dict.get("base_model_name_or_path"),
        )

    # ------------------------------------------------------------------
    # Repr / summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            "CriticalNeuronConfig",
            f"  row_modules      : {self.row_modules}",
            f"  column_modules   : {self.column_modules}",
            f"  norm_modules     : {self.norm_modules}",
            f"  embedding_modules: {self.embedding_modules}",
            f"  sparsity_ratio   : {self.sparsity_ratio}",
            f"  gate_combines_with: {self.gate_combines_with}",
            f"  base_model       : {self.base_model_name_or_path}",
        ]
        if self.neuron_indices is not None:
            n_modules = len(self.neuron_indices)
            n_neurons = sum(len(v) for v in self.neuron_indices.values())
            lines.append(f"  neuron_indices   : {n_neurons} neurons across {n_modules} modules")
        else:
            lines.append("  neuron_indices   : not loaded")
        return "\n".join(lines)

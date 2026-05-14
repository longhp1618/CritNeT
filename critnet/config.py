"""Configuration for :mod:`critnet`.

``CriticalNeuronConfig`` describes **which modules** in a transformer language
model are candidates for critical-neuron detection.  It is a pure
*architectural* description: there are deliberately no detection-run
hyperparameters (such as ``sparsity_ratio``) and no post-detection state
(such as ``neuron_indices``) on the config.  Those belong on
:class:`~critnet.detector.DetectionResult` instead.

Architectural defaults
----------------------
LLaMA, Mistral, and Qwen model families share an identical module-naming
convention for their linear projections and layer norms.  When a category
is left as ``None``, the config automatically fills it with sensible
defaults so users can get started with one line:

>>> config = CriticalNeuronConfig()  # full defaults

Override only what you need:

>>> config = CriticalNeuronConfig(
...     norm_modules=["input_layernorm", "post_attention_layernorm", "q_norm", "k_norm"],
... )
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
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
(output-dim neurons).  Importance is :math:`|W \\odot \\nabla_W \\mathcal{L}|`
reduced along ``dim=1``."""

DEFAULT_COLUMN_MODULES: List[str] = [
    "o_proj",
    "down_proj",
]
"""Linear modules whose neurons correspond to **columns** of the weight matrix
(input-dim neurons).  Importance reduces along ``dim=0``."""

DEFAULT_NORM_MODULES: List[str] = [
    "input_layernorm",
    "post_attention_layernorm",
]
"""1-D normalisation weights -- each scalar is an individual "neuron".
Add ``"q_norm"`` / ``"k_norm"`` for Qwen-3, ``"norm"`` for the final
RMSNorm."""

DEFAULT_GATE_COMBINES_WITH: Dict[str, str] = {
    "gate_proj": "up_proj",
}
"""Default pairing for SwiGLU architectures: ``gate_proj`` importance is
added element-wise to ``up_proj`` importance before global top-k, and both
modules share the resulting selected indices."""

_CONFIG_FILENAME = "critical_neuron_config.json"
_INDICES_FILENAME = "neuron_indices.json"


# ---------------------------------------------------------------------------
# Free-standing index I/O helpers (indices are no longer part of the config)
# ---------------------------------------------------------------------------


def save_neuron_indices(save_directory: str, indices: Dict[str, List[int]]) -> None:
    """Write ``neuron_indices.json`` to *save_directory*.

    The file is a plain JSON dump of ``{module_path: [neuron_idx, ...]}``.
    The directory is created if it does not exist.
    """
    os.makedirs(save_directory, exist_ok=True)
    path = os.path.join(save_directory, _INDICES_FILENAME)
    serialisable = {k: list(v) for k, v in indices.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)


def load_neuron_indices(load_directory: str) -> Dict[str, List[int]]:
    """Read ``neuron_indices.json`` from *load_directory*.

    Raises ``FileNotFoundError`` if the file is missing.
    """
    path = os.path.join(load_directory, _INDICES_FILENAME)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CriticalNeuronConfig
# ---------------------------------------------------------------------------


@dataclass
class CriticalNeuronConfig:
    """Architectural description of which modules carry "neurons".

    Modules are split into four categories that determine how a "neuron"
    is defined and how the delta-subspace wrapper operates:

    * **row_modules** -- ``nn.Linear`` layers where each row of the weight
      matrix is one neuron (e.g. ``q_proj``, ``up_proj``).
    * **column_modules** -- ``nn.Linear`` layers where each column is one
      neuron (e.g. ``o_proj``, ``down_proj``).
    * **norm_modules** -- normalisation layers (``RMSNorm`` / ``LayerNorm``)
      with a 1-D weight vector; each scalar element is one neuron.
    * **embedding_modules** -- ``nn.Embedding`` layers where each row (one
      vocab item) is one neuron.

    ``embed_tokens`` and ``lm_head`` are *not* excluded by the config
    itself; they are skipped by :func:`~critnet.model.get_neuron_model`
    via :data:`~critnet.model.DEFAULT_SKIP_MODULES` to avoid breaking
    fused kernels and weight-tying.  Pass
    ``modules_to_skip=set()`` to override.

    Smart defaults
    --------------
    Any category left as ``None`` is populated with the standard
    LLaMA / Mistral / Qwen suffixes.  Pass an explicit list (including
    an empty ``[]``) to override.

    Parameters
    ----------
    row_modules : list[str] or None
        Module-name suffixes for row-neuron linear layers.
        Default: ``["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]``.
    column_modules : list[str] or None
        Module-name suffixes for column-neuron linear layers.
        Default: ``["o_proj", "down_proj"]``.
    norm_modules : list[str] or None
        Module-name suffixes for 1-D norm weights.
        Default: ``["input_layernorm", "post_attention_layernorm"]``.
    embedding_modules : list[str] or None
        Module-name suffixes for embedding layers.
        Default: ``None`` (embeddings are off; pass a list to opt in).
    gate_combines_with : dict[str, str] or None
        Mapping from a "gate" suffix to a "partner" suffix.  The gate's
        importance is added element-wise into the partner before global
        top-k; both modules then share the resulting indices.
        Default: ``{"gate_proj": "up_proj"}`` when both appear in the
        linear lists.  ``{}`` disables gate-combination.
    base_model_name_or_path : str or None
        Optional metadata recording the HuggingFace model id / local
        path the config was built for.  Used by
        :meth:`~critnet.model.CriticalNeuronModel.from_pretrained` as a
        fallback when no explicit base path is provided.
    """

    row_modules: Optional[List[str]] = None
    column_modules: Optional[List[str]] = None
    norm_modules: Optional[List[str]] = None
    embedding_modules: Optional[List[str]] = None

    gate_combines_with: Optional[Dict[str, str]] = None

    base_model_name_or_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.row_modules is None:
            self.row_modules = list(DEFAULT_ROW_MODULES)
        if self.column_modules is None:
            self.column_modules = list(DEFAULT_COLUMN_MODULES)
        if self.norm_modules is None:
            self.norm_modules = list(DEFAULT_NORM_MODULES)
        # embedding_modules stays None unless explicitly provided

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
                        f"Module suffix '{s}' appears in both '{seen[s]}' and "
                        f"'{cat_name}'. Each suffix must belong to exactly one "
                        f"category."
                    )
                seen[s] = cat_name

        linear_suffixes = set(categories["row_modules"]) | set(categories["column_modules"])
        if self.gate_combines_with:
            for gate, partner in self.gate_combines_with.items():
                if gate not in linear_suffixes:
                    raise ValueError(
                        f"gate_combines_with key '{gate}' must appear in "
                        f"row_modules or column_modules; got {sorted(linear_suffixes)}."
                    )
                if partner not in linear_suffixes:
                    raise ValueError(
                        f"gate_combines_with value '{partner}' must appear in "
                        f"row_modules or column_modules; got {sorted(linear_suffixes)}."
                    )

    # ------------------------------------------------------------------
    # Classification (single source of truth for module typing)
    # ------------------------------------------------------------------

    def classify(self, module_name: str) -> Optional[str]:
        """Return the category of *module_name*, or ``None`` if not targeted.

        The *leaf* component of ``module_name`` (the part after the last
        ``'.'``) is matched against each category list.

        Returns
        -------
        ``"row"`` | ``"column"`` | ``"norm"`` | ``"embedding"`` | ``None``

        Examples
        --------
        >>> cfg = CriticalNeuronConfig()
        >>> cfg.classify("model.layers.0.self_attn.q_proj")
        'row'
        >>> cfg.classify("model.layers.0.self_attn.q_norm")  # not in defaults
        >>> # returns None
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
        return None

    # ------------------------------------------------------------------
    # Computed views
    # ------------------------------------------------------------------

    @property
    def target_modules(self) -> List[str]:
        """Flat union of every targeted module suffix across all four categories."""
        return (
            list(self.row_modules or [])
            + list(self.column_modules or [])
            + list(self.norm_modules or [])
            + list(self.embedding_modules or [])
        )

    @property
    def linear_modules(self) -> List[str]:
        """``row_modules`` + ``column_modules`` (every ``nn.Linear`` target)."""
        return list(self.row_modules or []) + list(self.column_modules or [])

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save_pretrained(
        self,
        save_directory: str,
        *,
        indices: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        """Persist this config to *save_directory*.

        Always writes ``critical_neuron_config.json``.  If ``indices`` is
        provided, also writes ``neuron_indices.json`` via
        :func:`save_neuron_indices` -- a convenience equivalent to::

            config.save_pretrained(save_directory)
            save_neuron_indices(save_directory, indices)
        """
        os.makedirs(save_directory, exist_ok=True)
        config_dict = {
            "row_modules": self.row_modules,
            "column_modules": self.column_modules,
            "norm_modules": self.norm_modules,
            "embedding_modules": self.embedding_modules,
            "gate_combines_with": self.gate_combines_with,
            "base_model_name_or_path": self.base_model_name_or_path,
        }
        path = os.path.join(save_directory, _CONFIG_FILENAME)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        if indices is not None:
            save_neuron_indices(save_directory, indices)

    @classmethod
    def from_pretrained(cls, load_directory: str) -> "CriticalNeuronConfig":
        """Load a config from *load_directory*.

        Only the config is read here.  To load the companion neuron
        indices use :func:`load_neuron_indices` separately::

            config = CriticalNeuronConfig.from_pretrained("./neurons")
            indices = load_neuron_indices("./neurons")
        """
        path = os.path.join(load_directory, _CONFIG_FILENAME)
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            row_modules=d.get("row_modules"),
            column_modules=d.get("column_modules"),
            norm_modules=d.get("norm_modules"),
            embedding_modules=d.get("embedding_modules"),
            gate_combines_with=d.get("gate_combines_with"),
            base_model_name_or_path=d.get("base_model_name_or_path"),
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable description."""
        lines = [
            "CriticalNeuronConfig",
            f"  row_modules       : {self.row_modules}",
            f"  column_modules    : {self.column_modules}",
            f"  norm_modules      : {self.norm_modules}",
            f"  embedding_modules : {self.embedding_modules}",
            f"  gate_combines_with: {self.gate_combines_with}",
            f"  base_model        : {self.base_model_name_or_path}",
        ]
        return "\n".join(lines)

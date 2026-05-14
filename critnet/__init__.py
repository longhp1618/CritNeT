"""CritNet -- parameter-efficient fine-tuning via critical-neuron selection.

Quickstart
----------
>>> from critnet import (
...     CriticalNeuronConfig, NeuronDetector,
...     get_neuron_model, load_neuron_indices,
... )
>>> # detect
>>> config = CriticalNeuronConfig(base_model_name_or_path="meta-llama/Llama-3.1-8B-Instruct")
>>> result = NeuronDetector(model, config).detect(loader, sparsity_ratio=0.01)
>>> config.save_pretrained("./neurons", indices=result.indices)
>>>
>>> # train
>>> config = CriticalNeuronConfig.from_pretrained("./neurons")
>>> indices = load_neuron_indices("./neurons")
>>> model = get_neuron_model(model, config, indices)

Module map
----------
* :mod:`config`       -- :class:`CriticalNeuronConfig`,
  :func:`save_neuron_indices`, :func:`load_neuron_indices`
* :mod:`detector`     -- :class:`NeuronDetector`, :class:`DetectionResult`,
  :func:`select_neurons_from_cache`
* :mod:`statistician` -- :class:`NeuronStatistician`, :class:`StatisticsResult`
* :mod:`model`        -- :class:`CriticalNeuronModel`,
  :func:`get_neuron_model`, :func:`freeze_neurons`,
  :class:`FrozenNeuronHandle`, :class:`LinearDeltaSubspace`,
  :class:`NormDeltaSubspace`, :class:`EmbeddingDeltaSubspace`,
  :data:`DEFAULT_SKIP_MODULES`
* :mod:`deactivator`  -- :class:`NeuronDeactivator`,
  :class:`DeactivationResult`
"""

from .config import (
    CriticalNeuronConfig,
    load_neuron_indices,
    save_neuron_indices,
)
from .deactivator import DeactivationResult, NeuronDeactivator
from .detector import (
    DetectionResult,
    NeuronDetector,
    select_neurons_from_cache,
)
from .model import (
    DEFAULT_SKIP_MODULES,
    CriticalNeuronModel,
    EmbeddingDeltaSubspace,
    FrozenNeuronHandle,
    LinearDeltaSubspace,
    NormDeltaSubspace,
    freeze_neurons,
    get_neuron_model,
)
from .statistician import NeuronStatistician, StatisticsResult

__all__ = [
    "CriticalNeuronConfig",
    "load_neuron_indices",
    "save_neuron_indices",
    "NeuronDetector",
    "DetectionResult",
    "select_neurons_from_cache",
    "NeuronStatistician",
    "StatisticsResult",
    "CriticalNeuronModel",
    "get_neuron_model",
    "freeze_neurons",
    "FrozenNeuronHandle",
    "DEFAULT_SKIP_MODULES",
    "LinearDeltaSubspace",
    "NormDeltaSubspace",
    "EmbeddingDeltaSubspace",
    "NeuronDeactivator",
    "DeactivationResult",
]

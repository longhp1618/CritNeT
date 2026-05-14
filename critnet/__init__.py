"""CritNet — Parameter-Efficient Fine-Tuning via critical-neuron selection.

Quick-start
-----------
>>> from critnet import CriticalNeuronConfig, get_neuron_model
>>> config = CriticalNeuronConfig.from_pretrained("./detected_neurons")
>>> model = get_neuron_model(model, config)

Modules
-------
* :mod:`config`       -- :class:`CriticalNeuronConfig`
* :mod:`detector`     -- :class:`NeuronDetector`
* :mod:`statistician` -- :class:`NeuronStatistician`, :class:`StatisticsResult`
* :mod:`model`        -- :class:`CriticalNeuronModel`, :func:`get_neuron_model`,
  :func:`freeze_neurons`, :class:`FrozenNeuronHandle`,
  :class:`LinearDeltaSubspace`, :class:`NormDeltaSubspace`,
  :class:`EmbeddingDeltaSubspace`
* :mod:`deactivator`  -- :class:`NeuronDeactivator`, :class:`DeactivationResult`
"""

from .config import CriticalNeuronConfig
from .detector import NeuronDetector
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
from .deactivator import NeuronDeactivator, DeactivationResult

__all__ = [
    "CriticalNeuronConfig",
    "NeuronDetector",
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

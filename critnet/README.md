# `critnet` — API reference

This is the per-symbol reference for the `critnet` package. For installation, repository layout, and a runnable Quickstart, see the root [README.md](../README.md).

> **TL;DR.** `critnet` ranks every "neuron" (row / column of a linear layer, or scalar of a norm) by the first-order Taylor importance $|w \odot \nabla_w \mathcal{L}|$, then lets you act on the global top-$k$ set: **analyse**, **deactivate**, **freeze**, or **fine-tune only those neurons**. One `CriticalNeuronConfig` describes the targeted modules; the per-run `sparsity_ratio` lives on `NeuronDetector.detect(...)`; the per-run indices live on a `DetectionResult`. Nothing else holds hidden state.

**Dependencies:** `torch`, `transformers`, `tqdm`. `safetensors` is optional (adapter and deactivated-checkpoint I/O fall back to `torch.save` when it is missing).

---

## Contents

1. [Public symbols at a glance](#public-symbols-at-a-glance)
2. [How importance scoring works](#how-importance-scoring-works)
3. [`CriticalNeuronConfig`](#criticalneuronconfig)
4. [`NeuronDetector` and `DetectionResult`](#neurondetector-and-detectionresult)
5. [`select_neurons_from_cache`](#select_neurons_from_cache)
6. [`NeuronStatistician` and `StatisticsResult`](#neuronstatistician-and-statisticsresult)
7. [`NeuronDeactivator` and `DeactivationResult`](#neurondeactivator-and-deactivationresult)
8. [`get_neuron_model`, `CriticalNeuronModel`, delta wrappers](#get_neuron_model-criticalneuronmodel-delta-wrappers)
9. [`freeze_neurons` and `FrozenNeuronHandle`](#freeze_neurons-and-frozenneuronhandle)
10. [Saved-artefact layout](#saved-artefact-layout)
11. [Cookbook](#cookbook)

---

## Public symbols at a glance

```python
from critnet import (
    # Architectural config + index I/O
    CriticalNeuronConfig,
    save_neuron_indices, load_neuron_indices,

    # Detection
    NeuronDetector, DetectionResult, select_neurons_from_cache,

    # Set algebra
    NeuronStatistician, StatisticsResult,

    # In-place ablation
    NeuronDeactivator, DeactivationResult,

    # Sparse PEFT + freezing
    get_neuron_model, CriticalNeuronModel,
    freeze_neurons, FrozenNeuronHandle,
    DEFAULT_SKIP_MODULES,
    LinearDeltaSubspace, NormDeltaSubspace, EmbeddingDeltaSubspace,
)
```

Every workflow follows the same shape:

```
build CriticalNeuronConfig
    │
    ▼
NeuronDetector(model, config).detect(loader, sparsity_ratio=...)  ─┐
                                                                    ├─►  DetectionResult
select_neurons_from_cache(cache_path, config, sparsity_ratio=...)  ─┘     (.indices, .sparsity_ratio,
                                                                            .gate_to_partner_path)
    │
    └─► result.indices is then consumed by ONE of:
            NeuronStatistician.analyze(...)         → StatisticsResult.save_report(dir)
            NeuronDeactivator.deactivate(indices)   → DeactivationResult
            get_neuron_model(model, config, indices) → CriticalNeuronModel  (+ HF Trainer)
            freeze_neurons(model, indices, config)  → FrozenNeuronHandle    (+ HF Trainer)
```

---

## How importance scoring works

### Per-neuron importance

For each targeted parameter tensor $W$, the detector computes the elementwise product $|W \odot \nabla_W \mathcal{L}|$ and **reduces it to one scalar per neuron** based on the module type:

| Module type | What is a "neuron"? | Reduction |
|-------------|--------------------|-----------|
| **row** (`q_proj`, `up_proj`, …) | one **row** of $W$ — an output channel | sum over the input dim |
| **column** (`o_proj`, `down_proj`) | one **column** of $W$ — an input channel | sum over the output dim |
| **norm** (`input_layernorm`, …) | one scalar of the 1-D weight | identity ($\|w_i \cdot \nabla w_i\|$) |
| **embedding** (opt-in) | one **row** of the embedding (a vocab item) | sum over the embed dim |

### Global top-$k$ selection

Scores from **all** targeted modules are pooled into a single list, then the top

```text
k = max(1, int(total_neuron_count * sparsity_ratio))
```

neurons survive. `sparsity_ratio` is therefore a **fraction of pooled neurons across the whole model**, not a per-layer fraction. This naturally lets attention QKV neurons compete with FFN gate/up/down neurons on the same scale.

### Gate / partner combination (SwiGLU)

In SwiGLU MLPs, `gate_proj` and `up_proj` are multiplied element-wise inside the activation, so picking row $i$ of one without the corresponding row $i$ of the other is rarely meaningful. The default config sets

```python
gate_combines_with = {"gate_proj": "up_proj"}
```

which makes the detector:

1. Add `gate_proj`'s row-scores into `up_proj`'s scores element-wise **before** top-$k$.
2. Mirror the resulting `up_proj` selection back onto `gate_proj` so the two modules share their selected indices.

This both halves the search space and guarantees that any active gate row has a matching up row.

---

## `CriticalNeuronConfig`

```python
from critnet import CriticalNeuronConfig
```

`CriticalNeuronConfig` is the single source of truth for **which modules** are candidates. It carries no detection hyperparameters and no detection output; those belong on `NeuronDetector.detect` / `DetectionResult`.

### Fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `row_modules` | `list[str] \| None` | `["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]` | Linear layers where rows are neurons. |
| `column_modules` | `list[str] \| None` | `["o_proj", "down_proj"]` | Linear layers where columns are neurons. |
| `norm_modules` | `list[str] \| None` | `["input_layernorm", "post_attention_layernorm"]` | 1-D norm weights — each scalar is one neuron. |
| `embedding_modules` | `list[str] \| None` | `None` | Embedding layers (rows = vocab items). Opt-in. |
| `gate_combines_with` | `dict[str, str] \| None` | `{"gate_proj": "up_proj"}` (when both are targeted) | SwiGLU pairing. |
| `base_model_name_or_path` | `str \| None` | `None` | Metadata used by `CriticalNeuronModel.from_pretrained` as a fallback. |

Any field left as `None` is replaced by a sensible LLaMA / Mistral / Qwen default in `__post_init__`. Pass an explicit empty list (`[]`) to opt out entirely.

### Methods

- **`classify(name) -> Optional[str]`** — single source of truth for module typing. Returns `"row"`, `"column"`, `"norm"`, `"embedding"`, or `None` based on the *leaf* component of `name`.
- **`target_modules`** — flat list of every targeted suffix.
- **`linear_modules`** — `row_modules + column_modules`.
- **`save_pretrained(save_directory, *, indices=None)`** — writes `critical_neuron_config.json`; also writes `neuron_indices.json` when `indices` is provided.
- **`CriticalNeuronConfig.from_pretrained(load_directory)`** — load the config (only; use `load_neuron_indices` for the indices).
- **`summary() -> str`** — human-readable formatting.

### Index I/O helpers (free functions)

```python
from critnet import save_neuron_indices, load_neuron_indices

save_neuron_indices("./neurons", indices)            # writes neuron_indices.json
indices = load_neuron_indices("./neurons")           # reads neuron_indices.json
```

These are deliberately separate from the config so neither owns the other.

---

## `NeuronDetector` and `DetectionResult`

```python
from critnet import NeuronDetector
```

### Constructor

```python
NeuronDetector(model: nn.Module, config: CriticalNeuronConfig)
```

`model` is required. Passing `None` raises `TypeError` — use `select_neurons_from_cache` for cache-only runs.

### `detect(...)`

```python
result = detector.detect(
    dataloader,
    *,
    sparsity_ratio: float,
    save_importance_cache_path: Optional[str] = None,
) -> DetectionResult
```

- **`dataloader`** — each batch must be a `dict` with at least `input_ids`, `attention_mask`, and `labels`. The collator is responsible for label preparation: mask prompt tokens to `-100` for chat SFT, or set `labels == input_ids` for pre-training.
- **`sparsity_ratio`** — fraction of pooled neurons across the whole model. Strictly in $(0, 1]$. Validated.
- **`save_importance_cache_path`** — when set, the post-gate-combined score tensors and gate map are pickled with `torch.save`, so a later run can call `select_neurons_from_cache` at a different ratio without the backward pass.

`detect()` is **non-mutating** to the caller's perspective:

- `model.training` and the `requires_grad` flags of every target parameter are snapshotted and restored on exit.
- `model.zero_grad()` is called before and after the backward loop.
- The caller's config is unchanged. The selected indices live only on the returned `DetectionResult`.

### `DetectionResult`

```python
@dataclass
class DetectionResult:
    indices: dict[str, list[int]]
    sparsity_ratio: float
    gate_to_partner_path: dict[str, str]
```

- `result.total_selected` — sum of every selected index across modules.
- `result.n_modules` — number of modules that received at least one index.
- `result.summary()` — one-line string.

To persist a result:

```python
config.save_pretrained("./neurons", indices=result.indices)
```

---

## `select_neurons_from_cache`

```python
from critnet import select_neurons_from_cache

result = select_neurons_from_cache(
    cache_path: str,
    config: CriticalNeuronConfig,
    *,
    sparsity_ratio: float,
) -> DetectionResult
```

Replays the global top-$k$ from a previously saved importance cache without loading a model. Cheap; safe to call in a tight loop while sweeping `sparsity_ratio`. The cache is the file written by `detect(..., save_importance_cache_path=...)`.

---

## `NeuronStatistician` and `StatisticsResult`

```python
from critnet import NeuronStatistician
```

```python
stats = NeuronStatistician(model=model, config=config)
result = stats.analyze({
    "utility": utility_result.indices,
    "refusal": refusal_result.indices,
})
result.save_report("./statistics")
```

Both `model` and `config` are optional. Without `model` you still get the union / intersection / exclusive / non-shared sets — but parameter coverage numbers will be zero (the statistician needs the real weight shapes to compute them).

### Computed sets

For each targeted module path $m$ and tasks $T = \{t_1, \dots, t_n\}$:

| Field on `StatisticsResult` | Definition |
|---|---|
| `union[m]` | $\bigcup_{t} N_t(m)$ |
| `intersection[m]` | $\bigcap_{t} N_t(m)$ |
| `non_shared[m]` | `union[m] - intersection[m]` |
| `exclusive[t][m]` | $N_t(m) \setminus \bigcap_{t'} N_{t'}(m)$ |
| `task_indices[t][m]` | the original $N_t(m)$, kept for reporting |

### Properties / methods

- `union_count`, `intersection_count`, `non_shared_count` — aggregate scalar counts.
- `task_count(t)`, `exclusive_count(t)` — per-task counts.
- `total_neurons_per_module` and `params_per_neuron` — derived from real weight shapes when `model` is set.
- `total_model_params` — `sum(p.numel() for p in model.parameters())`.
- **`param_coverage(indices) -> float`** — % of model parameters covered by an arbitrary indices dict.
- **`summary() -> str`** — formatted report with per-task lines.
- **`save_report(save_directory) -> None`** — writes `union_neurons.json`, `shared_neurons.json`, `non_shared_neurons.json`, one `exclusive_<task>_neurons.json` per task, and `statistics.csv`.

The statistician is stateless across `analyze` calls; you can reuse one instance for many runs.

---

## `NeuronDeactivator` and `DeactivationResult`

```python
from critnet import NeuronDeactivator

deactivator = NeuronDeactivator(model, config)
result = deactivator.deactivate(indices)
print(result.summary())
deactivator.save_pretrained("./deactivated_model", tokenizer=tokenizer, indices=indices)
```

### `deactivate(indices) -> DeactivationResult`

Zeroes the chosen neurons **in place** on the model's weight tensors. Per module type:

| Type | Action |
|---|---|
| `"row"` | `W[idx, :] = 0` |
| `"column"` | `W[:, idx] = 0` |
| `"norm"` | `w[idx] = 0` (and `bias[idx] = 0` when present) |
| `"embedding"` | `E[idx, :] = 0` |

**Strict.** If any module path in `indices` cannot be classified by `config.classify(...)`, `deactivate` raises `ValueError` with the offending paths. Silent mis-classification was the old default and could zero the wrong axis.

### `DeactivationResult`

`modules_affected`, `neurons_zeroed`, `total_weights_zeroed`, `per_module` (a dict per touched module), and `summary()`.

### `save_pretrained(save_directory, *, tokenizer=None, indices=None)`

Convenience over the three explicit calls (`model.save_pretrained`, `tokenizer.save_pretrained`, `config.save_pretrained(..., indices=...)`). Useful when you want a single drop-in directory.

---

## `get_neuron_model`, `CriticalNeuronModel`, delta wrappers

### `get_neuron_model(model, config, indices, *, modules_to_skip=None) -> CriticalNeuronModel`

Walks `model.named_modules()`, replaces every module path in `indices` with the appropriate delta wrapper, freezes every base parameter, unfreezes only the wrapper's `dW`, and (when the model exposes it) calls `enable_input_require_grads()` for gradient checkpointing.

- `indices` must include only modules that the config classifies. Otherwise `ValueError` is raised (no fallback guess on the axis).
- `modules_to_skip` defaults to `DEFAULT_SKIP_MODULES = {"lm_head", "embed_tokens"}` to avoid breaking fused kernels and weight-tying. Pass `set()` to override.
- Every key in `indices` must be a real `named_modules()` path. Unknown paths raise.

### `CriticalNeuronModel`

A thin `nn.Module` wrapper. Forwards every call to the inner model and adds:

- `model.neuron_indices` — live `dict[str, list[int]]` view (reads back from the wrappers).
- `model.config` — the public alias of the config. `model.peft_config` is the **HF-Trainer-compat alias** for the same object.
- `model.save_pretrained(save_directory)` — writes:
  - `adapter_model.safetensors` (`dW` and `idx` for every wrapper); falls back to `adapter_model.pt`.
  - `critical_neuron_config.json` + `neuron_indices.json`.
- `model.from_pretrained(adapter_path, *, base_model_name_or_path=None, model_kwargs=None, modules_to_skip=None)` — loads a saved adapter on top of a fresh base model. If `base_model_name_or_path` is `None`, it is read from the saved config; raising if absent.
- `model.merge_and_unload() -> nn.Module` — merges every `dW` into the base weights in place and returns the plain `nn.Module` (no wrappers left).
- `model.print_trainable_parameters()` — PEFT-style summary.

### Delta wrappers (advanced)

`LinearDeltaSubspace(base_linear, indices, mode={"row", "column"}, train_bias=False)`,
`NormDeltaSubspace(base_norm, indices)`,
`EmbeddingDeltaSubspace(base_embedding, indices)`.

All three:

- expose a 1-D `idx` buffer and a 2-D `dW` parameter,
- preserve the base module's surface (`.weight`, `.bias`, `.in_features`, …),
- have an in-place `merge_to_*_()` method that bakes `dW` into the base and returns the plain module.

With `dW = 0`, every wrapper is the identity on the forward pass.

---

## `freeze_neurons` and `FrozenNeuronHandle`

```python
from critnet import freeze_neurons

handle = freeze_neurons(model, indices, config)
print(handle.summary())

trainer = transformers.Trainer(
    model=model,
    args=training_args,
    callbacks=[handle.make_trainer_callback()],
    ...
)
trainer.train()
handle.remove()
```

Registers a backward hook on each affected weight tensor that zeroes the gradient slice corresponding to one frozen neuron. The forward pass is **unchanged**; the model remains structurally identical.

### Strictness

`freeze_neurons` is strict to avoid wrong-axis surprises:

1. If `config.classify(name)` returns a known type, that type is used.
2. Otherwise, if the module is structurally a 1-D norm (1-D `.weight` and a class name containing `norm`/`layernorm`/`rmsnorm`), it is auto-classified as `"norm"`. Norms have no axis ambiguity, so this fallback is safe.
3. Otherwise `freeze_neurons` raises `ValueError` — there is no `"row"` guess for `nn.Linear`. Add the leaf name to your config's category lists explicitly.

### Handle methods

- `handle.restore_frozen_weights()` — re-write the frozen weight slices back to their pre-training values. Use after every optimiser step when `weight_decay > 0`.
- `handle.remove()` — detach every registered hook.
- `handle.make_trainer_callback()` — returns an HF `TrainerCallback` that calls `restore_frozen_weights` at every `on_step_end`. Hand it to `Trainer(callbacks=[...])`.
- `handle.summary()` — one-line "frozen / trainable / all" formatted string.

---

## Saved-artefact layout

```
<save_directory>/
├── critical_neuron_config.json     # CriticalNeuronConfig fields
├── neuron_indices.json             # {module_path: [neuron_idx, ...]}
├── adapter_model.safetensors       # ONLY for CriticalNeuronModel; dW + idx
│   (or adapter_model.pt)
└── statistics.csv, *_neurons.json  # ONLY for StatisticsResult.save_report
```

- **Detection / deactivation output** uses `critical_neuron_config.json` + `neuron_indices.json`.
- **PEFT adapter output** additionally writes `adapter_model.safetensors` (or `.pt`).
- **Statistician output** uses the JSON-per-set + `statistics.csv` layout described above.

---

## Cookbook

### 1. Detect → sparse PEFT → merge to a flat checkpoint

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from critnet import (
    CriticalNeuronConfig, NeuronDetector, get_neuron_model,
    load_neuron_indices,
)

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# 1) detect once
model = AutoModelForCausalLM.from_pretrained(MODEL).cuda()
config = CriticalNeuronConfig(base_model_name_or_path=MODEL)
result = NeuronDetector(model, config).detect(loader, sparsity_ratio=0.05)
config.save_pretrained("./neurons", indices=result.indices)

# 2) sparse PEFT training (HF Trainer)
config = CriticalNeuronConfig.from_pretrained("./neurons")
indices = load_neuron_indices("./neurons")
model = AutoModelForCausalLM.from_pretrained(MODEL).cuda()
neuron_model = get_neuron_model(model, config, indices)
neuron_model.print_trainable_parameters()
# ... transformers.Trainer(model=neuron_model, ...).train()
neuron_model.save_pretrained("./adapter")

# 3) reload + merge to a vanilla HF checkpoint
from critnet import CriticalNeuronModel
reloaded = CriticalNeuronModel.from_pretrained("./adapter")  # base path read from config
merged = reloaded.merge_and_unload()
merged.save_pretrained("./merged")
AutoTokenizer.from_pretrained(MODEL).save_pretrained("./merged")
```

### 2. Freeze safety neurons during full fine-tuning

```python
import json
from critnet import CriticalNeuronConfig, freeze_neurons

model = AutoModelForCausalLM.from_pretrained(MODEL).cuda()
with open("./safety_indices.json") as f:
    safety_indices = json.load(f)

handle = freeze_neurons(model, safety_indices, CriticalNeuronConfig())
print(handle.summary())

trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=ft_dataset,
    callbacks=[handle.make_trainer_callback()],  # restores weights after each step
)
trainer.train()
handle.remove()
```

### 3. Sweep `sparsity_ratio` without rerunning the backward pass

```python
from critnet import (
    CriticalNeuronConfig, NeuronDetector, select_neurons_from_cache,
)

config = CriticalNeuronConfig(base_model_name_or_path=MODEL)

# Pay the gradient cost once; save the post-gate-combined scores.
NeuronDetector(model, config).detect(
    loader,
    sparsity_ratio=0.05,  # any value; only used to log a first selection
    save_importance_cache_path="./cache/scores.pt",
)

# Replay global top-k at any ratio, cheap.
for r in [0.001, 0.005, 0.01, 0.05, 0.10]:
    result = select_neurons_from_cache(
        "./cache/scores.pt", config, sparsity_ratio=r,
    )
    config.save_pretrained(f"./neurons/r{r}", indices=result.indices)
```

### 4. Deactivate a neuron set and compare

```python
from critnet import (
    CriticalNeuronConfig, NeuronDeactivator, load_neuron_indices,
)

config = CriticalNeuronConfig.from_pretrained("./neurons")
config.base_model_name_or_path = MODEL  # override if re-targeting
indices = load_neuron_indices("./neurons")

model = AutoModelForCausalLM.from_pretrained(MODEL).cuda()
deact = NeuronDeactivator(model, config)
print(deact.deactivate(indices).summary())
deact.save_pretrained("./deactivated", tokenizer=tokenizer, indices=indices)
```

### 5. Per-task statistics and reports

```python
from critnet import CriticalNeuronConfig, NeuronStatistician

config = CriticalNeuronConfig.from_pretrained("./neurons")
stats = NeuronStatistician(model=model, config=config).analyze({
    "en": load_neuron_indices("./neurons/en"),
    "zh": load_neuron_indices("./neurons/zh"),
    "ar": load_neuron_indices("./neurons/ar"),
})
print(stats.summary())
stats.save_report("./statistics")
```

### 6. Custom architecture — opt in to embeddings, add `q_norm`/`k_norm`

```python
from critnet import CriticalNeuronConfig

config = CriticalNeuronConfig(
    row_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
    column_modules=["o_proj", "down_proj"],
    norm_modules=[
        "input_layernorm",
        "post_attention_layernorm",
        "q_norm", "k_norm",       # Qwen-3
        "norm",                   # final RMSNorm
    ],
    embedding_modules=["embed_tokens"],   # opt in
    gate_combines_with={"gate_proj": "up_proj"},
)
```

To also train the embedding deltas, pass `modules_to_skip=set()` to `get_neuron_model`. Note that this may break fused-kernel pipelines (e.g. Liger): see `DEFAULT_SKIP_MODULES` in `critnet/model.py`.
